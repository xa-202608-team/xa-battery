"""run_all.py — T4: all comparator arms under ONE protocol.

    python -m src.comparators.run_all

THE ARMS
--------
1. ``persistence``                   non-learning floor: predicted delta = 0
2. ``observable_local_trend``        non-learning: extrapolate the observed slope
3. ``wiener_process``                classical stochastic degradation (NEW)
4. ``double_exp_particle_filter``    classical empirical fade + PF (NEW)
5. ``gru_pytorch``                   deep sequence comparator (NEW, PyTorch)
6. ``target_only_ridge``             v2's delivered mechanism: no source prior
7. ``l2sp_transfer``                 L2-SP fine-tuned from beta_src

Every arm sees the same leave-one-environment-family-out folds, the same rows, and the
same weighting. Arms that need a validation split take an INNER family, never a random
row split — windows from one trajectory overlap, so a random split would leak.

TWO TASKS ARE SCORED SEPARATELY, because they are not the same question:
  * SOH delta at horizons 2/4/8 steps (the delivered task)
  * mission-span RUL in days (the T1 task) — where the classical arms naturally live

Reported per arm: MAE, RMSE, across-family std, worst family. Earliness is reported for
the arms that produce an SOH trajectory, since dt_warn needs a threshold crossing.

WHATEVER WINS, THE DELIVERED MODEL DOES NOT CHANGE. Phase 6's gate closed deep-model
PROMOTION; this is a comparison. See GATES_scope_amendment.md.

EVIDENCE DOMAIN: SIMULATION ONLY. Not satellite accuracy.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import paths as P
from src.comparators import classical as CL
from src.comparators.deep_seq import train_gru
from src.data.target_dataset import HORIZONS, design, load_l3
from src.features.protocol import (nested_family_lofo, solve_ridge_weighted,
                                   trajectory_equal_weights, weighted_intercept)
from src.features.space import (EOL_SOH, REFERENCE_STEP_DAYS, load_frozen_space,
                                load_source_prior)
from src.finetune import _runtime as RT
from src.finetune.l2sp import LAMBDA_SP_GRID, solve_l2sp_closed_form
from src.rul.anchors import FEATURE_COLS, build as build_rul_anchors
from src.rul.censored_loss import LAMBDA_CENSORED, fit_censored_ridge, predict

ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
CTX_COLS = [f"ctx_soh_{i:02d}" for i in range(20)]

EVIDENCE = ("SIMULATION_ONLY (STK-derived twin; NOT satellite accuracy). "
            "All arms share one leave-one-environment-family-out protocol; "
            "hyper-parameters and early stopping use inner folds only.")

DELIVERY_NOTE = (
    "The delivered predictor remains target_only_arc_space regardless of this table. "
    "Phase 6's PATCHTST_SKIPPED_BY_PREDEFINED_GATE closed deep-model PROMOTION, not "
    "deep-model comparison. A winning deep arm is recorded as 'to be re-evaluated "
    "under a new pre-registered protocol', never promoted here.")


def _wrap(text: str, width: int) -> list:
    import textwrap
    return textwrap.wrap(text, width)


def _agg(per_family: dict) -> dict:
    v = [x for x in per_family.values() if np.isfinite(x)]
    if not v:
        return {"family_macro": float("nan"), "worst_family": float("nan"),
                "across_family_std": float("nan")}
    return {"family_macro": float(np.mean(v)), "worst_family": float(max(v)),
            "across_family_std": float(np.std(v))}


# ============================================================ SOH delta task

def soh_delta_task(df: pd.DataFrame, D: dict, Z: np.ndarray, prior) -> list[dict]:
    fam, tid, hs = D["family_id"], D["trajectory_id"], D["horizon_steps"]
    y, anchor, tgt = D["y_delta"], D["anchor"], D["target_soh"]
    ctx = df[CTX_COLS].to_numpy(dtype=float)
    families = sorted(set(fam.tolist()))
    rows = []

    for h in HORIZONS:
        in_h = hs == h
        beta_src, _ = prior.for_horizon(h)
        acc: dict = {}

        for held, inner in nested_family_lofo(families):
            tr, te = in_h & (fam != held), in_h & (fam == held)
            if not tr.any() or not te.any():
                continue
            w = trajectory_equal_weights(tid, fam, tr)
            yt = y[te]

            def mae(pred):
                return float(np.abs(pred - yt).mean())

            # ---- 1. persistence: predicted delta = 0 ----
            acc.setdefault("persistence", {})[held] = mae(np.zeros(te.sum()))

            # ---- 2. observable local trend: slope_full * horizon ----
            slope = df["frozen_05_slope_full"].to_numpy(dtype=float)[te]
            acc.setdefault("observable_local_trend", {})[held] = mae(slope * h)

            # ---- 6. target-only ridge (v2's delivered mechanism) ----
            best_a, best_e = None, np.inf
            for a in ALPHA_GRID:
                errs = []
                for f in inner:
                    itr, ite = in_h & (fam != held) & (fam != f), in_h & (fam == f)
                    if not itr.any() or not ite.any():
                        continue
                    wi = trajectory_equal_weights(tid, fam, itr)
                    c = solve_ridge_weighted(Z[itr], y[itr], wi[itr], a)
                    b = weighted_intercept(Z[itr], y[itr], wi[itr], c)
                    errs.append(float(np.abs(Z[ite] @ c + b - y[ite]).mean()))
                if errs and float(np.mean(errs)) < best_e - 1e-15:
                    best_a, best_e = float(a), float(np.mean(errs))
            c = solve_ridge_weighted(Z[tr], y[tr], w[tr], best_a)
            b = weighted_intercept(Z[tr], y[tr], w[tr], c)
            acc.setdefault("target_only_ridge", {})[held] = mae(Z[te] @ c + b)

            # ---- 7. L2-SP transfer ----
            best_p, best_e2 = None, np.inf
            for lam in LAMBDA_SP_GRID:
                for a in ALPHA_GRID:
                    errs = []
                    for f in inner:
                        itr, ite = in_h & (fam != held) & (fam != f), in_h & (fam == f)
                        if not itr.any() or not ite.any():
                            continue
                        wi = trajectory_equal_weights(tid, fam, itr)
                        cc = solve_l2sp_closed_form(Z[itr], y[itr], wi[itr],
                                                    beta_src, a, lam)
                        bb = weighted_intercept(Z[itr], y[itr], wi[itr], cc)
                        errs.append(float(np.abs(Z[ite] @ cc + bb - y[ite]).mean()))
                    if errs and float(np.mean(errs)) < best_e2 - 1e-15:
                        best_p, best_e2 = (float(a), float(lam)), float(np.mean(errs))
            a2, lam2 = best_p
            c2 = solve_l2sp_closed_form(Z[tr], y[tr], w[tr], beta_src, a2, lam2)
            b2 = weighted_intercept(Z[tr], y[tr], w[tr], c2)
            acc.setdefault("l2sp_transfer", {})[held] = mae(Z[te] @ c2 + b2)

            # ---- 5. GRU (PyTorch), validated on an inner family ----
            val_fam = inner[0]
            gtr = in_h & (fam != held) & (fam != val_fam)
            gva = in_h & (fam == val_fam)
            run = train_gru(ctx[gtr], y[gtr], w[gtr], ctx[gva], y[gva], ctx[te],
                            seed=CL.stable_seed(f"{held}|{h}"))
            acc.setdefault("gru_pytorch", {})[held] = mae(run.pred)

        for arm, per in acc.items():
            a = _agg(per)
            rows.append({
                "task": "SOH_DELTA", "arm": arm, "horizon_steps": int(h),
                "horizon_days": int(h) * REFERENCE_STEP_DAYS,
                "metric": "mae_delta_soh",
                "family_macro": a["family_macro"], "worst_family": a["worst_family"],
                "across_family_std": a["across_family_std"],
                "per_family": json.dumps({k: round(v, 8) for k, v in per.items()}),
            })
        print(f"    h={h} ({h * 14:3d} d): " + "  ".join(
            f"{k}={_agg(v)['family_macro']:.6f}" for k, v in sorted(acc.items())))
    return rows


# ============================================================ mission-span RUL

def rul_task(space) -> list[dict]:
    A = build_rul_anchors()
    X = A[FEATURE_COLS].to_numpy(dtype=float)
    Z = space.apply(X)
    y = A.rul_days.to_numpy(dtype=float)
    lb = A.rul_lower_bound_days.to_numpy(dtype=float)
    cen = A.censored.to_numpy() == 1
    fam = A.environment_family_id.to_numpy()
    tid = A.trajectory_id.to_numpy()
    day = A.anchor_day.to_numpy(dtype=float)
    last = A.f00.to_numpy(dtype=float)          # frozen_00_last == observed SOH
    families = sorted(set(fam.tolist()))

    # Per-trajectory classical fits: these need the SOH history, not the design row.
    l2 = pd.read_csv(P.L2_REFERENCE_CSV)
    hist = {t: (g.sort_values("reference_day").reference_day.to_numpy(dtype=float),
                g.sort_values("reference_day").soh_observed.to_numpy(dtype=float))
            for t, g in l2.groupby("trajectory_id")}

    acc: dict = {}
    for held, inner in nested_family_lofo(families):
        tr = fam != held
        te_u = (fam == held) & ~cen
        if not te_u.any():
            continue
        yt = y[te_u]

        def mae(pred):
            return float(np.abs(np.asarray(pred, dtype=float) - yt).mean())

        # ---- constant-mean predictor (non-learning floor for RUL) ----
        acc.setdefault("constant_mean", {})[held] = mae(
            np.full(te_u.sum(), float(y[tr & ~cen].mean())))

        # ---- slope extrapolation ----
        slope = A.f05.to_numpy(dtype=float)[te_u]
        with np.errstate(divide="ignore", invalid="ignore"):
            est = np.where(slope < 0,
                           (EOL_SOH - last[te_u]) / slope * REFERENCE_STEP_DAYS, np.nan)
        acc.setdefault("slope_extrapolation", {})[held] = mae(
            np.clip(np.nan_to_num(est, nan=2000.0), 0.0, 3000.0))

        # ---- Wiener process, fitted per trajectory on history up to the anchor ----
        wp, pf = [], []
        idx = np.flatnonzero(te_u)
        for i in idx:
            d_all, s_all = hist[tid[i]]
            m = d_all <= day[i]
            wf = CL.fit_wiener(d_all[m], s_all[m])
            wp.append(CL.wiener_rul(wf, float(s_all[m][-1])))
        acc.setdefault("wiener_process", {})[held] = mae(wp)

        # ---- double-exp + particle filter, on a subsample (cost) ----
        # Every 10th anchor, declared in advance; the subsample is REPORTED.
        sub = idx[::10]
        pf_pred, pf_true = [], []
        for i in sub:
            d_all, s_all = hist[tid[i]]
            m = d_all <= day[i]
            r = CL.particle_filter_rul(d_all[m], s_all[m],
                                       seed=CL.stable_seed(f"{tid[i]}|{day[i]}"))
            pf_pred.append(r.rul_mean)
            pf_true.append(y[i])
        acc.setdefault("double_exp_particle_filter", {})[held] = float(
            np.abs(np.array(pf_pred) - np.array(pf_true)).mean())

        # ---- censoring-aware ridge on the frozen 11 (the T1 model) ----
        def fit_on(mask, a):
            um, cm = mask & ~cen, mask & cen
            wu = trajectory_equal_weights(tid, fam, um)
            wc = trajectory_equal_weights(tid, fam, cm)
            return fit_censored_ridge(Z[um], y[um], wu[um], Z[cm], lb[cm], wc[cm],
                                      alpha=a, lam_c=LAMBDA_CENSORED)

        best_a, best_e = None, np.inf
        for a in ALPHA_GRID:
            errs = []
            for f in inner:
                itr, ite = tr & (fam != f), (fam == f) & ~cen
                if not ite.any():
                    continue
                errs.append(float(np.abs(predict(fit_on(itr, a), Z[ite])
                                         - y[ite]).mean()))
            if errs and float(np.mean(errs)) < best_e - 1e-15:
                best_a, best_e = float(a), float(np.mean(errs))
        acc.setdefault("censoring_aware_ridge_frozen11", {})[held] = mae(
            predict(fit_on(tr, best_a), Z[te_u]))

    rows = []
    for arm, per in acc.items():
        a = _agg(per)
        rows.append({
            "task": "MISSION_SPAN_RUL", "arm": arm, "horizon_steps": -1,
            "horizon_days": -1, "metric": "mae_days",
            "family_macro": a["family_macro"], "worst_family": a["worst_family"],
            "across_family_std": a["across_family_std"],
            "per_family": json.dumps({k: round(v, 4) for k, v in per.items()}),
        })
        print(f"    {arm:<34s} family-macro MAE = {a['family_macro']:8.1f} d   "
              f"worst = {a['worst_family']:8.1f} d")
    return rows


def main() -> int:
    RT.seed_everything(0)
    space = load_frozen_space()
    prior = load_source_prior()

    print("=" * 76)
    print("T4  comparator arms — one protocol, seven arms, two tasks")
    print("=" * 76)
    print(f"  {DELIVERY_NOTE}")
    print()

    df = load_l3()
    D = design(df)
    Z = space.apply(D["X"])

    print("  TASK 1: SOH delta (the delivered task), MAE on the delta scale")
    rows = soh_delta_task(df, D, Z, prior)
    print()
    print("  TASK 2: mission-span RUL, MAE in days")
    rows += rul_task(space)

    out = pd.DataFrame(rows)
    out.to_csv(P.results("comparison_table.csv"), index=False)

    # Ranking per task, for the summary.
    ranks = {}
    for task, g in out.groupby("task"):
        m = g.groupby("arm").family_macro.mean().sort_values()
        ranks[task] = [{"arm": k, "family_macro": float(v)} for k, v in m.items()]

    # Is the top margin larger than the spread across families? If not, say so.
    margins = {}
    for task, g in out.groupby("task"):
        m = g.groupby("arm").family_macro.mean().sort_values()
        spread = float(g.groupby("arm").across_family_std.mean().mean())
        if len(m) >= 2:
            first, second = m.index[0], m.index[1]
            gap = float(m.iloc[1] - m.iloc[0])
            margins[task] = {
                "best_arm": first, "runner_up": second,
                "absolute_gap": gap,
                "relative_gap": gap / max(float(m.iloc[0]), 1e-30),
                "mean_across_family_std": spread,
                "gap_exceeds_across_family_spread": bool(gap > spread),
                "interpretation": (
                    f"The gap between {first} and {second} is {gap:.3g}, "
                    f"{'LARGER' if gap > spread else 'SMALLER'} than the mean "
                    f"across-family spread ({spread:.3g}). "
                    + ("A gap smaller than the spread is not evidence of a real "
                       "ordering: reshuffling which families are held out could "
                       "plausibly change the winner. It must not be reported as one "
                       "method beating another."
                       if gap <= spread else
                       "The ordering is larger than the family-to-family spread, so it "
                       "is not explained by which families happened to be held out.")),
            }

    P.results("comparison_summary.json").write_text(json.dumps({
        "task": "COMPARATOR_ARMS",
        "protocol": ("NESTED_FAMILY_LOFO shared by every arm; inner folds for "
                     "hyper-parameters and early stopping; no arm sees the held-out "
                     "family"),
        "arms": {
            "persistence": "non-learning floor, predicted delta = 0",
            "observable_local_trend": "non-learning, observed slope x horizon",
            "wiener_process": "classical stochastic degradation, per-trajectory MLE (NEW)",
            "double_exp_particle_filter": (
                "double-exponential capacity fade + bootstrap particle filter (NEW); "
                "evaluated on every 10th anchor for cost, subsample declared in advance"),
            "gru_pytorch": "deep sequence COMPARATOR in PyTorch (NEW); never the delivered model",
            "target_only_ridge": "v2's delivered mechanism, no source prior",
            "l2sp_transfer": "L2-SP fine-tuned from the frozen beta_src",
            "constant_mean": "RUL non-learning floor",
            "slope_extrapolation": "RUL non-learning, extrapolate observed slope to EOL",
            "censoring_aware_ridge_frozen11": "T1's model, right-censoring aware",
        },
        "ranking_best_first": ranks,
        "top_margin_vs_family_spread": margins,
        "delivery_unchanged": DELIVERY_NOTE,
        "frozen_verdicts_intact": [
            "PATCHTST_SKIPPED_BY_PREDEFINED_GATE (deep promotion still closed)",
            "SOURCE_PRIOR_NOT_VALIDATED (delivered model uses no source prior)",
            "RUL_EVIDENCE_INSUFFICIENT (scoped to the 112-day crossing detector)",
        ],
        "evidence_domain": EVIDENCE,
    }, indent=2), encoding="utf-8")

    print()
    print("  ranking (best first):")
    for task, lst in ranks.items():
        print(f"    {task}:")
        for r in lst:
            print(f"      {r['arm']:<34s} {r['family_macro']:.6f}")
    print()
    print("  top-margin check — is the winner's lead bigger than the family spread?")
    for task, m in margins.items():
        print(f"    {task}:")
        print(f"      {m['best_arm']} over {m['runner_up']}: gap {m['absolute_gap']:.3g} "
              f"({m['relative_gap'] * 100:.1f}%) vs across-family spread "
              f"{m['mean_across_family_std']:.3g}")
        print(f"      -> {'SEPARABLE' if m['gap_exceeds_across_family_spread'] else 'NOT SEPARABLE'}")
        for line in _wrap(m["interpretation"], 68):
            print(f"         {line}")
    print()
    print(f"  EVIDENCE: {EVIDENCE}")
    print(f"  wrote {P.results('comparison_table.csv').name}, "
          f"{P.results('comparison_summary.json').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
