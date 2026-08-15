"""run_fault_evaluation.py — does the predictor still work on faulted trajectories?

    python -m src.twin.run_fault_evaluation

Injecting faults is only useful if a model is then run against them. This scores the
delivered mechanism (frozen ARC space + target-domain ridge head) on the faulted
subset and compares, per fault class, against the same model on the unfaulted
trajectories.

WHAT IS AND IS NOT BEING CLAIMED
--------------------------------
The model is trained ONLY on unfaulted data and evaluated on faulted trajectories under
the same leave-one-environment-family-out isolation. So this measures robustness to an
off-nominal excursion it never saw — which is the operationally interesting question.

It does NOT measure fault diagnosis (the model has no fault-type output), and the faults
are observable-signature perturbations rather than physics-resolved simulations
(``SPEC.md`` §4). Degradation in accuracy here is evidence about graceful degradation,
nothing more.

EVIDENCE DOMAIN: SIMULATION ONLY, and perturbed simulation at that.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import paths as P
from src.features.frozen11 import CONTEXT_LENGTH, common_features
from src.features.protocol import (nested_family_lofo, solve_ridge_weighted,
                                   trajectory_equal_weights, weighted_intercept)
from src.features.space import REFERENCE_STEP_DAYS, WARN_SOH, load_frozen_space
from src.metrics import prognostic as PG

HORIZON_STEPS = 8
ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

EVIDENCE = ("SIMULATION_ONLY, and PERTURBED simulation: faults are "
            "observable-signature perturbations, not physics-resolved fault "
            "simulations (src/twin/SPEC.md section 4). Model trained on unfaulted "
            "data only; leave-one-environment-family-out isolation preserved.")


def windows(day, soh_obs, soh_true, tid, fam):
    """Anchor rows at the warning horizon, from one trajectory's channels."""
    rows = []
    for i in range(CONTEXT_LENGTH - 1, len(day) - HORIZON_STEPS):
        j = i + HORIZON_STEPS
        f = common_features(soh_obs[i - CONTEXT_LENGTH + 1: i + 1])
        rows.append([tid, fam, day[i], soh_obs[i],
                     soh_true[j] - soh_obs[i], soh_true[j], *f])
    return rows


def build(faulted: bool) -> pd.DataFrame:
    """Anchor table over the 60 faulted trajectories, faulted or baseline channels."""
    F = pd.read_csv(P.results("twin_faulted_subset.csv"))
    oc = "soh_observed_faulted" if faulted else "soh_observed"
    tc = "soh_true_faulted" if faulted else "soh_true"
    rows, ft = [], {}
    for tid, g in F.groupby("trajectory_id", sort=True):
        g = g.sort_values("reference_day")
        ft[tid] = g.fault_type.iloc[0]
        rows += windows(g.reference_day.to_numpy(dtype=float),
                        g[oc].to_numpy(dtype=float),
                        g[tc].to_numpy(dtype=float),
                        tid, g.environment_family_id.iloc[0])
    cols = (["trajectory_id", "environment_family_id", "anchor_day",
             "anchor_soh_observed", "y_delta", "target_soh_true"]
            + [f"f{j:02d}" for j in range(11)])
    D = pd.DataFrame(rows, columns=cols)
    D["fault_type"] = D.trajectory_id.map(ft)
    return D


def train_on_unfaulted(space):
    """One ridge head per held-out family, trained on UNFAULTED trajectories only."""
    l2 = pd.read_csv(P.L2_REFERENCE_CSV)
    sm = pd.read_csv(P.SPLIT_MANIFEST_CSV)
    fam_of = dict(zip(sm.trajectory_id, sm.environment_family_id))

    rows = []
    for tid, g in l2.groupby("trajectory_id", sort=True):
        g = g.sort_values("reference_day")
        rows += windows(g.reference_day.to_numpy(dtype=float),
                        g.soh_observed.to_numpy(dtype=float),
                        g.soh_true.to_numpy(dtype=float),
                        tid, fam_of[tid])
    cols = (["trajectory_id", "environment_family_id", "anchor_day",
             "anchor_soh_observed", "y_delta", "target_soh_true"]
            + [f"f{j:02d}" for j in range(11)])
    T = pd.DataFrame(rows, columns=cols)

    Z = space.apply(T[[f"f{j:02d}" for j in range(11)]].to_numpy(dtype=float))
    y = T.y_delta.to_numpy(dtype=float)
    fam = T.environment_family_id.to_numpy()
    tid = T.trajectory_id.to_numpy()
    families = sorted(set(fam.tolist()))

    heads = {}
    for held, inner in nested_family_lofo(families):
        tr = fam != held
        best_a, best_e = None, np.inf
        for a in ALPHA_GRID:
            errs = []
            for f in inner:
                itr, ite = tr & (fam != f), fam == f
                wi = trajectory_equal_weights(tid, fam, itr)
                c = solve_ridge_weighted(Z[itr], y[itr], wi[itr], a)
                b = weighted_intercept(Z[itr], y[itr], wi[itr], c)
                errs.append(float(np.abs(Z[ite] @ c + b - y[ite]).mean()))
            if errs and float(np.mean(errs)) < best_e - 1e-15:
                best_a, best_e = float(a), float(np.mean(errs))
        w = trajectory_equal_weights(tid, fam, tr)
        c = solve_ridge_weighted(Z[tr], y[tr], w[tr], best_a)
        b = weighted_intercept(Z[tr], y[tr], w[tr], c)
        heads[held] = (c, b, best_a)
    return heads


def score(D: pd.DataFrame, space, heads) -> pd.DataFrame:
    Z = space.apply(D[[f"f{j:02d}" for j in range(11)]].to_numpy(dtype=float))
    pred_delta = np.full(len(D), np.nan)
    fam = D.environment_family_id.to_numpy()
    for held, (c, b, _) in heads.items():
        m = fam == held
        if m.any():
            pred_delta[m] = Z[m] @ c + b
    out = D.copy()
    out["pred_delta"] = pred_delta
    out["pred_soh"] = out.anchor_soh_observed + pred_delta
    out["abs_err_delta"] = (out.pred_delta - out.y_delta).abs()
    out["abs_err_soh"] = (out.pred_soh - out.target_soh_true).abs()
    return out


def alarm_stats(scored: pd.DataFrame, truth_col: str) -> dict:
    """Warning lead time on this subset, using each trajectory's own truth channel."""
    recs = []
    for tid, g in scored.groupby("trajectory_id", sort=True):
        g = g.sort_values("anchor_day")
        d = g.anchor_day.to_numpy(dtype=float)
        t_cross = PG.true_threshold_crossing_day(d, g[truth_col].to_numpy(dtype=float),
                                                 WARN_SOH)
        t_alarm = PG.first_alarm_day(d, g.pred_soh.to_numpy(dtype=float), WARN_SOH)
        crossed, alarmed = not np.isnan(t_cross), not np.isnan(t_alarm)
        if crossed and alarmed:
            dt = t_cross - t_alarm
            oc = "TRUE_POSITIVE" if dt >= 0 else "LATE"
        elif crossed:
            dt, oc = float("nan"), "MISSED"
        elif alarmed:
            dt, oc = float("nan"), "FALSE_POSITIVE"
        else:
            dt, oc = float("nan"), "TRUE_NEGATIVE"
        recs.append({"trajectory_id": tid, "fault_type": g.fault_type.iloc[0],
                     "dt_warn_days": dt, "outcome": oc})
    R = pd.DataFrame(recs)
    tp = R[R.outcome == "TRUE_POSITIVE"]
    return {"n": len(R), "n_true_positive": len(tp),
            "n_missed": int((R.outcome == "MISSED").sum()),
            "n_late": int((R.outcome == "LATE").sum()),
            "dt_warn_mean_days": float(tp.dt_warn_days.mean()) if len(tp) else float("nan"),
            "dt_warn_median_days": float(tp.dt_warn_days.median()) if len(tp) else float("nan"),
            "by_fault": R.to_dict(orient="records")}


def main() -> int:
    space = load_frozen_space()
    print("=" * 76)
    print("T5(C)  predictor evaluated on the FAULT-INJECTED subset")
    print("=" * 76)
    print("  training data : UNFAULTED trajectories only")
    print("  isolation     : leave-one-environment-family-out, preserved")
    print(f"  horizon       : {HORIZON_STEPS} steps "
          f"({HORIZON_STEPS * REFERENCE_STEP_DAYS} days)")
    print()

    heads = train_on_unfaulted(space)
    base = score(build(faulted=False), space, heads)
    falt = score(build(faulted=True), space, heads)

    print(f"  {'fault_type':<18s} {'n_rows':>7s} {'MAE base':>10s} {'MAE fault':>10s} "
          f"{'ratio':>7s}")
    rows = []
    for ft in sorted(falt.fault_type.unique()):
        b = base[base.fault_type == ft]
        f = falt[falt.fault_type == ft]
        mb, mf = float(b.abs_err_soh.mean()), float(f.abs_err_soh.mean())
        rows.append({"fault_type": ft, "n_rows": int(len(f)),
                     "mae_absolute_soh_baseline": mb,
                     "mae_absolute_soh_faulted": mf,
                     "degradation_ratio": mf / mb if mb else float("nan")})
        print(f"  {ft:<18s} {len(f):7d} {mb:10.6f} {mf:10.6f} {mf / mb:7.2f}x")

    mb_all = float(base.abs_err_soh.mean())
    mf_all = float(falt.abs_err_soh.mean())
    print(f"  {'ALL':<18s} {len(falt):7d} {mb_all:10.6f} {mf_all:10.6f} "
          f"{mf_all / mb_all:7.2f}x")
    print()

    a_base = alarm_stats(base, "target_soh_true")
    a_falt = alarm_stats(falt, "target_soh_true")
    print("  warning behaviour on the same trajectories:")
    print(f"    baseline : {a_base['n_true_positive']}/{a_base['n']} early warnings, "
          f"mean lead {a_base['dt_warn_mean_days']:.1f} d, "
          f"missed {a_base['n_missed']}, late {a_base['n_late']}")
    print(f"    faulted  : {a_falt['n_true_positive']}/{a_falt['n']} early warnings, "
          f"mean lead {a_falt['dt_warn_mean_days']:.1f} d, "
          f"missed {a_falt['n_missed']}, late {a_falt['n_late']}")
    print()
    print("  READING THIS: the model was never shown a fault, so a larger error on")
    print("  faulted trajectories is expected. What matters is that it still alarms")
    print("  before the (now earlier) threshold crossing rather than going silent.")
    print("  This is graceful-degradation evidence, NOT fault diagnosis: the model")
    print("  has no fault-type output.")

    pd.DataFrame(rows).to_csv(P.results("twin_fault_evaluation.csv"), index=False)
    P.results("twin_fault_evaluation.json").write_text(json.dumps({
        "mae_absolute_soh_baseline_all": mb_all,
        "mae_absolute_soh_faulted_all": mf_all,
        "degradation_ratio_all": mf_all / mb_all if mb_all else None,
        "per_fault": rows,
        "alarm_baseline": {k: v for k, v in a_base.items() if k != "by_fault"},
        "alarm_faulted": {k: v for k, v in a_falt.items() if k != "by_fault"},
        "training_data": "unfaulted trajectories only",
        "measures": "graceful degradation under unseen off-nominal excursions",
        "does_not_measure": ["fault diagnosis (no fault-type output exists)",
                             "physics-resolved fault propagation"],
        "evidence_domain": EVIDENCE,
    }, indent=2), encoding="utf-8")

    print()
    print(f"  wrote {P.results('twin_fault_evaluation.csv').name}, "
          f"{P.results('twin_fault_evaluation.json').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
