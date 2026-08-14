"""run_mission_span.py — T1: mission-span RUL under the frozen protocol.

    python -m src.rul.run_mission_span

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
IS:     a mission-span RUL regression, ``RUL(t) = t_EOL - t_anchor``, evaluated by
        leave-one-environment-family-out with inner-fold hyper-parameter selection.
IS NOT: a re-opening of the frozen ``RUL_EVIDENCE_INSUFFICIENT`` verdict, which is
        scoped to the 112-day finite-horizon crossing detector and stands unchanged.
        Both statements go in the report, side by side. See GATES_scope_amendment.md.

UPGRADES OVER THE FEASIBILITY PROBE
-----------------------------------
1. Right-censoring: the 28 never-reached-EOL trajectories are KEPT, with a one-sided
   loss on their observed follow-up (the probe dropped them, biasing toward fast cells).
2. Frozen ARC clip/scaler, applied not refitted — no per-fold standardisation.
3. Trajectory-equal + family-balanced weights, so a 137-anchor trajectory cannot
   outvote a 24-anchor one.
4. Family-balanced split-conformal intervals.
5. Baselines: slope extrapolation and a constant-mean predictor.

EVIDENCE DOMAIN: SIMULATION ONLY (STK-derived twin). These MAE values are NOT
satellite accuracy. Every quoted number carries the family-LOFO protocol and the
right-censoring treatment with it.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import paths as P
from src.features.protocol import (family_macro, nested_family_lofo,
                                   trajectory_equal_weights)
from src.features.space import EOL_SOH, REFERENCE_STEP_DAYS, load_frozen_space
from src.rul import conformal as CF
from src.rul.anchors import FEATURE_COLS, build, summarise
from src.rul.censored_loss import (LAMBDA_CENSORED, fit_censored_ridge, predict)

#: Locked before any fit, same ladder the probe used.
ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
COVERAGE = 0.90

EVIDENCE = ("SIMULATION_ONLY (STK-derived twin; NOT satellite accuracy). "
            "Protocol: leave-one-environment-family-out, alpha chosen on inner folds "
            "only. Right-censored trajectories retained via a one-sided loss on the "
            "observed follow-up.")


def _fit_and_predict(D: pd.DataFrame, Z: np.ndarray, tr: np.ndarray,
                     te: np.ndarray, alpha: float) -> np.ndarray:
    """Fit on ``tr`` (censoring-aware, weighted), predict on ``te``."""
    cen = D.censored.to_numpy() == 1
    tid = D.trajectory_id.to_numpy()
    fam = D.environment_family_id.to_numpy()
    y = D.rul_days.to_numpy(dtype=float)
    lb = D.rul_lower_bound_days.to_numpy(dtype=float)

    # Weights are computed within each stratum (uncensored / censored) separately so
    # that trajectory-equal weighting holds inside the group it is applied to.
    u_mask, c_mask = tr & ~cen, tr & cen
    wu = trajectory_equal_weights(tid, fam, u_mask)
    wc = trajectory_equal_weights(tid, fam, c_mask)

    fit = fit_censored_ridge(
        Z[u_mask], y[u_mask], wu[u_mask],
        Z[c_mask], lb[c_mask], wc[c_mask],
        alpha=alpha, lam_c=LAMBDA_CENSORED)
    return predict(fit, Z[te]), fit


def slope_extrapolation(X: np.ndarray) -> np.ndarray:
    """Non-learning comparator: extend the observed slope until it reaches EOL.

    ``f00 = last``, ``f05 = slope_full`` (per 14-day reference step). Same closed
    form the probe used, so the comparison is like-for-like.
    """
    last, slope = X[:, 0], X[:, 5]
    with np.errstate(divide="ignore", invalid="ignore"):
        est = np.where(slope < 0, (EOL_SOH - last) / slope * REFERENCE_STEP_DAYS, np.nan)
    return np.clip(np.nan_to_num(est, nan=2000.0), 0.0, 3000.0)


def main() -> int:
    space = load_frozen_space()
    D = build()
    stats = summarise(D)

    X = D[FEATURE_COLS].to_numpy(dtype=float)
    Z = space.apply(X)                      # FROZEN clip/scaler — applied, not refitted
    y = D.rul_days.to_numpy(dtype=float)
    lb = D.rul_lower_bound_days.to_numpy(dtype=float)
    cen = D.censored.to_numpy() == 1
    fam = D.environment_family_id.to_numpy()
    families = sorted(set(fam.tolist()))

    print("=" * 76)
    print("T1  mission-span RUL   RUL(t) = t_EOL - t_anchor")
    print("=" * 76)
    print(f"  frozen space        : {space.name} ({space.source_model_id}), applied not refitted")
    print(f"  anchors total       : {stats['anchors_total']}  "
          f"(uncensored {stats['anchors_uncensored']}, censored {stats['anchors_censored']})")
    print(f"  trajectories        : {stats['trajectories_total']}  "
          f"(uncensored {stats['trajectories_uncensored']}, right-censored "
          f"{stats['trajectories_censored']} RETAINED via one-sided loss)")
    print(f"  RUL label mean/std  : {stats['rul_mean_days']:.1f} / {stats['rul_std_days']:.1f} days")
    print(f"  RUL label min/max   : {stats['rul_min_days']:.1f} / {stats['rul_max_days']:.1f} days")
    print(f"  DEGENERACY CHECK    : std = {stats['rul_std_days']:.4f} days ({stats['degeneracy']})")
    print()

    rows, per_fam_model, per_fam_slope, per_fam_const = [], {}, {}, {}
    resid_all, resid_fam = [], []

    for held, inner in nested_family_lofo(families):
        tr, te = fam != held, fam == held

        # ---- inner-fold alpha selection: never sees the held-out family ----
        best_alpha, best_err = None, np.inf
        for a in ALPHA_GRID:
            errs = []
            for f in inner:
                itr, ite = tr & (fam != f), fam == f
                ite_u = ite & ~cen
                if not ite_u.any():
                    continue
                p, _ = _fit_and_predict(D, Z, itr, ite_u, a)
                errs.append(float(np.abs(p - y[ite_u]).mean()))
            if errs and float(np.mean(errs)) < best_err - 1e-15:
                best_alpha, best_err = float(a), float(np.mean(errs))

        # ---- outer evaluation on the held-out family, uncensored anchors ----
        te_u = te & ~cen
        te_c = te & cen
        p_u, fit = _fit_and_predict(D, Z, tr, te_u, best_alpha)
        mae = float(np.abs(p_u - y[te_u]).mean())
        rmse = float(np.sqrt(((p_u - y[te_u]) ** 2).mean()))

        base_s = float(np.abs(slope_extrapolation(X[te_u]) - y[te_u]).mean())
        # Constant-mean predictor: fitted on TRAIN uncensored labels only.
        const = float(y[tr & ~cen].mean())
        base_c = float(np.abs(const - y[te_u]).mean())

        p_c = (predict(fit, Z[te_c]) if te_c.any()
               else np.zeros(0))
        cen_chk = (CF.censored_one_sided_check(lb[te_c], p_c, 0.0)
                   if te_c.any() else {"censored_consistent_fraction": float("nan"),
                                       "n_censored": 0, "n_censored_violations": 0})

        per_fam_model[held], per_fam_slope[held], per_fam_const[held] = mae, base_s, base_c
        resid_all.append(p_u - y[te_u])
        resid_fam.append(fam[te_u])

        rows.append({
            "environment_family_id": held, "alpha_selected_on_inner_folds": best_alpha,
            "selector_saw_outer_test": False,
            "n_anchors_uncensored": int(te_u.sum()), "n_anchors_censored": int(te_c.sum()),
            "n_train_censored_active": fit.n_censored_active,
            "active_set_converged": fit.converged, "active_set_iters": fit.n_iters,
            "mae_days": mae, "rmse_days": rmse,
            "mae_days_slope_extrapolation": base_s,
            "mae_days_constant_mean": base_c,
            "censored_consistent_fraction_pointpred": cen_chk["censored_consistent_fraction"],
            "n_censored_violations_pointpred": cen_chk["n_censored_violations"],
        })
        print(f"  {held:<24s} alpha={best_alpha:<7g} n_unc={int(te_u.sum()):5d} "
              f"n_cen={int(te_c.sum()):5d}  ridge={mae:7.1f} d  "
              f"slope={base_s:7.1f} d  const={base_c:7.1f} d")

    # ---- family-balanced CROSS-FIT conformal ----
    # For each family, the half-width is calibrated on the OTHER families' LOFO
    # residuals and applied to this family's. Calibrating and evaluating on the same
    # residuals would be an in-sample width, which is not a coverage claim.
    resid = np.concatenate(resid_all)
    rfam = np.concatenate(resid_fam)
    cross_cov, cross_hw = {}, {}
    for held in families:
        cal = rfam != held
        ev = rfam == held
        if not cal.any() or not ev.any():
            continue
        hw, _ = CF.family_balanced_quantile(resid[cal], rfam[cal], COVERAGE)
        cross_hw[held] = hw
        cross_cov[held] = float(np.mean(np.abs(resid[ev]) <= hw))

    cf = CF.calibrate(resid, rfam, COVERAGE)   # reported width, all-family reference
    crossfit_coverage = float(np.mean(list(cross_cov.values())))
    worst_family_coverage = float(min(cross_cov.values()))

    macro = family_macro(per_fam_model)
    macro_s = family_macro(per_fam_slope)
    macro_c = family_macro(per_fam_const)
    worst = max(per_fam_model.values())

    print()
    print(f"  family-macro MAE, censoring-aware ridge on frozen 11 features : {macro:7.1f} days")
    print(f"  family-macro MAE, slope extrapolation                        : {macro_s:7.1f} days")
    print(f"  family-macro MAE, constant-mean predictor                    : {macro_c:7.1f} days")
    print(f"  worst family MAE                                             : {worst:7.1f} days")
    print(f"  improvement vs slope / vs constant                           : "
          f"{macro_s / macro:.2f}x / {macro_c / macro:.2f}x")
    print()
    print(f"  conformal {int(COVERAGE * 100)}% half-width (family-balanced, MARGINAL "
          f"per anchor) : +/-{cf.half_width_days:.1f} days")
    print(f"  CROSS-FIT coverage (width from other families, applied to held-out):")
    for f in sorted(cross_cov):
        print(f"      {f:<24s} half-width +/-{cross_hw[f]:6.1f} d   coverage {cross_cov[f]:.4f}")
    print(f"  cross-fit mean coverage                                      : "
          f"{crossfit_coverage:.4f}   (target {COVERAGE:.2f})")
    print(f"  cross-fit worst-family coverage                              : "
          f"{worst_family_coverage:.4f}")
    print()
    print(f"  EVIDENCE: {EVIDENCE}")

    # ------------------------------------------------------------------ outputs
    df = pd.DataFrame(rows)
    df.to_csv(P.results("rul_mission_span_by_family.csv"), index=False)

    summary = pd.DataFrame([{
        "task": "MISSION_SPAN_RUL_REGRESSION",
        "label_definition": "RUL(t) = t_EOL(trajectory) - t_anchor  (days, no horizon cap)",
        "distinct_from_frozen_verdict": (
            "RUL_EVIDENCE_INSUFFICIENT is scoped to the 112-day finite-horizon "
            "crossing detector and is NOT re-opened here"),
        "protocol": "NESTED_FAMILY_LOFO, alpha on inner folds only",
        "censoring_treatment": (
            f"28 right-censored trajectories RETAINED; one-sided squared hinge on "
            f"observed follow-up, lambda_censored={LAMBDA_CENSORED}"),
        "frozen_space": f"{space.name}/{space.source_model_id} applied, never refitted",
        "weighting": "trajectory-equal then family-balanced",
        **stats,
        "family_macro_mae_days": macro,
        "family_macro_mae_days_slope_extrapolation": macro_s,
        "family_macro_mae_days_constant_mean": macro_c,
        "worst_family_mae_days": worst,
        "across_family_std_days": float(np.std(list(per_fam_model.values()))),
        "conformal_coverage_target": COVERAGE,
        "conformal_half_width_days": cf.half_width_days,
        "conformal_method": cf.method,
        "conformal_marginal_only": True,
        "conformal_simultaneous_coverage_claimed": False,
        "conformal_crossfit_mean_coverage": crossfit_coverage,
        "conformal_crossfit_worst_family_coverage": worst_family_coverage,
        "evidence_domain": EVIDENCE,
    }])
    summary.to_csv(P.results("rul_mission_span.csv"), index=False)

    P.results("rul_mission_span_conformal.json").write_text(json.dumps({
        "coverage_target": cf.coverage_target,
        "half_width_days": cf.half_width_days,
        "method": cf.method,
        "per_family_half_width_days": cf.per_family_half_width,
        "crossfit_half_width_days": cross_hw,
        "crossfit_coverage_by_family": cross_cov,
        "crossfit_mean_coverage": crossfit_coverage,
        "crossfit_worst_family_coverage": worst_family_coverage,
        "marginal_only": True,
        "simultaneous_coverage_claimed": False,
        "calibrated_on": "uncensored anchors only (a censored anchor has no point label)",
        "crossfit_rule": ("each family's half-width comes from the OTHER families' "
                          "LOFO residuals; no family calibrates on itself"),
        "evidence_domain": EVIDENCE,
    }, indent=2), encoding="utf-8")

    print()
    print(f"  wrote {P.results('rul_mission_span.csv').name}, "
          f"{P.results('rul_mission_span_by_family.csv').name}, "
          f"{P.results('rul_mission_span_conformal.json').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
