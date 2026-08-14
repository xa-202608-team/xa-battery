"""run_prognostic.py — T2: earliness metrics, rolled over all 120 trajectories.

    python -m src.metrics.run_prognostic

Two predictors are needed, and they are the SAME two the rest of the package uses:

* SOH predictor (for the 0.80 warning): the frozen v2 delivery model's mechanism —
  frozen ARC space + target-domain ridge head on the delta label. Used to project
  SOH at the anchor's forecast reach.
* RUL predictor (for Prognostic Horizon): the T1 mission-span censoring-aware ridge.

Both are evaluated under leave-one-environment-family-out, so a trajectory is always
scored by a model that never saw its family. Without that, earliness numbers would be
in-sample and meaningless.

EVIDENCE DOMAIN: SIMULATION ONLY. These are twin-domain warning windows, not
satellite alarm performance.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import paths as P
from src.features.protocol import (family_macro, nested_family_lofo,
                                   solve_ridge_weighted, trajectory_equal_weights,
                                   weighted_intercept)
from src.features.space import (EOL_SOH, REFERENCE_STEP_DAYS, WARN_SOH,
                                load_frozen_space)
from src.metrics import prognostic as PG
from src.rul.anchors import FEATURE_COLS, build
from src.rul.censored_loss import LAMBDA_CENSORED, fit_censored_ridge, predict

#: The horizon the warning projection uses, in reference steps. 8 steps = 112 days is
#: v2's longest supported horizon; going further would extrapolate past the frozen
#: source model's support, which build_l3_v2.py bars.
WARN_HORIZON_STEPS = 8
ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

EVIDENCE = ("SIMULATION_ONLY (STK-derived twin; NOT satellite alarm performance). "
            "Rolling evaluation, leave-one-environment-family-out; every trajectory "
            "is scored by a model that never saw its family. dt_warn resolution is "
            "bounded by the 14-day reference step.")


def build_soh_delta_table(space) -> pd.DataFrame:
    """Anchor table for the SOH-delta task at the warning horizon.

    One row per (trajectory, anchor) that has both a full 20-point history and a
    real target ``WARN_HORIZON_STEPS`` ahead. The label is the v2 delta label
    ``target_soh_true - anchor_soh_observed``; features are the frozen 11.
    """
    from src.features.frozen11 import CONTEXT_LENGTH, common_features

    l2 = pd.read_csv(P.L2_REFERENCE_CSV)
    sm = pd.read_csv(P.SPLIT_MANIFEST_CSV)
    fam = dict(zip(sm.trajectory_id, sm.environment_family_id))

    rows = []
    for tid, d in l2.groupby("trajectory_id", sort=True):
        d = d.sort_values("reference_day").reset_index(drop=True)
        obs = d.soh_observed.to_numpy(dtype=float)
        tru = d.soh_true.to_numpy(dtype=float)
        day = d.reference_day.to_numpy(dtype=float)
        for i in range(CONTEXT_LENGTH - 1, len(d) - WARN_HORIZON_STEPS):
            j = i + WARN_HORIZON_STEPS
            f = common_features(obs[i - CONTEXT_LENGTH + 1: i + 1])
            rows.append([tid, fam[tid], day[i], day[j], obs[i],
                         tru[j] - obs[i], tru[j], *f])
    cols = (["trajectory_id", "environment_family_id", "anchor_day", "target_day",
             "anchor_soh_observed", "target_delta_soh_true", "target_soh_true"]
            + FEATURE_COLS)
    return pd.DataFrame(rows, columns=cols)


def lofo_predict_soh(S: pd.DataFrame, space) -> np.ndarray:
    """Out-of-family predicted ABSOLUTE SOH at the target point, for every row."""
    X = S[FEATURE_COLS].to_numpy(dtype=float)
    Z = space.apply(X)
    y = S.target_delta_soh_true.to_numpy(dtype=float)
    anchor = S.anchor_soh_observed.to_numpy(dtype=float)
    fam = S.environment_family_id.to_numpy()
    tid = S.trajectory_id.to_numpy()
    families = sorted(set(fam.tolist()))
    out = np.full(len(S), np.nan)

    for held, inner in nested_family_lofo(families):
        tr, te = fam != held, fam == held

        best_a, best_e = None, np.inf
        for a in ALPHA_GRID:
            errs = []
            for f in inner:
                itr, ite = tr & (fam != f), fam == f
                w = trajectory_equal_weights(tid, fam, itr)
                c = solve_ridge_weighted(Z[itr], y[itr], w[itr], a)
                b = weighted_intercept(Z[itr], y[itr], w[itr], c)
                errs.append(float(np.abs(Z[ite] @ c + b - y[ite]).mean()))
            if errs and float(np.mean(errs)) < best_e - 1e-15:
                best_a, best_e = float(a), float(np.mean(errs))

        w = trajectory_equal_weights(tid, fam, tr)
        c = solve_ridge_weighted(Z[tr], y[tr], w[tr], best_a)
        b = weighted_intercept(Z[tr], y[tr], w[tr], c)
        out[te] = anchor[te] + (Z[te] @ c + b)
    return out


def lofo_predict_rul(D: pd.DataFrame, space) -> np.ndarray:
    """Out-of-family predicted mission-span RUL, reusing T1's censoring-aware fit."""
    X = D[FEATURE_COLS].to_numpy(dtype=float)
    Z = space.apply(X)
    y = D.rul_days.to_numpy(dtype=float)
    lb = D.rul_lower_bound_days.to_numpy(dtype=float)
    cen = D.censored.to_numpy() == 1
    fam = D.environment_family_id.to_numpy()
    tid = D.trajectory_id.to_numpy()
    families = sorted(set(fam.tolist()))
    out = np.full(len(D), np.nan)

    for held, inner in nested_family_lofo(families):
        tr, te = fam != held, fam == held

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
                itr, ite_u = tr & (fam != f), (fam == f) & ~cen
                if not ite_u.any():
                    continue
                errs.append(float(np.abs(predict(fit_on(itr, a), Z[ite_u])
                                         - y[ite_u]).mean()))
            if errs and float(np.mean(errs)) < best_e - 1e-15:
                best_a, best_e = float(a), float(np.mean(errs))

        out[te] = predict(fit_on(tr, best_a), Z[te])
    return out


def main() -> int:
    space = load_frozen_space()

    print("=" * 76)
    print("T2  earliness metrics: warning lead time + Prognostic Horizon")
    print("=" * 76)
    print(f"  warning threshold   : SOH = {WARN_SOH}")
    print(f"  EOL threshold       : SOH = {EOL_SOH}")
    print(f"  warning projection  : {WARN_HORIZON_STEPS} steps "
          f"({WARN_HORIZON_STEPS * REFERENCE_STEP_DAYS} days), v2's longest supported horizon")
    print(f"  PH alpha            : {PG.ALPHA_PH}")
    print()

    S = build_soh_delta_table(space)
    S["pred_soh"] = lofo_predict_soh(S, space)
    D = build()
    D["pred_rul"] = lofo_predict_rul(D, space)

    l2 = pd.read_csv(P.L2_REFERENCE_CSV)
    sm = pd.read_csv(P.SPLIT_MANIFEST_CSV)
    fam_of = dict(zip(sm.trajectory_id, sm.environment_family_id))

    records, rows = [], []
    for tid, d in l2.groupby("trajectory_id", sort=True):
        d = d.sort_values("reference_day").reset_index(drop=True)
        day = d.reference_day.to_numpy(dtype=float)
        tru = d.soh_true.to_numpy(dtype=float)

        s = S[S.trajectory_id == tid].sort_values("anchor_day")
        # The true crossing must be read off the FULL day grid: anchors start at
        # reference index 19, so an early crossing would be invisible in the subset.
        t_cross = PG.true_threshold_crossing_day(day, tru, WARN_SOH)
        t_alarm = PG.first_alarm_day(s.anchor_day.to_numpy(dtype=float),
                                     s.pred_soh.to_numpy(dtype=float), WARN_SOH)
        crossed, alarmed = not np.isnan(t_cross), not np.isnan(t_alarm)
        if crossed and alarmed:
            dt = t_cross - t_alarm
            outcome = "TRUE_POSITIVE" if dt >= 0 else "LATE"
        elif crossed:
            dt, outcome = float("nan"), "MISSED"
        elif alarmed:
            dt, outcome = float("nan"), "FALSE_POSITIVE"
        else:
            dt, outcome = float("nan"), "TRUE_NEGATIVE"
        w = {"t_true_cross_day": t_cross, "t_alarm_day": t_alarm,
             "dt_warn_days": dt, "outcome": outcome,
             "crossed": crossed, "alarmed": alarmed}

        t_eol = PG.true_threshold_crossing_day(day, tru, EOL_SOH)
        r = D[D.trajectory_id == tid].sort_values("anchor_day")
        rd = r.anchor_day.to_numpy(dtype=float)
        ry = r.rul_days.to_numpy(dtype=float)
        rp = r.pred_rul.to_numpy(dtype=float)

        # STRICT Saxena: band = alpha*RUL, collapsing to 0 at EOL, raw predictions.
        ph = PG.prognostic_horizon(rd, ry, rp, t_eol)
        # GRID-FLOORED: band never finer than one 14-day reference step, and
        # predictions clamped at 0 as a deployed system would. Reported separately.
        ph_g = PG.prognostic_horizon(rd, ry, rp, t_eol,
                                     band_floor_days=REFERENCE_STEP_DAYS,
                                     clamp_nonnegative=True)
        cw = PG.credible_window(rd, ry, rp, t_eol)

        rec = {**w, **ph, "trajectory_id": tid,
               "environment_family_id": fam_of[tid], "t_eol_day": t_eol,
               "ph_days_gridfloor": ph_g["ph_days"],
               "ph_entry_day_gridfloor": ph_g["ph_entry_day"],
               "n_anchors_inside_band": ph.get("n_anchors_inside_band"),
               "n_anchors_inside_band_gridfloor": ph_g.get("n_anchors_inside_band"),
               "n_rul_anchors": ph.get("n_anchors"),
               "credible_window_days": cw.get("credible_window_days"),
               "credible_window_start_day": cw.get("credible_window_start_day"),
               "credible_window_end_day": cw.get("credible_window_end_day"),
               "credible_window_lead_before_eol_days":
                   cw.get("credible_window_lead_before_eol_days")}
        records.append(rec)
        rows.append(rec)

    agg = PG.aggregate(records)
    # Second PH variant, aggregated over the same EOL-reaching trajectories.
    phg = np.array([r["ph_days_gridfloor"] for r in records
                    if r.get("ph_defined") and not np.isnan(r["ph_days_gridfloor"])],
                   dtype=float)
    inside_strict = int(sum(r.get("n_anchors_inside_band") or 0 for r in records))
    inside_grid = int(sum(r.get("n_anchors_inside_band_gridfloor") or 0
                          for r in records))
    n_rul_anchors = int(sum(r.get("n_rul_anchors") or 0 for r in records))
    agg_g = {
        "ph_gridfloor_n": int(phg.size),
        "ph_gridfloor_mean_days": float(phg.mean()) if phg.size else float("nan"),
        "ph_gridfloor_median_days": (float(np.percentile(phg, 50)) if phg.size
                                     else float("nan")),
        "ph_gridfloor_p90_days": (float(np.percentile(phg, 90)) if phg.size
                                  else float("nan")),
        "ph_gridfloor_n_nonzero": int((phg > 0).sum()) if phg.size else 0,
        "anchors_inside_band_strict": inside_strict,
        "anchors_inside_band_gridfloor": inside_grid,
        "rul_anchors_total": n_rul_anchors,
    }
    cwv = np.array([r["credible_window_days"] for r in records
                    if r.get("credible_window_days") is not None
                    and not np.isnan(r["credible_window_days"])], dtype=float)
    cwl = np.array([r["credible_window_lead_before_eol_days"] for r in records
                    if r.get("credible_window_lead_before_eol_days") is not None
                    and not np.isnan(r["credible_window_lead_before_eol_days"])],
                   dtype=float)
    agg_cw = {
        "credible_window_n": int(cwv.size),
        "credible_window_mean_days": float(cwv.mean()) if cwv.size else float("nan"),
        "credible_window_median_days": (float(np.percentile(cwv, 50)) if cwv.size
                                        else float("nan")),
        "credible_window_p90_days": (float(np.percentile(cwv, 90)) if cwv.size
                                     else float("nan")),
        "credible_window_lead_mean_days": (float(cwl.mean()) if cwl.size
                                           else float("nan")),
        "credible_window_lead_median_days": (float(np.percentile(cwl, 50)) if cwl.size
                                            else float("nan")),
    }

    print(f"  trajectories                    : {agg['n_trajectories']}")
    print(f"  crossed SOH {WARN_SOH} (true)          : {agg['n_crossed_warning_threshold']}"
          f"   never crossed: {agg['n_never_crossed']}")
    print()
    print("  alarm outcomes")
    print(f"    TRUE_POSITIVE (early warning) : {agg['n_true_positive']}")
    print(f"    LATE (alarm after crossing)   : {agg['n_late']}")
    print(f"    MISSED (never alarmed)        : {agg['n_missed']}")
    print(f"    FALSE_POSITIVE                : {agg['n_false_positive']}")
    print(f"    TRUE_NEGATIVE                 : {agg['n_true_negative']}")
    print(f"    miss rate (of crossers)       : {agg['miss_rate_of_crossers']:.4f}")
    print(f"    late rate (of crossers)       : {agg['late_rate_of_crossers']:.4f}")
    print(f"    false-alarm rate (of non-x)   : {agg['false_alarm_rate_of_non_crossers']:.4f}")
    print()
    print(f"  warning lead time dt_warn (n={agg['dt_warn_n']}, TRUE_POSITIVE only)")
    print(f"    mean / median               : {agg['dt_warn_mean_days']:.1f} / "
          f"{agg['dt_warn_median_days']:.1f} days")
    print(f"    p10 / p90                   : {agg['dt_warn_p10_days']:.1f} / "
          f"{agg['dt_warn_p90_days']:.1f} days")
    print(f"    min / max                   : {agg['dt_warn_min_days']:.1f} / "
          f"{agg['dt_warn_max_days']:.1f} days")
    print()
    print(f"  Prognostic Horizon, STRICT Saxena (alpha={PG.ALPHA_PH}, band=alpha*RUL, "
          f"n={agg['ph_n_defined']} EOL-reaching)")
    print(f"    mean / median               : {agg['ph_mean_days']:.1f} / "
          f"{agg['ph_median_days']:.1f} days")
    print(f"    anchors inside band         : {agg_g['anchors_inside_band_strict']} "
          f"of {agg_g['rul_anchors_total']}")
    print("    INTERPRETATION: the strict band is alpha*RUL, so it collapses toward")
    print("    zero as RUL -> 0. At the final anchor (RUL = 14 d) it demands accuracy")
    print("    within +/-2.8 days, finer than the 14-day observation grid itself.")
    print("    PH = 0 therefore reports that the model does not hold a shrinking-to-zero")
    print("    tolerance through EOL. It is NOT a claim that the model is uninformative:")
    print("    lead time above is measured on the same predictions.")
    print()
    print(f"  Prognostic Horizon, GRID-FLOORED (band=alpha*max(RUL,{REFERENCE_STEP_DAYS} d), "
          f"predictions clamped at 0, n={agg_g['ph_gridfloor_n']})")
    print(f"    mean / median               : {agg_g['ph_gridfloor_mean_days']:.1f} / "
          f"{agg_g['ph_gridfloor_median_days']:.1f} days")
    print(f"    p90                         : {agg_g['ph_gridfloor_p90_days']:.1f} days")
    print(f"    trajectories with PH > 0    : {agg_g['ph_gridfloor_n_nonzero']} "
          f"of {agg_g['ph_gridfloor_n']}")
    print(f"    anchors inside band         : {agg_g['anchors_inside_band_gridfloor']} "
          f"of {agg_g['rul_anchors_total']}")
    print()

    df = pd.DataFrame(rows)
    keep = ["trajectory_id", "environment_family_id", "outcome", "crossed", "alarmed",
            "t_true_cross_day", "t_alarm_day", "dt_warn_days", "t_eol_day",
            "ph_days", "ph_entry_day", "ph_days_gridfloor", "ph_entry_day_gridfloor",
            "n_anchors_inside_band", "n_anchors_inside_band_gridfloor",
            "n_rul_anchors", "ph_defined", "reason",
            "credible_window_days", "credible_window_start_day",
            "credible_window_end_day", "credible_window_lead_before_eol_days"]
    df["dt_warn_family_macro_note"] = "see prognostic_metrics_by_family.json"
    df[keep].to_csv(P.results("prognostic_metrics_by_trajectory.csv"), index=False)

    per_fam = {}
    for f, g in df.groupby("environment_family_id"):
        tp = g[g.outcome == "TRUE_POSITIVE"]
        per_fam[f] = float(tp.dt_warn_days.mean()) if len(tp) else float("nan")

    pd.DataFrame([{
        "task": "EARLINESS_METRICS",
        "warning_threshold": WARN_SOH, "eol_threshold": EOL_SOH,
        "warning_projection_steps": WARN_HORIZON_STEPS,
        "warning_projection_days": WARN_HORIZON_STEPS * REFERENCE_STEP_DAYS,
        "protocol": "NESTED_FAMILY_LOFO rolling; no trajectory scored by its own family",
        **agg, **agg_g, **agg_cw,
        "ph_strict_definition": ("Saxena 2008 with band = alpha*RUL(t'), which "
                                 "collapses to zero at EOL and demands sub-grid "
                                 "accuracy at the final anchor"),
        "ph_gridfloor_definition": (f"band = alpha*max(RUL(t'), {REFERENCE_STEP_DAYS}) "
                                    "with predictions clamped at 0; tolerance never "
                                    "finer than the observation grid"),
        "dt_warn_family_macro_days": family_macro(per_fam),
        "evidence_domain": EVIDENCE,
    }]).to_csv(P.results("prognostic_metrics.csv"), index=False)

    P.results("prognostic_metrics_by_family.json").write_text(
        json.dumps({"dt_warn_mean_days_by_family": per_fam,
                    "evidence_domain": EVIDENCE}, indent=2), encoding="utf-8")

    print(f"  Credible window (longest contiguous in-band run, NOT anchored to EOL, "
          f"n={agg_cw['credible_window_n']})")
    print(f"    mean / median duration      : {agg_cw['credible_window_mean_days']:.1f} / "
          f"{agg_cw['credible_window_median_days']:.1f} days")
    print(f"    p90 duration                : {agg_cw['credible_window_p90_days']:.1f} days")
    print(f"    lead before EOL, mean/median: {agg_cw['credible_window_lead_mean_days']:.1f} / "
          f"{agg_cw['credible_window_lead_median_days']:.1f} days")
    print("    WEAKER CLAIM than PH: this run may end before EOL. Reported because")
    print("    PH = 0 says only 'not accurate through EOL', not 'never accurate'.")
    print()
    print(f"  EVIDENCE: {EVIDENCE}")
    print()
    print(f"  wrote {P.results('prognostic_metrics.csv').name}, "
          f"{P.results('prognostic_metrics_by_trajectory.csv').name}, "
          f"{P.results('prognostic_metrics_by_family.json').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
