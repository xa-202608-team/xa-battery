"""run_counterfactual_eval.py — equation-level counterfactual evaluation.

    python -m src.twin.run_counterfactual_eval

WHAT THIS IS
------------
Equation-level counterfactual ("温度/DoD/C-rate 方程级反事实"), NOT event-level
fault injection.  The stress drivers (temperature, DoD, C-rate) are scaled over
the *entire* trajectory, the ageing equation in :mod:`reconstruction` is
re-integrated, and the predictor is scored on the resulting perturbed L2
trajectories versus the baseline.

Three one-factor-at-a-time perturbations are evaluated:

* ``temperature_plus_5c``  — mean/max temperature +5 C
* ``dod_x1_2``             — DoD multiplied by 1.2
* ``crate_x1_2``           — mean C-rate multiplied by 1.2

CLAIM BOUNDARY
--------------
This is a **counterfactual stress scaling** applied at the equation level via
:func:`reconstruction.counterfactual_driver`.  It is NOT an event-level fault
model (no onset, no step, no single-event signature).  The ageing equation is
the calibrated reconstruction, not a flight-validated digital twin.

EVIDENCE DOMAIN: SIMULATION ONLY.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import paths as P
from src.features.frozen11 import CONTEXT_LENGTH, common_features
from src.features.protocol import (nested_family_lofo, solve_ridge_weighted,
                                   trajectory_equal_weights,
                                   weighted_intercept)
from src.features.space import REFERENCE_STEP_DAYS, load_frozen_space
from src.twin.reconstruction import (counterfactual_driver, load_config,
                                     reconstruct)


HORIZON_STEPS = 8
ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

COUNTERFACTUAL_CASES = {
    "temperature_plus_5c": {
        "label": "温度 +5 C（全轨迹）",
        "kwargs": {"temperature_delta_c": 5.0},
        "expected_direction": "faster_aging_lower_soh",
    },
    "dod_x1_2": {
        "label": "DoD x1.2（全轨迹）",
        "kwargs": {"dod_multiplier": 1.20},
        "expected_direction": "faster_aging_lower_soh",
    },
    "crate_x1_2": {
        "label": "C-rate x1.2（全轨迹）",
        "kwargs": {"crate_multiplier": 1.20},
        "expected_direction": "faster_aging_lower_soh",
    },
}

EVIDENCE = ("SIMULATION ONLY. Equation-level counterfactual stress scaling "
            "(temperature/DoD/C-rate over the full trajectory), NOT event-level "
            "fault injection. The ageing equation is the calibrated reconstruction "
            "(src/twin/reconstruction.py), not a flight-validated digital twin.")

CLAIM_BOUNDARY = (
    "方程级反事实（温度/DoD/C-rate 全轨迹缩放），非事件级故障。"
    " ageing equation = calibrated reconstruction, not original source code.")


def _windows(day, soh_obs, soh_true, tid, fam):
    """Anchor rows at the warning horizon, from one trajectory's channels."""
    rows = []
    for i in range(CONTEXT_LENGTH - 1, len(day) - HORIZON_STEPS):
        j = i + HORIZON_STEPS
        f = common_features(soh_obs[i - CONTEXT_LENGTH + 1: i + 1])
        rows.append([tid, fam, day[i], soh_obs[i],
                     soh_true[j] - soh_obs[i], soh_true[j], *f])
    return rows


def _build_anchor_table(generated: pd.DataFrame, split_manifest: pd.DataFrame
                        ) -> pd.DataFrame:
    """Build the predictor anchor table from a reconstructed L2 frame."""
    fam_of = dict(zip(split_manifest.trajectory_id,
                      split_manifest.environment_family_id))
    rows = []
    for tid, g in generated.groupby("trajectory_id", sort=True):
        g = g.sort_values("reference_index")
        day = (g["reference_index"].to_numpy(dtype=int)
               * REFERENCE_STEP_DAYS).astype(float)
        soh_obs = g["soh_observed"].to_numpy(dtype=float)
        soh_true = g["soh_true"].to_numpy(dtype=float)
        fam = fam_of.get(tid, "unknown")
        rows += _windows(day, soh_obs, soh_true, str(tid), fam)
    cols = (["trajectory_id", "environment_family_id", "anchor_day",
             "anchor_soh_observed", "y_delta", "target_soh_true"]
            + [f"f{j:02d}" for j in range(11)])
    return pd.DataFrame(rows, columns=cols)


def _train_heads(space, l2: pd.DataFrame, split_manifest: pd.DataFrame):
    """Train one ridge head per held-out family on the BASELINE L2 data."""
    fam_of = dict(zip(split_manifest.trajectory_id,
                      split_manifest.environment_family_id))
    rows = []
    for tid, g in l2.groupby("trajectory_id", sort=True):
        g = g.sort_values("reference_day")
        rows += _windows(g.reference_day.to_numpy(dtype=float),
                         g.soh_observed.to_numpy(dtype=float),
                         g.soh_true.to_numpy(dtype=float),
                         str(tid), fam_of[tid])
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


def _score(D: pd.DataFrame, space, heads) -> pd.DataFrame:
    """Apply the frozen predictor to an anchor table."""
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


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2)))


def _final_soh(generated: pd.DataFrame) -> pd.Series:
    """Final SOH per trajectory."""
    return (generated
            .groupby("trajectory_id", sort=False)
            .tail(1)
            .set_index("trajectory_id")["soh_true"])


def run(output_dir: Path) -> dict:
    # --- Load reconstruction inputs ---
    drivers = pd.read_csv(P.TWIN_STRESS_DRIVER_CSV)
    parameters = pd.read_csv(P.TWIN_TRAJECTORY_PARAMETERS_CSV)
    config = load_config(P.TWIN_RECONSTRUCTION_CONFIG_JSON)
    split_manifest = pd.read_csv(P.SPLIT_MANIFEST_CSV)
    l2 = pd.read_csv(P.L2_REFERENCE_CSV)

    # --- Baseline reconstruction ---
    baseline_gen = reconstruct(drivers, parameters, config,
                               add_observation_noise=False)
    base_final = _final_soh(baseline_gen)

    # --- Train predictor on the SHIPPED baseline L2 (unperturbed) ---
    space = load_frozen_space()
    heads = _train_heads(space, l2, split_manifest)

    # --- Score baseline reconstructed trajectories ---
    base_table = _build_anchor_table(baseline_gen, split_manifest)
    base_scored = _score(base_table, space, heads)
    base_rmse = _rmse(base_scored.abs_err_soh.to_numpy())

    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "title": "方程级反事实评估（温度/DoD/C-rate 全轨迹缩放）",
        "claim_boundary": CLAIM_BOUNDARY,
        "nature": (
            "Equation-level counterfactual: temperature/DoD/C-rate scaled over "
            "the full trajectory, ageing equation re-integrated via "
            "reconstruction.counterfactual_driver(). NOT event-level fault "
            "injection (no onset / step / single-event signature)."),
        "evidence_domain": EVIDENCE,
        "predictor": (
            "Frozen ARC space + per-family ridge head, trained on UNPERTURBED "
            "shipped L2 (leave-one-environment-family-out)."),
        "n_trajectories": int(baseline_gen["trajectory_id"].nunique()),
        "baseline_predictor_rmse_soh": base_rmse,
        "counterfactual_cases": {},
    }

    for name, spec in COUNTERFACTUAL_CASES.items():
        # Perturb drivers and re-integrate ageing
        perturbed_drivers = counterfactual_driver(drivers, **spec["kwargs"])
        perturbed_gen = reconstruct(perturbed_drivers, parameters, config,
                                    add_observation_noise=False)
        perturbed_final = _final_soh(perturbed_gen)

        # Score predictor on perturbed trajectories
        pert_table = _build_anchor_table(perturbed_gen, split_manifest)
        pert_scored = _score(pert_table, space, heads)
        pert_rmse = _rmse(pert_scored.abs_err_soh.to_numpy())

        # Direction check: all perturbed final SOH <= baseline (no violations)
        delta = perturbed_final - base_final
        n_violations = int((delta > 1e-12).sum())
        n_strict_faster = int((delta < -1e-8).sum())

        case_report = {
            "label": spec["label"],
            "perturbation": spec["kwargs"],
            "expected_direction": spec["expected_direction"],
            "n_trajectories": int(len(delta)),
            "mean_final_soh_change": float(delta.mean()),
            "max_final_soh_change": float(delta.max()),
            "n_direction_violations": n_violations,
            "n_strictly_faster_aging": n_strict_faster,
            "baseline_predictor_rmse_soh": base_rmse,
            "perturbed_predictor_rmse_soh": pert_rmse,
            "delta_rmse": float(pert_rmse - base_rmse),
        }
        report["counterfactual_cases"][name] = case_report

    # --- Write outputs ---
    report_path = output_dir / "counterfactual_eval_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    # Per-case trajectory-level detail
    detail_rows = []
    for name, spec in COUNTERFACTUAL_CASES.items():
        perturbed_drivers = counterfactual_driver(drivers, **spec["kwargs"])
        perturbed_gen = reconstruct(perturbed_drivers, parameters, config,
                                    add_observation_noise=False)
        perturbed_final = _final_soh(perturbed_gen)
        delta = perturbed_final - base_final
        for tid in delta.index:
            detail_rows.append({
                "case": name,
                "trajectory_id": tid,
                "baseline_final_soh": float(base_final[tid]),
                "perturbed_final_soh": float(perturbed_final[tid]),
                "delta_soh": float(delta[tid]),
            })
    pd.DataFrame(detail_rows).to_csv(
        output_dir / "counterfactual_trajectory_detail.csv", index=False)

    # --- Print summary ---
    print("=" * 76)
    print("P1-15  方程级反事实评估（温度/DoD/C-rate 全轨迹缩放）")
    print("=" * 76)
    print("  NATURE: 方程级反事实（temperature/DoD/C-rate 全轨迹缩放）")
    print("          非事件级故障（无 onset / step / single-event signature）")
    print()
    print(f"  trajectories          : {report['n_trajectories']}")
    print(f"  baseline predictor RMSE (SOH): {base_rmse:.6f}")
    print()
    print(f"  {'case':<22s} {'mean ΔSOH':>12s} {'violations':>11s} "
          f"{'pert RMSE':>11s} {'ΔRMSE':>10s}")
    for name, c in report["counterfactual_cases"].items():
        print(f"  {name:<22s} {c['mean_final_soh_change']:12.6f} "
              f"{c['n_direction_violations']:11d} "
              f"{c['perturbed_predictor_rmse_soh']:11.6f} "
              f"{c['delta_rmse']:+10.6f}")
    print()
    print("  READING: mean ΔSOH < 0 confirms faster aging under the stress")
    print("  perturbation. n_direction_violations = 0 means every trajectory")
    print("  aged at least as fast as baseline. ΔRMSE shows how much the")
    print("  predictor's error changes on perturbed trajectories.")
    print()
    print(f"  report : {report_path}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="方程级反事实评估（温度/DoD/C-rate 全轨迹缩放）")
    parser.add_argument(
        "--output-dir", type=Path,
        default=P.RESULTS_DIR / "twin_counterfactual")
    args = parser.parse_args()
    report = run(args.output_dir)

    failures = []
    for name, c in report["counterfactual_cases"].items():
        if c["n_direction_violations"] > 0:
            failures.append(f"{name}: {c['n_direction_violations']} direction violations")
        if c["n_strictly_faster_aging"] == 0:
            failures.append(f"{name}: no trajectory aged strictly faster")
    if failures:
        print("COUNTERFACTUAL_FAIL: %s" % failures)
        return 1
    print("COUNTERFACTUAL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
