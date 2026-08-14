"""Run the self-contained mission-scale ageing reconstruction and verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import paths as P
from src.twin.reconstruction import (
    MODEL_STATUS, compare_with_reference, counterfactual_driver, load_config,
    reconstruct,
)


def monotonicity_violations(frame: pd.DataFrame) -> int:
    ordered = frame.sort_values(["trajectory_id", "reference_index"])
    diff = ordered.groupby("trajectory_id", sort=False)["soh_true"].diff()
    return int((diff > 1e-12).sum())


def _counterfactual_summary(drivers: pd.DataFrame, parameters: pd.DataFrame,
                            config) -> dict:
    baseline = reconstruct(drivers, parameters, config, add_observation_noise=False)
    cases = {
        "temperature_plus_5c": counterfactual_driver(drivers, temperature_delta_c=5.0),
        "dod_plus_20pct": counterfactual_driver(drivers, dod_multiplier=1.20),
        "crate_plus_20pct": counterfactual_driver(drivers, crate_multiplier=1.20),
    }
    out = {}
    base_final = baseline.groupby("trajectory_id", sort=False).tail(1).set_index(
        "trajectory_id")["soh_true"]
    for name, perturbed_driver in cases.items():
        perturbed = reconstruct(
            perturbed_driver, parameters, config, add_observation_noise=False)
        perturbed_final = perturbed.groupby(
            "trajectory_id", sort=False).tail(1).set_index("trajectory_id")["soh_true"]
        delta = perturbed_final - base_final
        out[name] = {
            "n_trajectories": int(len(delta)),
            "mean_final_soh_change": float(delta.mean()),
            "max_final_soh_change": float(delta.max()),
            "n_direction_violations": int((delta > 1e-12).sum()),
            "expected_direction": "non-positive",
        }
    return out


def run(output_dir: Path) -> dict:
    drivers = pd.read_csv(P.TWIN_STRESS_DRIVER_CSV)
    parameters = pd.read_csv(P.TWIN_TRAJECTORY_PARAMETERS_CSV)
    config = load_config(P.TWIN_RECONSTRUCTION_CONFIG_JSON)
    generated = reconstruct(drivers, parameters, config, add_observation_noise=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_path = output_dir / "generated_l2_reference_points.csv"
    generated.to_csv(generated_path, index=False)

    report = {
        "model_id": config.model_id,
        "status": config.status,
        "claim_boundary": (
            "Mission-scale SOH is re-integrated from stress-only drivers and the "
            "recorded trajectory parameters. This is a calibrated reconstruction of "
            "the missing program, not the lost original source code or a real-flight "
            "validated digital twin."),
        "n_rows": int(len(generated)),
        "n_trajectories": int(generated["trajectory_id"].nunique()),
        "monotonicity_violations": monotonicity_violations(generated),
        "counterfactual_direction_checks": _counterfactual_summary(
            drivers, parameters, config),
    }

    if P.TWIN_CALIBRATION_REFERENCE_CSV.exists():
        reference = pd.read_csv(P.TWIN_CALIBRATION_REFERENCE_CSV)
        metrics, by_family = compare_with_reference(generated, reference)
        report["soh_reconstruction_metrics"] = metrics
        by_family.to_csv(output_dir / "reconstruction_metrics_by_family.csv", index=False)

    report_path = output_dir / "reconstruction_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    heldout = report.get("soh_reconstruction_metrics", {}).get(
        "heldout_target_test", {})
    print("TWIN_RECONSTRUCTION_OK")
    print("  status                  : %s" % report["status"])
    print("  rows / trajectories     : %d / %d" % (
        report["n_rows"], report["n_trajectories"]))
    print("  monotonicity violations : %d" % report["monotonicity_violations"])
    if heldout:
        print("  heldout test MAE        : %.8f" % heldout["mae"])
        print("  heldout test R2         : %.8f" % heldout["r2"])
    print("  generated L2            : %s" % generated_path)
    print("  report                  : %s" % report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=P.RESULTS_DIR / "twin_reconstruction")
    args = parser.parse_args()
    report = run(args.output_dir)

    heldout = report.get("soh_reconstruction_metrics", {}).get(
        "heldout_target_test", {})
    failures = []
    if report["status"] != MODEL_STATUS:
        failures.append("unexpected status")
    if report["monotonicity_violations"] != 0:
        failures.append("SOH monotonicity")
    if heldout and heldout["mae"] > 0.0035:
        failures.append("heldout MAE")
    if heldout and heldout["r2"] < 0.995:
        failures.append("heldout R2")
    for name, value in report["counterfactual_direction_checks"].items():
        if value["n_direction_violations"]:
            failures.append(name)
    if failures:
        print("TWIN_RECONSTRUCTION_FAIL: %s" % failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

