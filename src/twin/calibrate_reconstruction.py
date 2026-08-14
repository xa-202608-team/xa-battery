"""Re-fit the four global reconstruction corrections without target-test leakage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src import paths as P
from src.twin.reconstruction import (
    MODEL_ID, MODEL_STATUS, ReconstructionConfig, compare_with_reference,
    compute_base_increments, reconstruct, validate_inputs,
)


def calibrate(drivers: pd.DataFrame, parameters: pd.DataFrame,
              reference: pd.DataFrame) -> dict:
    validate_inputs(drivers, parameters)
    d = drivers.sort_values(
        ["trajectory_id", "reference_index"], kind="mergesort").reset_index(drop=True)
    p = parameters.set_index("trajectory_id")
    inc = compute_base_increments(d, parameters)
    cal = inc["calendar_increment_uncalibrated"].to_numpy(dtype=float)
    cyc = inc["cycle_increment_uncalibrated"].to_numpy(dtype=float)
    group = d["trajectory_id"].astype(str)
    knee = group.map(p["knee_start_loss"]).to_numpy(dtype=float)
    gain = group.map(p["knee_gain"]).to_numpy(dtype=float)

    ref = d[["trajectory_id", "reference_index", "split"]].merge(
        reference[["trajectory_id", "reference_index", "soh_true_reference"]],
        on=["trajectory_id", "reference_index"], how="left", validate="one_to_one")
    if ref["soh_true_reference"].isna().any():
        raise ValueError("reference truth does not cover every driver row")
    truth_loss = 1.0 - ref["soh_true_reference"].to_numpy(dtype=float)
    calibration_mask = ref["split"].ne("target_test").to_numpy()

    def predict_loss(theta: np.ndarray) -> np.ndarray:
        calendar_scale, cycle_scale, knee_scale, knee_offset = theta
        raw = pd.Series(
            calendar_scale * cal + cycle_scale * cyc).groupby(
                group, sort=False).cumsum().to_numpy(dtype=float)
        return raw + knee_scale * gain * np.maximum(0.0, raw - (knee + knee_offset))

    fit = least_squares(
        lambda theta: (predict_loss(theta) - truth_loss)[calibration_mask],
        x0=np.asarray([1.0, 1.0, 0.60, -0.018], dtype=float),
        bounds=(np.asarray([0.5, 0.5, 0.3, -0.05]),
                np.asarray([1.5, 1.5, 2.0, 0.05])),
        loss="linear", max_nfev=500)

    # Global resistance scale is separately fitted on calibration families.  It is
    # reported as an approximate secondary channel; SOH is the primary reconstructed
    # state and the acceptance thresholds are defined on SOH.
    ref_r = d[["trajectory_id", "reference_index", "split"]].merge(
        reference[["trajectory_id", "reference_index", "rint_ohm_reference"]],
        on=["trajectory_id", "reference_index"], validate="one_to_one")
    r0 = group.map(p["initial_rint_ohm"]).to_numpy(dtype=float)
    r_exp = group.map(p["rint_exponent"]).to_numpy(dtype=float)
    x = np.power(np.clip(truth_loss, 0.0, None), r_exp)
    y = ref_r["rint_ohm_reference"].to_numpy(dtype=float) / r0 - 1.0
    rint_scale = float(
        np.dot(x[calibration_mask], y[calibration_mask])
        / np.dot(x[calibration_mask], x[calibration_mask]))

    cfg = ReconstructionConfig(
        calendar_scale=float(fit.x[0]), cycle_scale=float(fit.x[1]),
        knee_gain_scale=float(fit.x[2]), knee_start_offset=float(fit.x[3]),
        rint_growth_scale=rint_scale)
    generated = reconstruct(d, parameters, cfg, add_observation_noise=False)
    metrics, by_family = compare_with_reference(generated, reference)

    return {
        "model_id": MODEL_ID,
        "status": MODEL_STATUS,
        "claim_boundary": (
            "Recorded trajectory coefficients are used directly. The four global "
            "corrections reconstruct the missing integration program and were fitted "
            "without target-test families. This is not the lost original source code."),
        "calibration_protocol": {
            "fit_splits": ["target_train", "target_calibration"],
            "heldout_split": "target_test",
            "objective": "row-level SOH reconstruction residual",
            "optimizer": "scipy.optimize.least_squares, linear loss",
        },
        "global_coefficients": {
            "calendar_scale": cfg.calendar_scale,
            "cycle_scale": cfg.cycle_scale,
            "knee_gain_scale": cfg.knee_gain_scale,
            "knee_start_offset": cfg.knee_start_offset,
            "rint_growth_scale": cfg.rint_growth_scale,
            "temperature_reference_c": cfg.temperature_reference_c,
            "sensor_noise_seed": cfg.sensor_noise_seed,
        },
        "thresholds": {"eol": cfg.eol_threshold, "warning": cfg.warning_threshold},
        "soh_reconstruction_metrics": metrics,
        "per_family_metrics": json.loads(by_family.to_json(orient="records")),
        "acceptance_thresholds": {
            "heldout_target_test_mae_max": 0.0035,
            "heldout_target_test_r2_min": 0.995,
            "monotonicity_violations_max": 0,
        },
        "rint_note": (
            "rint_growth_scale is a global secondary-channel approximation; it is not "
            "used to claim exact resistance reconstruction."),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drivers", type=Path, default=P.TWIN_STRESS_DRIVER_CSV)
    parser.add_argument("--parameters", type=Path,
                        default=P.TWIN_TRAJECTORY_PARAMETERS_CSV)
    parser.add_argument("--reference", type=Path,
                        default=P.TWIN_CALIBRATION_REFERENCE_CSV)
    # Read-only protection: the shipped reference config (TWIN_RECONSTRUCTION_CONFIG_JSON
    # in data/twin_reconstruction/) must never be overwritten.  Default output goes to
    # the writable results directory instead.
    parser.add_argument("--output", type=Path,
                        default=P.RESULTS_DIR / "twin_reconstruction"
                        / "reconstruction_config_calibrated.json")
    args = parser.parse_args()
    report = calibrate(
        pd.read_csv(args.drivers), pd.read_csv(args.parameters),
        pd.read_csv(args.reference))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    heldout = report["soh_reconstruction_metrics"]["heldout_target_test"]
    print("CALIBRATION_OK")
    print("  status              : %s" % report["status"])
    print("  heldout test MAE    : %.8f" % heldout["mae"])
    print("  heldout test R2     : %.8f" % heldout["r2"])
    print("  output              : %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

