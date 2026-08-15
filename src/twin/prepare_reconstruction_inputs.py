"""Prepare compact, leakage-separated reconstruction inputs from full afterstk.

Usage:
    python -m src.twin.prepare_reconstruction_inputs --afterstk-dir X --output-dir Y

The command is included for provenance.  The prepared files already ship in
``data/twin_reconstruction`` so normal users do not need the 253.4 MB source archive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.twin.reconstruction import DRIVER_COLUMNS, PARAMETER_COLUMNS


EXTRA_PARAMETER_COLUMNS = (
    "reference_step_days", "mission_limit_days", "long_life_censoring_seed",
    "altitude_km", "inclination_deg", "raan_deg", "load_multiplier",
    "panel_multiplier", "temperature_offset_c", "comm_multiplier",
    "solar_degradation_per_year", "adapt_rank_within_target_train",
    "in_adapt_10pct", "in_adapt_25pct", "in_adapt_50pct", "in_adapt_100pct",
)


def prepare(afterstk_dir: Path, output_dir: Path) -> dict:
    l2_path = afterstk_dir / "L2_multi_scenario_reference_points.csv"
    manifest_path = afterstk_dir / "trajectory_manifest.csv"
    calendar_path = afterstk_dir / "annual_environment_family_calendar.csv"
    for path in (l2_path, manifest_path, calendar_path):
        if not path.exists():
            raise FileNotFoundError(path)

    l2 = pd.read_csv(l2_path, encoding="utf-8-sig")
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    calendar = pd.read_csv(calendar_path, encoding="utf-8-sig")

    mean_soc = calendar[["environment_family_id", "day_of_year", "mean_soc"]]
    drivers = l2.merge(
        mean_soc, on=["environment_family_id", "day_of_year"], how="left",
        validate="many_to_one")
    drivers = drivers.loc[:, list(DRIVER_COLUMNS)]
    if drivers["mean_soc"].isna().any():
        raise ValueError("annual calendar did not cover every L2 day_of_year")

    parameter_cols = list(dict.fromkeys(
        list(PARAMETER_COLUMNS) + list(EXTRA_PARAMETER_COLUMNS)))
    parameters = manifest.loc[:, parameter_cols]

    reference = l2.loc[:, [
        "trajectory_id", "environment_family_id", "split", "reference_index",
        "age_days", "soh_true", "rint_ohm", "capacity_ah"]].rename(columns={
            "soh_true": "soh_true_reference",
            "rint_ohm": "rint_ohm_reference",
            "capacity_ah": "capacity_ah_reference",
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    drivers.to_csv(output_dir / "stress_driver_v1.csv", index=False)
    parameters.to_csv(output_dir / "trajectory_parameters_v1.csv", index=False)
    reference.to_csv(output_dir / "calibration_reference_truth.csv", index=False)
    calendar.to_csv(output_dir / "annual_environment_family_calendar.csv", index=False)

    summary = {
        "status": "PREPARED",
        "source_afterstk_dir": str(afterstk_dir),
        "stress_driver_rows": int(len(drivers)),
        "trajectory_parameter_rows": int(len(parameters)),
        "reference_rows": int(len(reference)),
        "annual_calendar_rows": int(len(calendar)),
        "stress_driver_truth_columns": [],
    }
    (output_dir / "input_preparation_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--afterstk-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = prepare(args.afterstk_dir, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

