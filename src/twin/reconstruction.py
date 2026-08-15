"""Auditable mission-scale battery ageing reconstruction.

This module closes a precise gap in the original delivery: the surviving
``trajectory_manifest.csv`` contains the 120 trajectory-specific ageing parameters,
but the program that combined them with the environmental stress drivers was absent.

The implementation here is deliberately named a *reconstruction*, not the original
generator.  The exact historic source code and its sensor-noise sequence are still
unavailable.  The structural equation below was selected on target-train and
target-calibration families, while the two target-test families remained held out.
Every global correction is stored in ``reconstruction_config_v1.json`` and can be
re-fitted with ``python -m src.twin.calibrate_reconstruction``.

Inputs are separated by role:

* ``stress_driver_v1.csv`` contains only environment/exposure channels; it contains no
  SOH, capacity, resistance, EOL or final-life labels.
* ``trajectory_parameters_v1.csv`` contains the recorded per-trajectory coefficients.
* ``calibration_reference_truth.csv`` is optional verification evidence and is never
  read by :func:`reconstruct`.

Consequently the generated ``soh_true`` is actually re-integrated from recorded
drivers and parameters; it is not copied from the reference table.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


MODEL_STATUS = "CALIBRATED_RECONSTRUCTION_NOT_ORIGINAL_SOURCE"
MODEL_ID = "stk_battery_mission_ageing_reconstruction_v1"

DRIVER_COLUMNS = (
    "trajectory_id", "environment_family_id", "split", "reference_index",
    "timestamp_utc", "age_days", "mean_temp_c", "max_temp_c", "mean_c_rate",
    "dod", "ah_throughput", "wh_throughput", "efc", "eclipse_h_per_day",
    "mission_comm_ratio", "mean_beta_deg", "mean_abs_beta_deg",
    "mean_solar_power_w", "max_solar_power_w", "mean_effective_area_m2",
    "altitude_km", "inclination_deg", "raan_deg", "day_of_year", "mean_soc",
)

PARAMETER_COLUMNS = (
    "trajectory_id", "environment_family_id", "split", "seed_index", "q0_ah",
    "initial_rint_ohm", "k_cal", "k_cyc", "dod_exponent", "crate_exponent",
    "temperature_sensitivity", "knee_start_loss", "knee_gain", "beta_cal",
    "beta_cyc", "rint_exponent", "sensor_soh_sigma", "start_day_of_year",
)

FORBIDDEN_DRIVER_COLUMNS = {
    "soh", "soh_observed", "soh_true", "capacity_ah", "rint_ohm",
    "eol_reached", "eol_day_exact", "final_soh_true", "target_soh",
}


@dataclass(frozen=True)
class ReconstructionConfig:
    """Global coefficients for the recorded per-trajectory parameterization."""

    calendar_scale: float
    cycle_scale: float
    knee_gain_scale: float
    knee_start_offset: float
    rint_growth_scale: float
    temperature_reference_c: float = 25.0
    sensor_noise_seed: int = 20260807
    eol_threshold: float = 0.70
    warning_threshold: float = 0.80
    model_id: str = MODEL_ID
    status: str = MODEL_STATUS

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ReconstructionConfig":
        g = value.get("global_coefficients", value)
        thresholds = value.get("thresholds", {})
        return cls(
            calendar_scale=float(g["calendar_scale"]),
            cycle_scale=float(g["cycle_scale"]),
            knee_gain_scale=float(g["knee_gain_scale"]),
            knee_start_offset=float(g["knee_start_offset"]),
            rint_growth_scale=float(g["rint_growth_scale"]),
            temperature_reference_c=float(g.get("temperature_reference_c", 25.0)),
            sensor_noise_seed=int(g.get("sensor_noise_seed", 20260807)),
            eol_threshold=float(thresholds.get("eol", 0.70)),
            warning_threshold=float(thresholds.get("warning", 0.80)),
            model_id=str(value.get("model_id", MODEL_ID)),
            status=str(value.get("status", MODEL_STATUS)),
        )


def load_config(path: Path) -> ReconstructionConfig:
    return ReconstructionConfig.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def _missing(columns: Iterable[str], required: Iterable[str]) -> list:
    present = set(columns)
    return sorted(set(required) - present)


def validate_inputs(drivers: pd.DataFrame, parameters: pd.DataFrame) -> None:
    """Fail early on schema, leakage, identity and ordering mistakes."""
    missing_driver = _missing(drivers.columns, DRIVER_COLUMNS)
    missing_param = _missing(parameters.columns, PARAMETER_COLUMNS)
    if missing_driver:
        raise ValueError("stress driver missing columns: %s" % missing_driver)
    if missing_param:
        raise ValueError("trajectory parameters missing columns: %s" % missing_param)

    leaked = sorted(FORBIDDEN_DRIVER_COLUMNS.intersection(drivers.columns))
    if leaked:
        raise ValueError(
            "stress driver contains forbidden truth/output columns: %s" % leaked)

    if parameters["trajectory_id"].duplicated().any():
        dup = parameters.loc[parameters["trajectory_id"].duplicated(),
                             "trajectory_id"].tolist()
        raise ValueError("duplicate trajectory parameter rows: %s" % dup[:5])

    driver_ids = set(drivers["trajectory_id"].astype(str))
    parameter_ids = set(parameters["trajectory_id"].astype(str))
    if driver_ids != parameter_ids:
        raise ValueError(
            "trajectory identity mismatch: missing parameters=%s, unused parameters=%s"
            % (sorted(driver_ids - parameter_ids)[:5],
               sorted(parameter_ids - driver_ids)[:5]))

    keyed = drivers.sort_values(["trajectory_id", "reference_index"])
    duplicate_key = keyed.duplicated(["trajectory_id", "reference_index"])
    if duplicate_key.any():
        raise ValueError("duplicate trajectory/reference_index rows in stress driver")

    starts = keyed.groupby("trajectory_id", sort=False).first()
    if not np.all(starts["reference_index"].to_numpy(dtype=int) == 0):
        raise ValueError("every trajectory must start at reference_index=0")
    if not np.allclose(starts["age_days"].to_numpy(dtype=float), 0.0):
        raise ValueError("every trajectory must start at age_days=0")

    for name in ("q0_ah", "initial_rint_ohm", "k_cal", "k_cyc",
                 "dod_exponent", "crate_exponent", "temperature_sensitivity"):
        if (parameters[name].astype(float) <= 0).any():
            raise ValueError("parameter %s must be strictly positive" % name)


def _stable_seed(text: str, base: int) -> int:
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "little") + int(base)) % (2 ** 32)


def _merge(drivers: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
    parameter_payload = [c for c in PARAMETER_COLUMNS
                         if c not in ("environment_family_id", "split")]
    merged = drivers.merge(
        parameters.loc[:, parameter_payload], on="trajectory_id", how="left",
        validate="many_to_one")
    return merged.sort_values(
        ["trajectory_id", "reference_index"], kind="mergesort").reset_index(drop=True)


def compute_base_increments(
        drivers: pd.DataFrame,
        parameters: pd.DataFrame,
        temperature_reference_c: float = 25.0) -> pd.DataFrame:
    """Return the calendar and cycling loss increments before global calibration.

    Structural equation (all quantities dimensionless except noted):

    ``dL_cal = k_cal * d(sqrt(day)) * exp(k_T*(Tmax-Tref))
                * (1 + beta_cal*mean_SOC)``

    ``dL_cyc = k_cyc * dEFC * DoD**p * C_rate**q
                * exp(k_T*(Tmean-Tref))
                * (1 + beta_cyc*|beta_angle|/90)``

    Calendar ageing uses maximum temperature, cycling ageing uses mean temperature.
    All exponents and sensitivities are the recorded values from the manifest.
    """
    validate_inputs(drivers, parameters)
    d = _merge(drivers, parameters)
    grouped = d.groupby("trajectory_id", sort=False)
    prev_age = grouped["age_days"].shift().fillna(0.0).to_numpy(dtype=float)
    prev_efc = grouped["efc"].shift().fillna(0.0).to_numpy(dtype=float)

    age = d["age_days"].to_numpy(dtype=float)
    efc = d["efc"].to_numpy(dtype=float)
    delta_sqrt_day = np.sqrt(age) - np.sqrt(prev_age)
    delta_efc = efc - prev_efc
    if (delta_sqrt_day < -1e-12).any() or (delta_efc < -1e-9).any():
        raise ValueError("age_days and efc must be non-decreasing within trajectory")

    k_t = d["temperature_sensitivity"].to_numpy(dtype=float)
    calendar_temperature = np.exp(
        k_t * (d["max_temp_c"].to_numpy(dtype=float) - temperature_reference_c))
    cycle_temperature = np.exp(
        k_t * (d["mean_temp_c"].to_numpy(dtype=float) - temperature_reference_c))

    calendar_soc = 1.0 + (
        d["beta_cal"].to_numpy(dtype=float)
        * d["mean_soc"].to_numpy(dtype=float))
    cycle_beta = 1.0 + (
        d["beta_cyc"].to_numpy(dtype=float)
        * d["mean_abs_beta_deg"].to_numpy(dtype=float) / 90.0)

    calendar = (
        d["k_cal"].to_numpy(dtype=float)
        * delta_sqrt_day * calendar_temperature * calendar_soc)
    cycle = (
        d["k_cyc"].to_numpy(dtype=float)
        * delta_efc
        * np.power(d["dod"].to_numpy(dtype=float),
                   d["dod_exponent"].to_numpy(dtype=float))
        * np.power(d["mean_c_rate"].to_numpy(dtype=float),
                   d["crate_exponent"].to_numpy(dtype=float))
        * cycle_temperature * cycle_beta)

    first = d["reference_index"].to_numpy(dtype=int) == 0
    calendar[first] = 0.0
    cycle[first] = 0.0
    if (calendar < -1e-15).any() or (cycle < -1e-15).any():
        raise ValueError("ageing increments must be non-negative")

    return pd.DataFrame({
        "trajectory_id": d["trajectory_id"].astype(str),
        "reference_index": d["reference_index"].astype(int),
        "calendar_increment_uncalibrated": calendar,
        "cycle_increment_uncalibrated": cycle,
    })


def reconstruct(
        drivers: pd.DataFrame,
        parameters: pd.DataFrame,
        config: ReconstructionConfig,
        add_observation_noise: bool = True) -> pd.DataFrame:
    """Re-integrate capacity loss and return generated mission reference points."""
    increments = compute_base_increments(
        drivers, parameters, config.temperature_reference_c)
    d = _merge(drivers, parameters)
    if not np.array_equal(
            d["trajectory_id"].astype(str).to_numpy(),
            increments["trajectory_id"].to_numpy()):
        raise RuntimeError("internal row-order mismatch")

    step_loss = (
        config.calendar_scale
        * increments["calendar_increment_uncalibrated"].to_numpy(dtype=float)
        + config.cycle_scale
        * increments["cycle_increment_uncalibrated"].to_numpy(dtype=float))
    raw_loss = pd.Series(step_loss).groupby(
        d["trajectory_id"], sort=False).cumsum().to_numpy(dtype=float)

    knee_start = (
        d["knee_start_loss"].to_numpy(dtype=float) + config.knee_start_offset)
    knee_extra = (
        config.knee_gain_scale
        * d["knee_gain"].to_numpy(dtype=float)
        * np.maximum(0.0, raw_loss - knee_start))
    total_loss = np.maximum(0.0, raw_loss + knee_extra)
    # The clipping only removes possible negative numerical noise.  Groupwise
    # monotonicity follows from non-negative step_loss and a monotone knee transform.
    soh_true = np.clip(1.0 - total_loss, 0.0, 1.0)
    capacity_ah = d["q0_ah"].to_numpy(dtype=float) * soh_true
    rint = d["initial_rint_ohm"].to_numpy(dtype=float) * (
        1.0 + config.rint_growth_scale
        * np.power(total_loss, d["rint_exponent"].to_numpy(dtype=float)))

    observed = soh_true.copy()
    if add_observation_noise:
        for tid, idx in d.groupby("trajectory_id", sort=False).groups.items():
            loc = np.asarray(list(idx), dtype=int)
            sigma = float(d.loc[loc[0], "sensor_soh_sigma"])
            rng = np.random.default_rng(_stable_seed(str(tid), config.sensor_noise_seed))
            observed[loc] = np.clip(
                soh_true[loc] + rng.normal(0.0, sigma, size=len(loc)), 0.0, 1.05)

    out = drivers.copy()
    out = out.sort_values(
        ["trajectory_id", "reference_index"], kind="mergesort").reset_index(drop=True)
    out["capacity_ah"] = capacity_ah
    out["soh_true"] = soh_true
    out["soh_observed"] = observed
    out["rint_ohm"] = rint
    out["raw_loss_before_knee"] = raw_loss
    out["calendar_increment"] = (
        config.calendar_scale
        * increments["calendar_increment_uncalibrated"].to_numpy(dtype=float))
    out["cycle_increment"] = (
        config.cycle_scale
        * increments["cycle_increment_uncalibrated"].to_numpy(dtype=float))
    out["eol_threshold"] = config.eol_threshold
    out["warning_threshold"] = config.warning_threshold
    out["generator_model_id"] = config.model_id
    out["generator_status"] = config.status
    return out


def counterfactual_driver(
        drivers: pd.DataFrame,
        temperature_delta_c: float = 0.0,
        dod_multiplier: float = 1.0,
        crate_multiplier: float = 1.0) -> pd.DataFrame:
    """Return a reversible one-factor-at-a-time stress perturbation."""
    if dod_multiplier <= 0 or crate_multiplier <= 0:
        raise ValueError("DoD and C-rate multipliers must be positive")
    out = drivers.copy()
    out["mean_temp_c"] = out["mean_temp_c"].astype(float) + temperature_delta_c
    out["max_temp_c"] = out["max_temp_c"].astype(float) + temperature_delta_c
    out["dod"] = np.clip(out["dod"].astype(float) * dod_multiplier, 0.0, 1.0)
    out["mean_c_rate"] = out["mean_c_rate"].astype(float) * crate_multiplier
    return out


def compare_with_reference(
        generated: pd.DataFrame,
        reference: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Compute transparent replay metrics; never used by the generator itself."""
    required = {"trajectory_id", "reference_index", "soh_true_reference", "split",
                "environment_family_id"}
    missing = sorted(required - set(reference.columns))
    if missing:
        raise ValueError("calibration reference missing columns: %s" % missing)
    merged = generated[["trajectory_id", "reference_index", "soh_true"]].merge(
        reference, on=["trajectory_id", "reference_index"], how="inner",
        validate="one_to_one")
    merged["error"] = merged["soh_true"] - merged["soh_true_reference"]
    merged["absolute_error"] = merged["error"].abs()
    merged["squared_error"] = merged["error"] ** 2

    rows = []
    for (split, family), g in merged.groupby(
            ["split", "environment_family_id"], sort=True):
        truth = g["soh_true_reference"].to_numpy(dtype=float)
        pred = g["soh_true"].to_numpy(dtype=float)
        denom = float(np.sum((truth - truth.mean()) ** 2))
        rows.append({
            "split": split, "environment_family_id": family, "n_rows": len(g),
            "mae": float(np.mean(np.abs(pred - truth))),
            "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
            "max_abs_error": float(np.max(np.abs(pred - truth))),
            "r2": float(1.0 - np.sum((pred - truth) ** 2) / denom)
            if denom > 0 else 1.0,
        })
    by_family = pd.DataFrame(rows)

    def summary(g: pd.DataFrame) -> Dict[str, float]:
        truth = g["soh_true_reference"].to_numpy(dtype=float)
        pred = g["soh_true"].to_numpy(dtype=float)
        denom = float(np.sum((truth - truth.mean()) ** 2))
        return {
            "n_rows": int(len(g)),
            "mae": float(np.mean(np.abs(pred - truth))),
            "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
            "max_abs_error": float(np.max(np.abs(pred - truth))),
            "r2": float(1.0 - np.sum((pred - truth) ** 2) / denom)
            if denom > 0 else 1.0,
        }

    report = {
        "all": summary(merged),
        "calibration_splits": summary(merged[merged["split"] != "target_test"]),
        "heldout_target_test": summary(merged[merged["split"] == "target_test"]),
    }
    return report, by_family
