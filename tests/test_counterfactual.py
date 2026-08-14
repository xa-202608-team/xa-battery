# tests/test_counterfactual.py
"""Tests for the equation-level counterfactual evaluation (P1-15 收窄版).

Verifies that:
1. Temperature +5C / DoD x1.2 / C-rate x1.2 all accelerate aging (lower SOH).
2. No direction violations (every trajectory ages at least as fast).
3. The predictor produces finite RMSE on both baseline and perturbed trajectories.
4. The claim boundary string is present and clearly states "方程级" not "事件级".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths as P  # noqa: E402
from src.twin.reconstruction import (  # noqa: E402
    counterfactual_driver, load_config, reconstruct,
)
from src.twin.run_counterfactual_eval import (  # noqa: E402
    COUNTERFACTUAL_CASES, CLAIM_BOUNDARY, run,
)


def _load():
    drivers = pd.read_csv(P.TWIN_STRESS_DRIVER_CSV)
    parameters = pd.read_csv(P.TWIN_TRAJECTORY_PARAMETERS_CSV)
    config = load_config(P.TWIN_RECONSTRUCTION_CONFIG_JSON)
    return drivers, parameters, config


def test_counterfactual_cases_definition():
    """The three declared perturbations match the spec."""
    assert set(COUNTERFACTUAL_CASES.keys()) == {
        "temperature_plus_5c", "dod_x1_2", "crate_x1_2"}
    for name, spec in COUNTERFACTUAL_CASES.items():
        assert "kwargs" in spec
        assert spec["expected_direction"] == "faster_aging_lower_soh"


def test_claim_boundary_states_equation_level():
    """The claim boundary must say '方程级' and explicitly disambiguate from
    event-level faults (must contain '非事件级')."""
    assert "方程级" in CLAIM_BOUNDARY
    assert "非事件级" in CLAIM_BOUNDARY


def test_temperature_plus_5c_accelerates_aging():
    drivers, parameters, config = _load()
    base = reconstruct(drivers, parameters, config, add_observation_noise=False)
    pert = reconstruct(
        counterfactual_driver(drivers, temperature_delta_c=5.0),
        parameters, config, add_observation_noise=False)
    base_final = base.groupby("trajectory_id", sort=False).tail(1).set_index(
        "trajectory_id")["soh_true"]
    pert_final = pert.groupby("trajectory_id", sort=False).tail(1).set_index(
        "trajectory_id")["soh_true"]
    delta = pert_final - base_final
    assert (delta <= 1e-12).all(), "温度 +5C 不应出现方向违反"
    assert (delta < -1e-8).any(), "至少一条轨迹应严格加速老化"


def test_dod_x12_accelerates_aging():
    drivers, parameters, config = _load()
    base = reconstruct(drivers, parameters, config, add_observation_noise=False)
    pert = reconstruct(
        counterfactual_driver(drivers, dod_multiplier=1.20),
        parameters, config, add_observation_noise=False)
    base_final = base.groupby("trajectory_id", sort=False).tail(1).set_index(
        "trajectory_id")["soh_true"]
    pert_final = pert.groupby("trajectory_id", sort=False).tail(1).set_index(
        "trajectory_id")["soh_true"]
    delta = pert_final - base_final
    assert (delta <= 1e-12).all(), "DoD x1.2 不应出现方向违反"
    assert (delta < -1e-8).any(), "至少一条轨迹应严格加速老化"


def test_crate_x12_accelerates_aging():
    drivers, parameters, config = _load()
    base = reconstruct(drivers, parameters, config, add_observation_noise=False)
    pert = reconstruct(
        counterfactual_driver(drivers, crate_multiplier=1.20),
        parameters, config, add_observation_noise=False)
    base_final = base.groupby("trajectory_id", sort=False).tail(1).set_index(
        "trajectory_id")["soh_true"]
    pert_final = pert.groupby("trajectory_id", sort=False).tail(1).set_index(
        "trajectory_id")["soh_true"]
    delta = pert_final - base_final
    assert (delta <= 1e-12).all(), "C-rate x1.2 不应出现方向违反"
    assert (delta < -1e-8).any(), "至少一条轨迹应严格加速老化"


def test_run_produces_valid_report(tmp_path):
    """Full run produces a JSON report with finite metrics and zero violations."""
    report = run(tmp_path)
    assert "方程级" in report["claim_boundary"]
    assert report["evidence_domain"].startswith("SIMULATION")
    assert report["n_trajectories"] == 120
    assert np.isfinite(report["baseline_predictor_rmse_soh"])

    for name, case in report["counterfactual_cases"].items():
        assert case["n_direction_violations"] == 0, (
            f"{name}: 不应有方向违反")
        assert case["n_strictly_faster_aging"] > 0, (
            f"{name}: 至少一条轨迹严格加速老化")
        assert case["mean_final_soh_change"] < 0, (
            f"{name}: 平均 ΔSOH 应为负（加速老化）")
        assert np.isfinite(case["perturbed_predictor_rmse_soh"])
        assert np.isfinite(case["delta_rmse"])

    # Verify output files were created
    assert (tmp_path / "counterfactual_eval_report.json").exists()
    assert (tmp_path / "counterfactual_trajectory_detail.csv").exists()

    # Verify JSON is valid
    j = json.loads(
        (tmp_path / "counterfactual_eval_report.json").read_text(encoding="utf-8"))
    assert set(j["counterfactual_cases"].keys()) == set(COUNTERFACTUAL_CASES.keys())
