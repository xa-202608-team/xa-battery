"""Acceptance tests for the self-contained mission-scale ageing reconstruction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths as P  # noqa: E402
from src.twin.reconstruction import (  # noqa: E402
    FORBIDDEN_DRIVER_COLUMNS, MODEL_STATUS, compare_with_reference,
    counterfactual_driver, load_config, reconstruct, validate_inputs,
)


def _load():
    drivers = pd.read_csv(P.TWIN_STRESS_DRIVER_CSV)
    parameters = pd.read_csv(P.TWIN_TRAJECTORY_PARAMETERS_CSV)
    reference = pd.read_csv(P.TWIN_CALIBRATION_REFERENCE_CSV)
    config = load_config(P.TWIN_RECONSTRUCTION_CONFIG_JSON)
    return drivers, parameters, reference, config


def test_01_inputs_are_complete_and_truth_is_separated():
    drivers, parameters, _, _ = _load()
    validate_inputs(drivers, parameters)
    assert len(drivers) == 14285
    assert drivers["trajectory_id"].nunique() == 120
    assert len(parameters) == 120
    assert not FORBIDDEN_DRIVER_COLUMNS.intersection(drivers.columns)
    assert parameters["q0_ah"].between(36.0, 44.1).all()
    assert not np.isclose(parameters["q0_ah"], 100.0).any(), (
        "the obsolete 100 Ah demonstration value must never enter reconstruction")


def test_02_reconstruction_is_deterministic_and_monotone():
    drivers, parameters, _, config = _load()
    a = reconstruct(drivers, parameters, config, add_observation_noise=True)
    b = reconstruct(drivers, parameters, config, add_observation_noise=True)
    np.testing.assert_array_equal(a["soh_true"], b["soh_true"])
    np.testing.assert_array_equal(a["soh_observed"], b["soh_observed"])
    diff = a.sort_values(["trajectory_id", "reference_index"]).groupby(
        "trajectory_id", sort=False)["soh_true"].diff()
    assert int((diff > 1e-12).sum()) == 0
    assert a["generator_status"].eq(MODEL_STATUS).all()


def test_03_heldout_environment_families_pass_replay_gate():
    drivers, parameters, reference, config = _load()
    generated = reconstruct(drivers, parameters, config, add_observation_noise=False)
    report, by_family = compare_with_reference(generated, reference)
    heldout = report["heldout_target_test"]
    assert heldout["mae"] <= 0.0035
    assert heldout["r2"] >= 0.995
    assert set(by_family[by_family["split"].eq("target_test")][
        "environment_family_id"]) == {
            "h500_i070_raan000", "h550_i053_raan180"}


def test_04_one_factor_counterfactuals_have_physical_direction():
    drivers, parameters, _, config = _load()
    base = reconstruct(drivers, parameters, config, add_observation_noise=False)
    base_final = base.groupby("trajectory_id", sort=False).tail(1).set_index(
        "trajectory_id")["soh_true"]
    cases = (
        counterfactual_driver(drivers, temperature_delta_c=5.0),
        counterfactual_driver(drivers, dod_multiplier=1.20),
        counterfactual_driver(drivers, crate_multiplier=1.20),
    )
    for perturbed_driver in cases:
        perturbed = reconstruct(
            perturbed_driver, parameters, config, add_observation_noise=False)
        final = perturbed.groupby("trajectory_id", sort=False).tail(1).set_index(
            "trajectory_id")["soh_true"]
        assert (final <= base_final + 1e-12).all()
        assert (final < base_final - 1e-8).any()


def test_05_calibration_protocol_does_not_fit_target_test():
    cfg = json.loads(P.TWIN_RECONSTRUCTION_CONFIG_JSON.read_text(encoding="utf-8"))
    protocol = cfg["calibration_protocol"]
    assert protocol["heldout_split"] == "target_test"
    assert "target_test" not in protocol["fit_splits"]
    assert cfg["status"] == MODEL_STATUS

