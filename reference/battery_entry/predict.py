"""predict.py — the ``predict`` subcommand: point prediction from an observed history.

Given exactly 20 observed-SOH reference points per battery, produces:

* the point prediction of absolute SOH at days 28, 56 and 112 (H = 2, 4, 8) — the
  DEFAULT output;
* on request, per-horizon intervals, every one labelled
  ``EXPERIMENTAL_SIMULATION_INTERVAL``;
* on request, the finite-horizon RUL read-out, labelled
  ``FINITE_HORIZON_EVIDENCE_INSUFFICIENT``, which returns a STATUS rather than a number
  when the forecast path does not reach 0.70 inside 112 days.

The deployment head is fitted once, on all six environment families of the shipped
reference slice, at the frozen fixed-holdout alpha. That is the honest deployment
configuration: at inference time there is no held-out family to respect, but there is
also no licence to choose a new alpha, so the frozen one is replayed.

Exit codes are part of the interface, because "no crossing" is a real answer that must
not be mistaken for a failure:

* ``0`` — every battery produced a point prediction, and every requested RUL crossed
  inside 112 days;
* ``3`` — predictions succeeded, but at least one battery's path did not reach the EOL
  threshold within 112 days (``NO_CROSSING_WITHIN_FORECAST_HORIZON``). Nothing is
  extrapolated;
* ``2`` — the input was refused by the schema validator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from battery_entry import FROZEN_STATUS, OUTPUT_LABELS, __version__
from battery_entry import paths as P
from battery_entry.conformal import (
    COVERAGE_LEVELS, HORIZON_DAYS, MATCHED_HORIZONS, interval, load_quantiles)
from battery_entry.features import FROZEN_COLUMNS, common_features
from battery_entry.model import (
    load_frozen_space, fit_head, observable_local_trend, predict_delta,
    trajectory_equal_weights)
from battery_entry.reproduce import load_reference_slice
from battery_entry.rul import readout as rul_readout
from battery_entry.schema import ValidatedInput, read_input_csv

EXIT_OK = 0
EXIT_SCHEMA_REFUSED = 2
EXIT_NO_CROSSING = 3

LABEL_COLUMN = "target_delta_soh_true"
ANCHOR_COLUMN = "anchor_soh_observed"


@dataclass
class DeploymentModel:
    """One head per horizon, fitted on the full reference slice at the frozen alpha."""
    heads: dict[int, Any]
    space: Any
    alpha_by_horizon: dict[int, float]
    n_train_rows: dict[int, int]
    n_train_trajectories: dict[int, int]
    provenance: str


def build_deployment_model(solver: str = "numpy") -> DeploymentModel:
    """Fit the deployment head per horizon at the FROZEN fixed-holdout alpha.

    No alpha is chosen here. The value comes off the frozen Phase 4 selection, which is
    why ``alpha_selected_in_this_package`` is false in ``models/frozen_alpha.json``.
    """
    df = load_reference_slice()
    space = load_frozen_space()
    alpha_tab = json.loads(
        P.FROZEN_ALPHA_JSON.read_text(encoding="utf-8"))["fixed_holdout"]

    X = df[list(FROZEN_COLUMNS)].to_numpy(dtype=float)
    label = df[LABEL_COLUMN].to_numpy(dtype=float)
    traj = df["trajectory_id"].to_numpy()
    fam = df["environment_family_id"].to_numpy()
    h_all = df["horizon_steps"].to_numpy()

    heads: dict[int, Any] = {}
    alphas: dict[int, float] = {}
    n_rows: dict[int, int] = {}
    n_traj: dict[int, int] = {}
    for h in MATCHED_HORIZONS:
        alpha = float(alpha_tab[str(h)])
        train = (h_all == h)
        w = trajectory_equal_weights(traj, fam, train)
        head = fit_head(X, label, w, train, space, alpha, traj, solver=solver)
        heads[h] = head
        alphas[h] = alpha
        n_rows[h] = int(train.sum())
        n_traj[h] = head.n_train_trajectories
    return DeploymentModel(
        heads=heads, space=space, alpha_by_horizon=alphas,
        n_train_rows=n_rows, n_train_trajectories=n_traj,
        provenance=("head refitted on all 6 simulated environment families of the "
                    "shipped reference slice, at the FROZEN Phase 4 fixed-holdout "
                    "alpha; the frozen ARC clip/scaler is reused unchanged"))


def predict_one(vi: ValidatedInput, dm: DeploymentModel,
                with_intervals: bool, with_rul: bool,
                coverage: float) -> dict[str, Any]:
    """Predict for one validated battery history."""
    feats = common_features(vi.history).reshape(1, -1)
    anchor = np.array([vi.anchor_soh], dtype=float)

    row: dict[str, Any] = {
        "battery_id": vi.battery_id,
        "anchor_reference_index": vi.anchor_reference_index,
        "anchor_soh_observed": vi.anchor_soh,
        "context_length": vi.context_length,
        "model_id": FROZEN_STATUS["point_model"],
        "transfer_mechanism": FROZEN_STATUS["transfer_mechanism"],
        "evidence_domain": FROZEN_STATUS["evidence_domain"],
    }

    abs_by_h: dict[int, float] = {}
    for h in MATCHED_HORIZONS:
        head = dm.heads[h]
        delta = float(predict_delta(feats, dm.space, head)[0])
        soh = float(anchor[0] + delta)
        abs_by_h[h] = soh
        d = HORIZON_DAYS[h]
        row[f"predicted_delta_soh_day{d}"] = delta
        row[f"predicted_soh_day{d}"] = soh
        row[f"alpha_frozen_h{h}"] = dm.alpha_by_horizon[h]

    # the non-learning engineering baseline, always reported beside the fitted model
    slope_full = float(feats[0, 5])
    for h in MATCHED_HORIZONS:
        d = HORIZON_DAYS[h]
        row[f"baseline_predicted_soh_day{d}"] = float(
            anchor[0] + observable_local_trend(np.array([slope_full]),
                                               np.array([h]))[0])

    if with_intervals:
        q = load_quantiles()
        row["interval_label"] = OUTPUT_LABELS["interval"]
        row["interval_gate_status"] = FROZEN_STATUS["CONFORMAL_GATE"]
        row["interval_coverage_nominal"] = coverage
        row["interval_is_per_horizon_marginal"] = True
        row["interval_simultaneous_coverage_claimed"] = False
        for h in MATCHED_HORIZONS:
            d = HORIZON_DAYS[h]
            hw = q.half_width(coverage, h, protocol="PROTOCOL_FIXED_HOLDOUT")
            lo, hi = interval(np.array([abs_by_h[h]]), hw)
            row[f"interval_lo_day{d}"] = float(lo[0])
            row[f"interval_hi_day{d}"] = float(hi[0])
            row[f"interval_half_width_day{d}"] = float(hw)

    if with_rul:
        r = rul_readout(vi.anchor_soh, abs_by_h[2], abs_by_h[4], abs_by_h[8])
        row["rul_label"] = r["label"]
        row["rul_evidence_status"] = r["evidence_status"]
        row["rul_days"] = r["rul_days"]
        row["rul_status"] = r["rul_status"]
        row["rul_segment"] = r["rul_segment"]
        row["rul_crossed_within_112d"] = r["crossed_within_112d"]
        row["warning_days"] = r["warning_days"]
        row["warning_status"] = r["warning_status"]
        row["warning_is_forecast_not_restatement"] = r[
            "warning_is_forecast_not_restatement"]
        row["max_forecast_days"] = r["max_forecast_days"]
        row["extrapolated_beyond_horizon"] = False

    return row


def run_predict(input_csv: str, output_csv: str,
                with_intervals: bool = False,
                with_rul: bool = False,
                coverage: float = 0.90,
                solver: str = "numpy") -> tuple[int, dict[str, Any]]:
    """Validate, predict, write. Returns ``(exit_code, summary)``."""
    from battery_entry.schema import SchemaError

    try:
        batteries = read_input_csv(input_csv)
    except SchemaError as e:
        return EXIT_SCHEMA_REFUSED, {
            "status": "INPUT_REFUSED_BY_SCHEMA",
            "error": str(e),
            "n_batteries": 0,
            "note": ("the input is refused rather than coerced: silent repair of a "
                     "mis-scaled or mis-ordered history would produce a well-formed "
                     "wrong answer"),
        }

    if coverage not in COVERAGE_LEVELS:
        return EXIT_SCHEMA_REFUSED, {
            "status": "COVERAGE_LEVEL_NOT_CALIBRATED",
            "error": (f"coverage {coverage} was not calibrated; the package ships "
                      f"{list(COVERAGE_LEVELS)} and refuses to interpolate a quantile"),
            "n_batteries": len(batteries),
        }

    dm = build_deployment_model(solver=solver)
    rows = [predict_one(vi, dm, with_intervals, with_rul, coverage)
            for vi in batteries]
    df = pd.DataFrame(rows)

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_no_cross = 0
    if with_rul:
        n_no_cross = int((~df["rul_crossed_within_112d"]).sum())

    summary = {
        "status": "OK",
        "n_batteries": len(rows),
        "output": str(out),
        "with_intervals": with_intervals,
        "with_rul": with_rul,
        "coverage_nominal": coverage if with_intervals else None,
        "n_no_crossing_within_112d": n_no_cross,
        "model_id": FROZEN_STATUS["point_model"],
        "transfer_mechanism": FROZEN_STATUS["transfer_mechanism"],
        "deployment_head_provenance": dm.provenance,
        "alpha_by_horizon": dm.alpha_by_horizon,
        "alpha_selected_here": False,
        "evidence_domain": FROZEN_STATUS["evidence_domain"],
        "labels": {
            "interval": OUTPUT_LABELS["interval"] if with_intervals else None,
            "rul": OUTPUT_LABELS["rul"] if with_rul else None,
        },
        "caveat": ("Every number is produced by a model fitted and evaluated on an "
                   "STK-derived simulated twin. This is NOT real-satellite accuracy and "
                   "no on-orbit guarantee follows from it."),
    }
    code = EXIT_NO_CROSSING if n_no_cross else EXIT_OK
    return code, summary
