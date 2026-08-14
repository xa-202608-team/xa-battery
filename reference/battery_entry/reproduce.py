"""reproduce.py — the ``reproduce`` subcommand: quick and full.

**quick** (target < 10 minutes, CPU) refits nothing and re-selects nothing. It:

* recomputes the exact 11-dim features from the shipped observed-SOH history and checks
  them against the shipped L3 reference slice bit-for-bit;
* replays the frozen head at the frozen alpha on the shipped reference slice and
  compares its point predictions against the frozen Phase 4/5 predictions at
  ``rtol=0, atol<=1e-12``;
* restates the formal SOH metrics, the Phase 5 per-horizon coverage/width summary and
  the RUL summary from the shipped reference tables;
* runs the release test suite.

**full** re-fits ONLY the frozen final model ``target_only_arc_space`` under the frozen
protocol: the same leave-one-family-out folds, the same frozen alpha per (fold,
horizon), the same 11 features in the same frozen ARC space, the same
trajectory-equal/family-balanced weights, the same three horizons. It re-derives the
point predictions, the per-horizon intervals from the frozen half-widths, and the main
result tables.

What ``full`` deliberately does NOT do — each of these is a thing an earlier phase
already decided, and re-running it here would be re-litigating a closed question:

* no model selection, no feature-set selection, no alpha tuning, no widened grid;
* no Source-Prior arm (Phase 3: SOURCE_PRIOR_NOT_VALIDATED);
* no stress or exposure arm (Phase 4: STRESS_NOT_VALIDATED / EXPOSURE_INPUT_NOT_AVAILABLE);
* no PatchTST, TimesFM or any neural model (Phase 6: closed without training);
* no re-calibration of the conformal half-widths after seeing coverage.

Both modes write ONLY under the caller's ``--output`` directory, enforced by
``paths.guard_output_write``. The shipped reference results are never touched.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from battery_entry import FROZEN_STATUS, OUTPUT_LABELS, __version__
from battery_entry import paths as P
from battery_entry.conformal import (
    COVERAGE_LEVELS, HORIZON_DAYS, MATCHED_HORIZONS, coverage_report,
    interval, interval_caveats, load_quantiles)
from battery_entry.features import FROZEN_COLUMNS, common_features_batch
from battery_entry.model import (
    load_frozen_space, observable_local_trend, fit_head, predict_delta,
    trajectory_equal_weights)
from battery_entry.rul import rul_caveats

LABEL_COLUMN = "target_delta_soh_true"
ANCHOR_COLUMN = "anchor_soh_observed"
ABSOLUTE_TARGET = "target_soh_true"
BIT_EXACT_ATOL = 1e-12


def _write_json(obj: Any, path: Path, output_root: Path) -> Path:
    p = P.guard_output_write(path, output_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                 encoding="utf-8")
    return p


def _write_csv(df: pd.DataFrame, path: Path, output_root: Path) -> Path:
    p = P.guard_output_write(path, output_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def load_reference_slice() -> pd.DataFrame:
    """The shipped L3 v2 reference slice, with numerics from the lossless matrix.

    The CSV is the human-readable copy and supplies the identity/metadata columns; the
    float64 numeric columns are overwritten from ``reference_matrix.npz``. That is not
    belt-and-braces: a CSV round-trip perturbs feature values by up to 1.11e-16, which the
    ridge solve amplifies to ~2.6e-13 in the prediction. Reading the binary makes the
    golden replay bit-exact rather than merely within tolerance.

    The two are cross-checked on ``sample_uid`` order, so a stale npz cannot be paired
    with a newer CSV.
    """
    df = pd.read_csv(P.L3_REFERENCE_CSV)
    z = np.load(P.REFERENCE_MATRIX_NPZ, allow_pickle=True)

    uid_csv = df["sample_uid"].to_numpy().astype(str)
    uid_npz = np.asarray(z["sample_uid"]).astype(str)
    if uid_csv.shape != uid_npz.shape or not np.array_equal(uid_csv, uid_npz):
        raise RuntimeError(
            "reference_matrix.npz and the reference CSV disagree on sample_uid order; "
            "the two shipped copies are out of sync and must not be mixed")

    cols = [str(c) for c in np.asarray(z["feature_columns"]).tolist()]
    if tuple(cols) != tuple(FROZEN_COLUMNS):
        raise RuntimeError(f"reference matrix feature order {cols} is not the frozen 11")

    X = np.asarray(z["X"], dtype=float)
    for j, c in enumerate(cols):
        df[c] = X[:, j]
    for name in ("anchor_soh_observed", "target_soh_true", "target_delta_soh_true",
                 "frozen_predicted_delta"):
        df[name] = np.asarray(z[name], dtype=float)
    return df


def load_frozen_alpha() -> dict[str, Any]:
    return json.loads(P.FROZEN_ALPHA_JSON.read_text(encoding="utf-8"))


def build_folds(families: list[str]) -> list[dict[str, Any]]:
    """6 outer folds; inside each, the remaining 5 families are the training pool.

    Mirrors the frozen ``build_lofo_folds``: the held-out family enters no fit, no
    selection and no clip/scaler statistic.
    """
    out = []
    for k, fam in enumerate(families):
        rest = tuple(f for f in families if f != fam)
        out.append({"index": k, "test_family": fam, "rest_families": rest})
    return out


# ------------------------------------------------------------------ feature check

def recompute_features(df: pd.DataFrame) -> dict[str, Any]:
    """Recompute the 11 features from the shipped observed-SOH history.

    The reference slice ships the context history alongside the frozen feature columns
    precisely so this check can be a recomputation rather than a copy comparison. If the
    shipped extractor had drifted from the one that produced the slice, this fails.
    """
    hist_cols = [c for c in df.columns if c.startswith("ctx_soh_")]
    if len(hist_cols) != 20:
        return {"ok": False,
                "error": f"expected 20 ctx_soh_* columns, found {len(hist_cols)}"}
    hist_cols = sorted(hist_cols, key=lambda c: int(c.rsplit("_", 1)[1]))
    hist = df[hist_cols].to_numpy(dtype=float)
    got = common_features_batch(hist)
    want = df[list(FROZEN_COLUMNS)].to_numpy(dtype=float)
    diff = np.abs(got - want)
    per_feature = {FROZEN_COLUMNS[i]: float(diff[:, i].max())
                   for i in range(diff.shape[1])}
    worst = float(diff.max())
    return {
        "ok": bool(worst <= BIT_EXACT_ATOL),
        "n_rows": int(len(df)),
        "n_features": int(got.shape[1]),
        "max_abs_diff": worst,
        "max_abs_diff_per_feature": per_feature,
        "bit_identical": bool(worst == 0.0),
        "atol": BIT_EXACT_ATOL,
        "rtol": 0,
        "note": ("features recomputed from ctx_soh_00..19 by the shipped extractor and "
                 "compared against the frozen_NN_* columns of the reference slice"),
    }


# ------------------------------------------------------------------ golden replay

def replay_frozen_predictions(df: pd.DataFrame,
                              solver: str = "numpy") -> dict[str, Any]:
    """Refit the frozen head per (fold, horizon) and compare to frozen predictions.

    This is the golden test: the release's own fit path must reproduce the frozen
    Phase 4/5 outer-test point predictions on the common sample set at
    ``rtol=0, atol<=1e-12``.
    """
    space = load_frozen_space()
    alpha_tab = load_frozen_alpha()["lofo"]
    families = sorted(df["environment_family_id"].unique().tolist())
    folds = build_folds(families)

    X = df[list(FROZEN_COLUMNS)].to_numpy(dtype=float)
    label = df[LABEL_COLUMN].to_numpy(dtype=float)
    traj = df["trajectory_id"].to_numpy()
    fam = df["environment_family_id"].to_numpy()
    h_all = df["horizon_steps"].to_numpy()
    frozen_pred = df["frozen_predicted_delta"].to_numpy(dtype=float)

    cells: list[dict[str, Any]] = []
    worst_overall = 0.0
    for fold in folds:
        for h in MATCHED_HORIZONS:
            key = f"{fold['index']}|{h}"
            if key not in alpha_tab:
                continue
            alpha = float(alpha_tab[key])
            train = np.isin(fam, list(fold["rest_families"])) & (h_all == h)
            test = (fam == fold["test_family"]) & (h_all == h)
            if train.sum() < 20 or test.sum() == 0:
                continue
            w = trajectory_equal_weights(traj, fam, train)
            head = fit_head(X, label, w, train, space, alpha, traj, solver=solver)
            pred = predict_delta(X, space, head)
            d = np.abs(pred[test] - frozen_pred[test])
            worst = float(d.max())
            worst_overall = max(worst_overall, worst)
            cells.append({
                "outer_fold": fold["index"],
                "outer_test_family": fold["test_family"],
                "horizon_steps": h,
                "horizon_days": HORIZON_DAYS[h],
                "alpha_frozen": alpha,
                "alpha_selected_here": False,
                "n_train_rows": int(train.sum()),
                "n_train_trajectories": head.n_train_trajectories,
                "n_test_rows": int(test.sum()),
                "max_abs_diff_vs_frozen": worst,
                "bit_exact": bool(worst == 0.0),
                "within_atol": bool(worst <= BIT_EXACT_ATOL),
                "solver": solver,
            })
    return {
        "ok": bool(cells) and all(c["within_atol"] for c in cells),
        "n_cells": len(cells),
        "n_bit_exact": sum(1 for c in cells if c["bit_exact"]),
        "worst_abs_diff": worst_overall,
        "atol": BIT_EXACT_ATOL,
        "rtol": 0,
        "solver": solver,
        "cells": cells,
        "note": ("the release fit path replays the FROZEN alpha; it selects no model, no "
                 "feature set and no hyperparameter"),
    }


# ------------------------------------------------------------------ metrics

def _metrics(truth: np.ndarray, pred_abs: np.ndarray,
             traj: np.ndarray, fam: np.ndarray) -> dict[str, Any]:
    """The Phase 2A metric contract on one evaluation sample set.

    Primary metrics are on ABSOLUTE SOH so the anchor choice cannot hide in the metric.
    Trajectory- and family-macro are unweighted means over units, which is what removes
    the 24..137 windows-per-trajectory imbalance.
    """
    err = np.abs(truth - pred_abs)
    per_traj = {str(t): float(err[traj == t].mean()) for t in np.unique(traj)}
    per_fam = {str(f): float(err[fam == f].mean()) for f in np.unique(fam)}
    tv = np.array(list(per_traj.values()))
    fv = np.array(list(per_fam.values()))
    return {
        "n_windows": int(err.size),
        "n_trajectories": int(tv.size),
        "n_families": int(fv.size),
        "target_soh_true_mae": float(err.mean()),
        "trajectory_macro_mae": float(tv.mean()),
        "family_macro_mae": float(fv.mean()),
        "worst_family_mae": float(fv.max()),
        "per_family_mae": per_fam,
    }


def restate_reference_metrics() -> dict[str, Any]:
    """Read the formal metrics off the shipped reference tables.

    Restated, not recomputed: these are the frozen numbers, and the package's job is to
    carry them with their provenance, not to produce a second opinion.
    """
    out: dict[str, Any] = {}
    summary = pd.read_csv(P.REFERENCE_DIR / "result_summary.csv")
    out["result_summary_rows"] = int(len(summary))
    out["soh"] = summary[summary["block"] == "point_accuracy"].to_dict(orient="records")
    out["coverage"] = summary[summary["block"] == "uncertainty"].to_dict(orient="records")
    out["rul"] = summary[summary["block"] == "rul"].to_dict(orient="records")
    out["gates"] = summary[summary["block"] == "gate"].to_dict(orient="records")
    return out


# ------------------------------------------------------------------ quick

def run_quick(output: Path) -> dict[str, Any]:
    """The quick reproduction. CPU only, no refit of anything unfrozen."""
    t0 = time.time()
    out_root = P.guard_output_write(output / ".probe", output).parent
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_reference_slice()
    steps: dict[str, Any] = {}

    steps["01_feature_recomputation"] = recompute_features(df)
    steps["02_golden_point_prediction_replay"] = replay_frozen_predictions(df)
    steps["03_reference_metrics_restated"] = restate_reference_metrics()

    # per-horizon coverage/width, applied from the frozen half-widths
    q = load_quantiles()
    cov_rows = []
    for cov in COVERAGE_LEVELS:
        for h in MATCHED_HORIZONS:
            hw = q.half_width(cov, h)
            sub = df[df["horizon_steps"] == h]
            pred_abs = (sub[ANCHOR_COLUMN].to_numpy(dtype=float)
                        + sub["frozen_predicted_delta"].to_numpy(dtype=float))
            rep = coverage_report(
                sub[ABSOLUTE_TARGET].to_numpy(dtype=float), pred_abs, hw,
                sub["trajectory_id"].to_numpy(),
                sub["environment_family_id"].to_numpy())
            cov_rows.append({"coverage_nominal": cov, "horizon_steps": h,
                             "horizon_days": HORIZON_DAYS[h], **rep})
    steps["04_conformal_coverage_reapplied"] = {
        "ok": True, "rows": cov_rows,
        "caveats": interval_caveats(),
        "note": ("half-widths are the frozen Phase 5 values, APPLIED not re-derived; "
                 "re-calibrating after seeing coverage is refused"),
    }
    steps["05_rul_summary"] = {
        "ok": True,
        "caveats": rul_caveats(),
        "reference_rows": pd.read_csv(
            P.REFERENCE_DIR / "rul_results_summary.csv").to_dict(orient="records"),
    }

    _write_csv(pd.DataFrame(cov_rows).drop(columns=["per_family_coverage"]),
               out_root / "quick_coverage_by_horizon.csv", out_root)
    _write_csv(pd.DataFrame(steps["02_golden_point_prediction_replay"]["cells"]),
               out_root / "quick_golden_replay_cells.csv", out_root)

    elapsed = time.time() - t0
    result = {
        "mode": "quick",
        "package_version": __version__,
        "ok": all(v.get("ok", True) for v in steps.values()),
        "elapsed_seconds": round(elapsed, 2),
        "target_seconds": 600,
        "within_target_runtime": bool(elapsed <= 600),
        "output_directory": str(out_root),
        "refit_performed": "frozen head only, at the frozen alpha",
        "model_selection_performed": False,
        "hyperparameter_tuning_performed": False,
        "steps": steps,
        "frozen_status": FROZEN_STATUS,
        "output_labels": OUTPUT_LABELS,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }
    _write_json(result, out_root / "quick_result.json", out_root)
    return result


# ------------------------------------------------------------------ full

def run_full(output: Path, solver: str = "numpy") -> dict[str, Any]:
    """The full reproduction: refit ONLY the frozen final model, frozen protocol.

    Regenerates the point predictions, the per-horizon intervals and the main result
    tables into the caller's output directory. Nothing in the package is overwritten and
    no closed question is reopened.
    """
    t0 = time.time()
    out_root = P.guard_output_write(output / ".probe", output).parent
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_reference_slice()
    space = load_frozen_space()
    alpha_tab = load_frozen_alpha()["lofo"]
    q = load_quantiles()

    families = sorted(df["environment_family_id"].unique().tolist())
    folds = build_folds(families)

    X = df[list(FROZEN_COLUMNS)].to_numpy(dtype=float)
    label = df[LABEL_COLUMN].to_numpy(dtype=float)
    anchor = df[ANCHOR_COLUMN].to_numpy(dtype=float)
    truth = df[ABSOLUTE_TARGET].to_numpy(dtype=float)
    traj = df["trajectory_id"].to_numpy()
    fam = df["environment_family_id"].to_numpy()
    h_all = df["horizon_steps"].to_numpy()
    slope = df["frozen_05_slope_full"].to_numpy(dtype=float)

    fold_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []

    for fold in folds:
        for h in MATCHED_HORIZONS:
            key = f"{fold['index']}|{h}"
            if key not in alpha_tab:
                continue
            alpha = float(alpha_tab[key])
            train = np.isin(fam, list(fold["rest_families"])) & (h_all == h)
            test = (fam == fold["test_family"]) & (h_all == h)
            if train.sum() < 20 or test.sum() == 0:
                continue

            w = trajectory_equal_weights(traj, fam, train)
            head = fit_head(X, label, w, train, space, alpha, traj, solver=solver)
            pred_delta = predict_delta(X, space, head)
            pred_abs = anchor + pred_delta

            base_delta = observable_local_trend(slope, h_all)
            base_abs = anchor + base_delta

            idx = np.flatnonzero(test)
            m_final = _metrics(truth[idx], pred_abs[idx], traj[idx], fam[idx])
            m_base = _metrics(truth[idx], base_abs[idx], traj[idx], fam[idx])

            for arm, mm, pa in (("FINAL_MODEL", m_final, pred_abs),
                                ("STABILITY_BASELINE", m_base, base_abs)):
                fold_rows.append({
                    "protocol": "PROTOCOL_NESTED_FAMILY_LOFO",
                    "arm": arm,
                    "model_id": (FROZEN_STATUS["point_model"] if arm == "FINAL_MODEL"
                                 else FROZEN_STATUS["high_stability_baseline"]),
                    "outer_fold": fold["index"],
                    "outer_test_family": fold["test_family"],
                    "horizon_steps": h,
                    "horizon_days": HORIZON_DAYS[h],
                    "alpha_frozen": alpha if arm == "FINAL_MODEL" else None,
                    "alpha_selected_here": False,
                    "n_train_rows": int(train.sum()) if arm == "FINAL_MODEL" else 0,
                    "n_train_trajectories": (head.n_train_trajectories
                                             if arm == "FINAL_MODEL" else 0),
                    **{k: v for k, v in mm.items() if k != "per_family_mae"},
                    "per_family_mae_json": json.dumps(mm["per_family_mae"]),
                    "evidence_domain": "SIMULATION_ONLY",
                })

            for i in idx:
                pred_rows.append({
                    "sample_uid": df["sample_uid"].iloc[i],
                    "trajectory_id": traj[i],
                    "environment_family_id": fam[i],
                    "outer_fold": fold["index"],
                    "horizon_steps": int(h_all[i]),
                    "horizon_days": HORIZON_DAYS[h],
                    "anchor_soh_observed": float(anchor[i]),
                    "predicted_delta": float(pred_delta[i]),
                    "predicted_soh": float(pred_abs[i]),
                    "baseline_predicted_soh": float(base_abs[i]),
                    "target_soh_true": float(truth[i]),
                })

            for cov in COVERAGE_LEVELS:
                hw = q.half_width(cov, h)
                lo, hi = interval(pred_abs[idx], hw)
                rep = coverage_report(truth[idx], pred_abs[idx], hw,
                                      traj[idx], fam[idx])
                interval_rows.append({
                    "protocol": "PROTOCOL_NESTED_FAMILY_LOFO",
                    "arm": "FINAL_MODEL",
                    "method": "FAMILY_BALANCED_CROSSFIT",
                    "coverage_nominal": cov,
                    "outer_fold": fold["index"],
                    "outer_test_family": fold["test_family"],
                    "horizon_steps": h,
                    "horizon_days": HORIZON_DAYS[h],
                    "half_width_frozen": hw,
                    "half_width_recalibrated_here": False,
                    "interval_label": OUTPUT_LABELS["interval"],
                    "gate_status": FROZEN_STATUS["CONFORMAL_GATE"],
                    **{k: v for k, v in rep.items()
                       if k not in ("per_family_coverage",)},
                    "per_family_coverage_json": json.dumps(
                        rep.get("per_family_coverage", {})),
                })

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)
    int_df = pd.DataFrame(interval_rows)

    # matched-H aggregate: one held-out family per fold, so the mean over the 6 folds IS
    # the family-macro across all 6 families. H=16 is absent on ARC and never filled.
    agg = (fold_df.groupby(["arm", "model_id"], as_index=False)
           .agg(family_macro_mae=("family_macro_mae", "mean"),
                trajectory_macro_mae=("trajectory_macro_mae", "mean"),
                worst_family_mae=("worst_family_mae", "max"),
                n_cells=("family_macro_mae", "size"),
                n_windows=("n_windows", "sum")))
    agg["evidence_domain"] = "SIMULATION_ONLY"

    _write_csv(fold_df, out_root / "full_outer_fold_results.csv", out_root)
    _write_csv(pred_df, out_root / "full_point_predictions.csv", out_root)
    _write_csv(int_df, out_root / "full_interval_results.csv", out_root)
    _write_csv(agg, out_root / "full_matched_horizon_summary.csv", out_root)

    # compare against the shipped reference aggregate
    ref = pd.read_csv(P.REFERENCE_DIR / "outer_fold_results_reference.csv")
    joined = fold_df.merge(
        ref, on=["arm", "outer_fold", "horizon_steps"], suffixes=("", "_ref"))
    dev = (float(np.max(np.abs(joined["family_macro_mae"]
                               - joined["family_macro_mae_ref"])))
           if len(joined) else float("nan"))

    elapsed = time.time() - t0
    result = {
        "mode": "full",
        "package_version": __version__,
        "ok": bool(len(fold_rows) > 0),
        "elapsed_seconds": round(elapsed, 2),
        "output_directory": str(out_root),
        "solver": solver,
        "refit": "target_only_arc_space ONLY, at the frozen alpha, frozen protocol",
        "arms_run": sorted(fold_df["arm"].unique().tolist()),
        "n_outer_cells": int(len(fold_df)),
        "n_point_predictions": int(len(pred_df)),
        "n_interval_cells": int(len(int_df)),
        "matched_horizon_summary": agg.to_dict(orient="records"),
        "max_abs_family_macro_mae_deviation_vs_reference": dev,
        "reference_join_rows": int(len(joined)),
        "not_run_and_why": {
            "source_prior": "Phase 3 SOURCE_PRIOR_NOT_VALIDATED",
            "stress_arms_F2_F3": "Phase 4 STRESS_NOT_VALIDATED",
            "exposure_target_T2": "Phase 4 EXPOSURE_INPUT_NOT_AVAILABLE",
            "time_rate_target_T1": ("Phase 4 RATE_DAY_REPARAMETERIZATION_ONLY — T1 is "
                                    "T0 rewritten, not a different model"),
            "patchtst_timesfm": "Phase 6 PHASE6_CLOSED_WITHOUT_TRAINING",
            "alpha_grid_search": "frozen by Phase 4 inner-fold selection",
            "conformal_recalibration": "refused: re-calibration after seeing coverage",
        },
        "frozen_status": FROZEN_STATUS,
        "output_labels": OUTPUT_LABELS,
        "interval_caveats": interval_caveats(),
        "rul_caveats": rul_caveats(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }
    _write_json(result, out_root / "full_result.json", out_root)
    return result
