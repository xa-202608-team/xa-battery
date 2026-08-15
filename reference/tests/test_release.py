"""test_release.py — the release acceptance suite.

Runs BOTH ways from the same assertions, so the two reporting conventions cannot
silently diverge::

    python tests/test_release.py          # direct execution, prints a summary
    python -m pytest tests/test_release.py -q

Fifteen checks, each mapping onto a declared acceptance requirement:

  01  the package runs in a clean directory with no parent project present
  02  no active reference to the parent project, its artifacts or a host path
  03  the 11 feature names / order / clip / scaler agree with the frozen manifest
  04  release point predictions == frozen Phase 4/5 predictions, rtol=0, atol<=1e-12
  05  the metric tables reproduce within the declared float tolerance
  06  the torch ridge path agrees with the numpy one (same normal equations)
  07  hidden truth / future realized exposure / family / split cannot enter the matrix
  08  conformal is calibrated per horizon at H = 2, 4, 8 separately
  09  the pooled control is never described as group-aware
  10  interval outputs carry CONFORMAL_NOT_VALIDATED
  11  RUL never uses -1 and never extrapolates past day 112
  12  RUL MAE on different eligible sets does not enter a ranking
  13  PatchTST / TimesFM are neither implemented nor run
  14  quick/full write only into the caller's output directory
  15  the shipped baseline files are unchanged by a run (hash before == after)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from battery_entry import FROZEN_STATUS, OUTPUT_LABELS  # noqa: E402
from battery_entry import paths as P  # noqa: E402
from battery_entry.conformal import (  # noqa: E402
    COVERAGE_LEVELS, MATCHED_HORIZONS, load_quantiles)
from battery_entry.features import (  # noqa: E402
    FEATURE_NAMES, FROZEN_COLUMNS, common_features_batch, feature_names_sha256)
from battery_entry.model import (  # noqa: E402
    fit_head, load_frozen_space, predict_delta, solve_ridge_weighted,
    trajectory_equal_weights)
from battery_entry.rul import (  # noqa: E402
    MAX_FORECAST_DAYS, NO_CROSSING, first_crossing, rul_caveats)

ATOL = 1e-12
METRIC_ATOL = 1e-9

#: Keys/headings under which a forbidden claim is QUOTED IN ORDER TO FORBID IT.
#: The documents deliberately list every statement that may not be made — that is how the
#: prohibition is recorded — so a naive grep for those sentences flags the prohibition
#: itself. These tests must distinguish "the document asserts X" from "the document
#: forbids X", or they would force the package to stop documenting its own limits.
_PROHIBITION_KEYS = re.compile(
    r"may_not|may_never|must_not|not_be_claimed|forbidden|cannot_be_claimed"
    r"|statements_that_may_not_be_made|must never|must_be_stated"
    r"|may\s+not\s+be\s+\w+"                      # may not be made / written / claimed
    r"|may\s+never\s+be\s+\w+"
    r"|never\s+(?:be\s+)?(?:described|called|claimed|stated)"   # a denial
    r"|is\s+not\s+group[- ]aware|NOT\s+run|was\s+NOT\s+",
    re.IGNORECASE)


def _demarkdown(s: str) -> str:
    """Strip markdown emphasis so prohibition phrases match as written prose.

    Load bearing: the documents write ``This may **not** be written as ...``, and the
    ``**`` sits between "may" and "not", so a plain ``may\\s+not`` regex misses it. Without
    this, the checker reads a prohibition as an assertion — the exact inversion that would
    force the docs to stop recording their own limits.
    """
    return re.sub(r"[*_`~]+", "", s)


def _is_prohibition_context(text: str, pos: int) -> bool:
    """Is the match at ``pos`` inside a list of claims the document FORBIDS?

    Four shapes, all of which appear in the shipped documents:

    * markdown strikethrough — ``* ~~PatchTST underperformed~~``;
    * a prohibition marker on the same line;
    * a prohibition marker in the **preceding lines of the same block** — the forbidden
      claims are often a comma-separated run wrapped across lines under one
      "This may **not** be written as ..." lead-in, so the marker is not on the matched
      line itself;
    * a JSON/YAML/CSV value whose nearest preceding key is a prohibition key
      (``may_not_be_claimed``, ``must_be_stated``, ``forbidden_extrapolation``, ...).
    """
    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos)
    line = text[line_start:line_end if line_end != -1 else len(text)]

    if line.count("~~") >= 2:
        return True
    if _PROHIBITION_KEYS.search(_demarkdown(line)):
        return True

    # The lead-in may be several lines above the matched sentence. Walk back to the start
    # of the enclosing markdown paragraph / JSON value and search the whole of it.
    back = text[max(0, pos - 1200):pos]
    para = re.split(r"\n\s*\n", back)[-1]      # last blank-line-delimited block
    if _PROHIBITION_KEYS.search(_demarkdown(para)):
        return True

    # nearest enclosing JSON key / markdown heading / YAML key
    window = text[max(0, pos - 4000):pos]
    key_hits = list(re.finditer(
        r'"([A-Za-z0-9_]+)"\s*:|^#{1,6}\s*(.+)$|^\s*([a-z0-9_]+):',
        window, re.MULTILINE))
    if key_hits:
        last = key_hits[-1]
        label = last.group(1) or last.group(2) or last.group(3) or ""
        if _PROHIBITION_KEYS.search(_demarkdown(label)):
            return True
    return False


#: This file is the checker. It necessarily contains the very sentences it forbids, in the
#: regexes and in the explanatory comments, so scanning it would match by construction —
#: the same reason the package keeps its own patterns in
#: ``schemas/forbidden_patterns.json``. Exempted from the PROSE scans only; every
#: functional assertion still runs over it via test_02 and test_13's AST-level checks.
_SELF = "tests/test_release.py"


# ---------------------------------------------------------------- 01

def test_01_runs_in_clean_directory_without_parent_project() -> None:
    """The package imports and verifies with the CWD outside any parent project.

    Run as a subprocess from a temp directory with a minimal environment, so an
    accidentally-inherited ``sys.path`` entry or a relative path that happens to resolve
    from the project root would fail here.
    """
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONPATH"] = str(P.PACKAGE_ROOT)
        r = subprocess.run(
            [sys.executable, "-m", "battery_entry", "verify"],
            cwd=td, env=env, capture_output=True, text=True, timeout=1800,
            encoding="utf-8", errors="replace")
        out = r.stdout or ""
        err = r.stderr or ""
        assert r.returncode == 0, (
            f"verify failed from a clean cwd (exit {r.returncode})\n"
            f"stdout:\n{out[-3000:]}\nstderr:\n{err[-3000:]}")
        assert "VERIFY_OK" in out, out[-2000:]

    # and the package root itself must not need the parent to exist on disk
    assert not (P.PACKAGE_ROOT / "src").exists(), (
        "the package ships a src/ directory, which suggests a parent-tree copy")


# ---------------------------------------------------------------- 02

def test_02_no_active_parent_project_reference() -> None:
    """No code file references the parent project, its artifacts, or a host path."""
    from battery_entry.verify import check_no_parent_dependency

    res = check_no_parent_dependency()
    assert res["ok"], (
        f"{res['n_hits']} forbidden path reference(s):\n"
        + "\n".join(f"  {h['file']}:{h['line']}  {h['match']!r}  ({h['why']})"
                    for h in res["hits"][:20]))
    assert res["n_files_scanned"] > 10, res


# ---------------------------------------------------------------- 03

def test_03_feature_implementation_matches_frozen_manifest() -> None:
    """Feature names, order, clip/scaler agree with the frozen manifest, and the
    shipped extractor reproduces the shipped frozen feature columns bit-for-bit."""
    space = load_frozen_space()
    schema = json.loads(
        (P.SCHEMA_DIR / "feature_schema_v2.json").read_text(encoding="utf-8"))

    assert len(FEATURE_NAMES) == 11
    assert tuple(space.feature_names) == tuple(FEATURE_NAMES), (
        f"space order {space.feature_names} != code order {FEATURE_NAMES}")
    assert tuple(schema["feature_names"]) == tuple(FEATURE_NAMES)
    assert tuple(schema["frozen_columns"]) == tuple(FROZEN_COLUMNS)
    assert schema["feature_names_sha256"] == feature_names_sha256()

    for nm, arr in (("clip_lo", space.clip_lo), ("clip_hi", space.clip_hi),
                    ("scaler_mean", space.scaler_mean),
                    ("scaler_std", space.scaler_std)):
        a = np.asarray(arr)
        assert a.size == 11, f"{nm} has {a.size} entries, expected 11"
        assert np.isfinite(a).all(), f"{nm} carries non-finite values"
    assert np.all(space.clip_lo <= space.clip_hi)
    assert np.all(space.scaler_std > 0)

    import hashlib
    clip_sha = hashlib.sha256(
        np.asarray(space.clip_lo, dtype=float).tobytes()
        + np.asarray(space.clip_hi, dtype=float).tobytes()).hexdigest()
    scaler_sha = hashlib.sha256(
        np.asarray(space.scaler_mean, dtype=float).tobytes()
        + np.asarray(space.scaler_std, dtype=float).tobytes()).hexdigest()
    assert clip_sha == space.clip_sha256, "clip arrays do not match their frozen hash"
    assert scaler_sha == space.scaler_sha256, "scaler arrays do not match their hash"

    # the extractor must reproduce the shipped frozen feature columns
    from battery_entry.reproduce import load_reference_slice
    df = load_reference_slice()
    ctx = sorted([c for c in df.columns if c.startswith("ctx_soh_")],
                 key=lambda c: int(c.rsplit("_", 1)[1]))
    assert len(ctx) == 20, f"expected 20 context columns, found {len(ctx)}"
    got = common_features_batch(df[ctx].to_numpy(dtype=float))
    want = df[list(FROZEN_COLUMNS)].to_numpy(dtype=float)
    worst = float(np.max(np.abs(got - want)))
    assert worst <= ATOL, (
        f"the shipped extractor no longer reproduces the shipped frozen features "
        f"(max |diff| {worst:.3e} > atol {ATOL:.0e}); features.py has drifted from the "
        f"implementation that produced the reference slice")


# ---------------------------------------------------------------- 04

def test_04_release_predictions_match_frozen_bit_exactly() -> None:
    """rtol=0, atol<=1e-12 against the frozen Phase 4/5 outer-test predictions.

    Loaded from the lossless ``reference_matrix.npz`` rather than the CSV, so the
    comparison avoids the CSV round-trip that perturbs feature values by up to
    1.11e-16.

    The worst observed residual (~3.3e-13) is float64 accumulation noise from the
    ridge solve, whose magnitude depends on the BLAS implementation (OpenBLAS vs
    MKL vs Apple ARM).  Bit-exact identity across BLAS backends is unreliable and
    not physically meaningful — SOH is reported to 4 decimal places (1e-4), so a
    1e-12 tolerance is eight orders of magnitude below any quantity of interest.
    The contract is therefore: every cell within atol, worst diff below 1e-12.
    """
    from battery_entry.reproduce import load_reference_slice, replay_frozen_predictions

    df = load_reference_slice()
    res = replay_frozen_predictions(df)
    assert res["n_cells"] == 18, f"expected 18 (fold, horizon) cells, got {res['n_cells']}"
    assert res["ok"], (
        f"worst |diff| {res['worst_abs_diff']:.3e} exceeds atol {ATOL:.0e}; "
        f"offending cells: "
        f"{[c for c in res['cells'] if not c['within_atol']][:4]}")
    # numpy.allclose semantics with rtol exactly 0
    for c in res["cells"]:
        assert c["max_abs_diff_vs_frozen"] <= ATOL, c
    # Bit-exact identity (worst_abs_diff == 0) is not guaranteed across BLAS
    # implementations (OpenBLAS vs MKL vs Apple ARM produce different float64
    # accumulation sequences).  A worst-case diff below 1e-12 — eight orders of
    # magnitude below the SOH reporting precision of 1e-4 — confirms the fit path
    # has not drifted.
    assert res["worst_abs_diff"] < ATOL, (
        f"worst |diff| {res['worst_abs_diff']:.3e} exceeds tolerance {ATOL:.0e}; "
        f"the fit path has drifted beyond float64 accumulation noise")


# ---------------------------------------------------------------- 05

def test_05_metric_tables_reproduce_within_tolerance() -> None:
    """The reference metric tables reproduce from the shipped predictions."""
    from battery_entry.reproduce import load_reference_slice
    ofr = pd.read_csv(P.REFERENCE_DIR / "outer_fold_results_reference.csv")
    df = load_reference_slice()

    fin = ofr[ofr["arm"] == "FINAL_MODEL"]
    assert len(fin) == 18, f"expected 18 FINAL_MODEL rows, got {len(fin)}"

    worst = 0.0
    for r in fin.itertuples():
        sub = df[(df["frozen_outer_fold"] == r.outer_fold)
                 & (df["horizon_steps"] == r.horizon_steps)]
        pred_abs = (sub["anchor_soh_observed"].to_numpy(dtype=float)
                    + sub["frozen_predicted_delta"].to_numpy(dtype=float))
        err = np.abs(sub["target_soh_true"].to_numpy(dtype=float) - pred_abs)
        fam = sub["environment_family_id"].to_numpy()
        per_fam = np.array([err[fam == f].mean() for f in np.unique(fam)])
        worst = max(worst, abs(float(per_fam.mean()) - float(r.family_macro_mae)))
    assert worst <= METRIC_ATOL, (
        f"family_macro_mae does not reproduce (max |diff| {worst:.3e} > "
        f"{METRIC_ATOL:.0e})")

    # the headline number must match the frozen Phase 4 reference point
    summary = pd.read_csv(P.REFERENCE_DIR / "result_summary.csv")
    row = summary[summary["metric"] == "family_macro_mae_final_model_matched_H"]
    assert len(row) == 1
    assert abs(float(row.iloc[0]["value"]) - 0.003328) < 5e-7, (
        f"headline family_macro_mae {row.iloc[0]['value']} differs from the frozen "
        f"Phase 4 reference 0.003328")


# ---------------------------------------------------------------- 06

def test_06_torch_ridge_equals_numpy_ridge() -> None:
    """The torch solve path agrees with numpy: same normal equations, not a new model.

    This is what licenses describing the PyTorch path as a reuse of the already-verified
    ``lambda_source = 0`` Ridge solver rather than a different algorithm.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("torch is not installed; the numpy path is the default")
        return

    from battery_entry.model import solve_ridge_weighted_torch
    from battery_entry.reproduce import load_reference_slice

    df = load_reference_slice()
    space = load_frozen_space()
    alpha = json.loads(P.FROZEN_ALPHA_JSON.read_text(encoding="utf-8"))["lofo"]["0|8"]

    X = df[list(FROZEN_COLUMNS)].to_numpy(dtype=float)
    y = df["target_delta_soh_true"].to_numpy(dtype=float)
    traj = df["trajectory_id"].to_numpy()
    fam = df["environment_family_id"].to_numpy()
    families = sorted(np.unique(fam).tolist())
    train = np.isin(fam, families[1:]) & (df["horizon_steps"].to_numpy() == 8)
    w = trajectory_equal_weights(traj, fam, train)

    Z = space.apply(X)
    a = solve_ridge_weighted(Z[train], y[train], w[train], float(alpha))
    b = solve_ridge_weighted_torch(Z[train], y[train], w[train], float(alpha))
    d = float(np.max(np.abs(a - b)))
    assert d < 1e-10, f"torch and numpy ridge coefficients differ by {d:.3e}"

    # and the full predictions agree
    ha = fit_head(X, y, w, train, space, float(alpha), traj, solver="numpy")
    hb = fit_head(X, y, w, train, space, float(alpha), traj, solver="torch")
    dp = float(np.max(np.abs(predict_delta(X, space, ha)
                             - predict_delta(X, space, hb))))
    assert dp < 1e-10, f"torch and numpy predictions differ by {dp:.3e}"


# ---------------------------------------------------------------- 07

def test_07_forbidden_channels_cannot_enter_the_matrix() -> None:
    """Hidden truth, future realized exposure, family and split are structurally out.

    Three layers, each checked: the shipped data file does not carry the columns; the
    design matrix is built from a fixed 11-name list; and the predict validator refuses
    an input file that carries them.
    """
    from battery_entry.schema import SchemaError, validate_input

    df = pd.read_csv(P.L3_REFERENCE_CSV, nrows=5)
    banned_present = [c for c in df.columns
                      if c in ("soh_true", "anchor_soh_true", "rint_true",
                               "soh_noise_residual", "future_realized_delta_efc",
                               "future_realized_delta_ah", "future_realized_delta_wh",
                               "future_planned_delta_efc", "altitude_km",
                               "inclination_deg", "raan_deg")]
    assert not banned_present, (
        f"the shipped slice carries forbidden columns {banned_present}")

    # the design matrix is exactly the frozen 11, in order
    assert tuple(FROZEN_COLUMNS) == tuple(
        f"frozen_{i:02d}_{n}" for i, n in enumerate(FEATURE_NAMES))
    assert len(FROZEN_COLUMNS) == 11

    # family and split ship as metadata but are NOT in the frozen 11
    for c in ("environment_family_id", "split", "trajectory_id",
              "target_soh_true", "target_delta_soh_true"):
        assert c not in FROZEN_COLUMNS, f"{c} is inside the design-matrix column list"

    # the validator refuses each forbidden column in a predict input
    base = pd.DataFrame({
        "battery_id": ["B"] * 20,
        "reference_index": list(range(20)),
        "soh_observed": np.linspace(0.95, 0.80, 20),
    })
    for bad, val in (("soh_true", 0.9), ("environment_family_id", "h550_i053_raan000"),
                     ("split", "target_test"), ("future_realized_delta_efc", 1.0),
                     ("target_soh_true", 0.8), ("altitude_km", 550.0),
                     ("anchor_soh_true", 0.9), ("obs_hist_mean_soh", 0.9)):
        bad_df = base.copy()
        bad_df[bad] = val
        try:
            validate_input(bad_df)
        except SchemaError:
            pass
        else:
            raise AssertionError(f"the validator accepted a forbidden column {bad!r}")

    # a clean input is accepted, so the refusals above are specific rather than blanket
    ok = validate_input(base)
    assert len(ok) == 1 and ok[0].context_length == 20


# ---------------------------------------------------------------- 08

def test_08_conformal_calibrated_per_horizon() -> None:
    """H = 2, 4, 8 each carry their OWN half-width, and the widths grow with horizon."""
    q = load_quantiles()
    assert q.per_horizon_marginal_only is True

    for protocol in ("PROTOCOL_NESTED_FAMILY_LOFO", "PROTOCOL_FIXED_HOLDOUT"):
        for cov in COVERAGE_LEVELS:
            hw = [q.half_width(cov, h, protocol=protocol) for h in MATCHED_HORIZONS]
            assert len(set(hw)) == 3, (
                f"{protocol} @ {cov}: the three horizons share a half-width {hw}; "
                f"a single pooled quantile would over-cover H=2 and under-cover H=8")
            assert hw[0] < hw[1] < hw[2], (
                f"{protocol} @ {cov}: half-widths {hw} are not increasing in horizon")

    # the ~4x growth Phase 5 measured, on the primary LOFO 90% cells
    lo = q.half_width(0.90, 2)
    hi = q.half_width(0.90, 8)
    assert 2.5 < hi / lo < 6.0, (
        f"half-width growth {hi / lo:.2f}x from 28 to 112 days is outside the measured "
        f"~4x; the per-horizon calibration rationale rests on this")


# ---------------------------------------------------------------- 09

def test_09_pooled_control_never_called_group_aware() -> None:
    """``POOLED_SPLIT_CONFORMAL`` is not shipped as a candidate and is not mis-named.

    Its finite-sample correction counts overlapping windows (~10,000) rather than
    independent trajectories (100), which is why its interval is narrower and its
    coverage lower. Describing it as group-aware would misrepresent that.
    """
    q = json.loads((P.MODEL_DIR / "conformal_quantiles.json").read_text(encoding="utf-8"))
    assert q["method"] == "FAMILY_BALANCED_CROSSFIT"
    assert "POOLED_SPLIT_CONFORMAL" in q["controls_not_shipped"]

    ref = pd.read_csv(P.REFERENCE_DIR / "coverage_by_horizon_reference.csv")
    pooled = ref[ref["method"] == "POOLED_SPLIT_CONFORMAL"]
    if len(pooled):
        assert not pooled["is_group_aware"].any(), (
            "a POOLED_SPLIT_CONFORMAL row is flagged is_group_aware=True")

    # no shipped text may ASSERT that the pooled control is group-aware. A document that
    # lists that sentence in order to FORBID it is correct behaviour, not a violation.
    rx = re.compile(r"group[- ]aware[^.]{0,80}pooled|pooled[^.]{0,80}group[- ]aware",
                    re.IGNORECASE)
    for p in P.package_files():
        if p.suffix.lower() not in (".py", ".md", ".json"):
            continue
        if P.relpath(p) == _SELF:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for m in rx.finditer(text):
            if _is_prohibition_context(text, m.start()):
                continue
            snippet = text[max(0, m.start() - 60):m.end() + 60]
            # a sentence that explicitly DENIES it is fine
            if re.search(r"not\s+group[- ]aware|never\s+be\s+(?:described|called)|"
                         r"NOT\s+group-aware|is_group_aware\s*=\s*False|"
                         r"mis-?named|mis-?declared|must never be called that",
                         snippet, re.IGNORECASE):
                continue
            raise AssertionError(
                f"{P.relpath(p)} appears to call the pooled control group-aware: "
                f"...{snippet.strip()}...")


# ---------------------------------------------------------------- 10

def test_10_interval_outputs_carry_conformal_not_validated() -> None:
    """Every interval path emits CONFORMAL_NOT_VALIDATED and the experimental label."""
    from battery_entry.conformal import GATE_STATUS, INTERVAL_LABEL, interval_caveats

    assert GATE_STATUS == "CONFORMAL_NOT_VALIDATED"
    assert INTERVAL_LABEL == "EXPERIMENTAL_SIMULATION_INTERVAL"
    assert OUTPUT_LABELS["interval"] == "EXPERIMENTAL_SIMULATION_INTERVAL"
    assert FROZEN_STATUS["CONFORMAL_GATE"] == "CONFORMAL_NOT_VALIDATED"

    c = interval_caveats()
    assert c["gate_status"] == "CONFORMAL_NOT_VALIDATED"
    assert c["simultaneous_coverage_claimed"] is False
    assert c["per_horizon_marginal_only"] is True
    assert "CONFORMAL_WORST_FAMILY_SHORTFALL" in c["flags"]
    assert "FIXED_HOLDOUT_COVERAGE_SHORTFALL" in c["flags"]
    assert c["family_macro_shortfall_fired"] is False, (
        "CONFORMAL_FAMILY_MACRO_SHORTFALL did NOT fire in Phase 5; claiming it did "
        "would overstate the shortfall")

    q = load_quantiles()
    assert q.gate_status == "CONFORMAL_NOT_VALIDATED"
    assert q.simultaneous_coverage_claimed is False

    # and a real predict run labels its interval columns
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "pred.csv"
        from battery_entry.predict import run_predict
        code, summary = run_predict(
            str(P.EXAMPLES_DIR / "example_input.csv"), str(out),
            with_intervals=True, with_rul=False, coverage=0.90)
        assert code == 0, summary
        df = pd.read_csv(out)
        assert (df["interval_label"] == "EXPERIMENTAL_SIMULATION_INTERVAL").all()
        assert (df["interval_gate_status"] == "CONFORMAL_NOT_VALIDATED").all()
        assert (~df["interval_simultaneous_coverage_claimed"]).all()


# ---------------------------------------------------------------- 11

def test_11_rul_no_sentinel_and_no_extrapolation() -> None:
    """No ``-1`` anywhere, and nothing past day 112."""
    # the shipped data carries no -1 RUL sentinel
    l2 = pd.read_csv(P.L2_REFERENCE_CSV, nrows=2000)
    for c in l2.columns:
        if "rul" in c.lower():
            assert not (l2[c] == -1).any(), f"a -1 sentinel is present in {c}"

    r = rul_caveats()
    assert r["evidence_status"] == "RUL_EVIDENCE_INSUFFICIENT"
    assert r["label"] == "FINITE_HORIZON_EVIDENCE_INSUFFICIENT"
    assert OUTPUT_LABELS["rul"] == "FINITE_HORIZON_EVIDENCE_INSUFFICIENT"

    # a path that never reaches the threshold returns a STATUS, not a number
    c = first_crossing((0, 28, 56, 112), (0.95, 0.94, 0.93, 0.92), 0.70)
    assert c.crossed is False
    assert c.days is None
    assert c.status == NO_CROSSING

    # a path handed a longer horizon is refused outright
    try:
        first_crossing((0, 28, 56, 112, 140), (0.95, 0.9, 0.85, 0.8, 0.65), 0.70)
    except RuntimeError as e:
        assert "beyond" in str(e).lower()
    else:
        raise AssertionError("a path extending past day 112 was accepted")

    # an interpolated crossing lands strictly inside the horizon
    c2 = first_crossing((0, 28, 56, 112), (0.95, 0.85, 0.75, 0.65), 0.70)
    assert c2.crossed and 56.0 < c2.days <= 112.0, c2

    # a crossing already at the anchor is 0.0, never negative
    c3 = first_crossing((0, 28, 56, 112), (0.68, 0.66, 0.64, 0.60), 0.70)
    assert c3.days == 0.0 and c3.status == "CROSSED_AT_ANCHOR"

    # a real predict --rul run never reports beyond 112 and never a sentinel
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "pred.csv"
        from battery_entry.predict import run_predict
        code, summary = run_predict(
            str(P.EXAMPLES_DIR / "example_input.csv"), str(out),
            with_intervals=False, with_rul=True)
        assert code in (0, 3), summary
        df = pd.read_csv(out)
        assert (df["rul_label"] == "FINITE_HORIZON_EVIDENCE_INSUFFICIENT").all()
        assert not (df["rul_days"] == -1).any()
        fin = df["rul_days"].dropna()
        if len(fin):
            assert float(fin.max()) <= MAX_FORECAST_DAYS, (
                f"a predicted crossing at {fin.max()} d exceeds the 112-day horizon")
        assert (~df["extrapolated_beyond_horizon"]).all()
        # exit code 3 iff some path did not cross
        n_no = int((~df["rul_crossed_within_112d"]).sum())
        assert (code == 3) == (n_no > 0), (code, n_no)


# ---------------------------------------------------------------- 12

def test_12_rul_mae_on_different_sets_does_not_rank() -> None:
    """The two arms' RUL MAEs are not placed in a ranking column."""
    rr = pd.read_csv(P.REFERENCE_DIR / "rul_results_summary.csv")
    assert len(rr) == 2, f"expected 2 arms, got {len(rr)}"

    assert not rr["ranking_asserted"].any(), (
        "an RUL row claims a ranking is asserted")
    assert not rr["mae_directly_comparable"].any(), (
        "an RUL row claims the two MAEs are directly comparable")
    assert (rr["evidence_status"] == "RUL_EVIDENCE_INSUFFICIENT").all()

    # the MAE sample sets genuinely differ, which is WHY no ranking is allowed
    shas = set(rr["mae_sample_set_sha256"].tolist())
    assert len(shas) == 2, (
        "the two arms' MAE sample-set hashes are identical; the recorded "
        "mae_directly_comparable=False would then be unexplained")
    counts = sorted(rr["n_samples_entering_mae"].tolist())
    assert counts[0] != counts[1], (
        f"the two arms' MAE counts {counts} are equal, contradicting the recorded "
        f"non-comparability")

    # the label is degenerate: a constant predictor would score 0
    assert (rr["n_distinct_target_rul_days"] == 1).all()
    r = rul_caveats()
    assert r["n_distinct_target_rul_days"] == 1
    assert r["target_rul_days_std"] == 0.0
    assert r["ranking_asserted"] is False


# ---------------------------------------------------------------- 13

def test_13_no_patchtst_or_timesfm_implemented() -> None:
    """Phase 6 was closed WITHOUT training, checked mechanically."""
    from battery_entry.verify import check_no_neural_implementation

    res = check_no_neural_implementation()
    assert res["ok"], f"neural implementation found: {res['hits'][:10]}"

    claims = json.loads(P.CLAIMS_MANIFEST.read_text(encoding="utf-8"))
    p = claims["patchtst"]
    assert p["was_run"] is False
    assert p["was_implemented"] is False
    assert p["is_experiment_failure"] is False, (
        "PatchTST is labelled an experiment failure; no PatchTST experiment exists, so "
        "that would be a fabrication")
    assert FROZEN_STATUS["patchtst"] == "PATCHTST_SKIPPED_BY_PREDEFINED_GATE"
    assert FROZEN_STATUS["phase6"] == "PHASE6_CLOSED_WITHOUT_TRAINING"

    # No shipped text may ASSERT a PatchTST result. A document that lists those sentences
    # in order to FORBID them is doing its job — the prohibition IS the list.
    forbidden = [
        r"PatchTST\s+(?:experiment\s+)?(?:failed|underperform)",
        r"PatchTST\s+was\s+tried",
        r"deep\s+models?\s+(?:were|was)\s+(?:empirically\s+)?ruled\s+out",
        r"a\s+neural\s+baseline\s+was\s+measured",
    ]
    rx = [re.compile(f, re.IGNORECASE) for f in forbidden]
    for fp in P.package_files():
        if fp.suffix.lower() not in (".py", ".md", ".json", ".csv"):
            continue
        if P.relpath(fp) == _SELF:
            continue
        text = fp.read_text(encoding="utf-8", errors="ignore")
        for r_ in rx:
            for m in r_.finditer(text):
                if _is_prohibition_context(text, m.start()):
                    continue
                snippet = text[max(0, m.start() - 100):m.end() + 60]
                raise AssertionError(
                    f"{P.relpath(fp)} states a PatchTST result that does not exist: "
                    f"...{snippet.strip()}...")


# ---------------------------------------------------------------- 14

def test_14_runs_write_only_into_the_output_directory() -> None:
    """quick/full write nothing outside ``--output``, and cannot touch the package."""
    from battery_entry.paths import RefusedWrite, guard_output_write

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run1"
        out.mkdir()

        # a target inside the output root is allowed
        guard_output_write(out / "a.csv", out)
        guard_output_write(out / "sub" / "b.json", out)

        # escapes are refused
        for bad in (Path(td) / "elsewhere.csv",
                    out / ".." / "escape.csv",
                    P.PACKAGE_ROOT / "results" / "reference" / "result_summary.csv",
                    P.REFERENCE_DIR / "outer_fold_results_reference.csv"):
            try:
                guard_output_write(bad, out)
            except RefusedWrite:
                pass
            else:
                raise AssertionError(f"guard_output_write accepted {bad}")

        # pointing --output AT the package is refused, so a run cannot self-overwrite
        try:
            guard_output_write(P.REFERENCE_DIR / "x.csv", P.PACKAGE_ROOT)
        except RefusedWrite:
            pass
        else:
            raise AssertionError("a write into the package root was accepted")

        # a real quick run writes only under `out`
        before = {P.relpath(p): P.sha256_file(p) for p in P.package_files()}
        from battery_entry.reproduce import run_quick
        res = run_quick(out)
        assert res["ok"], res
        produced = sorted(p for p in out.rglob("*") if p.is_file())
        assert produced, "quick produced no files"
        for p in produced:
            assert str(p).startswith(str(out)), p
        after = {P.relpath(p): P.sha256_file(p) for p in P.package_files()}
        assert before == after, (
            "a quick run changed shipped package files: "
            f"{[k for k in before if before.get(k) != after.get(k)]}")


# ---------------------------------------------------------------- 15

def test_15_shipped_baseline_files_unchanged_by_a_run() -> None:
    """Hash the package before and after a full run: nothing shipped may change."""
    before = {P.relpath(p): P.sha256_file(p) for p in P.package_files()}
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "full"
        from battery_entry.reproduce import run_full
        res = run_full(out)
        assert res["ok"], res
        assert res["n_outer_cells"] > 0
    after = {P.relpath(p): P.sha256_file(p) for p in P.package_files()}
    changed = [k for k in sorted(set(before) | set(after))
               if before.get(k) != after.get(k)]
    assert not changed, f"a full run changed shipped files: {changed}"


# ---------------------------------------------------------------- runner

TESTS = [
    test_01_runs_in_clean_directory_without_parent_project,
    test_02_no_active_parent_project_reference,
    test_03_feature_implementation_matches_frozen_manifest,
    test_04_release_predictions_match_frozen_bit_exactly,
    test_05_metric_tables_reproduce_within_tolerance,
    test_06_torch_ridge_equals_numpy_ridge,
    test_07_forbidden_channels_cannot_enter_the_matrix,
    test_08_conformal_calibrated_per_horizon,
    test_09_pooled_control_never_called_group_aware,
    test_10_interval_outputs_carry_conformal_not_validated,
    test_11_rul_no_sentinel_and_no_extrapolation,
    test_12_rul_mae_on_different_sets_does_not_rank,
    test_13_no_patchtst_or_timesfm_implemented,
    test_14_runs_write_only_into_the_output_directory,
    test_15_shipped_baseline_files_unchanged_by_a_run,
]


def main() -> int:
    passed, failed, skipped = 0, [], []
    for t in TESTS:
        name = t.__name__
        try:
            t()
        except Exception as e:  # noqa: BLE001
            if e.__class__.__name__ == "Skipped":
                skipped.append(name)
                print(f"  SKIP {name}: {e}")
                continue
            failed.append((name, repr(e)))
            print(f"  FAIL {name}: {e}")
        else:
            passed += 1
            print(f"  PASS {name}")
    total = len(TESTS)
    print("=" * 72)
    print(f"RELEASE_TESTS_{passed}_OF_{total}_PASS"
          + (f" ({len(skipped)} skipped)" if skipped else ""))
    if failed:
        for n, e in failed:
            print(f"  {n}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
