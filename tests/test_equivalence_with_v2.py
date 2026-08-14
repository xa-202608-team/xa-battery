"""test_equivalence_with_v2.py — v3 is a strict superset of v2, proven bit-for-bit.

This file is the load-bearing wall of the two-package structure. It proves v3 did not
start over: the 11-dim features and the ARC space it computes are IDENTICAL to v2's,
so every v2 result remains the reference for v3's.

It also guards the hard boundary: v2 must be byte-identical before and after, and must
contain no neural-network code (which is why the torch arms live only in v3).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths as P                                            # noqa: E402
from src.features import frozen11 as F3                               # noqa: E402
from src.features.space import load_frozen_space, load_source_prior    # noqa: E402

#: v2's own extractor, imported directly from the frozen package.
sys.path.insert(0, str(P.V2_ROOT))
from battery_entry import features as F2                              # noqa: E402
from battery_entry import model as M2                                 # noqa: E402

PY = sys.executable


# ------------------------------------------------------------------ feature parity

def test_01_feature_module_is_byte_identical_to_v2():
    """v3's extractor is a verbatim copy — compared by file hash, not by reading."""
    a = hashlib.sha256((P.V2_ROOT / "battery_entry" / "features.py").read_bytes()).hexdigest()
    b = hashlib.sha256(Path(F3.__file__).with_suffix(".py").read_bytes()).hexdigest()
    assert a == b, "v3 frozen11.py has drifted from v2 features.py"


def test_02_feature_order_and_hash_match():
    assert F3.FEATURE_NAMES == F2.FEATURE_NAMES
    assert F3.FROZEN_COLUMNS == F2.FROZEN_COLUMNS
    assert F3.CONTEXT_LENGTH == F2.CONTEXT_LENGTH == 20
    assert F3.feature_names_sha256() == F2.feature_names_sha256()


def test_03_features_recomputed_from_l3_context_match_shipped_columns_exactly():
    """Recompute all 11 features from ctx_soh_00..19 and require exact agreement.

    "Exact" here means to the precision the CSV itself carries. The shipped columns
    were written with ``%.17g``, which round-trips a float64 — but the ctx_soh_NN
    columns were written the same way, so recomputing from them reproduces the
    features only up to the rounding of their own inputs. The measured residual is
    ~7e-16, i.e. a few ULP at this magnitude, which is float64 round-trip noise and
    not an implementation difference.

    The genuinely bit-exact statement is
    :func:`test_04_v3_and_v2_extractors_agree_on_random_histories`: given the SAME
    in-memory history, the two extractors return bit-identical arrays. That is the
    claim that establishes v3's design matrix is v2's.
    """
    df = pd.read_csv(P.L3_WINDOWS_CSV, float_precision="round_trip")
    ctx = df[[f"ctx_soh_{i:02d}" for i in range(20)]].to_numpy(dtype=float)
    got = F3.common_features_batch(ctx)
    want = df[list(F3.FROZEN_COLUMNS)].to_numpy(dtype=float)

    assert got.shape == want.shape
    diff = np.max(np.abs(got - want))
    # 8 ULP at the O(1) scale of SOH. Tight enough that a real formula change
    # (a different slope convention, a shifted index) could not slip through.
    assert diff < 8 * np.finfo(float).eps, (
        f"recomputed features differ from the shipped frozen columns by {diff:.3e}, "
        "which is larger than float64 round-trip noise")


def test_04_v3_and_v2_extractors_agree_on_random_histories():
    rng = np.random.default_rng(12345)
    for _ in range(200):
        h = np.cumsum(-np.abs(rng.normal(0, 0.002, size=20))) + 1.0
        assert np.array_equal(F3.common_features(h), F2.common_features(h))


# -------------------------------------------------------------------- ARC space

def test_05_frozen_space_arrays_match_v2_exactly():
    s3 = load_frozen_space()
    s2 = M2.load_frozen_space()
    for k in ("clip_lo", "clip_hi", "scaler_mean", "scaler_std"):
        assert np.array_equal(getattr(s3, k), getattr(s2, k)), f"{k} differs"
    assert s3.name == s2.name
    assert s3.source_model_id == s2.source_model_id
    assert s3.feature_names == s2.feature_names


def test_06_space_apply_is_bit_identical_to_v2():
    """The transferred coordinate system produces identical outputs in both packages."""
    df = pd.read_csv(P.L3_WINDOWS_CSV, float_precision="round_trip")
    X = df[list(F3.FROZEN_COLUMNS)].to_numpy(dtype=float)
    s3, s2 = load_frozen_space(), M2.load_frozen_space()
    assert np.array_equal(s3.apply(X), s2.apply(X))


# ------------------------------------------------------- protocol / solver parity

def test_07_weighted_ridge_matches_v2_solver():
    from src.features.protocol import solve_ridge_weighted as s3
    rng = np.random.default_rng(7)
    Z = rng.normal(size=(300, 11))
    y = rng.normal(size=300) * 0.01
    w = rng.uniform(0.5, 2.0, size=300)
    for a in (1e-4, 1.0, 100.0):
        assert np.allclose(s3(Z, y, w, a), M2.solve_ridge_weighted(Z, y, w, a),
                           rtol=0, atol=0), f"ridge solve differs at alpha={a}"


def test_08_trajectory_weights_match_v2():
    from src.features.protocol import trajectory_equal_weights as w3
    df = pd.read_csv(P.L3_WINDOWS_CSV, usecols=[
        "trajectory_id", "environment_family_id", "horizon_steps"])
    tid = df.trajectory_id.to_numpy()
    fam = df.environment_family_id.to_numpy()
    for h in (2, 4, 8):
        m = (df.horizon_steps.to_numpy() == h)
        assert np.array_equal(w3(tid, fam, m),
                              M2.trajectory_equal_weights(tid, fam, m))


def test_09_l2sp_at_lambda_zero_reduces_to_plain_weighted_ridge():
    """lam_SP = 0 must be EXACTLY v2's ridge — the limit that anchors the spectrum."""
    from src.finetune.l2sp import solve_l2sp_closed_form
    rng = np.random.default_rng(99)
    Z = rng.normal(size=(400, 11))
    y = rng.normal(size=400) * 0.01
    w = rng.uniform(0.5, 2.0, size=400)
    bs = rng.normal(size=11)
    for a in (1e-3, 1.0, 100.0):
        got = solve_l2sp_closed_form(Z, y, w, bs, alpha=a, lambda_sp=0.0)
        want = M2.solve_ridge_weighted(Z, y, w, a)
        assert np.allclose(got, want, rtol=0, atol=1e-12), (
            f"lam_SP=0 does not reduce to weighted ridge at alpha={a}")


def test_10_l2sp_large_lambda_drives_beta_to_the_prior():
    """The other end of the spectrum: lam_SP -> inf is zero-shot transfer."""
    from src.finetune.l2sp import solve_l2sp_closed_form
    rng = np.random.default_rng(101)
    Z = rng.normal(size=(200, 11))
    y = rng.normal(size=200) * 0.01
    w = np.ones(200)
    bs = rng.normal(size=11)
    far = solve_l2sp_closed_form(Z, y, w, bs, alpha=0.0, lambda_sp=1e12)
    assert np.max(np.abs(far - bs)) < 1e-6, "large lam_SP did not approach beta_src"


# ---------------------------------------------------------------- source prior

def test_11_source_prior_sha256_matches_the_frozen_record():
    """beta_src must come from the fit that produced v2's frozen space."""
    prior = load_source_prior()
    space = load_frozen_space()
    assert prior.npz_sha256 == space.model_npz_sha256
    assert prior.coef.shape[1] == 11
    assert prior.feature_names == space.feature_names


def test_12_source_prior_bars_horizon_extrapolation():
    prior = load_source_prior()
    for h in (2, 4, 8):
        b, b0 = prior.for_horizon(h)
        assert b.shape == (11,) and np.isfinite(b).all() and np.isfinite(b0)
    with pytest.raises(ValueError):
        prior.for_horizon(prior.coef.shape[0] + 1)


# ------------------------------------------------------------ the hard boundary

def test_13_v2_contains_no_neural_implementation():
    """The reason every nn.Module lives in v3: v2's verify scans for exactly this."""
    banned = ("nn.Module", "torch.nn", "import torch.nn")
    hits = []
    for p in P.V2_ROOT.rglob("*.py"):
        t = p.read_text(encoding="utf-8", errors="ignore")
        for b in banned:
            if b in t:
                hits.append((str(p.relative_to(P.V2_ROOT)), b))
    assert not hits, f"neural code found inside the frozen package: {hits}"


def test_14_v3_neural_code_is_confined_to_v3():
    """And the converse: v3 really does carry the nn.Module."""
    found = any("nn.Module" in p.read_text(encoding="utf-8", errors="ignore")
                for p in (P.V3_ROOT / "src").rglob("*.py"))
    assert found, "expected at least one nn.Module in battery_pytorch_v3"


def test_15_v2_sha256sums_still_verify():
    """v2's own SHA256SUMS must still match — the byte-level read-only guarantee."""
    lines = (P.V2_ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    bad = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        want, rel = parts[0], parts[1].strip().lstrip("*")
        f = P.V2_ROOT / rel
        if not f.exists():
            bad.append((rel, "MISSING"))
            continue
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        if h.hexdigest() != want:
            bad.append((rel, "HASH MISMATCH"))
    assert not bad, f"v2 files changed: {bad}"


def test_16_v2_verify_still_reports_7_of_7():
    """The hard acceptance item, asserted from inside the test suite.

    Decoding is forced to UTF-8 with replacement: this box's console default is GBK,
    which cannot decode v2's output and would fail the test for a reason that has
    nothing to do with v2's integrity.
    """
    r = subprocess.run([PY, "-m", "battery_entry", "verify"],
                       cwd=str(P.V2_ROOT), capture_output=True,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    assert "VERIFY_OK" in out, f"v2 verify did not report VERIFY_OK:\n{out}"
    assert "7/7" in out, f"v2 verify did not report 7/7:\n{out}"


def test_17_v3_writes_nothing_into_v2():
    """No v3 output path may resolve inside the frozen package."""
    v2 = P.V2_ROOT.resolve()
    for p in (P.RESULTS_DIR, P.DOCS_DIR, P.DOCKER_DIR):
        assert v2 not in p.resolve().parents and p.resolve() != v2
