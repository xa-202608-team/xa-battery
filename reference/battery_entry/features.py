"""features.py — the frozen 11-dim SOH-geometry extractor.

**This is a verbatim copy of the authoritative implementation**
``src/production_model/features.py::common_features`` from the research project,
carried here because the delivery package may not import the parent project. The copy
is byte-compared against the original's behaviour by
``tests/test_release.py::test_03_feature_implementation_matches_frozen_manifest``,
which re-derives the feature values from the shipped L3 reference slice: if this file
ever drifted from the implementation that produced that slice, the recomputation would
stop matching and the test would fail.

Feature order is LOAD BEARING. It comes from the frozen source model manifest's
``feature_names`` and is the position of each column in the model's design matrix:

  last, diff1, diff2, diff_long, slope5, slope_full,
  curvature, ctx_mean, ctx_std, ctx_min, ctx_max

The extractor reads ONLY an observed-SOH history. It consumes no future point, no
hidden-truth channel, no exposure column and no family identifier — which is why the
11-dim space is safe to compute at prediction time.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES: list[str] = [
    "last", "diff1", "diff2", "diff_long", "slope5", "slope_full",
    "curvature", "ctx_mean", "ctx_std", "ctx_min", "ctx_max",
]

#: The L3 v2 column names for the same 11 features, in the same order. The
#: ``frozen_NN_`` prefix carries the position, so the two orders cannot drift.
FROZEN_COLUMNS: tuple[str, ...] = tuple(
    f"frozen_{i:02d}_{n}" for i, n in enumerate(FEATURE_NAMES))

FEATURE_SCHEMA_VERSION = "frozen_exact_11d_v1"
IMPLEMENTATION = "src/production_model/features.py::common_features (verbatim copy)"

#: The route context length. ARC routes use 20 reference points; this is read from the
#: frozen source manifest, not chosen here.
CONTEXT_LENGTH = 20


def _slope(seq: np.ndarray) -> float:
    k = len(seq)
    if k < 2:
        return 0.0
    x = np.arange(k, dtype=float); xm = x.mean()
    den = float(((x - xm) ** 2).sum())
    return float(((seq - seq.mean()) * (x - xm)).sum() / den) if den else 0.0


def common_features(soh_hist: np.ndarray) -> np.ndarray:
    """soh_hist:(L,) -> (11,). Single sample."""
    h = np.asarray(soh_hist, dtype=float)
    L = len(h)
    last = float(h[-1])
    return np.array([
        last,
        float(h[-1] - h[-2]) if L >= 2 else 0.0,
        float(h[-1] - h[-3]) if L >= 3 else 0.0,
        float(h[-1] - h[0]) if L >= 2 else 0.0,
        _slope(h[-min(5, L):]),
        _slope(h),
        float(h[-1] - 2 * h[-2] + h[-3]) if L >= 3 else 0.0,
        float(h.mean()), float(h.std()), float(h.min()), float(h.max()),
    ], dtype=float)


def common_features_batch(soh_hist_batch: np.ndarray) -> np.ndarray:
    """soh_hist:(n,L) -> (n,11)."""
    return np.stack(
        [common_features(soh_hist_batch[i]) for i in range(soh_hist_batch.shape[0])],
        axis=0)


def feature_names_sha256() -> str:
    """SHA-256 over the newline-joined frozen feature order.

    Matches the ``source_feature_names_sha256`` the Phase 1B layer records, so a
    reordering of the 11 is detectable by hash rather than by inspection.
    """
    import hashlib
    return hashlib.sha256("\n".join(FEATURE_NAMES).encode("utf-8")).hexdigest()
