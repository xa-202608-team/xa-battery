"""space.py — the frozen ARC clip/scaler, and the frozen source coefficients.

Two halves, with different provenance and different rules:

* :class:`FrozenSpace` — clip/scaler, loaded from v2's own
  ``models/arc_clean_fixed_space.npz``. **Applied, never refitted.** This mirrors
  ``battery_release_v2/battery_entry/model.py::FrozenSpace`` and is the half that
  transfers across the domain boundary.
* :func:`load_source_prior` — the frozen source coefficients ``beta_src``. These are
  DELIBERATELY ABSENT from v2's npz; its own ``note`` field says shipping them
  "would invite that misuse", because Phase 3 ruled ``SOURCE_PRIOR_NOT_VALIDATED``.
  L2-SP needs them, so they are read from the authoritative source-model artifact
  identified by SHA-256 match (see ``paths.SOURCE_MODEL_NPZ``).

The Phase 3 verdict is NOT overturned by reading them: lambda_SP = 0 remains a
legitimate search outcome, and the delivered predictor is unchanged. See
``GATES_scope_amendment.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import paths as P

#: v2 horizons, in steps of the 14-day reference grid.
HORIZON_STEPS: tuple[int, ...] = (2, 4, 8)
REFERENCE_STEP_DAYS = 14
EOL_SOH = 0.70
WARN_SOH = 0.80


@dataclass(frozen=True)
class FrozenSpace:
    """The frozen source ARC clip/scaler. Applied, never refitted."""
    name: str
    clip_lo: np.ndarray
    clip_hi: np.ndarray
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    feature_names: tuple[str, ...]
    source_model_id: str
    model_npz_sha256: str

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Clip to the frozen bounds, then z-score with the frozen statistics.

        Bit-identical to v2's ``FrozenSpace.apply``; asserted by
        ``tests/test_equivalence_with_v2.py``.
        """
        Xc = np.clip(X, self.clip_lo, self.clip_hi)
        return (Xc - self.scaler_mean) / self.scaler_std


def load_frozen_space() -> FrozenSpace:
    """Load the frozen ARC space from v2 (read-only)."""
    z = np.load(P.ARC_SPACE_NPZ, allow_pickle=True)
    return FrozenSpace(
        name=str(z["space_name"]),
        clip_lo=np.asarray(z["clip_lo"], dtype=float),
        clip_hi=np.asarray(z["clip_hi"], dtype=float),
        scaler_mean=np.asarray(z["scaler_mean"], dtype=float),
        scaler_std=np.asarray(z["scaler_std"], dtype=float),
        feature_names=tuple(str(x) for x in np.asarray(z["feature_names"]).tolist()),
        source_model_id=str(z["source_model_id"]),
        model_npz_sha256=str(z["model_npz_sha256"]))


def _sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class SourcePrior:
    """The frozen source-domain head: ``beta_src`` for L2-SP.

    ``coef`` is ``(H_src, 11)`` and ``intercept`` is ``(H_src,)`` in the source
    model's own horizon layout (H_src = 10 one-step-ahead horizons). ``for_horizon``
    picks the column L2-SP should pull toward for a given v2 horizon.
    """
    coef: np.ndarray
    intercept: np.ndarray
    feature_names: tuple[str, ...]
    alpha: float
    context_length: int
    npz_sha256: str
    provenance: str

    def for_horizon(self, horizon_steps: int) -> tuple[np.ndarray, float]:
        """``(beta_src, b_src)`` for one v2 horizon.

        The source model was fitted on ARC cycle-indexed horizons 1..10; v2's
        horizons are 2/4/8 reference steps. The mapping is positional and declared
        here rather than searched: horizon ``h`` steps uses source row ``h - 1``,
        i.e. the source head trained to predict ``h`` units ahead. No horizon is
        extrapolated (v2's max is 8 <= H_src = 10), which is the same
        no-extrapolation rule ``build_l3_v2.py`` applies.
        """
        i = int(horizon_steps) - 1
        if not (0 <= i < self.coef.shape[0]):
            raise ValueError(
                f"horizon {horizon_steps} outside source support "
                f"1..{self.coef.shape[0]} — extrapolation is barred")
        return self.coef[i].astype(float), float(self.intercept[i])


def load_source_prior() -> SourcePrior:
    """Load ``beta_src`` and verify it is THE model that produced v2's ARC space.

    The verification is the point: v2 records ``model_npz_sha256`` for the source
    model it derived clip/scaler from. If this artifact's hash matches, the prior is
    provably from the same fit — not a lookalike from another sensitivity arm.
    """
    space = load_frozen_space()
    sha = _sha256(P.SOURCE_MODEL_NPZ)
    if sha != space.model_npz_sha256:
        raise RuntimeError(
            "source-model SHA-256 mismatch: this artifact did not produce v2's "
            f"frozen ARC space.\n  expected {space.model_npz_sha256}\n  got      {sha}")

    z = np.load(P.SOURCE_MODEL_NPZ, allow_pickle=True)
    names = tuple(str(x) for x in np.asarray(z["feature_names"]).tolist())
    if names != space.feature_names:
        raise RuntimeError(f"feature order differs: {names} vs {space.feature_names}")

    return SourcePrior(
        coef=np.asarray(z["coef"], dtype=float),
        intercept=np.asarray(z["intercept"], dtype=float),
        feature_names=names,
        alpha=float(z["alpha"]),
        context_length=int(z["L"]),
        npz_sha256=sha,
        provenance=(
            "artifacts/experimental_arc_sensitivity/arc_clean_fixed_alpha10/model.npz; "
            "SHA-256 matches battery_release_v2/models/arc_clean_fixed_space.npz"
            "::model_npz_sha256, i.e. the source fit that produced the frozen space. "
            "v2 omits these coefficients by design (Phase 3 "
            "SOURCE_PRIOR_NOT_VALIDATED); reading them here does not overturn that "
            "verdict — see GATES_scope_amendment.md."))
