"""model.py — the frozen point model, its space, and the ridge solve path.

Three things live here, and each is a REUSE of a frozen research-project component
rather than a new algorithm:

1. :class:`FrozenSpace` — the frozen source ARC clip/scaler. Loaded from the shipped
   ``models/arc_clean_fixed_space.npz``, which is a copy of the frozen Task B package's
   own arrays. It is applied, never refitted. This is the transferred half.
2. :func:`solve_ridge_weighted` — the weighted ridge closed form, mirroring
   ``src/stk_transfer_v2/baselines/engine_p2b.py::_solve_ridge_weighted`` exactly,
   including the mean-1.0 weight normalisation that keeps ``alpha`` comparable with the
   frozen unweighted fit.
3. :func:`fit_head` / :func:`predict_delta` — the target-domain head refit. This is the
   half that does NOT transfer: Phase 3 ruled SOURCE_PRIOR_NOT_VALIDATED, so the frozen
   source coefficients are never used as a prior, and ``lambda_source`` does not exist
   in this package.

On the PyTorch question (competition requirement): :func:`solve_ridge_weighted_torch`
solves the SAME normal equations on the SAME float64 CPU path, and is the
``lambda_source = 0`` limit of the already-verified Phase 3 Source-Prior solver — it is
not a different algorithm. ``tests/test_release.py::test_06_torch_ridge_equals_numpy_ridge``
proves the two agree, and the default path stays NumPy so torch is optional.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from battery_entry import paths as P

#: Trajectory-equal weighting: every training trajectory carries total weight 1.0
#: inside a horizon stratum, then every family is balanced. Uniform row weighting
#: would let a 137-window trajectory count ~5.7x a 24-window one.
WEIGHTING_RULE = (
    "w_i = 1/n_windows(trajectory_i, horizon) then /= n_trajectories(family_i); "
    "each family sums to 1.0")


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
    clip_sha256: str
    scaler_sha256: str

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Clip to the frozen bounds, then z-score with the frozen statistics."""
        Xc = np.clip(X, self.clip_lo, self.clip_hi)
        return (Xc - self.scaler_mean) / self.scaler_std


def load_frozen_space() -> FrozenSpace:
    """Load the frozen ARC space from inside the package."""
    z = np.load(P.ARC_SPACE_NPZ, allow_pickle=True)
    names = tuple(str(x) for x in np.asarray(z["feature_names"]).tolist())
    return FrozenSpace(
        name=str(z["space_name"]),
        clip_lo=np.asarray(z["clip_lo"], dtype=float),
        clip_hi=np.asarray(z["clip_hi"], dtype=float),
        scaler_mean=np.asarray(z["scaler_mean"], dtype=float),
        scaler_std=np.asarray(z["scaler_std"], dtype=float),
        feature_names=names,
        source_model_id=str(z["source_model_id"]),
        model_npz_sha256=str(z["model_npz_sha256"]),
        clip_sha256=str(z["clip_sha256"]),
        scaler_sha256=str(z["scaler_sha256"]))


# ------------------------------------------------------------------ weights

def trajectory_equal_weights(trajectory_id: np.ndarray,
                             family_id: np.ndarray,
                             mask: np.ndarray) -> np.ndarray:
    """Row weights: trajectory-equal inside a horizon stratum, then family-balanced.

    Mirrors ``engine_p2b.trajectory_equal_weights``. The caller is responsible for
    passing a ``mask`` that is already restricted to ONE horizon; mixing horizons here
    would silently break the budget ladder's premise.
    """
    w = np.zeros(len(trajectory_id), dtype=float)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return w

    sub_traj = trajectory_id[idx]
    uniq, inv, counts = np.unique(sub_traj, return_inverse=True, return_counts=True)
    w[idx] = 1.0 / counts[inv]

    traj_family = {t: f for t, f in zip(sub_traj, family_id[idx])}
    fam_traj_count: dict[Any, int] = {}
    for t in uniq:
        fam_traj_count[traj_family[t]] = fam_traj_count.get(traj_family[t], 0) + 1
    denom = np.array([fam_traj_count[traj_family[t]] for t in sub_traj], dtype=float)
    w[idx] = w[idx] / denom
    return w


# ------------------------------------------------------------------ ridge

def solve_ridge_weighted(Z: np.ndarray, y: np.ndarray, w: np.ndarray,
                         alpha: float) -> np.ndarray:
    """Weighted ridge closed form. Mirrors ``engine_p2b._solve_ridge_weighted``.

    Weights are normalised to mean 1.0 so ``alpha`` keeps the meaning it had in the
    frozen unweighted fit — otherwise the trajectory weights (which sum to n_families,
    not n_rows) would silently rescale the regularisation.
    """
    ww = w / w.mean()
    Zw = Z * ww[:, None]
    A = Z.T @ Zw + alpha * np.eye(Z.shape[1])
    return np.linalg.solve(A, Zw.T @ y)


def solve_ridge_weighted_torch(Z: np.ndarray, y: np.ndarray, w: np.ndarray,
                               alpha: float) -> np.ndarray:
    """The SAME normal equations, solved on torch's float64 CPU path.

    This exists because the competition may require a PyTorch code path. It is the
    ``lambda_source = 0`` limit of the Phase 3 Source-Prior solver
    (``src/stk_transfer_v2/models/source_prior_ridge.py``), which that phase already
    verified reduces to plain weighted ridge — so this is a reuse of a validated path,
    not a second algorithm. float64, CPU, no RNG.
    """
    import torch

    dtype, device = torch.float64, torch.device("cpu")
    tZ = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float64)).to(device, dtype)
    ty = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float64)).to(device, dtype)
    tw = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float64)).to(device, dtype)
    tww = tw / tw.mean()
    Zw = tZ * tww[:, None]
    A = tZ.T @ Zw + float(alpha) * torch.eye(tZ.shape[1], dtype=dtype, device=device)
    B = Zw.T @ ty
    X = torch.linalg.solve(A, B)
    return X.detach().cpu().numpy().astype(np.float64)


@dataclass
class Head:
    """A target-domain regression head: what the refit produces."""
    coef: np.ndarray
    intercept: float
    alpha: float
    n_train_rows: int
    n_train_trajectories: int
    space_name: str
    space_owner: str = "REUSED_FROZEN_ARC_CLIP_SCALER"
    solver: str = "numpy"


def fit_head(X: np.ndarray,
             y: np.ndarray,
             w: np.ndarray,
             train_mask: np.ndarray,
             space: FrozenSpace,
             alpha: float,
             trajectory_id: np.ndarray,
             solver: str = "numpy") -> Head:
    """Refit the regression head on target-domain rows only.

    ``space`` is REUSED unchanged — this function never fits a clip or a scaler. The
    intercept convention matches Phase 4's ``fit_candidate``: a weight-normalised
    centring, so the head is comparable with the frozen Phase 4/5 predictions.
    """
    Z = space.apply(X)
    Ztr, ytr, wtr = Z[train_mask], y[train_mask], w[train_mask]
    if wtr.sum() <= 0:
        raise RuntimeError("training weights sum to zero")

    if solver == "torch":
        coef = solve_ridge_weighted_torch(Ztr, ytr, wtr, alpha)
    elif solver == "numpy":
        coef = solve_ridge_weighted(Ztr, ytr, wtr, alpha)
    else:
        raise ValueError(f"unknown solver {solver!r}")

    ww = wtr / wtr.sum()
    intercept = float(ytr @ ww - (Ztr * ww[:, None]).sum(axis=0) @ coef)
    return Head(coef=coef, intercept=intercept, alpha=float(alpha),
                n_train_rows=int(train_mask.sum()),
                n_train_trajectories=int(len(set(
                    trajectory_id[train_mask].tolist()))),
                space_name=space.name, solver=solver)


def predict_delta(X: np.ndarray, space: FrozenSpace, head: Head) -> np.ndarray:
    """Predicted SOH CHANGE over the horizon, on the delta scale."""
    return head.intercept + space.apply(X) @ head.coef


def predict_absolute(X: np.ndarray, anchor: np.ndarray,
                     space: FrozenSpace, head: Head) -> np.ndarray:
    """Predicted ABSOLUTE SOH at the target point.

    The anchor is ``anchor_soh_observed`` — the NOISY observed SOH, which is what a
    deployed predictor actually reads. It is an anchor, never a design column: it is
    bit-identical to ``frozen_00_last``, so including it would add a perfectly
    collinear duplicate rather than information.
    """
    return anchor + predict_delta(X, space, head)


# ------------------------------------------------------------------ baseline

def observable_local_trend(slope_full: np.ndarray,
                           horizon_steps: np.ndarray) -> np.ndarray:
    """The high-stability engineering baseline. No fit, no label, no future point.

    Closed-form extrapolation of the OBSERVABLE slope over the horizon: reads
    ``frozen_05_slope_full`` and the horizon only. Phase 2B's lowest across-family std,
    which is why it ships beside the fitted model rather than behind it.
    """
    return np.asarray(slope_full, dtype=float) * np.asarray(horizon_steps, dtype=float)


def persistence(n: int) -> np.ndarray:
    """The non-learning floor: predicted delta == 0 from the observable anchor."""
    return np.zeros(int(n), dtype=float)
