"""protocol.py — the frozen evaluation protocol, reused verbatim in intent.

Three things, all mirroring the frozen research components named in ``REUSE_MAP.md``:

* :func:`trajectory_equal_weights` — mirrors
  ``battery_release_v2/battery_entry/model.py::trajectory_equal_weights``, itself a
  mirror of ``engine_p2b.trajectory_equal_weights``.
* :func:`solve_ridge_weighted` — mirrors ``engine_p2b._solve_ridge_weighted``,
  including the mean-1.0 weight normalisation that keeps ``alpha`` comparable with
  the frozen unweighted fit.
* :func:`nested_family_lofo` — the leave-one-environment-family-out outer loop with
  inner-fold hyper-parameter selection. ``environment_family_id`` is the strict
  target-domain isolation key; nothing selects on the held-out family.
"""
from __future__ import annotations

from typing import Callable, Iterator, Sequence

import numpy as np

#: Documented rule, carried over from v2 so the two cannot drift silently.
WEIGHTING_RULE = (
    "w_i = 1/n_windows(trajectory_i, stratum) then /= n_trajectories(family_i); "
    "each family sums to 1.0")


def trajectory_equal_weights(trajectory_id: np.ndarray,
                             family_id: np.ndarray,
                             mask: np.ndarray) -> np.ndarray:
    """Trajectory-equal inside a stratum, then family-balanced.

    The caller must pass a ``mask`` restricted to ONE stratum (one horizon, or one
    RUL anchor set). Mixing strata here would break the premise that every
    trajectory carries equal total weight within the stratum.
    """
    w = np.zeros(len(trajectory_id), dtype=float)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return w

    sub_traj = trajectory_id[idx]
    uniq, inv, counts = np.unique(sub_traj, return_inverse=True, return_counts=True)
    w[idx] = 1.0 / counts[inv]

    traj_family = {t: f for t, f in zip(sub_traj, family_id[idx])}
    fam_traj_count: dict = {}
    for t in uniq:
        fam_traj_count[traj_family[t]] = fam_traj_count.get(traj_family[t], 0) + 1
    denom = np.array([fam_traj_count[traj_family[t]] for t in sub_traj], dtype=float)
    w[idx] = w[idx] / denom
    return w


def solve_ridge_weighted(Z: np.ndarray, y: np.ndarray, w: np.ndarray,
                         alpha: float) -> np.ndarray:
    """Weighted ridge closed form, no intercept column.

    Weights are normalised to mean 1.0 so ``alpha`` keeps the meaning it had in the
    frozen unweighted fit.
    """
    ww = w / w.mean()
    Zw = Z * ww[:, None]
    A = Z.T @ Zw + alpha * np.eye(Z.shape[1])
    return np.linalg.solve(A, Zw.T @ y)


def weighted_intercept(Z: np.ndarray, y: np.ndarray, w: np.ndarray,
                       coef: np.ndarray) -> float:
    """v2's intercept convention: weight-normalised centring (``fit_head``)."""
    ww = w / w.sum()
    return float(y @ ww - (Z * ww[:, None]).sum(axis=0) @ coef)


def nested_family_lofo(families: Sequence,
                       ) -> Iterator[tuple[object, list]]:
    """Yield ``(held_out_family, inner_families)`` for the outer LOFO loop.

    The inner list never contains the held-out family, which is the whole point:
    hyper-parameters are chosen on inner folds only, so
    ``selector_saw_outer_test`` stays False as it does in ``frozen_alpha.json``.
    """
    fams = sorted(families)
    for held in fams:
        yield held, [f for f in fams if f != held]


def select_on_inner_folds(inner_families: list,
                          family_of_row: np.ndarray,
                          outer_train: np.ndarray,
                          grid: Sequence[float],
                          score: Callable[[np.ndarray, np.ndarray, float], float],
                          ) -> tuple[float, float]:
    """Pick one hyper-parameter by mean inner-fold score. Lower is better.

    ``score(inner_train_mask, inner_test_mask, value)`` is supplied by the caller so
    this stays agnostic about what is being fitted. Ties resolve to the FIRST grid
    entry, so the choice is a deterministic function of the declared grid order.
    """
    best_v, best_e = None, np.inf
    for v in grid:
        errs = []
        for inner in inner_families:
            itr = outer_train & (family_of_row != inner)
            ite = outer_train & (family_of_row == inner)
            if not itr.any() or not ite.any():
                continue
            errs.append(score(itr, ite, v))
        if not errs:
            continue
        m = float(np.mean(errs))
        if m < best_e - 1e-15:
            best_v, best_e = float(v), m
    if best_v is None:
        raise RuntimeError("no inner fold produced a score")
    return best_v, best_e


def family_macro(per_family: dict) -> float:
    """Macro average over families: every family counts once, regardless of size."""
    vals = [v for v in per_family.values() if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))
