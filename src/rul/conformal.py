"""conformal.py — split-conformal RUL intervals, family-balanced.

Method mirrors ``src/stk_transfer_v2/uncertainty/conformal_p5.py``'s
``FAMILY_BALANCED_CROSSFIT`` in the part that matters here: the calibration quantile
is taken so that every family contributes equally, rather than letting the families
with more anchors dominate the quantile.

SCOPE LIMITS — these are the ones the report must carry
-------------------------------------------------------
* The interval is MARGINAL per anchor. It is NOT a simultaneous band over a
  trajectory, and must never be described as one.
* Calibration uses UNCENSORED anchors only, because a censored anchor has no point
  label to score a residual against. Coverage is therefore an uncensored-anchor
  statement, and the censored subset is reported separately as a one-sided check
  (does the interval's lower end respect the observed follow-up?).
* Simulation domain. Not satellite accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalRul:
    coverage_target: float
    half_width_days: float
    method: str
    n_calibration: int
    n_families: int
    per_family_half_width: dict
    marginal_only: bool = True
    simultaneous_coverage_claimed: bool = False


def family_balanced_quantile(residuals: np.ndarray, family: np.ndarray,
                            coverage: float) -> tuple[float, dict]:
    """Per-family quantile, then the macro (mean) across families.

    Averaging family quantiles rather than pooling residuals keeps a large family
    from setting the width for everyone. Uses the finite-sample conformal index
    ``ceil((n+1) * coverage) / n`` per family.
    """
    per = {}
    for f in sorted(set(family.tolist())):
        r = np.sort(np.abs(residuals[family == f]))
        n = len(r)
        if n == 0:
            continue
        k = int(np.ceil((n + 1) * coverage))
        per[f] = float(r[min(k, n) - 1])
    if not per:
        raise RuntimeError("no calibration residuals")
    return float(np.mean(list(per.values()))), per


def calibrate(residuals: np.ndarray, family: np.ndarray,
              coverage: float = 0.90) -> ConformalRul:
    hw, per = family_balanced_quantile(residuals, family, coverage)
    return ConformalRul(
        coverage_target=float(coverage), half_width_days=hw,
        method="FAMILY_BALANCED_SPLIT_CONFORMAL_ON_UNCENSORED_ANCHORS",
        n_calibration=int(len(residuals)), n_families=len(per),
        per_family_half_width=per)


def evaluate_coverage(y_true: np.ndarray, y_pred: np.ndarray,
                      half_width: float) -> dict:
    """Empirical marginal coverage and mean interval width, on uncensored anchors."""
    lo, hi = y_pred - half_width, y_pred + half_width
    cov = float(np.mean((y_true >= lo) & (y_true <= hi)))
    return {"empirical_coverage": cov, "mean_width_days": float(2 * half_width),
            "n": int(len(y_true))}


def censored_one_sided_check(lower_bound: np.ndarray, y_pred: np.ndarray,
                             half_width: float) -> dict:
    """For censored anchors: does the interval's UPPER end reach the observed bound?

    A censored anchor's truth is ``RUL >= lower_bound``. The interval is consistent
    with that truth whenever ``y_pred + half_width >= lower_bound``. Falling short is
    a genuine miss (the model asserts the cell died before we last saw it alive);
    exceeding it says nothing, which is why this is one-sided.
    """
    ok = (y_pred + half_width) >= lower_bound
    return {"censored_consistent_fraction": float(np.mean(ok)),
            "n_censored": int(len(ok)),
            "n_censored_violations": int((~ok).sum())}
