"""conformal.py — per-horizon marginal conformal intervals, and what they do NOT mean.

The half-widths this module applies are the ones Phase 5 calibrated. They are carried
in ``models/conformal_quantiles.json`` and are APPLIED here, never re-derived from any
data the caller supplies: re-calibrating after seeing coverage is exactly the move
Phase 5's ``allow_recalibrate_after_seeing_coverage`` refuses.

Four statements that must travel with every interval this module produces, and which
are attached in code (``INTERVAL_LABEL``, :func:`interval_caveats`) rather than left to
documentation:

1. **Per-horizon MARGINAL, no simultaneous guarantee.** Each horizon is calibrated
   separately because the half-width grows ~4x from 28 to 112 days (0.00341 -> 0.01406
   at 90%). Three intervals that each cover with probability 0.90 have JOINT coverage no
   greater than 0.90 and possibly far less. These are not "a 112-day band".
2. **The gate reads CONFORMAL_NOT_VALIDATED.** At 90% nominal, family-macro coverage is
   0.9045 while the worst family (``h550_i053_raan180``) covers 0.7641; the fixed
   holdout covers 0.8683. Split conformal promises a MARGINAL rate and delivered one; it
   never promised family-conditional coverage.
3. **Simulation only.** The coverage was measured on an STK-derived twin. It is not
   real-satellite coverage and certifies nothing on orbit.
4. **The pooled control is not group-aware.** ``POOLED_SPLIT_CONFORMAL`` counts
   overlapping windows (~10,000) rather than independent trajectories (100) in its
   finite-sample correction, which is why its interval is narrower and its coverage
   lower. It quantifies the cost of ignoring the group structure; it is not a candidate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from battery_entry import paths as P

#: Attached to every interval column this package emits. Non-negotiable.
INTERVAL_LABEL = "EXPERIMENTAL_SIMULATION_INTERVAL"

PRIMARY_METHOD = "FAMILY_BALANCED_CROSSFIT"
GATE_STATUS = "CONFORMAL_NOT_VALIDATED"
COVERAGE_LEVELS: tuple[float, ...] = (0.90, 0.95)
MATCHED_HORIZONS: tuple[int, ...] = (2, 4, 8)
HORIZON_DAYS: dict[int, int] = {2: 28, 4: 56, 8: 112}

SIMULTANEOUS_COVERAGE_CLAIMED = False


@dataclass(frozen=True)
class Quantiles:
    """The frozen half-widths, keyed by (protocol, arm, coverage, horizon)."""
    table: dict[str, float]
    provenance: str
    gate_status: str
    flags: tuple[str, ...]
    per_horizon_marginal_only: bool
    simultaneous_coverage_claimed: bool

    def half_width(self, coverage: float, horizon: int,
                   protocol: str = "PROTOCOL_NESTED_FAMILY_LOFO",
                   arm: str = "FINAL_MODEL") -> float:
        key = f"{protocol}|{arm}|{coverage:g}|h{horizon}"
        if key not in self.table:
            raise KeyError(
                f"no frozen half-width for {key}; the package ships calibrated "
                f"half-widths for {sorted(self.table)} and refuses to invent one")
        return float(self.table[key])


def load_quantiles() -> Quantiles:
    d = json.loads((P.MODEL_DIR / "conformal_quantiles.json").read_text(encoding="utf-8"))
    return Quantiles(
        table={k: float(v) for k, v in d["half_widths"].items()},
        provenance=d["provenance"],
        gate_status=d["gate_status"],
        flags=tuple(d["flags"]),
        per_horizon_marginal_only=bool(d["per_horizon_marginal_only"]),
        simultaneous_coverage_claimed=bool(d["simultaneous_coverage_claimed"]))


def interval(predicted_soh: np.ndarray, half_width: float
             ) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric interval about the point prediction.

    Symmetric BY CONSTRUCTION: the non-conformity score is the absolute residual
    ``|target_soh_true - predicted_soh|``, so the calibrated quantity is a half-width
    and the interval width is directly readable as +/- SOH.
    """
    p = np.asarray(predicted_soh, dtype=float)
    return p - float(half_width), p + float(half_width)


def coverage_report(truth: np.ndarray, predicted_soh: np.ndarray,
                    half_width: float,
                    trajectory_id: np.ndarray | None = None,
                    family_id: np.ndarray | None = None) -> dict[str, Any]:
    """Coverage decomposed the four ways Phase 5 reports it.

    A pooled (window-micro) number can sit exactly on nominal while one family is 20
    points short, because the families contribute 1343..2434 windows each. So
    family-macro and worst-family are always reported beside it.
    """
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(predicted_soh, dtype=float)
    lo, hi = interval(pred, half_width)
    inside = (truth >= lo) & (truth <= hi)

    out: dict[str, Any] = {
        "n_windows": int(inside.size),
        "window_micro_coverage": float(inside.mean()) if inside.size else float("nan"),
        "half_width": float(half_width),
        "mean_interval_width": 2.0 * float(half_width),
        "interval_label": INTERVAL_LABEL,
        "gate_status": GATE_STATUS,
    }
    if trajectory_id is not None:
        per = {str(t): float(inside[trajectory_id == t].mean())
               for t in np.unique(trajectory_id)}
        tv = np.array(list(per.values()))
        out["trajectory_macro_coverage"] = float(tv.mean())
        out["worst_trajectory_coverage"] = float(tv.min())
        out["n_trajectories"] = int(tv.size)
    if family_id is not None:
        per = {str(f): float(inside[family_id == f].mean())
               for f in np.unique(family_id)}
        fv = np.array(list(per.values()))
        out["family_macro_coverage"] = float(fv.mean())
        out["worst_family_coverage"] = float(fv.min())
        out["worst_family"] = min(per, key=lambda k: per[k])
        out["n_families"] = int(fv.size)
        out["per_family_coverage"] = per
    return out


def interval_caveats() -> dict[str, Any]:
    """The caveats that must accompany every interval. Emitted into every report."""
    return {
        "label": INTERVAL_LABEL,
        "gate_status": GATE_STATUS,
        "flags": ["CONFORMAL_WORST_FAMILY_SHORTFALL",
                  "FIXED_HOLDOUT_COVERAGE_SHORTFALL"],
        "family_macro_shortfall_fired": False,
        "per_horizon_marginal_only": True,
        "simultaneous_coverage_claimed": False,
        "primary_method": PRIMARY_METHOD,
        "primary_method_fixed_before_results": True,
        "measured_on": "STK-derived simulated twin",
        "may_not_be_claimed": [
            "family-conditional coverage is guaranteed",
            "coverage is guaranteed on a real satellite",
            "coverage is guaranteed out of distribution",
            "coverage is guaranteed simultaneously across the three horizons",
            "the family-balanced result proves conditional validity",
            "STK simulation coverage is real-satellite coverage",
            "the robustness control independently confirms the family-balanced result",
        ],
        "why_worst_family_shortfall_is_not_a_bug": (
            "Split conformal promises a MARGINAL rate and delivered one (family-macro "
            "0.9045 at 90% nominal). It never promised family-conditional coverage, and "
            "five of six families sit at or above nominal. Widening the interval after "
            "seeing the shortfall would be re-calibration after the fact, which Phase 5 "
            "refuses."),
        "why_the_group_aware_control_is_not_corroboration": (
            "Every family holds exactly 20 trajectories, so n_fam * n_traj_in_fam == "
            "n_traj_total and the family-balanced and trajectory-equal weight vectors "
            "are identical element-wise (measured max |delta| = 0.0). Their agreement is "
            "arithmetic, not independent evidence. The informative contrast is against "
            "POOLED_SPLIT_CONFORMAL."),
    }
