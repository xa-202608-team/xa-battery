"""rul.py — the finite-horizon RUL read-out, and why it ranks nothing.

RUL here is a **secondary display**, not a model. Nothing is fitted on an RUL label and
no RUL model exists: the read-out is derived from the same H in {2,4,8} SOH point
predictions the rest of the package produces.

The four rules that make this a read-out rather than a guess, all enforced in code:

1. The discrete forecast path has exactly four nodes — the anchor at day 0 and the
   predicted absolute SOH at days 28, 56 and 112. Nothing exists past day 112.
2. A crossing is located ONLY between two adjacent nodes, by linear interpolation in
   days. No curve is fitted and no segment slope is extended.
3. If the path does not reach 0.70 by day 112, the answer is the status
   ``NO_CROSSING_WITHIN_FORECAST_HORIZON`` — never an extrapolated number.
   :func:`first_crossing` raises if handed a longer path.
4. No ``-1`` sentinel appears anywhere. A sentinel is a number, and any arithmetic that
   touches it silently produces a wrong RUL.

**Why the evidence is insufficient, structurally.** Measured on the Phase 1B layer,
``k*`` equals the number of remaining reference steps for every uncensored anchor,
because the simulation terminates at EOL (92/120 trajectories cross 0.70, and the
crossing is at the final recorded index in 92/92 cases). So "true EOL within 112 days"
forces ``k* <= 8`` while "all three horizons available" forces ``k* >= 8``; the
intersection is ``k* == 8`` exactly. All 92 eligible anchors therefore carry
``target_rul_days == 112.0``, std 0.0 — **a constant 112-day predictor would score
MAE = 0**. RUL MAE here measures how near the forecast path crosses to the end of the
record, not remaining-life discrimination. Nothing may be ranked on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

#: Attached to every RUL output this package emits. Non-negotiable.
RUL_LABEL = "FINITE_HORIZON_EVIDENCE_INSUFFICIENT"
EVIDENCE_STATUS = "RUL_EVIDENCE_INSUFFICIENT"
LABEL_STATUS = "RUL_LABEL_DEGENERATE_SINGLE_VALUE"

EOL_THRESHOLD = 0.70
WARNING_THRESHOLD = 0.80
MAX_FORECAST_DAYS = 112

#: Node 0 is the anchor, which is OBSERVED rather than predicted, so it anchors the
#: path without being part of the forecast.
NODE_DAYS: tuple[int, ...] = (0, 28, 56, 112)
NODE_HORIZONS: tuple[int | None, ...] = (None, 2, 4, 8)

NO_CROSSING = "NO_CROSSING_WITHIN_FORECAST_HORIZON"
CROSSED_AT_ANCHOR = "CROSSED_AT_ANCHOR"
CROSSED_BY_INTERPOLATION = "CROSSED_BY_INTERPOLATION"

#: The truth channel for a TRUE EOL is soh_true. Phase 1B measured that applying the
#: same rule to soh_observed moves 1,194 windows across the censoring boundary and
#: changes the trajectory split from 92/28 to 89/31, so the observed channel may not
#: decide the true EOL. (A deployed predictor only ever sees soh_observed as INPUT;
#: this rule is about defining the LABEL.)
TRUTH_CHANNEL = "soh_true"


@dataclass(frozen=True)
class Crossing:
    """First threshold crossing on one discrete forecast path."""
    crossed: bool
    days: float | None
    segment: str | None
    status: str


def first_crossing(path_days: Sequence[float],
                   path_soh: Sequence[float],
                   threshold: float) -> Crossing:
    """First crossing of ``threshold``, by linear interpolation between two nodes.

    Walks adjacent node pairs in time order and returns the first pair that brackets the
    threshold. A node already at or below the threshold counts as a crossing at that
    node's own day, so the answer is 0.0 rather than a negative number.

    Deliberately does NOT handle "the path ends above the threshold" by extending the
    last segment: that is the extrapolation the contract forbids. It returns
    :data:`NO_CROSSING` instead.
    """
    d = np.asarray(path_days, dtype=float)
    s = np.asarray(path_soh, dtype=float)
    if d.size != s.size or d.size < 2:
        raise ValueError("path must have at least two aligned nodes")
    if float(d[-1]) > MAX_FORECAST_DAYS:
        raise RuntimeError(
            f"path extends to day {d[-1]}, beyond the {MAX_FORECAST_DAYS}-day forecast "
            f"horizon; extrapolation past the final node is refused")

    if s[0] <= threshold:
        return Crossing(True, 0.0, "anchor", CROSSED_AT_ANCHOR)

    for i in range(d.size - 1):
        a, b = float(s[i]), float(s[i + 1])
        if b <= threshold:
            # bracketed by (i, i+1); a > threshold >= b so the denominator is non-zero
            frac = (a - threshold) / (a - b)
            day = float(d[i]) + frac * (float(d[i + 1]) - float(d[i]))
            return Crossing(True, day, f"{int(d[i])}->{int(d[i + 1])}d",
                            CROSSED_BY_INTERPOLATION)

    return Crossing(False, None, None, NO_CROSSING)


def readout(anchor_soh: float, soh_at_28: float, soh_at_56: float,
            soh_at_112: float) -> dict[str, Any]:
    """The full RUL read-out for one anchor, with both thresholds and every caveat.

    Returns a dict rather than a bare number precisely so a caller cannot pick the
    number up without the status and the label travelling with it.
    """
    path = [float(anchor_soh), float(soh_at_28), float(soh_at_56), float(soh_at_112)]
    eol = first_crossing(NODE_DAYS, path, EOL_THRESHOLD)
    warn = first_crossing(NODE_DAYS, path, WARNING_THRESHOLD)

    #: A warning "crossing" at day 0 is not a forecast: the anchor is ALREADY at or
    #: below 0.80, so nothing was predicted. Reporting it as lead time would restate
    #: the present observation as if it were a prediction.
    warning_is_forecast = bool(warn.crossed and warn.days is not None and warn.days > 0.0)

    return {
        "label": RUL_LABEL,
        "evidence_status": EVIDENCE_STATUS,
        "rul_days": eol.days,
        "rul_status": eol.status,
        "rul_segment": eol.segment,
        "crossed_within_112d": bool(eol.crossed),
        "warning_days": warn.days,
        "warning_status": warn.status,
        "warning_is_forecast_not_restatement": warning_is_forecast,
        "eol_threshold": EOL_THRESHOLD,
        "warning_threshold": WARNING_THRESHOLD,
        "max_forecast_days": MAX_FORECAST_DAYS,
        "path_soh": path,
        "path_days": list(NODE_DAYS),
        "sentinel_used": False,
        "extrapolated_beyond_horizon": False,
    }


def rul_caveats() -> dict[str, Any]:
    """The caveats that must accompany every RUL output. Emitted into every report."""
    return {
        "label": RUL_LABEL,
        "evidence_status": EVIDENCE_STATUS,
        "label_status": LABEL_STATUS,
        "status": "SECONDARY_DISPLAY",
        "ranking_asserted": False,
        "mae_directly_comparable": False,
        "n_eligible_anchors_in_study": 92,
        "n_distinct_target_rul_days": 1,
        "distinct_target_rul_days": [112.0],
        "target_rul_days_std": 0.0,
        "structural_cause": (
            "k* equals the number of remaining reference steps for every uncensored "
            "anchor because the simulation terminates at EOL (92/120 trajectories cross "
            "0.70, and the crossing is at the final recorded index in 92/92 cases). "
            "'True EOL within 112 days' forces k* <= 8; 'all three horizons available' "
            "forces k* >= 8. The intersection is k* == 8 exactly."),
        "consequence": (
            "A constant predictor returning 112 days would achieve MAE = 0 on the "
            "eligible set. The reported MAE measures how near the forecast path crosses "
            "to the end of the record, not remaining-life discrimination."),
        "why_the_eligible_set_was_not_widened": (
            "Relaxing 'all three horizons' would admit anchors whose day-112 node does "
            "not exist, and scoring them would require extrapolating past the forecast "
            "horizon. Relaxing 'true EOL within 112 days' would require predicting a "
            "crossing the path cannot reach. Both are refused; the degeneracy is "
            "reported instead."),
        "warning_lead_time_is_nearly_vacuous": (
            "90 of the 92 eligible anchors are ALREADY at or below the 0.80 warning "
            "threshold when observed (eligible anchors sit near end of life by "
            "construction, mean anchor SOH ~= 0.755). For those, 'the warning fires' "
            "restates the present observation rather than forecasting anything, so they "
            "are excluded from the lead-time mean — leaving 2 anchors behind each "
            "reported mean. It may not be quoted as a capability."),
        "may_not_be_claimed": [
            "RUL is validated",
            "the RUL MAE of two arms computed on different eligible sets may be compared",
            "an RUL beyond 112 days was predicted",
            "a censored trajectory's RUL is known",
            "soh_observed established the true EOL",
            "STK simulation RUL error is real-satellite RUL error",
        ],
        "data_shape_limitation_not_a_modelling_one": (
            "A dataset whose trajectories continue past EOL, or a shorter forecast "
            "horizon paired with a shorter EOL definition, would be needed to evaluate "
            "RUL meaningfully. This is a property of the data, not of the model."),
    }
