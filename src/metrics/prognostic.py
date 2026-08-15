"""prognostic.py — earliness metrics: warning lead time and Prognostic Horizon.

Both definitions come from the closeout plan's Appendix B; PH follows Saxena et al.
(2008). Neither existed in the historical assets (see ``REUSE_MAP.md`` N2), so this
module is NEW.

WARNING LEAD TIME
-----------------
    dt_warn = t_true(SOH = 0.80) - t_alarm

``t_alarm`` is the first anchor at which the model predicts SOH will fall below 0.80
within its forecast reach. ``dt_warn > 0`` is a useful early warning; ``< 0`` is a
late (post-hoc) alarm. Resolution is bounded by the 14-day reference step, so a
lead time is only meaningful to +/-14 days and is reported as a distribution, never
as a single headline number.

Three outcomes per trajectory, and all three are counted:
  * ``TRUE_POSITIVE``  — crossed 0.80, and the model alarmed at or before the crossing
  * ``LATE``           — crossed 0.80, model alarmed after the crossing (dt_warn < 0)
  * ``MISSED``         — crossed 0.80, model never alarmed
  * ``FALSE_POSITIVE`` — never crossed 0.80, but the model alarmed
  * ``TRUE_NEGATIVE``  — never crossed, never alarmed

PROGNOSTIC HORIZON
------------------
    PH = t_EOL - min{ t : |RUL_hat(t') - RUL(t')| <= alpha * RUL(t')  for all t' >= t }

The quantifier is "for all t' >= t", so PH is found by scanning BACKWARD from EOL and
stopping at the first anchor that leaves the alpha band — an earlier accidental hit
inside the band does not count if the model later leaves it. alpha = 0.2.

A larger PH means credible predictions arrive earlier. PH is only defined for
trajectories that reach EOL, so censored trajectories are excluded from PH (and that
exclusion is reported, not silent).
"""
from __future__ import annotations

import numpy as np

ALPHA_PH = 0.2
WARN_SOH = 0.80
EOL_SOH = 0.70


def true_threshold_crossing_day(day: np.ndarray, soh_true: np.ndarray,
                                threshold: float) -> float:
    """First reference day at which the TRUE SOH is at or below ``threshold``.

    Returns NaN when the trajectory never crosses within the mission (censored at
    this threshold). No interpolation: the reference grid is the observable one, and
    inventing sub-step precision would overstate the resolution.
    """
    idx = np.flatnonzero(np.asarray(soh_true) <= threshold)
    return float(day[idx[0]]) if len(idx) else float("nan")


def first_alarm_day(day: np.ndarray, predicted_soh: np.ndarray,
                    threshold: float = WARN_SOH) -> float:
    """First anchor day whose PREDICTED SOH is at or below ``threshold``.

    ``predicted_soh`` is aligned to ``day``: entry ``i`` is what the model, standing
    at anchor ``i``, says SOH will be at the end of its forecast reach. NaN entries
    (anchors without enough history) are ignored rather than treated as no-alarm.
    """
    p = np.asarray(predicted_soh, dtype=float)
    ok = ~np.isnan(p)
    idx = np.flatnonzero(ok & (p <= threshold))
    return float(day[idx[0]]) if len(idx) else float("nan")


def warning_lead_time(day: np.ndarray, soh_true: np.ndarray,
                      predicted_soh: np.ndarray,
                      threshold: float = WARN_SOH) -> dict:
    """dt_warn for one trajectory, plus its confusion-matrix outcome."""
    t_true = true_threshold_crossing_day(day, soh_true, threshold)
    t_alarm = first_alarm_day(day, predicted_soh, threshold)
    crossed = not np.isnan(t_true)
    alarmed = not np.isnan(t_alarm)

    if crossed and alarmed:
        dt = t_true - t_alarm
        outcome = "TRUE_POSITIVE" if dt >= 0 else "LATE"
    elif crossed and not alarmed:
        dt, outcome = float("nan"), "MISSED"
    elif not crossed and alarmed:
        dt, outcome = float("nan"), "FALSE_POSITIVE"
    else:
        dt, outcome = float("nan"), "TRUE_NEGATIVE"

    return {"t_true_cross_day": t_true, "t_alarm_day": t_alarm,
            "dt_warn_days": dt, "outcome": outcome,
            "crossed": bool(crossed), "alarmed": bool(alarmed)}


def prognostic_horizon(day: np.ndarray, rul_true: np.ndarray,
                       rul_pred: np.ndarray, t_eol: float,
                       alpha: float = ALPHA_PH,
                       band_floor_days: float = 0.0,
                       clamp_nonnegative: bool = False) -> dict:
    """PH for one trajectory that reaches EOL.

    Scans backward from EOL; the returned entry time is the earliest anchor from
    which the prediction stays inside the alpha band for the whole remainder.

    ``band_floor_days``
        Floors the tolerance: ``band = max(alpha * RUL, band_floor_days)``. With the
        default 0 this is the STRICT Saxena definition, in which the tolerance
        collapses to zero as RUL -> 0 and therefore demands near-exact accuracy at
        the final anchor. Setting it to one reference step (14 days) asks instead for
        accuracy no finer than the observation grid itself. Both are reported;
        neither is substituted for the other.
    ``clamp_nonnegative``
        Clamps predictions at 0. A negative remaining life is not a physical
        quantity, so a deployed system would clamp; the strict variant leaves the raw
        prediction alone so the underlying behaviour stays visible.
    """
    if np.isnan(t_eol):
        return {"ph_days": float("nan"), "ph_entry_day": float("nan"),
                "ph_defined": False,
                "reason": "trajectory never reaches EOL (right-censored)"}

    d = np.asarray(day, dtype=float)
    yt = np.asarray(rul_true, dtype=float)
    yp = np.asarray(rul_pred, dtype=float)
    ok = ~(np.isnan(yt) | np.isnan(yp))
    if not ok.any():
        return {"ph_days": float("nan"), "ph_entry_day": float("nan"),
                "ph_defined": False, "reason": "no valid anchor"}

    d, yt, yp = d[ok], yt[ok], yp[ok]
    if clamp_nonnegative:
        yp = np.maximum(yp, 0.0)
    order = np.argsort(d)
    d, yt, yp = d[order], yt[order], yp[order]
    band = np.maximum(alpha * yt, band_floor_days)
    inside = np.abs(yp - yt) <= np.maximum(band, 1e-12)

    # Backward scan: the last contiguous run of in-band anchors ending at EOL.
    entry = None
    for i in range(len(d) - 1, -1, -1):
        if inside[i]:
            entry = d[i]
        else:
            break
    if entry is None:
        return {"ph_days": 0.0, "ph_entry_day": float("nan"), "ph_defined": True,
                "reason": "never enters the alpha band before EOL",
                "n_anchors_inside_band": int(inside.sum()),
                "n_anchors": int(len(d))}
    return {"ph_days": float(t_eol - entry), "ph_entry_day": float(entry),
            "ph_defined": True, "reason": "",
            "n_anchors_inside_band": int(inside.sum()), "n_anchors": int(len(d))}


def credible_window(day: np.ndarray, rul_true: np.ndarray, rul_pred: np.ndarray,
                    t_eol: float, alpha: float = ALPHA_PH) -> dict:
    """The LONGEST contiguous in-band run, wherever it sits — not anchored to EOL.

    PH asks a strict question: does accuracy hold from some point all the way to EOL?
    When PH = 0 that answer is "no", and it stops there. This complements it by
    reporting WHERE the model is inside the alpha band and for how long, which is the
    quantity an operator planning a replacement window actually reads.

    Reported alongside PH, never as a substitute: this is a weaker claim (a run that
    ends before EOL) and is labelled as such.
    """
    if np.isnan(t_eol):
        return {"credible_window_days": float("nan"), "defined": False}

    d = np.asarray(day, dtype=float)
    yt = np.asarray(rul_true, dtype=float)
    yp = np.asarray(rul_pred, dtype=float)
    ok = ~(np.isnan(yt) | np.isnan(yp))
    if not ok.any():
        return {"credible_window_days": float("nan"), "defined": False}

    d, yt, yp = d[ok], yt[ok], yp[ok]
    order = np.argsort(d)
    d, yt, yp = d[order], yt[order], yp[order]
    inside = np.abs(yp - yt) <= np.maximum(alpha * yt, 1e-12)

    best_len, best_lo, best_hi = 0.0, np.nan, np.nan
    i = 0
    while i < len(d):
        if not inside[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(d) and inside[j + 1]:
            j += 1
        span = float(d[j] - d[i])
        if span > best_len:
            best_len, best_lo, best_hi = span, float(d[i]), float(d[j])
        i = j + 1

    return {"credible_window_days": best_len,
            "credible_window_start_day": best_lo,
            "credible_window_end_day": best_hi,
            "credible_window_lead_before_eol_days": (
                float(t_eol - best_lo) if not np.isnan(best_lo) else float("nan")),
            "defined": True}


def aggregate(records: list[dict]) -> dict:
    """Distribution of dt_warn and PH, plus the alarm rates.

    False-alarm rate is over trajectories that never crossed; miss rate is over
    trajectories that did. Reporting either over the whole 120 would hide which
    denominator moved.
    """
    outcomes = [r["outcome"] for r in records]
    n_crossed = sum(1 for r in records if r["crossed"])
    n_not = len(records) - n_crossed

    dt = np.array([r["dt_warn_days"] for r in records
                   if r["outcome"] == "TRUE_POSITIVE"], dtype=float)
    ph = np.array([r["ph_days"] for r in records
                   if r.get("ph_defined") and not np.isnan(r.get("ph_days", np.nan))],
                  dtype=float)

    def q(a, p):
        return float(np.percentile(a, p)) if a.size else float("nan")

    n_missed = outcomes.count("MISSED")
    n_late = outcomes.count("LATE")
    n_fp = outcomes.count("FALSE_POSITIVE")

    return {
        "n_trajectories": len(records),
        "n_crossed_warning_threshold": n_crossed,
        "n_never_crossed": n_not,
        "n_true_positive": outcomes.count("TRUE_POSITIVE"),
        "n_late": n_late,
        "n_missed": n_missed,
        "n_false_positive": n_fp,
        "n_true_negative": outcomes.count("TRUE_NEGATIVE"),
        "miss_rate_of_crossers": (n_missed / n_crossed) if n_crossed else float("nan"),
        "late_rate_of_crossers": (n_late / n_crossed) if n_crossed else float("nan"),
        "false_alarm_rate_of_non_crossers": (n_fp / n_not) if n_not else float("nan"),
        "dt_warn_n": int(dt.size),
        "dt_warn_mean_days": float(dt.mean()) if dt.size else float("nan"),
        "dt_warn_median_days": q(dt, 50),
        "dt_warn_p10_days": q(dt, 10),
        "dt_warn_p90_days": q(dt, 90),
        "dt_warn_min_days": float(dt.min()) if dt.size else float("nan"),
        "dt_warn_max_days": float(dt.max()) if dt.size else float("nan"),
        "ph_n_defined": int(ph.size),
        "ph_mean_days": float(ph.mean()) if ph.size else float("nan"),
        "ph_median_days": q(ph, 50),
        "ph_p10_days": q(ph, 10),
        "ph_p90_days": q(ph, 90),
        "ph_alpha": ALPHA_PH,
        "resolution_note": ("dt_warn resolution is bounded by the 14-day reference "
                            "step; values are meaningful to +/-14 days"),
    }
