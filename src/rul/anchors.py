"""anchors.py — mission-span RUL anchors, censored trajectories KEPT.

The label is the mission-span one, NOT the finite-horizon crossing label::

    RUL(t) = t_EOL(trajectory) - t_anchor          (days, no horizon cap)

The frozen ``RUL_EVIDENCE_INSUFFICIENT`` verdict is scoped to the 112-day
finite-horizon crossing detector, where every eligible anchor carries
``target_rul_days == 112.0`` (std 0). That verdict stands. This is a different task
with a different label — see ``GATES_scope_amendment.md``.

WHAT CHANGED VS THE PROBE
-------------------------
``files/rul_longhorizon_feasibility.py`` DROPPED the 28 right-censored trajectories,
which biases the sample toward faster-degrading cells. Here they are KEPT and carry
a lower bound instead of a point label:

* uncensored anchor: ``rul_days`` is known exactly, ``censored = 0``
* censored anchor:   ``rul_days`` is NaN, ``rul_lower_bound_days`` is the observed
  follow-up ``t_last - t_anchor``, ``censored = 1``

The censoring bookkeeping mirrors ``src/stk_transfer_v2/data/labels_v2.py``
``rul_and_censoring``: an EOL crossing is the FIRST reference point at or below the
threshold, and a trajectory that never crosses within the mission is right-censored
with the final observed day as its bound.

CHANNEL POLICY (unchanged from the probe and from v2)
-----------------------------------------------------
  features  <- soh_observed   (what a deployed system reads)
  EOL label <- soh_true       (evaluation truth only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import paths as P
from src.features.frozen11 import CONTEXT_LENGTH, common_features
from src.features.space import EOL_SOH

FEATURE_COLS = [f"f{j:02d}" for j in range(11)]


def load_l2() -> tuple[pd.DataFrame, dict, dict]:
    """L2 reference points + the family map and the censoring flags from v2."""
    l2 = pd.read_csv(P.L2_REFERENCE_CSV)
    sm = pd.read_csv(P.SPLIT_MANIFEST_CSV)
    fam = dict(zip(sm.trajectory_id, sm.environment_family_id))
    censored_flag = dict(zip(sm.trajectory_id, sm.right_censored_on_soh_true))
    return l2, fam, censored_flag


def build_anchors(l2: pd.DataFrame, fam_map: dict) -> pd.DataFrame:
    """One row per (trajectory, anchor) with a full 20-point history.

    Anchors at or after the EOL crossing are excluded for uncensored trajectories:
    predicting remaining life after death is not the task. For censored
    trajectories every anchor with enough history is kept, since none of them has
    passed EOL by construction.
    """
    rows = []
    for tid, d in l2.groupby("trajectory_id", sort=True):
        d = d.sort_values("reference_day").reset_index(drop=True)
        obs = d.soh_observed.to_numpy(dtype=float)
        tru = d.soh_true.to_numpy(dtype=float)
        day = d.reference_day.to_numpy(dtype=float)

        crossed = np.flatnonzero(tru <= EOL_SOH)
        is_censored = len(crossed) == 0
        t_eol = np.nan if is_censored else float(day[crossed[0]])
        t_last = float(day[-1])

        for i in range(CONTEXT_LENGTH - 1, len(d)):
            if not is_censored and day[i] >= t_eol:
                break
            f = common_features(obs[i - CONTEXT_LENGTH + 1: i + 1])
            rows.append([
                tid, fam_map[tid], day[i],
                np.nan if is_censored else t_eol - day[i],   # rul_days
                t_last - day[i],                             # observed follow-up
                int(is_censored),
                *f,
            ])

    cols = (["trajectory_id", "environment_family_id", "anchor_day",
             "rul_days", "rul_lower_bound_days", "censored"] + FEATURE_COLS)
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["trajectory_id", "anchor_day"],
                           kind="stable").reset_index(drop=True)


def build() -> pd.DataFrame:
    """The anchor table, with a consistency check against v2's own flags."""
    l2, fam, censored_flag = load_l2()
    D = build_anchors(l2, fam)

    # Cross-check our EOL derivation against the manifest's own censoring column.
    ours = D.groupby("trajectory_id")["censored"].max().to_dict()
    drift = {t: (v, int(censored_flag[t])) for t, v in ours.items()
             if int(censored_flag[t]) != int(v)}
    if drift:
        raise RuntimeError(
            "censoring disagrees with split_manifest_v2.right_censored_on_soh_true "
            f"for {len(drift)} trajectories: {list(drift.items())[:5]}")
    return D


def summarise(D: pd.DataFrame) -> dict:
    """Label statistics, reported on the uncensored subset only (labels exist there)."""
    unc = D[D.censored == 0]
    y = unc.rul_days.to_numpy(dtype=float)
    return {
        "anchors_total": int(len(D)),
        "anchors_uncensored": int(len(unc)),
        "anchors_censored": int((D.censored == 1).sum()),
        "trajectories_total": int(D.trajectory_id.nunique()),
        "trajectories_uncensored": int(unc.trajectory_id.nunique()),
        "trajectories_censored": int(D[D.censored == 1].trajectory_id.nunique()),
        "rul_mean_days": float(y.mean()),
        "rul_std_days": float(y.std()),
        "rul_min_days": float(y.min()),
        "rul_max_days": float(y.max()),
        "degeneracy": "NON-DEGENERATE" if y.std() > 1.0 else "DEGENERATE",
    }
