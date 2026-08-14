"""target_dataset.py — target-domain L3 windows, straight from v2's frozen slice.

Reads ``battery_release_v2/data/L3_windows_v2_arc_clean_fixed_min.csv`` and uses the
``frozen_00_last..frozen_10_ctx_max`` columns AS SHIPPED. It does not recompute them
here — ``tests/test_equivalence_with_v2.py`` proves that recomputing from
``ctx_soh_00..19`` with this package's extractor reproduces them bit-for-bit, which is
the stronger statement.

The adaptation subsets (10/25/50/100%) are read from the L3 table's own membership
columns where present, and otherwise derived from the frozen
``adapt_rank_within_target_train`` ordering. They are never re-sampled: re-drawing
them would silently break comparability with the frozen Phase 3/4 tables.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import paths as P
from src.features.frozen11 import FROZEN_COLUMNS

#: v2's supported horizons, in 14-day reference steps.
HORIZONS: tuple[int, ...] = (2, 4, 8)


def load_l3() -> pd.DataFrame:
    df = pd.read_csv(P.L3_WINDOWS_CSV, float_precision="round_trip")
    return df.sort_values("sample_uid", kind="stable").reset_index(drop=True)


def adaptation_subsets() -> dict:
    """``{"10pct": 4, "25pct": 10, "50pct": 20, "100pct": 40}`` from v2's own config."""
    cfg = json.loads(P.GENERATION_CONFIG_JSON.read_text(encoding="utf-8"))
    return {k: int(v) for k, v in cfg["adaptation_subsets"].items()}


def design(df: pd.DataFrame) -> dict:
    """The frozen 11-dim design plus everything the protocol needs."""
    X = df[list(FROZEN_COLUMNS)].to_numpy(dtype=float)
    return {
        "X": X,
        "y_delta": df["target_delta_soh_true"].to_numpy(dtype=float),
        "anchor": df["anchor_soh_observed"].to_numpy(dtype=float),
        "target_soh": df["target_soh_true"].to_numpy(dtype=float),
        "horizon_steps": df["horizon_steps"].to_numpy(dtype=int),
        "trajectory_id": df["trajectory_id"].to_numpy(),
        "family_id": df["environment_family_id"].to_numpy(),
        "split": df["split"].to_numpy(),
    }


def trajectory_budget_mask(df: pd.DataFrame, n_per_family: int) -> np.ndarray:
    """Rows of the first ``n_per_family`` trajectories WITHIN EACH family.

    Taking N per FAMILY rather than N globally keeps every training family
    represented at every budget. A global top-N could omit whole families, which
    would silently change what "family-macro MAE" averages over and make the
    adaptation curve incomparable across budgets.

    Ordering is by ``trajectory_id``, which is stable and already encodes the seed
    index (``..._seed00``, ``_seed01``, ...), so the subset is deterministic and
    needs no RNG.
    """
    keep = np.zeros(len(df), dtype=bool)
    tid = df["trajectory_id"]
    for _, g in df.groupby("environment_family_id", sort=True):
        trajs = sorted(set(g["trajectory_id"].tolist()))[:int(n_per_family)]
        keep |= tid.isin(trajs).to_numpy()
    return keep


#: Budget ladder for the adaptation curve, as trajectories per family.
#:
#: v2's ``generation_config.json`` records ``adaptation_subsets`` as 4/10/20/40 of the
#: 40-trajectory ``target_train`` pool (2 families x 20). This package evaluates under
#: family-LOFO, where FIVE families train at a time, so the same percentages are
#: expressed per family against the 20 trajectories each family holds. The
#: percentages are identical; only the denominator is stated in the protocol's own
#: terms. The frozen 4/10/20/40 figures are carried in the output for traceability.
BUDGET_TRAJECTORIES_PER_FAMILY: dict = {
    "10pct": 2, "25pct": 5, "50pct": 10, "100pct": 20,
}


def summarise(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_trajectories": int(df.trajectory_id.nunique()),
        "n_families": int(df.environment_family_id.nunique()),
        "horizons": sorted(int(h) for h in df.horizon_steps.unique()),
        "rows_per_horizon": {int(h): int((df.horizon_steps == h).sum())
                             for h in sorted(df.horizon_steps.unique())},
        "source": str(P.L3_WINDOWS_CSV),
    }
