"""source_dataset.py — the NASA PCoE public degradation dataset, as 20-point windows.

The public source data IS present on disk: 34 cleaned cells under
``data/processed/clean/arc/`` (see ``REUSE_MAP.md`` 1.3). Each npz carries the
per-cycle discharge record — ``cycle_idx``, ``capacity``, ``voltage``, ``current``,
``temperature``, ``ambient_temperature`` — so source-domain pretraining runs on real
measurements, not a placeholder.

WHAT THIS BUILDER DOES
----------------------
1. capacity -> SOH by dividing by the cell's own reference capacity, so cells with
   different nominal capacities land on one comparable scale.
2. 20-point context windows (``CONTEXT_LENGTH``, the frozen route context) with a
   target ``horizon`` cycles ahead.
3. The frozen 11-dim geometry on the SOH history — the SAME extractor v2 ships, so
   the source and target designs are literally the same function of a SOH history.

The label is the delta ``SOH(t+h) - SOH(t)``, matching v2's ``T0_DELTA_REFERENCE``.

UNITS DIFFER BETWEEN DOMAINS, AND THAT IS THE POINT
---------------------------------------------------
Source horizons are in CYCLES; target horizons are in 14-day reference steps. They
are not the same physical quantity, and this module never pretends otherwise. What
transfers is the geometry of a degradation curve and the coordinate system fitted to
it (clip/scaler) — which is exactly the "observable SOH geometry as cross-domain
invariant" claim, and exactly why lam_SP has to be searched rather than assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import paths as P
from src.features.frozen11 import CONTEXT_LENGTH, common_features

#: Cells whose SOH history is too short to yield a single 20-point window are skipped
#: and REPORTED, never silently dropped.
MIN_CYCLES = CONTEXT_LENGTH + 1


def list_cells() -> list[Path]:
    return sorted(P.PUBLIC_SOURCE_DIR.glob("*.npz"))


def load_cell_soh(npz_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(cycle_idx, soh, meta)`` for one cell.

    SOH is capacity normalised by the cell's own reference capacity. The reference is
    the FIRST recorded capacity, which is the convention the cleaned cache was built
    on (``src/data/clean.py``); using a per-cell reference is what makes cells with
    different nominal Ah comparable at all.
    """
    z = np.load(npz_path, allow_pickle=True)
    cyc = np.asarray(z["cycle_idx"], dtype=float)
    cap = np.asarray(z["capacity"], dtype=float)

    ok = np.isfinite(cap) & (cap > 0)
    cyc, cap = cyc[ok], cap[ok]
    order = np.argsort(cyc)
    cyc, cap = cyc[order], cap[order]

    meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    if len(cap) == 0:
        return cyc, np.zeros(0), meta
    return cyc, cap / float(cap[0]), meta


def build_windows(horizon: int = 8) -> dict:
    """Frozen-11 design matrix over all source cells.

    ``horizon`` is in CYCLES (see module docstring). The default 8 mirrors the target
    domain's longest supported horizon in step count, so the source head being pulled
    toward is the one trained on a comparable number of steps ahead.
    """
    X, y, cell_ids, anchors = [], [], [], []
    skipped = []
    for f in list_cells():
        cyc, soh, _ = load_cell_soh(f)
        if len(soh) < MIN_CYCLES + horizon:
            skipped.append({"cell": f.stem, "n_cycles": int(len(soh))})
            continue
        for i in range(CONTEXT_LENGTH - 1, len(soh) - horizon):
            hist = soh[i - CONTEXT_LENGTH + 1: i + 1]
            X.append(common_features(hist))
            y.append(soh[i + horizon] - soh[i])
            cell_ids.append(f.stem)
            anchors.append(cyc[i])

    return {
        "X": np.asarray(X, dtype=float),
        "y": np.asarray(y, dtype=float),
        "cell_id": np.asarray(cell_ids),
        "anchor_cycle": np.asarray(anchors, dtype=float),
        "horizon_cycles": int(horizon),
        "n_cells_used": int(len(set(cell_ids))),
        "n_cells_skipped": len(skipped),
        "skipped": skipped,
        "context_length": CONTEXT_LENGTH,
        "label": "SOH(t+h) - SOH(t), delta scale (matches v2 T0_DELTA_REFERENCE)",
        "soh_definition": "capacity / capacity[first recorded cycle], per cell",
        "provenance": str(P.PUBLIC_SOURCE_DIR),
        "dataset": "NASA PCoE battery degradation (cleaned cache)",
    }


def summarise(d: dict) -> dict:
    return {
        "dataset": d["dataset"],
        "n_windows": int(len(d["y"])),
        "n_cells_used": d["n_cells_used"],
        "n_cells_skipped": d["n_cells_skipped"],
        "horizon_cycles": d["horizon_cycles"],
        "context_length": d["context_length"],
        "label_mean": float(d["y"].mean()) if len(d["y"]) else float("nan"),
        "label_std": float(d["y"].std()) if len(d["y"]) else float("nan"),
    }
