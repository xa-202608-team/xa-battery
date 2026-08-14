"""schema.py — strict input validation for ``predict``.

A prediction request is refused unless it is unambiguously well formed. Silent coercion
is the failure mode this module exists to prevent: a mis-ordered history or a
percentage-scale SOH would still produce a number, and that number would be wrong
without anything looking wrong.

What is validated, and why each check earns its place:

* **columns / dtypes** — declared in ``schemas/predict_input_schema.json``, so the
  contract is data rather than prose;
* **units** — SOH must be dimensionless in (0, 1.5]. A caller passing percent (0..100)
  is refused explicitly rather than clipped, because clipping would silently move every
  feature;
* **time order** — the history must be strictly increasing in ``reference_index``. The
  11 features read positions (``last``, ``diff1``, ``h[0]``), so a shuffled history
  yields well-formed nonsense;
* **context length** — exactly 20 reference points, from the frozen ARC route manifest.
  Truncating or padding would change the feature semantics, so a short history is
  refused rather than zero-filled;
* **forbidden columns** — any hidden-truth, future-realized-exposure, family or split
  column present in the input is refused. This is the structural half of the leakage
  guarantee: the design matrix is built from a fixed 11-name list, and additionally the
  loader refuses to even accept a file carrying those names, so a leaked column cannot
  arrive by a later code change either.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from battery_entry import paths as P
from battery_entry.features import CONTEXT_LENGTH

#: Required input columns. One row per (battery_id, reference_index) observation.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "battery_id", "reference_index", "soh_observed")

#: Optional, and only ever carried through to the output for traceability.
OPTIONAL_COLUMNS: tuple[str, ...] = ("timestamp_utc",)

#: Refused BY NAME if present in the input file. Each is a channel an earlier phase
#: declared label-only, audit-only, oracle-only or a family-identity proxy.
FORBIDDEN_INPUT_COLUMNS: tuple[str, ...] = (
    "soh_true", "anchor_soh_true", "rint_true", "soh_noise_residual",
    "target_soh_true", "target_soh_observed", "target_delta_soh_true",
    "target_delta_soh_observed", "audit_upper_bound_delta_soh_true_to_true",
    "target_rul_days", "target_rul_reference_steps", "censored",
    "rul_lower_bound_days", "rul_lower_bound_reference_steps",
    "future_realized_delta_efc", "future_realized_delta_ah",
    "future_realized_delta_wh",
    "future_planned_delta_efc", "future_planned_delta_ah", "future_planned_delta_wh",
    "environment_family_id", "split", "trajectory_id",
    "altitude_km", "inclination_deg", "raan_deg",
)
FORBIDDEN_INPUT_PREFIXES: tuple[str, ...] = (
    "target_", "future_realized_", "future_planned_", "audit_",
    "legacy_proxy_", "obs_hist_", "stress_",
)
#: ``soh_observed`` is the ONE SOH channel a predictor may read. Anything ending in
#: ``_true`` is a hidden-truth channel.
FORBIDDEN_INPUT_SUFFIXES: tuple[str, ...] = ("_true",)

SOH_MIN_EXCLUSIVE = 0.0
SOH_MAX_INCLUSIVE = 1.5

#: Why these columns are refused, so the error message teaches rather than just blocks.
REFUSAL_REASONS: dict[str, str] = {
    "soh_true": "hidden-truth channel; label/audit only (Phase 1B prohibition)",
    "anchor_soh_true": "audit only — removes the anchor noise a deployed system faces",
    "rint_true": "audit only (Phase 1B prohibition)",
    "environment_family_id": "the strict target-domain isolation key; an input leaks it",
    "split": "data-source metadata, and a family-identity proxy",
    "trajectory_id": "identity, not a feature",
    "altitude_km": "orbit triplet maps one-to-one onto the 6 families — family proxy",
    "inclination_deg": "orbit triplet maps one-to-one onto the 6 families — family proxy",
    "raan_deg": "orbit triplet maps one-to-one onto the 6 families — family proxy",
    "future_realized_delta_efc": "ORACLE_AUDIT_ONLY — reads the trajectory after cutoff",
    "future_planned_delta_efc": "EXPOSURE_INPUT_NOT_AVAILABLE (NULL on 100% of rows)",
}


class SchemaError(ValueError):
    """Raised when an input file does not satisfy the declared contract."""


@dataclass
class ValidatedInput:
    """One validated prediction request per battery."""
    battery_id: str
    reference_index: np.ndarray
    soh_observed: np.ndarray
    timestamp_utc: list[str] | None
    context_length: int

    @property
    def history(self) -> np.ndarray:
        """The context window: exactly ``CONTEXT_LENGTH`` points ending at the anchor."""
        return self.soh_observed[-CONTEXT_LENGTH:]

    @property
    def anchor_soh(self) -> float:
        return float(self.soh_observed[-1])

    @property
    def anchor_reference_index(self) -> int:
        return int(self.reference_index[-1])


def load_schema() -> dict[str, Any]:
    return json.loads(
        (P.SCHEMA_DIR / "predict_input_schema.json").read_text(encoding="utf-8"))


def _refuse_forbidden(columns: list[str]) -> None:
    hits: list[str] = []
    for c in columns:
        low = c.strip().lower()
        if low in {x.lower() for x in FORBIDDEN_INPUT_COLUMNS}:
            hits.append(f"{c} ({REFUSAL_REASONS.get(low, 'declared non-input channel')})")
            continue
        for pre in FORBIDDEN_INPUT_PREFIXES:
            if low.startswith(pre):
                hits.append(f"{c} (matches forbidden prefix {pre!r})")
                break
        else:
            for suf in FORBIDDEN_INPUT_SUFFIXES:
                if low.endswith(suf):
                    hits.append(f"{c} (matches forbidden suffix {suf!r} — hidden truth)")
                    break
    if hits:
        raise SchemaError(
            "the input carries columns that may never reach a model input:\n  "
            + "\n  ".join(hits)
            + "\nRemove them. The 11 features are computed from soh_observed alone.")


def validate_input(df: pd.DataFrame) -> list[ValidatedInput]:
    """Validate a prediction input frame and split it per battery.

    Raises :class:`SchemaError` with a specific message on the first violation. Every
    check is a refusal, never a repair.
    """
    if df.empty:
        raise SchemaError("input is empty")

    cols = [str(c) for c in df.columns]
    if len(set(c.lower() for c in cols)) != len(cols):
        raise SchemaError(f"duplicate column names (case-insensitive): {cols}")

    _refuse_forbidden(cols)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"missing required columns {missing}; required: {list(REQUIRED_COLUMNS)}")

    unknown = [c for c in cols
               if c not in REQUIRED_COLUMNS and c not in OPTIONAL_COLUMNS]
    if unknown:
        raise SchemaError(
            f"unknown columns {unknown}; allowed: "
            f"{list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS)}. Unknown columns are "
            f"refused rather than ignored, so a mis-named feature cannot pass silently.")

    # dtypes
    try:
        ref = df["reference_index"].to_numpy()
        if not np.issubdtype(ref.dtype, np.integer):
            ref_f = df["reference_index"].astype(float).to_numpy()
            if not np.all(ref_f == np.floor(ref_f)):
                raise SchemaError("reference_index must be integral")
    except (TypeError, ValueError) as e:
        raise SchemaError(f"reference_index is not integral: {e}") from e
    try:
        soh_all = df["soh_observed"].astype(float).to_numpy()
    except (TypeError, ValueError) as e:
        raise SchemaError(f"soh_observed is not numeric: {e}") from e

    if not np.isfinite(soh_all).all():
        n_bad = int((~np.isfinite(soh_all)).sum())
        raise SchemaError(f"soh_observed carries {n_bad} non-finite value(s)")

    # units
    if float(np.max(soh_all)) > SOH_MAX_INCLUSIVE:
        mx = float(np.max(soh_all))
        hint = (" Values above 1.5 look like a PERCENTAGE scale (0..100); SOH must be "
                "dimensionless. Divide by 100." if mx > 1.5 else "")
        raise SchemaError(
            f"soh_observed max {mx:g} exceeds {SOH_MAX_INCLUSIVE}.{hint}")
    if float(np.min(soh_all)) <= SOH_MIN_EXCLUSIVE:
        raise SchemaError(
            f"soh_observed min {float(np.min(soh_all)):g} is <= {SOH_MIN_EXCLUSIVE}; "
            f"SOH must be strictly positive")

    out: list[ValidatedInput] = []
    for bid, g in df.groupby("battery_id", sort=True):
        ridx = g["reference_index"].astype(int).to_numpy()
        order_ok = bool(np.all(np.diff(ridx) > 0))
        if not order_ok:
            raise SchemaError(
                f"battery {bid!r}: reference_index is not strictly increasing "
                f"({ridx[:8].tolist()}...). The 11 features read positions (last, "
                f"diff1, h[0]), so a shuffled or duplicated history would produce "
                f"well-formed nonsense. Sort the input and remove duplicates.")
        if len(g) < CONTEXT_LENGTH:
            raise SchemaError(
                f"battery {bid!r}: history has {len(g)} reference points, but the frozen "
                f"ARC route requires exactly {CONTEXT_LENGTH}. Padding or truncating "
                f"would change the feature semantics, so a short history is refused.")
        soh = g["soh_observed"].astype(float).to_numpy()
        ts = (g["timestamp_utc"].astype(str).tolist()
              if "timestamp_utc" in g.columns else None)
        out.append(ValidatedInput(
            battery_id=str(bid), reference_index=ridx, soh_observed=soh,
            timestamp_utc=ts, context_length=CONTEXT_LENGTH))
    return out


def read_input_csv(path: str) -> list[ValidatedInput]:
    """Read and validate a prediction input CSV."""
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        raise SchemaError(f"cannot read {path}: {e}") from e
    return validate_input(df)
