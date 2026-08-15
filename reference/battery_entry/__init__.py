"""battery_entry — self-contained delivery package for the STK-transfer SOH model.

This package has NO dependency on the parent research project. It carries its own
copies of the frozen 11-dim feature extractor, the frozen ARC clip/scaler, the
reference data slice and the reference results, and it resolves every path relative
to its own location.

Four subcommands, all implemented in :mod:`battery_entry.cli`::

    python -m battery_entry verify
    python -m battery_entry reproduce --mode quick --output <dir>
    python -m battery_entry reproduce --mode full  --output <dir>
    python -m battery_entry predict --input <csv> --output <csv>

Scope of what this package can honestly claim, carried in code so it cannot drift
from the documentation:

* the point model is ``target_only_arc_space``, frozen by Phase 3/4 and replayed here;
* the transfer mechanism is FEATURE_SPACE_TRANSFER_WITH_TARGET_DOMAIN_HEAD_REFIT —
  the frozen source ARC clip/scaler defines the feature space and is reused unchanged,
  while the regression head is refitted on target-domain rows only. The frozen source
  COEFFICIENTS are not used; Phase 3 ruled SOURCE_PRIOR_NOT_VALIDATED;
* every number is measured on an STK-derived simulated twin. None of it is
  real-satellite accuracy, and no on-orbit guarantee follows from any of it;
* intervals are per-horizon MARGINAL simulation intervals carrying
  CONFORMAL_NOT_VALIDATED; RUL carries RUL_EVIDENCE_INSUFFICIENT.
"""
from __future__ import annotations

__version__ = "2.0.0-candidate"

#: Frozen status strings. Every report this package emits must carry these verbatim;
#: they are the Phase 3/4/5/6 verdicts and they are NOT re-decided here.
FROZEN_STATUS = {
    "point_model": "target_only_arc_space",
    "high_stability_baseline": "observable_local_trend_extrapolation",
    "transfer_mechanism": "FEATURE_SPACE_TRANSFER_WITH_TARGET_DOMAIN_HEAD_REFIT",
    "phase3_verdict": "SOURCE_PRIOR_NOT_VALIDATED",
    "TIME_NORMALIZATION_GATE": "RATE_DAY_REPARAMETERIZATION_ONLY",
    "EXPOSURE_GATE": "EXPOSURE_INPUT_NOT_AVAILABLE",
    "STRESS_GATE_LOW_DATA": "STRESS_NOT_VALIDATED",
    "STRESS_GATE_FULL_DATA": "STRESS_NOT_VALIDATED",
    "CONFORMAL_GATE": "CONFORMAL_NOT_VALIDATED",
    "rul_evidence": "RUL_EVIDENCE_INSUFFICIENT",
    "phase6": "PHASE6_CLOSED_WITHOUT_TRAINING",
    "patchtst": "PATCHTST_SKIPPED_BY_PREDEFINED_GATE",
    "candidate_status": "CANDIDATE_NOT_PROMOTED",
    "evidence_domain": "SIMULATION_ONLY",
}

#: Labels that must be attached to the corresponding outputs, always.
OUTPUT_LABELS = {
    "interval": "EXPERIMENTAL_SIMULATION_INTERVAL",
    "rul": "FINITE_HORIZON_EVIDENCE_INSUFFICIENT",
    "no_crossing": "NO_CROSSING_WITHIN_FORECAST_HORIZON",
}

__all__ = ["__version__", "FROZEN_STATUS", "OUTPUT_LABELS"]
