# Data lineage: STK -> L0 -> L1 -> L2 -> L3 v2 -> model

Generated from the frozen manifests. Row counts and hashes are read, not restated.

---

## The chain

```
  24 STK scenarios (72 h, 30 s step)
        |  report exports: panel power/area, beta angle, lighting, access, environment
        v
  L0  environment time series                    207,384 rows
        |  battery model: SOC, DoD, temperature, throughput
        v
  L1  battery telemetry (72 h)                   207,384 rows
        |  multi-year extrapolation onto the 14-day reference grid
        v
  L2  reference points                           14,285 rows
        |  windowing: (context_end, horizon) -> one row per sample
        v
  L3  training windows (frozen v1 layer)         50,180 rows
        |  v2 semantic layer: exact 11-dim features recomputed from soh_observed,
        |  channels renamed so observed/true can never be confused,
        |  legacy proxy features isolated under legacy_proxy_*
        v
  L2 v2 / L3 v2 / source_prediction_v2
        |  allow-listed column subset + 20-point context history + frozen prediction
        v
  battery_release_v2/data/L3_windows_v2_arc_clean_fixed_min.csv   34,335 rows
```

## Layer semantics

| layer | one row is | key |
|---|---|---|
| L0 | one 30-second environment sample of one scenario | (scenario_id, time) |
| L1 | one 30-second battery telemetry sample | (scenario_id, time) |
| L2 | one reference point on the 14-day grid | (trajectory_id, reference_index) |
| L3 | one (context_end, horizon) prediction sample | sample_uid |

## Derived dataset properties

| property | value |
|---|---|
| trajectories | 120 |
| trajectory split | {"target_test": 40, "target_train": 40, "target_calibration": 40} |
| windows per horizon | {"2": 13205, "4": 12965, "8": 12485, "16": 11525} |
| EOL reached | 92 trajectories |
| right-censored | 28 trajectories |
| EOL day min / median / max | 690 / 1493.0 / 2141 |
| soh_true monotonicity violations | 0 |

H=16 exists in the upstream L3 for the RW route only. It is **absent on ARC** and is
never filled, imputed or extrapolated; the formal matched-horizon set is {2, 4, 8}.

## The feature layer, and one thing that must not be confused

The v2 layer recomputes the **exact frozen 11-dim features** from `soh_observed`
using the authoritative extractor, and carries them as `frozen_00_last` ...
`frozen_10_ctx_max`.

The upstream L3's original `feature_00` ... `feature_10` were a **`proxy_geometry_v2`
placeholder** at L=8 — an interface stub, **not** the formal frozen features. They are
retained upstream under the `legacy_proxy_*` prefix with model input disabled, so the
two can never be selected together by accident. This package ships only the frozen
features and none of the proxy layer.

## Integrity manifests

* raw export layer: `data/afterstk/sha256_manifest.csv` — 44 entries, sha256 `e2676b5dce103cfc8deac3ef7c7825b68ffaec538a6404b7dd0319ae36c3a8f1`
* v2 semantic layer: `data/stk_transfer_v2/sha256_manifest_v2.csv` — 17 entries, sha256 `79db43b73ebd4d2d30ec4d4ae00ddbc4b3b69bed50ac08609881c6db97d6525b`
* this package: `SHA256SUMS` + `release_manifest.json`, re-verified by
  `python -m battery_entry verify`

## What the chain cannot deliver

The **generator** that turned L1 telemetry into multi-year degradation is not in the
project. `data/afterstk` is its *output*. So degradation cannot be re-integrated under
a modified factor, no strict paired counterfactual exists, and every factor-direction
claim is capped at `OBSERVATIONAL_EVIDENCE` / `NON_CAUSAL`.

