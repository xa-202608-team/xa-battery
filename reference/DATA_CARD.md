# Data card

## What the data is

**Simulated.** Every row in this package descends from an STK-driven spaceflight battery
digital twin. The `source_kind` column in the upstream layer reads `simulated`, and that
value is load bearing: it is what forbids describing any metric here as real-satellite
accuracy.

No real on-orbit telemetry entered any stage of this work.

## Two domains, and what crossed between them

| domain | role | what came from it |
|---|---|---|
| **public battery data** (source) | established the degradation feature space and pre-processing convention | the frozen 11-dim extractor, the frozen clip/scaler (`arc_clean_fixed`) |
| **STK spaceflight twin** (target) | the domain the model actually predicts in | every trajectory, every label, every metric in this package |

The **feature space** crossed. The **prediction head** did not — it is estimated entirely
from target-domain rows. The frozen source coefficients are not used and are not shipped.

## The target-domain dataset

| property | value |
|---|---|
| environment families | 6 |
| trajectories | 120 (exactly 20 per family) |
| reference grid | 14 days per step |
| horizons | H ∈ {2, 4, 8} = 28 / 56 / 112 days |
| windows in the shipped slice | 34,335 |
| trajectories crossing EOL (0.70) | 92 of 120; the remaining 28 are right-censored |
| EOL threshold | 0.70 SOH |
| warning threshold | 0.80 SOH |

`environment_family_id` is the **strict target-domain isolation key**. Splits are assigned
per family, never per date.

That every family holds exactly 20 trajectories is not incidental — it makes the two
group-aware conformal weightings algebraically identical, so their agreement is arithmetic
rather than corroboration. See `LIMITATIONS.md`.

## Channels: which SOH may be read

This distinction is the whole leakage story, so it is stated in the data rather than left
to convention.

| channel | may a predictor read it? | why |
|---|---|---|
| `soh_observed` | **yes** — the only SOH a predictor may read | sensor-noise observation; what a deployed system actually sees |
| `soh_true` | **no** — label / evaluation only | noiseless simulated truth |
| `anchor_soh_true` | **no** — audit only | would remove the anchor noise a deployed system faces |
| `rint_true` | **no** — audit only | identical to `rint_observed` by construction in this dataset |

The main task label is `target_delta_soh_true = target_soh_true - anchor_soh_observed`.
This deliberately mixes an **observable present** with a **true future**: that is the
"observable history and current observed anchor → future simulated truth" formulation.
Because the observed anchor carries sensor noise, a small number of rows are positive;
that is a property of the definition, not a data defect.

## What is shipped, and the allow-list that governs it

`data/L3_windows_v2_arc_clean_fixed_min.csv` — 34,335 rows. Columns are selected by an
**allow-list**, not a deny-list, which is what makes the leakage guarantee structural: a
column not named in the builder simply is not in the file.

| group | columns | role |
|---|---|---|
| keys | `sample_uid`, `trajectory_id`, `environment_family_id`, `split`, `horizon_steps`, `horizon_days`, context/target indices, `context_length` | identity and bookkeeping; never features |
| anchor | `anchor_soh_observed` | `ANCHOR_ROLE_ONLY`; == `frozen_00_last` |
| features | `frozen_00_last` … `frozen_10_ctx_max` | the 11 design columns, in frozen order |
| context history | `ctx_soh_00` … `ctx_soh_19` | the 20-point observed-SOH window, so the feature check is a **recomputation** |
| labels | `target_soh_true`, `target_delta_soh_true` | evaluation truth and training label |
| frozen prediction | `frozen_predicted_delta`, `frozen_outer_fold` | the bit-exact golden-test target |

`target_soh_true` **is** shipped: a reproduction package that cannot score itself is not a
reproduction package. It is declared label-only in the schema and the design matrix is
built from a fixed 11-name list, so it cannot reach a model input.

**Absent by construction:** `soh_noise_residual`, `anchor_soh_true`, `rint_true`, every
`future_realized_*` and `future_planned_*` column, every `legacy_proxy_*` column, every
raw `stress_*` and derived `obs_hist_*` column, and `altitude_km` / `inclination_deg` /
`raan_deg`.

Also shipped: `data/L2_reference_points_v2_min.csv` (the reference-point grid, for
rebuilding a context window) and `data/split_manifest_v2.csv` (family / trajectory / split
map).

## Columns permanently excluded, and why

| column(s) | reason |
|---|---|
| `altitude_km`, `inclination_deg`, `raan_deg` | the orbit triplet maps **one-to-one** onto the 6 families, so each is a family-identity proxy. Excluded by measurement, not by taste — and any near-constant re-encoding is equally barred |
| `future_realized_delta_{efc,ah,wh}` | `ORACLE_AUDIT_ONLY` — reads the trajectory *after* the cutoff |
| `future_planned_delta_{efc,ah,wh}` | NULL on **100 %** of rows; evidence status `INSUFFICIENT_EVIDENCE_NOT_DETERMINABLE_AT_PREDICTION_TIME` |
| raw `stress_*` | point values at `context_end`, **not** history summaries — verified against every candidate aggregation window |
| derived `obs_hist_*` | built and audited in Phase 4 (25 of 29 cleared identifiability); neither F2 nor F3 passed a gate, so the formal model is the frozen 11 and nothing else |
| `legacy_proxy_feature_*` | the original L=8 placeholder layer, retained upstream for provenance only; never a frozen feature |

On planned exposure specifically: the dataset has no mission-plan table. The per-family
annual calendar reproduces realized one-step EFC only to ~5.8 % median / 15.0 % max
relative error, because each trajectory carries its own load, panel and comm multipliers,
temperature offset and solar degradation rate. That error is the bar it failed. **Realized
exposure may never be substituted for it.**

## STK provenance

24 STK scenarios across 2 altitude/inclination shells × 3 RAAN values × 4 epochs. The
export chain and the scenario→family→file mapping are in `docs/STK_SCENARIOS.md`;
the L0→L1→L2→L3 lineage is in `docs/DATA_LINEAGE.md`.

**A downstream reproduction needs no STK licence and no STK installation.** This package
depends only on data already exported from STK.

**The digital-twin generator itself is absent from the project.** `data/afterstk` is
*output*; nothing in the codebase consumes the physics parameters the trajectory manifest
records. Consequently degradation cannot be re-integrated and **no strict counterfactual
is producible** — every factor-direction claim is capped at `OBSERVATIONAL_EVIDENCE` and
labelled `NON_CAUSAL`.

## Known properties that look like defects and are not

* **989 non-monotone steps in `soh_observed`.** Expected sensor noise. `soh_true` has 0
  monotonicity violations.
* **Some `target_delta_soh_*` rows are positive.** Sensor noise on the endpoints; a
  property of the label definition.
* **`realized_elapsed_days` is 8 days short of `horizon_days` on 112 upstream rows.** Those
  targets land on the truncated final point of a mission-limited trajectory.
  `horizon_days` remains the definitional horizon = `horizon_steps × 14`.
* **All 92 eligible RUL anchors carry exactly 112 days.** Structural: the simulation stops
  at EOL. See `LIMITATIONS.md`.

## Splits

Three family-level splits: `target_train`, `target_calibration`, `target_test` (2 families
each). The primary protocol is nested leave-one-family-out; the fixed holdout is secondary.

**The fixed holdout is not a virgin test set.** It may not be called never-seen, virgin or
a final independent external validation: the frozen upstream task never read the `split`
column, and family-LOFO trains on those families in 6/6 folds.

## Licence and redistribution

The simulated data in this package was generated for this project. The public source
dataset that established the feature space is not redistributed here — only the frozen
clip/scaler statistics derived from it, which are summary statistics rather than data.
