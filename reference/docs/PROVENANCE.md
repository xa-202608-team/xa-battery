# Provenance of every frozen input

Each row names the upstream artifact, its sha256, and what this package took from it.
Hashes are recomputed at build time, and `verify` re-checks the ones carried inside
the package.

---

## 1. The frozen source space

| item | value |
|---|---|
| source package | `artifacts/experimental_arc_sensitivity/arc_clean_fixed_alpha10` |
| `model.npz` sha256 | `fcce4e530ace44a4f6b93fd98c695f0d0e6798144b5e159286ba9c13fbf2a979` |
| `manifest.json` sha256 | `143c3a11fe9e70d31491900cf3c3e5f152ddfecdfe02f88005f794fbae22d868` |
| model id | `arc_clean_fixed_alpha10` |
| carried into this package as | `models/arc_clean_fixed_space.npz` |

**What was taken:** `clip_lo`, `clip_hi`, `scaler_mean`, `scaler_std`,
`feature_names`, and the provenance hashes.

**What was deliberately NOT taken:** `coef` and `intercept`. Phase 3 ruled
`SOURCE_PRIOR_NOT_VALIDATED`, so the frozen source coefficients are not used as a
prior — and shipping them would invite exactly that misuse. The package therefore
*cannot* silently start using them.

The ARC package carries 10 coefficient rows for its own native horizon grid. There is
**no H=16 row**, and one may never be constructed, extrapolated or filled.

## 2. The frozen 11-dim extractor

| item | value |
|---|---|
| authoritative implementation | `src/production_model/features.py::common_features` |
| carried here as | `battery_entry/features.py::common_features` (verbatim copy) |
| schema version | `frozen_exact_11d_v1` |
| context length | 20 reference points (from the frozen ARC route manifest) |

The copy is not trusted on its word: `tests/test_release.py::test_03` recomputes the
features from the shipped 20-point history and compares against the shipped
`frozen_NN_*` columns at `atol <= 1e-12`. If the copy had drifted from the
implementation that produced the reference slice, that check fails.

Upstream, the same 11 features were shown **implementation-independent**: the
fitting-side and inference-side extractors agree bit-for-bit over 1000 comparisons.

## 3. The frozen alphas

| protocol | source | value |
|---|---|---|
| LOFO, per (fold, horizon) | `reports/stk_transfer_v2/04_exposure_stress/selected_configs.csv` | 18 cells |
| fixed holdout, per horizon | the same report's `run_manifest.json::frozen_alpha_cells` | 100.0 at H = 2, 4, 8 |

Chosen on **inner folds only** — `selector_saw_outer_test` is `False` on every row,
and the build script refuses to proceed if it is not. The grid (7 candidates) was
locked before any fit, inherited from `configs/stk_transfer.yaml::p3.alpha_grid`.

`models/frozen_alpha.json` records `alpha_selected_in_this_package: false`.

## 4. The frozen conformal half-widths

Extracted from `reports/stk_transfer_v2/05_uncertainty_rul/coverage_by_horizon.csv`,
**primary method rows only** (`FAMILY_BALANCED_CROSSFIT`), as
`mean_interval_width / 2` — the interval is symmetric about the point prediction
because the non-conformity score is an absolute residual.

24 cells: 2 protocols x 2 arms x 2 coverage levels x 3 horizons.

The two control methods' half-widths are **not** shipped. The primary method was
pre-declared, and shipping a menu would invite picking whichever covers best on new
data — the method switch Phase 5 forbids.

## 5. The reference data slice

| item | value |
|---|---|
| upstream | `data/stk_transfer_v2/L3_windows_v2_arc_clean_fixed.csv` |
| shipped as | `data/L3_windows_v2_arc_clean_fixed_min.csv` (34,335 rows) |
| column policy | **allow-list** — a column not named in the builder is absent |

Additions made at build time, each for a stated reason:

* `ctx_soh_00` ... `ctx_soh_19` — the 20-point observed-SOH context window, so the
  feature check is a genuine **recomputation** rather than a copy comparison. Verified
  at build time: `ctx_soh_19 == frozen_00_last` exactly.
* `frozen_predicted_delta` — the frozen Phase 4 outer-test point prediction, joined
  from `artifacts/stk_transfer_v2/04_exposure_stress/outer_test_predictions.npz` by
  `sample_uid`, so the golden test has a **bit-exact** target.
* `frozen_outer_fold` — which LOFO fold held this row's family out.

A measured note on the join: the `.npz` arrays are the bit-exact authority. The frozen
`interval_predictions.csv` agrees with them to **9.996e-17**, which is a CSV float
round-trip; the package joins the `.npz`-derived values.

## 6. The reference result tables

| shipped file | upstream source |
|---|---|
| `results/reference/outer_fold_results_reference.csv` | recomputed from `05_uncertainty_rul/interval_predictions.csv` (primary method, 90 %) |
| `results/reference/coverage_by_horizon_reference.csv` | `05_uncertainty_rul/coverage_by_horizon.csv` |
| `results/reference/conformal_results_reference.csv` | `05_uncertainty_rul/conformal_results.csv` (carries the `ALL_MATCHED` aggregates) |
| `results/reference/coverage_by_family_reference.csv` | `05_uncertainty_rul/coverage_by_family.csv` |
| `results/reference/rul_results_summary.csv` | `05_uncertainty_rul/rul_results.csv` |
| `results/reference/result_summary.csv` | generated from all of the above |
| `results/reference/claims_manifest.csv` | generated; every claim bound to a hash |

## 7. What no upstream artifact was allowed to become

* no `artifacts/production_model_v2` was created;
* no formal `recommended_config.json` was modified;
* no `PROJECT_STATUS.md` was modified;
* no Phase 0-6 source, config, report, prediction, model or gate was modified.

This package is downstream of all of them and writes into none of them.

