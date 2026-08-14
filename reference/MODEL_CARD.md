# Model card — `target_only_arc_space`

## Identity

| field | value |
|---|---|
| model id | `target_only_arc_space` |
| transfer mechanism | `FEATURE_SPACE_TRANSFER_WITH_TARGET_DOMAIN_HEAD_REFIT` |
| high-stability baseline shipped alongside | `observable_local_trend_extrapolation` |
| status | `CANDIDATE_NOT_PROMOTED` |
| evidence domain | `SIMULATION_ONLY` |
| target | `T0_DELTA_REFERENCE` — future simulated truth minus the observable anchor |
| feature set | `F0_geometry11` — 11 observable SOH-geometry columns |
| horizons | H ∈ {2, 4, 8} reference steps = 28 / 56 / 112 days |
| source route | `arc_clean_fixed` |
| stress features | **disabled** |
| prediction-time exposure | **disabled** |
| source coefficient prior | **disabled** |

## What the model is, structurally

A weighted ridge regression on 11 features, in a coordinate system borrowed from a
frozen public-data model.

Concretely, two halves with different provenance:

**The half that transfers — the feature space.** The frozen source ARC model's
`clip_lo`/`clip_hi` and `scaler_mean`/`scaler_std` define the coordinate system. They are
reused **unchanged** and are never refitted. This is what crossed the domain gap.

**The half that does not transfer — the prediction head.** The regression coefficients and
intercept are estimated **entirely from target-domain rows**. The frozen source
coefficients are not used as a prior, not used as an initialisation, and are not even
shipped in this package.

That second point is a measured finding, not a design preference. Phase 3 fitted a
Source-Prior Ridge-Delta with a pre-locked six-value λ grid and let inner folds choose
freely. They chose **λ_source = 0 in 44 of 72 cells (61.1 %)**, and every non-zero choice
was the grid *minimum* (0.01); 0.1, 1, 10 and 100 were never selected. Fitted `W` moved a
full `‖W_source‖` or more away from the prior (retention ratio 0.00–0.065). Verdict:
`SOURCE_PRIOR_NOT_VALIDATED` on all three formal priors.

**This may not be restated as "frozen source coefficients provided additional gain."**
Phase 2B's ARC advantage was, and remains, a **pre-processing** effect.

## The 11 features

Order is load bearing — it comes from the frozen source manifest's `feature_names`, and
position in the list is position in the design matrix.

```
last, diff1, diff2, diff_long, slope5, slope_full,
curvature, ctx_mean, ctx_std, ctx_min, ctx_max
```

Computed from a 20-point observed-SOH history and **nothing else**. No future reference
point, no hidden-truth channel, no exposure column, no environment-family or orbit
column. That is why the space is safe to compute at prediction time, and it is enforced
at a single choke point rather than by discipline.

`anchor_soh_observed` is an **anchor, never a design column**: it is bit-identical to
`frozen_00_last`, so including it would add a perfectly collinear duplicate rather than
information.

Implementation: `battery_entry/features.py::common_features`, a verbatim copy of the
research project's `src/production_model/features.py::common_features` — the same
extractor that produced the shipped reference slice. `tests/test_release.py::test_03`
recomputes the features from the shipped history and fails if the copy has drifted.

## Training and hyperparameters

| item | value |
|---|---|
| estimator | weighted ridge, closed form (`Z^T S Z + αI`)⁻¹ `Z^T S y` |
| weights | trajectory-equal inside a horizon stratum, then family-balanced |
| α, LOFO | frozen per (outer fold, horizon) — 18 cells |
| α, fixed holdout / deployment | frozen at 100.0 for every horizon |
| α selection | **not performed here** — read from the frozen Phase 4 inner-fold selection |
| α grid | locked before any fit: 7 candidates from `configs/stk_transfer.yaml::p3.alpha_grid` |

Weights matter: windows per trajectory span 24–137 inside one horizon, so uniform row
weighting would let one trajectory count 5.7× another. Weights are normalised to mean 1.0
so α keeps the meaning it had in the frozen unweighted fit.

Two solve paths are provided and are the *same* normal equations: NumPy (default) and
torch float64 CPU (`--solver torch`). The torch path is the `λ_source = 0` limit of the
already-verified Phase 3 solver, not a second algorithm;
`tests/test_release.py::test_06` proves they agree.

## Performance

STK-derived simulated twin. Leave-one-family-out over 6 held-out environment families,
120 simulated trajectories, matched horizons.

| metric | final model | non-learning baseline |
|---|---|---|
| family-macro MAE (absolute SOH) | **0.003328** | 0.003588 |
| margin | — | fitted model is ≈ 7.2 % better |

Report `n_trajectories` as the sample size, never window counts: windows overlap in time
within a trajectory. Bootstrap is over trajectories (2000 replicates, seed 42).

**These are simulation numbers on an uncalibrated twin. They are not satellite accuracy.**

## Uncertainty

Split conformal, primary method `FAMILY_BALANCED_CROSSFIT`, pre-declared before any
residual existed. Non-conformity score is the absolute residual on absolute SOH, so the
interval is symmetric by construction.

| nominal | family-macro | worst family | mean width |
|---|---|---|---|
| 90 % | 0.9045 | **0.7641** (`h550_i053_raan180`) | 0.016053 |
| 95 % | 0.9530 | 0.8628 | 0.023213 |

Gate: **`CONFORMAL_NOT_VALIDATED`**. Flags fired: `CONFORMAL_WORST_FAMILY_SHORTFALL`,
`FIXED_HOLDOUT_COVERAGE_SHORTFALL`. `CONFORMAL_FAMILY_MACRO_SHORTFALL` did **not** fire.

Three things this does and does not mean:

1. **Per-horizon MARGINAL, no simultaneous guarantee.** Calibrated separately per horizon
   because the half-width grows ~4× from 28 to 112 days. Joint coverage over the three is
   ≤ 0.90 and possibly far lower.
2. **The worst-family shortfall is the honest signature of a marginal guarantee.** Split
   conformal promises a marginal rate and delivered one; five of six families sit at or
   above nominal. It never promised family-conditional coverage, and widening the
   interval after seeing the shortfall would be re-calibration after the fact.
3. **Exchangeability is violated by design.** Under leave-one-family-out the held-out
   family is a genuinely new environment, so even the marginal guarantee is not certified
   out of distribution — which is what the shortfalls show.

## RUL

`SECONDARY_DISPLAY`, evidence status **`RUL_EVIDENCE_INSUFFICIENT`**. Nothing is fitted on
an RUL label and no RUL model exists; the read-out is derived from the same SOH point
predictions.

The eligible set is **structurally degenerate**: all 92 eligible anchors carry
`target_rul_days == 112.0`, std 0.0, because the simulation terminates at EOL. **A
constant 112-day predictor would score MAE = 0.** The reported MAEs (33.36 d vs 30.96 d)
are also on *different* anchor sets (32 vs 13, the latter a strict subset), so
`mae_directly_comparable = False`. **Nothing may be ranked on RUL.**

The warning lead time is nearly vacuous: 90 of the 92 eligible anchors are already at or
below the 0.80 threshold when observed (mean anchor SOH ≈ 0.755), leaving **2 anchors**
behind each reported mean.

## Intended use

Method and protocol demonstration on simulated spaceflight battery degradation. Suitable
for: evaluating the transfer approach, reproducing the reported simulation results,
sanity-checking a pipeline on synthetic histories.

**Not suitable for:** flight decisions, mission planning, on-orbit health management, or
any use where a wrong SOH or RUL carries operational consequence. Real-satellite
performance was never tested.

## Out-of-scope inputs

* histories shorter or longer than 20 reference points on a 14-day grid;
* SOH on a percentage scale, or outside (0, 1.5];
* non-monotone-in-time or duplicated reference indices;
* chemistries, form factors or duty cycles unlike the simulated twin's;
* any request for an RUL beyond 112 days.

## Deep sequence models

**PatchTST was not run and not implemented.** No PatchTST result exists in this project,
favourable or unfavourable. What exists is a **decision not to run one**, taken on
pre-declared grounds (`PHASE6_CLOSED_WITHOUT_TRAINING`,
`PATCHTST_SKIPPED_BY_PREDEFINED_GATE`).

This may **not** be written as "the PatchTST experiment failed", "PatchTST
underperformed", "deep models were empirically ruled out on this dataset" or "a neural
baseline was measured". All four would be fabrications.

The load-bearing ground for the decision: a deeper model cannot repair what actually
binds here — the absence of a digital-twin generator (so no strict counterfactual is
producible) and the absence of real on-orbit data (so more capacity buys a better number
about the *simulator*, not about a satellite). `tests/test_release.py::test_13` greps for
neural layers, autograd and optimisers so this is checked mechanically.

## Provenance

Frozen ARC source package: `arc_clean_fixed_alpha10`. Its `model.npz` sha256 and the
clip/scaler array hashes are carried inside
`models/arc_clean_fixed_space.npz` and re-verified by `verify`. Full lineage in
`docs/PROVENANCE.md`; per-claim evidence with hashes in `CLAIMS_MANIFEST.json`.
