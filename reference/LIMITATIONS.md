# Limitations

Ordered by how much each one binds. The first is not a caveat on the results — it is the
frame the results sit inside.

---

## 1. No real on-orbit data exists in this project

Every metric — MAE, coverage, RUL — was measured on an STK-derived simulated twin. **Real
on-orbit performance was NEVER TESTED.**

A better number here is a better number about the *simulator*, not about a satellite. This
is not a hedge; it is the literal scope of the evidence.

## 2. The twin is uncalibrated against a real satellite

Nothing establishes that its degradation rates, noise level or failure modes match flight
hardware. The error magnitudes therefore carry **no on-orbit meaning even as an order of
magnitude**. A family-macro MAE of 0.0033 SOH is a statement about this simulator's
trajectories.

## 3. No digital-twin generator, so no causal claim

The simulator that produced the data is absent from the project: `data/afterstk` is
*output*, and nothing in the codebase consumes the physics parameters the trajectory
manifest records. Degradation cannot be re-integrated, so **no strict counterfactual is
producible**.

Every factor-direction claim is capped at `OBSERVATIONAL_EVIDENCE` and labelled
`NON_CAUSAL`. Fabricating a contrast by editing a stress column is refused by the upstream
guard, not merely discouraged.

## 4. Exchangeability is violated by design

Under leave-one-family-out the held-out family is a genuinely **new environment**. Split
conformal's guarantee assumes exchangeability, so even the *marginal* guarantee is not
certified out of distribution — and the worst-family and fixed-holdout shortfalls are
exactly what that violation looks like when measured.

## 5. The intervals are per-horizon marginal, and one family is 14 points short

Gate: **`CONFORMAL_NOT_VALIDATED`**.

At 90 % nominal, family-macro coverage is 0.9045 while `h550_i053_raan180` covers
**0.7641**. At 95 %, that family covers 0.8628. The fixed holdout covers 0.8683
(window-micro) at 90 % nominal.

Two flags fired: `CONFORMAL_WORST_FAMILY_SHORTFALL`, `FIXED_HOLDOUT_COVERAGE_SHORTFALL`.
`CONFORMAL_FAMILY_MACRO_SHORTFALL` did **not** fire.

The shortfall is the honest signature of a **marginal** guarantee, not a bug: split
conformal promised a marginal rate and delivered one, five of six families sit at or above
nominal, and it never promised family-conditional coverage. It is **not** repaired by
widening the interval after the fact — that would be re-calibration after seeing coverage,
which the protocol refuses.

Half-widths grow ~4× from 28 to 112 days (0.00341 → 0.01406 at 90 %), which is why each
horizon is calibrated separately. **Three separately-calibrated marginal intervals are not
a simultaneous band.** Joint coverage over the path is ≤ 0.90 and possibly far lower.

## 6. RUL cannot be evaluated on this dataset at all

Evidence status: **`RUL_EVIDENCE_INSUFFICIENT`**. Label status:
`RUL_LABEL_DEGENERATE_SINGLE_VALUE`.

The degeneracy is structural, not a tuning problem. Measured on the upstream layer, `k*`
equals the number of remaining reference steps for **every** uncensored anchor, because the
simulated trajectory terminates at EOL (92/120 trajectories cross 0.70, and the crossing is
at the final recorded index in 92/92 cases). So:

* "true EOL within 112 days" forces `k* ≤ 8`;
* "predictions available at all of H = 2, 4, 8" forces `k* ≥ 8`;
* the intersection is `k* == 8` **exactly**.

All 92 eligible anchors therefore carry `target_rul_days == 112.0`, std 0.0. **A constant
112-day predictor would score MAE = 0.** The reported MAEs measure how near the forecast
path crosses to the end of the record, not remaining-life discrimination. They are also
computed on *different* anchor sets (32 vs 13, the latter a strict subset), so
`mae_directly_comparable = False`.

**No ranking may rest on RUL.**

The eligible set was **not** widened to escape this, and must not be: relaxing "all three
horizons" would admit anchors whose day-112 node does not exist (scoring them needs
extrapolation past the horizon), and relaxing "true EOL within 112 days" would require
predicting a crossing the path cannot reach. The degeneracy is reported instead.

This is a **data-shape** limitation, not a modelling one. A dataset whose trajectories
continue past EOL, or a shorter forecast horizon paired with a shorter EOL definition,
would be needed.

## 7. The warning lead time is nearly vacuous

90 of the 92 eligible anchors are **already** at or below the 0.80 warning threshold when
observed (eligible anchors sit near end of life by construction; mean anchor SOH ≈ 0.755).
For those, "the warning fires" restates the present observation rather than forecasting
anything, so they are excluded from the lead-time mean — leaving **2 anchors** behind each
reported mean. It may not be quoted as a capability.

## 8. The margin over a non-learning baseline is small

Family-macro MAE 0.003328 (fitted) vs 0.003588 (closed-form extrapolation of the observable
slope, **no fitting at all**). The fitted model is ≈ 7.2 % better.

That is the honest headline. A reader entitled to ask "what did the learning buy?" gets
"about 7 % on a simulator", which is why the non-learning baseline ships beside the model
rather than behind it.

## 9. Only 6 environment families, 120 trajectories

Family-macro statistics rest on **6** units. Windows overlap in time within a trajectory,
so window counts are not sample sizes: `n_trajectories` is the independent-unit count, and
bootstrapping is over trajectories.

## 10. The fixed holdout is not a virgin test set

It may not be called never-seen, virgin, or a final independent external validation. The
frozen upstream task never read the `split` column, and family-LOFO trains on those
families in 6/6 folds.

## 11. What the negative results are, and are not

`SOURCE_PRIOR_NOT_VALIDATED`, `RATE_DAY_REPARAMETERIZATION_ONLY`,
`EXPOSURE_INPUT_NOT_AVAILABLE`, `STRESS_NOT_VALIDATED` (×2), `CONFORMAL_NOT_VALIDATED`,
`RUL_EVIDENCE_INSUFFICIENT`.

**None of these is an execution failure.** Each is a measurement taken under a method fixed
*before* the data was seen. Two specifically:

* **T1 is not a failed time experiment.** Weighted ridge is linear in the label and α sits
  on the label-free side of the normal equations, so under per-horizon separate fitting,
  dividing the label by a per-stratum constant and multiplying back is an *identity*.
  Measured: worst |Δprediction| 2.8e-13 over 72 cells at the same α, against a 1e-10
  tolerance, with a deliberately-mismatched control arm that does *not* close. **No MAE
  difference between T0 and T1 may be described as a time-normalisation gain.**
* **Stress was measured, not merely excluded.** It was built, audited (25 of 29 derived
  fields cleared identifiability) and run. Neither candidate passed a gate. Adding *more*
  stress columns is not the indicated next step.

## 12. PatchTST was not run

`PHASE6_CLOSED_WITHOUT_TRAINING` / `PATCHTST_SKIPPED_BY_PREDEFINED_GATE`.
`was_run = false`, `was_implemented = false`, `is_experiment_failure = false`.

No PatchTST result exists in this project, favourable or unfavourable. What exists is a
**decision not to run one**, taken on pre-declared grounds.

---

## Statements that may NOT be made

* ~~Source-Prior is validated after all~~
* ~~the `SOURCE_PRIOR_NOT_VALIDATED` verdict is overturned~~
* ~~frozen source coefficients provided additional gain~~
* ~~a source model generalised into the space domain~~
* ~~a gate was overturned~~
* ~~stress features help~~
* ~~time normalisation improved the model~~
* ~~planned mission exposure is available~~
* ~~family-conditional coverage is guaranteed~~
* ~~coverage is guaranteed on a real satellite~~
* ~~coverage is guaranteed out of distribution~~
* ~~coverage holds simultaneously across the three horizons~~
* ~~the 90 % band covers the 112-day trajectory with 90 % probability~~
* ~~the family-balanced result proves conditional validity~~
* ~~the robustness control independently confirms the family-balanced result~~
* ~~the pooled control is group-aware~~
* ~~RUL is validated~~
* ~~the RUL MAE of two arms computed on different eligible sets may be compared~~
* ~~an RUL beyond 112 days was predicted~~
* ~~a censored trajectory's RUL is known~~
* ~~`soh_observed` established the true EOL~~
* ~~the fixed holdout is a virgin / never-seen test set~~
* ~~PatchTST was tried and failed~~
* ~~PatchTST underperformed~~
* ~~deep models were empirically ruled out on this dataset~~
* ~~a neural baseline was measured~~
* ~~production v2 is complete~~
* ~~this model is promoted / production~~
* ~~STK simulation metrics are real-satellite accuracy~~
* ~~STK simulation coverage is real-satellite coverage~~
* ~~STK simulation RUL error is real-satellite RUL error~~

Two that are especially easy to slip into, spelled out:

**"The 90 % band covers the 112-day trajectory with 90 % probability."** The three horizons
are calibrated **separately**, so each is a marginal interval. Joint coverage over the path
is ≤ 0.90 and possibly far less.

**"The robustness control confirms the family-balanced result."** Every family holds exactly
20 trajectories, so `n_fam × n_traj_in_fam == n_traj_total` and the two group-aware weight
vectors are identical element-wise (measured max |Δ| = 0.0). Their agreement is
**arithmetic, not corroboration**. The informative contrast is against
`POOLED_SPLIT_CONFORMAL`, which differs in both the weighting and the finite-sample unit.

---

## Consequence for deployment

The deliverable is a **method and protocol validated in simulation**, not a flight-qualified
predictor. Real telemetry would re-open every gate from Phase 3 onward, and the acceptance
bars are already written down — see `docs/GATES.md` and the real-data validation plan in the
project report tree.

**Engineering completeness is not scientific validation.** Even with the package built,
tested and containerised, these remain: `CONFORMAL_NOT_VALIDATED`,
`RUL_EVIDENCE_INSUFFICIENT`, `CANDIDATE_NOT_PROMOTED`, `SIMULATION_ONLY`.
