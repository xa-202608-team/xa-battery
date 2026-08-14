# Gates and verdicts

Every verdict below was decided by its own phase under a method fixed **before** the
data was seen, and none may be softened, omitted or restated as a success here.

**A negative result is not an execution failure.** Each of these is a measurement.

---

## Summary

| gate | verdict | phase |
|---|---|---|
| source **parameter** prior (3 formal priors) | `SOURCE_PRIOR_NOT_VALIDATED` | 3 |
| `TIME_NORMALIZATION_GATE` | `RATE_DAY_REPARAMETERIZATION_ONLY` | 4 |
| `EXPOSURE_GATE` | `EXPOSURE_INPUT_NOT_AVAILABLE` | 4 |
| `STRESS_GATE_LOW_DATA` | `STRESS_NOT_VALIDATED` | 4 |
| `STRESS_GATE_FULL_DATA` | `STRESS_NOT_VALIDATED` | 4 |
| `CONFORMAL_GATE` | `CONFORMAL_NOT_VALIDATED` | 5 |
| RUL evidence | `RUL_EVIDENCE_INSUFFICIENT` | 5 |
| PatchTST / deep models | `PATCHTST_SKIPPED_BY_PREDEFINED_GATE` | 6 |
| real on-orbit performance | **NEVER TESTED** | — |
| causal direction of any factor | `OBSERVATIONAL_EVIDENCE` ceiling | 4 |

What **is** validated, and must not be merged with the first row above:

| finding | evidence |
|---|---|
| source **pre-processing** transfer has value | the ARC source space beat a target-native Ridge on **6/6** families (Phase 2B); Phase 4's F4 control re-confirmed it (0/6 wins for target-native at full budget) |

---

## 1. `SOURCE_PRIOR_NOT_VALIDATED` (Phase 3)

Inner folds were free to choose any prior strength from a **pre-locked** grid of six
values. They chose **lambda_source = 0 in 44 of 72 cells (61.1 %)**, and where
non-zero, **always the smallest value, 0.01**. `0.1`, `1`, `10` and `100` were never
selected. Fitted `W` moved a full `||W_source||` or more from the prior (retention
ratio 0.00-0.065). On the fixed holdout `lambda_source = 0` was selected at every
budget, so prior and comparator are literally the same model in 3 of 4 cells (diff
exactly 0.000000).

**The source parameter prior did not add value beyond the source pre-processing.**

This may **not** be reopened by widening the grid upward — the selected values sit at
the *bottom* of it, so widening answers a question the data already answered. A
different prior *form* would be a new named experiment with its own contract.

## 2. `RATE_DAY_REPARAMETERIZATION_ONLY` (Phase 4)

**T1 is not a failed time experiment; T1 *is* T0.** Weighted ridge is linear in the
label and alpha sits on the label-free side of the normal equations. Under
per-horizon separate fitting `horizon_days` is constant within each stratum
(H=2 -> 28 d, 4 -> 56 d, 8 -> 112 d), so dividing the label by it and multiplying
back is an **identity**.

Measured: worst |delta prediction| **2.8e-13** over 72 cells at the same alpha,
against a `1e-10` tolerance. A deliberately-mismatched control arm (alpha * d^2) has
minimum difference `1.2e-3`, which proves the tolerance is tight enough to detect a
real difference.

**No MAE difference between T0 and T1 may be described as a time-normalisation gain.**

## 3. `EXPOSURE_INPUT_NOT_AVAILABLE` (Phase 4)

Planned future exposure **does not exist** in this dataset: `future_planned_delta_efc`
is NULL on **100 %** of rows, evidence status
`INSUFFICIENT_EVIDENCE_NOT_DETERMINABLE_AT_PREDICTION_TIME`. T2 was never run.

The enabling criterion is already written down: a planned channel must reproduce
realized exposure at prediction time, and the current climatology's **5.8 % median /
15.0 % max** relative error is the bar it failed.

**Realized exposure may never be substituted** — it reads the trajectory *after* the
cutoff and is `ORACLE_AUDIT_ONLY`.

## 4. `STRESS_NOT_VALIDATED` x2 (Phase 4)

Stress was **measured, not merely excluded**. Phase 2B kept it out by guardrail;
Phase 4 built it, audited it and ran it:

* the raw `stress_*` columns are **point values at `context_end`**, not history
  summaries — verified against every candidate aggregation window;
* a derived `obs_hist_*` layer was built with a structurally-guaranteed
  `<= context_end` prefix bound; **25 of 29** fields cleared the identifiability audit;
* neither F2 (core) nor F3 (full) passed a gate at either budget.

**Adding more stress columns is not the indicated next step.** The orbit triplet is
permanently excluded as a family proxy.

## 5. `CONFORMAL_NOT_VALIDATED` (Phase 5)

Flags fired: `CONFORMAL_WORST_FAMILY_SHORTFALL`, `FIXED_HOLDOUT_COVERAGE_SHORTFALL`.
`CONFORMAL_FAMILY_MACRO_SHORTFALL` did **not** fire.

At 90 % nominal: family-macro **0.9045**, worst family (`h550_i053_raan180`)
**0.7641**; at 95 % that family covers 0.8628. Fixed-holdout window-micro coverage
0.8683 at 90 % nominal. Five of six families sit at or above nominal.

The shortfall is the **honest signature of a marginal guarantee**, not a bug: split
conformal promised a marginal rate and delivered one; it never promised
family-conditional coverage. It may **not** be repaired by widening the interval post
hoc.

The 90 % / 95 % figures are **per-horizon MARGINAL** intervals. Half-widths grow ~4x
from 28 to 112 days (0.00341 -> 0.01406 at 90 %), which is why each horizon is
calibrated separately. **Three separately-calibrated marginal intervals are not a
simultaneous band.**

Two method notes that prevent over-reading the result:

* **The two group-aware methods coincide by arithmetic here.** Every family holds
  exactly 20 trajectories, so `n_fam * n_traj_in_fam == n_traj_total` and the two
  weight vectors are identical element-wise (measured max |delta| = 0.0). The
  robustness control **cannot** disagree, so its agreement is not corroboration.
* **`POOLED_SPLIT_CONFORMAL` is not group-aware and must never be called that.** Its
  finite-sample correction counts ~10,000 overlapping windows rather than 100
  independent trajectories, which is why its interval is narrower and its coverage
  lower. It quantifies the cost of ignoring group structure.

The cross-fit is **leave-one-FAMILY-out**, deliberately: leave-one-trajectory-out
would let the calibration model train on 19 siblings from the scored trajectory's own
family, understating the error on a new family. The deployment question is a new
family.

## 6. `RUL_EVIDENCE_INSUFFICIENT` (Phase 5)

Structurally degenerate, not under-tuned. `k*` equals the number of remaining
reference steps for **every** uncensored anchor because the simulation terminates at
EOL (92/120 trajectories cross 0.70; the crossing is at the final recorded index in
92/92 cases). 'True EOL within 112 days' forces `k* <= 8`; 'all three horizons
available' forces `k* >= 8`; the intersection is `k* == 8` exactly.

All 92 eligible anchors carry `target_rul_days == 112.0`, std 0.0. **A constant
112-day predictor would score MAE = 0.** The two arms' MAEs are on *different* anchor
sets (32 vs 13, the latter a strict subset), so `mae_directly_comparable = False`.

**No ranking may rest on RUL.** The eligible set was not widened to escape the
degeneracy and must not be. This is a **data-shape** limitation, not a modelling one.

The warning lead time is nearly vacuous: 90 of 92 eligible anchors are already at or
below 0.80 when observed (mean anchor SOH ~ 0.755), leaving **2 anchors** behind each
reported mean.

## 7. `PHASE6_CLOSED_WITHOUT_TRAINING` (Phase 6)

* `was_run` = **False**
* `was_implemented` = **False**
* `is_experiment_failure` = **False**

**PatchTST was not run, and no PatchTST result exists — favourable or unfavourable.**
What exists is a *decision not to run one*, taken on pre-declared grounds.

This may **not** be written as any of:

* ~~PatchTST underperformed~~
* ~~PatchTST was tried and failed~~
* ~~deep models were empirically ruled out on this dataset~~
* ~~a neural baseline was measured~~

All would be fabrications.

**The load-bearing ground is the last one, not the first three.** The two binding
constraints are *identification* and *data*, and neither is a function-class problem:

1. **Causal identification.** The simulator is not in the repository, so degradation
   cannot be re-integrated and no strict counterfactual is producible. A larger
   hypothesis class does not change what is identifiable from the data it is given.
2. **Absence of real on-orbit data.** Fitting higher capacity to the same simulation
   buys a better number about the *simulator*, not about a satellite.

Against a 100-trajectory target domain, adding capacity would raise complexity and
overfitting risk while leaving both constraints untouched. Phase 4 already measured
that 25 audited observable-history stress columns failed to pass a gate; the indicated
next step is **better data, not a bigger model**.

**What would reopen the question:** real on-orbit telemetry, or a digital-twin
generator permitting strict counterfactuals. Neither would revive *this* gate — a
deep-model evaluation would need a fresh pre-registered protocol with its own budget
ladder, isolation keys and acceptance criteria.

---

## Engineering completeness is not scientific validation

Even with this package built, tested and containerised, all of these still hold:

* `CONFORMAL_NOT_VALIDATED`
* `RUL_EVIDENCE_INSUFFICIENT`
* `CANDIDATE_NOT_PROMOTED`
* `SIMULATION_ONLY`

