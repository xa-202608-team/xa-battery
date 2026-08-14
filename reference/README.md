# battery_release_v2 — STK-transfer battery SOH prediction

Self-contained delivery package. It carries its own copy of the frozen 11-dim feature
extractor, the frozen ARC clip/scaler, a reference data slice, the frozen reference
results and the full test suite. It does **not** import the research project, does not
need an STK installation, and does not touch the network at run time.

```
python -m battery_entry verify
python -m battery_entry reproduce --mode quick --output ./out
python -m battery_entry reproduce --mode full  --output ./out_full
python -m battery_entry predict --input examples/example_input.csv --output ./pred.csv
```

---

## 1. What this is, in one paragraph

A public battery dataset was used to establish a unified degradation **feature space**
and pre-processing convention. That space was then transferred to an STK-driven
spaceflight battery digital-twin domain, and the **prediction head was re-estimated on
target-domain trajectories**. Strict nested evaluation showed that the source-domain
pre-processing space has value, but the frozen source **coefficients** provided no
verifiable additional gain — so the final configuration is feature-space transfer plus
target-domain head refit. **Every conclusion is confined to the simulated environment
and awaits validation on real on-orbit or ground-equivalent data.**

In one line: **a coordinate system crossed the domain gap; a predictive function did
not.**

---

## 2. What ships, and what each part is for

| path | role |
|---|---|
| `battery_entry/` | the CLI, features, model, conformal, RUL, schema validation, verify |
| `models/arc_clean_fixed_space.npz` | the frozen source ARC clip/scaler + provenance hashes |
| `models/frozen_alpha.json` | the frozen alpha per (fold, horizon); read, never chosen |
| `models/conformal_quantiles.json` | the frozen Phase 5 half-widths; applied, never re-derived |
| `data/` | the reference L3 slice (allow-listed columns + 20-point context + frozen prediction), L2 grid, split manifest |
| `schemas/` | the input contract and the feature schema |
| `results/reference/` | the frozen metric tables `quick` restates |
| `tests/test_release.py` | 15 acceptance checks |
| `docs/` | gates, provenance, data lineage, STK scenarios |
| `examples/` | a minimal input and its expected output |
| `Dockerfile`, `requirements.lock` | minimal CPU image, pinned dependencies |
| `CLAIMS_MANIFEST.json` | every claim bound to evidence file, field, sha256, allowed scope |
| `release_manifest.json`, `SHA256SUMS` | what is here and what it hashes to |

The frozen source **coefficients** are deliberately **not** shipped. Phase 3 ruled
`SOURCE_PRIOR_NOT_VALIDATED`, so they are not used as a prior, and shipping them would
invite exactly that misuse.

---

## 3. The four commands

### `verify`

Checks the release manifest, recomputes **every** SHA256 from disk, greps every shipped
file for parent-project paths / `parents[N]` escapes / absolute host paths, and confirms
the model, clip/scaler, feature order and configs all agree. Also confirms the frozen
verdicts ship verbatim and that no neural sequence model is implemented.

Exit `0` on `VERIFY_OK`, `1` otherwise.

### `reproduce --mode quick --output <dir>`

CPU, target under 10 minutes. Refits nothing unfrozen and re-selects nothing. It:

1. recomputes the 11 features from the shipped observed-SOH history and checks them
   against the shipped frozen feature columns;
2. replays the frozen head at the frozen alpha and compares the point predictions
   against the frozen Phase 4/5 predictions at `rtol=0, atol<=1e-12`;
3. restates the formal SOH metrics, the per-horizon coverage/width summary and the RUL
   summary from the shipped reference tables.

### `reproduce --mode full --output <dir>`

Re-fits **only** the frozen final model `target_only_arc_space`, under the frozen
protocol: same leave-one-family-out folds, same frozen alpha per (fold, horizon), same 11
features in the same frozen ARC space, same trajectory-equal/family-balanced weights,
same three horizons. Regenerates point predictions, per-horizon intervals and the main
result tables into `<dir>`.

It does **not** run Source-Prior, stress, exposure, T1, PatchTST, any new alpha grid, or
any re-calibration. Each of those is a question an earlier phase closed.

### `predict --input <csv> --output <csv>`

Point prediction of absolute SOH at days 28 / 56 / 112 from exactly 20 observed-SOH
reference points per battery. Add `--intervals` for interval bounds (labelled
`EXPERIMENTAL_SIMULATION_INTERVAL`) and `--rul` for the finite-horizon read-out (labelled
`FINITE_HORIZON_EVIDENCE_INSUFFICIENT`).

Exit codes: `0` all predicted; `2` input refused by the schema validator; `3` predicted,
but at least one path did not reach 0.70 inside 112 days
(`NO_CROSSING_WITHIN_FORECAST_HORIZON` — nothing is extrapolated).

Input is **refused, never repaired**. A percentage-scale SOH, a shuffled history or a
short context would each still produce a number, and that number would be wrong without
anything looking wrong.

---

## 4. Headline numbers

All measured on an **STK-derived simulated twin**, leave-one-family-out over 6 held-out
environment families, 120 simulated trajectories, H ∈ {2, 4, 8} = 28 / 56 / 112 days.

| quantity | value |
|---|---|
| family-macro MAE, final model | **0.003328** absolute SOH |
| family-macro MAE, non-learning baseline | 0.003588 |
| margin of the fitted model over a closed-form slope extrapolation | **≈ 7.2 %** |
| 90 % interval: family-macro coverage | 0.9045 |
| 90 % interval: worst family (`h550_i053_raan180`) | **0.7641** |
| 90 % mean interval width | 0.016053 SOH |
| RUL eligible anchors | 92, all with `target_rul_days == 112.0` |

The ~7 % margin is the honest headline. A closed-form extrapolation of the observable
slope, with no fitting at all, gets most of the way — which is why the non-learning
baseline ships beside the fitted model rather than behind it.

---

## 5. What is NOT validated

| item | verdict |
|---|---|
| source **parameter** prior (3 formal priors) | `SOURCE_PRIOR_NOT_VALIDATED` |
| time normalisation | `RATE_DAY_REPARAMETERIZATION_ONLY` (T1 *is* T0, algebraically) |
| prediction-time exposure | `EXPOSURE_INPUT_NOT_AVAILABLE` (NULL on 100 % of rows) |
| observable stress features | `STRESS_NOT_VALIDATED` (both budgets) |
| interval coverage | `CONFORMAL_NOT_VALIDATED` |
| RUL | `RUL_EVIDENCE_INSUFFICIENT` |
| PatchTST / deep models | `PATCHTST_SKIPPED_BY_PREDEFINED_GATE` — **not run, not implemented, not a failed experiment** |
| real on-orbit performance | **NEVER TESTED** |
| causal direction of any stress factor | `OBSERVATIONAL_EVIDENCE` ceiling |

Source **pre-processing** transfer *is* validated: the ARC source space beat a
target-native Ridge on 6/6 families. That is a different claim from the parameter prior,
and the two must not be merged.

See `LIMITATIONS.md` for the full list of statements that may not be made, and
`docs/GATES.md` for each verdict with its evidence.

---

## 6. Two things that are especially easy to get wrong

**The 90 % intervals are per-horizon MARGINAL.** Each horizon is calibrated separately
because the half-width grows ~4× from 28 to 112 days. Three intervals that each cover
with probability 0.90 have *joint* coverage no greater than 0.90 and possibly far less.
They are **not** "a 112-day band".

**The robustness control does not corroborate the primary method.** Every family holds
exactly 20 trajectories, which makes the two group-aware weightings algebraically
identical (measured max |Δ| = 0.0). Their agreement is arithmetic, not evidence.

---

## 7. Status

`CANDIDATE_NOT_PROMOTED` · `SIMULATION_ONLY`

This package promotes nothing, creates no production v2, and modifies no formal
recommended configuration. Engineering completeness is not scientific validation: the
gates above remain as their phases decided them.
