# Reproducibility

## Environment

Every frozen number in this package was produced in this environment:

| item | value |
|---|---|
| Python | 3.9.25 |
| numpy | 2.0.2 |
| pandas | 2.3.3 |
| PyYAML | 6.0.2 |
| platform | Windows 11 (win32) |
| device | CPU only — no GPU was used at any stage |
| dtype | float64 throughout |

Pinned exactly in `requirements.lock`. Only three packages are needed to run
`verify` / `reproduce` / `predict`; `matplotlib` is for figure regeneration only, and
`torch` is optional (see §5).

No network access is required at run time. Every input this package needs is inside it.

## 1. Verify first

```
python -m battery_entry verify
```

Seven check groups: manifest coverage, SHA256 of every shipped file (recomputed from
disk), no-parent-dependency grep, model/space/feature-order consistency, schema parity
with the code, frozen verdicts present verbatim, and no neural implementation.

Expect `VERIFY_OK` and exit `0`.

## 2. Quick reproduction

```
python -m battery_entry reproduce --mode quick --output ./out_quick
```

CPU, target under 10 minutes. Refits nothing unfrozen; selects nothing.

| step | what it establishes |
|---|---|
| feature recomputation | the shipped extractor reproduces the shipped frozen feature columns from the 20-point history, `atol ≤ 1e-12` |
| golden replay | the release fit path reproduces the frozen Phase 4/5 outer-test point predictions on the common sample set, **`rtol = 0, atol ≤ 1e-12`**, 18 (fold, horizon) cells |
| reference metrics | the formal SOH / coverage / RUL numbers restated from the shipped tables with their provenance |
| coverage re-application | the frozen half-widths applied per horizon, four-way coverage decomposition |

Writes `quick_result.json`, `quick_coverage_by_horizon.csv`,
`quick_golden_replay_cells.csv` into `--output` and nowhere else.

## 3. Full reproduction

```
python -m battery_entry reproduce --mode full --output ./out_full
```

Re-fits **only** the frozen final model `target_only_arc_space`, under the frozen protocol:

* the same leave-one-family-out folds (6 outer folds, one held-out family each);
* the same frozen α per (fold, horizon), read from `models/frozen_alpha.json`;
* the same 11 features in the same frozen ARC clip/scaler space;
* the same trajectory-equal then family-balanced weights;
* the same three horizons (28 / 56 / 112 days).

Regenerates `full_outer_fold_results.csv`, `full_point_predictions.csv`,
`full_interval_results.csv`, `full_matched_horizon_summary.csv`, `full_result.json`.

**What `full` deliberately does not do**, and why each is closed rather than skipped:

| not run | reason |
|---|---|
| Source-Prior arm | Phase 3 `SOURCE_PRIOR_NOT_VALIDATED` |
| stress arms F2 / F3 | Phase 4 `STRESS_NOT_VALIDATED` (both budgets) |
| exposure target T2 | Phase 4 `EXPOSURE_INPUT_NOT_AVAILABLE` (NULL on 100 % of rows) |
| time-rate target T1 | Phase 4 `RATE_DAY_REPARAMETERIZATION_ONLY` — T1 *is* T0, algebraically |
| PatchTST / TimesFM | Phase 6 `PHASE6_CLOSED_WITHOUT_TRAINING` |
| any new α grid | frozen by Phase 4 inner-fold selection; the grid was locked before any fit |
| conformal re-calibration | refused — re-calibrating after seeing coverage |

`full` compares its own per-fold `family_macro_mae` against
`results/reference/outer_fold_results_reference.csv` and reports the maximum deviation.

## 4. Numerical acceptance criteria

| comparison | tolerance |
|---|---|
| release point predictions vs frozen Phase 4/5 predictions | `rtol = 0`, `atol ≤ 1e-12` |
| feature recomputation vs shipped frozen columns | `atol ≤ 1e-12` |
| metric tables | `atol ≤ 1e-9` |
| torch vs numpy ridge coefficients | `< 1e-10` |

`rtol = 0` is deliberate: a relative tolerance on a quantity near 0.003 would hide an
absolute error that matters at the SOH scale.

**A measured note on the frozen artifacts.** The frozen `.npz` prediction arrays are the
bit-exact authority. The frozen `interval_predictions.csv` agrees with them to
**9.996e-17**, which is a CSV float round-trip and nothing more; the package joins against
the `.npz`-derived values, so the golden test is genuinely bit-level.

## 5. Two solve paths

The default is NumPy. `--solver torch` solves the **same normal equations** on torch's
float64 CPU path, and exists because the competition may require a PyTorch code path.

It is the `λ_source = 0` limit of the already-verified Phase 3 Source-Prior solver — not a
second algorithm, and not a re-implementation of a different model.
`tests/test_release.py::test_06` proves the coefficients and the predictions agree to
`< 1e-10`, and skips cleanly when torch is absent.

`torch.linalg.solve` is a closed-form linear solve. No neural layer, autograd pass or
gradient optimiser exists anywhere in this package, and `test_13` greps for all three.

## 6. Tests

```
python tests/test_release.py                 # direct execution
python -m pytest tests/test_release.py -q    # pytest, same assertions
```

Fifteen checks. Both invocation styles run from the same assertions, so the two reporting
conventions cannot silently diverge.

| # | check |
|---|---|
| 01 | runs in a clean directory with no parent project present |
| 02 | no active reference to the parent project, its artifacts or a host path |
| 03 | the 11 feature names / order / clip / scaler agree with the frozen manifest |
| 04 | release predictions == frozen predictions, `rtol=0, atol≤1e-12` |
| 05 | metric tables reproduce within the declared tolerance |
| 06 | the torch ridge path agrees with the numpy one |
| 07 | hidden truth / realized exposure / family / split cannot enter the matrix |
| 08 | conformal is calibrated per horizon at H = 2, 4, 8 separately |
| 09 | the pooled control is never described as group-aware |
| 10 | interval outputs carry `CONFORMAL_NOT_VALIDATED` |
| 11 | RUL never uses `-1` and never extrapolates past day 112 |
| 12 | RUL MAE on different eligible sets does not enter a ranking |
| 13 | PatchTST / TimesFM are neither implemented nor run |
| 14 | quick/full write only into the caller's output directory |
| 15 | shipped baseline files are unchanged by a run (hash before == after) |

Note on the environment: `KMP_DUPLICATE_LIB_OK` is **not** set anywhere, and must not be.
Intel documents that it can silently produce wrong results. The parent project splits
certain test processes for a measured OpenMP reason; this package does not import that
runtime, so no split is needed here.

## 7. Isolation

The package resolves every path from its own root
(`battery_entry/paths.py::PACKAGE_ROOT`). There is no reference anywhere in the code to the
research project, `parents[N]` for N ≥ 1, an absolute host path, or an STK installation
directory — and `verify`'s no-parent-dependency group greps every shipped file to prove it
rather than asserting it. Documentation files that *describe* where the frozen inputs came
from are exempt by name; every `.py` file is checked with no exemptions.

Writes are confined to the caller's `--output` directory by
`paths.guard_output_write`, which additionally refuses any target inside the package root —
so a run cannot overwrite the shipped reference results even if `--output` were pointed at
the package by mistake.

## 8. Docker

```
docker build -t battery-soh-v2 .
docker run --rm battery-soh-v2 python -m battery_entry verify
docker run --rm -v "$PWD/out:/out" battery-soh-v2 \
    python -m battery_entry reproduce --mode quick --output /out
```

Minimal CPU image on `python:3.9.25-slim-bookworm`. Contains no STK, no parent project, no
CUDA wheels, and needs no network at run time. Runs as a non-root user (UID 10001) so files
written into a mounted volume are not root-owned on the host. The build runs
`verify` as its final step, so a broken image cannot be produced silently.

**Docker was not available in the environment where this package was built** — no Docker
CLI, no Docker Desktop, and no docker/podman inside WSL. The specification above is
therefore recorded as `DOCKER_SPEC_READY_NOT_EXECUTED`: it has **not** been executed, and no
build or run result is claimed for it. Anyone with a Docker host should run the three
commands above; if the build or run fails, the fault is in this package and should be fixed
here, never by altering a frozen upstream result.

## 9. Determinism

No RNG is consulted by `verify`, `reproduce` or `predict`. The budget subsets upstream are
constructed by lexicographic prefix rather than sampling, and the bootstrap in the frozen
tables used seed 42 with 2000 trajectory-level replicates. float64 throughout, CPU only, so
results do not depend on thread count or device.

## 10. What reproduction establishes, and what it does not

Reproducing these numbers establishes that the pipeline is faithful to the frozen
configuration and that the reported simulation metrics are real.

It establishes **nothing** about on-orbit performance. Every number is measured on an
STK-derived simulated twin, and `CONFORMAL_NOT_VALIDATED`, `RUL_EVIDENCE_INSUFFICIENT`,
`CANDIDATE_NOT_PROMOTED` and `SIMULATION_ONLY` all still hold. See `LIMITATIONS.md`.
