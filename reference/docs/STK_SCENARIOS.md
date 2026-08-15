# STK scenarios and the export chain

Generated from the frozen audit manifests. No scenario count, orbit parameter or
hash below is hand-entered.

> **A downstream reproduction needs no STK licence and no STK installation.** This
> package depends only on data ALREADY EXPORTED from STK. The Docker image contains
> no STK component.

---

## 1. What exists, and what does not

Stated first, because the honest boundary matters more than the inventory.

| item | present in this project? | consequence |
|---|---|---|
| exported per-scenario CSVs (`stk_external_data/`) | **yes** — 24 files | the data chain is reproducible from the exports |
| per-scenario report directories (`data/stk_report/`) | **yes** — 24 directories | beta angle, lighting, panel power/area, access |
| scenario audit manifest with all 24 scenarios | **yes** | orbit parameters, quality flags, physics checks |
| SHA256 manifest of the raw exports | **yes** | integrity of the chain is checkable |
| STK `.sc` scenario definition files | **NO** — 0 found | the scenarios cannot be re-run inside STK from this project |
| STK `.vdf` bundles | **NO** — 0 found | same |
| the digital-twin degradation generator | **NO** | degradation cannot be re-integrated, so **no strict counterfactual is producible** |

The last two absences are load bearing and are **not** worked around:

* Because the `.sc` files are absent, this package does **not** claim the STK
  scenarios can be regenerated. It claims only that the already-exported data
  reproduces the reported results, which is what `reproduce` demonstrates.
* Because the generator is absent, every factor-direction claim is capped at
  `OBSERVATIONAL_EVIDENCE` and labelled `NON_CAUSAL`. `data/afterstk` is *output*;
  nothing in the codebase consumes the physics parameters the trajectory manifest
  records. Fabricating a contrast by editing a stress column is refused upstream.

---

## 2. Scenario design

| property | value |
|---|---|
| exact STK scenarios | 24 |
| environment families | 6 |
| trajectories per family | 20 |
| reference step | 14 days |
| mission span | 2190 days |
| EOL threshold | 0.7 SOH |
| warning threshold | 0.8 SOH |

The 24 scenarios are 2 orbit shells x 3 RAAN values x 4 seasonal epochs, each
simulated for 72 hours at a 30-second step. The 72-hour scenario supplies the
environment; multi-year trajectories are derived from it on the 14-day grid.

### The 24 scenarios

| scenario_id | alt (km) | incl (deg) | RAAN (deg) | epoch | family | eclipse | mean DoD | EFC/72h |
|---|---|---|---|---|---|---|---|---|
| `T001_h550_i053_raan000_20260115_d72h_s30` | 550 | 53 | 0 | 20260115 | `h550_i053_raan000` | 0.3567 | 0.2572 | 12.153 |
| `T002_h550_i053_raan090_20260115_d72h_s30` | 550 | 53 | 90 | 20260115 | `h550_i053_raan090` | 0.3604 | 0.2628 | 12.200 |
| `T003_h550_i053_raan180_20260115_d72h_s30` | 550 | 53 | 180 | 20260115 | `h550_i053_raan180` | 0.2489 | 0.1523 | 7.187 |
| `T004_h550_i053_raan000_20260415_d72h_s30` | 550 | 53 | 0 | 20260415 | `h550_i053_raan000` | 0.3612 | 0.2612 | 12.216 |
| `T005_h550_i053_raan090_20260415_d72h_s30` | 550 | 53 | 90 | 20260415 | `h550_i053_raan090` | 0.2797 | 0.1768 | 8.369 |
| `T006_h550_i053_raan180_20260415_d72h_s30` | 550 | 53 | 180 | 20260415 | `h550_i053_raan180` | 0.2039 | 0.1097 | 5.183 |
| `T007_h550_i053_raan000_20260701_d72h_s30` | 550 | 53 | 0 | 20260701 | `h550_i053_raan000` | 0.3519 | 0.2532 | 11.790 |
| `T008_h550_i053_raan090_20260715_d72h_s30` | 550 | 53 | 90 | 20260715 | `h550_i053_raan090` | 0.3476 | 0.2485 | 11.615 |
| `T009_h550_i053_raan180_20260715_d72h_s30` | 550 | 53 | 180 | 20260715 | `h550_i053_raan180` | 0.3684 | 0.2636 | 12.324 |
| `T010_h550_i053_raan000_20261015_d72h_s30` | 550 | 53 | 0 | 20261015 | `h550_i053_raan000` | 0.3453 | 0.2516 | 11.729 |
| `T011_h550_i053_raan090_20261015_d72h_s30` | 550 | 53 | 90 | 20261015 | `h550_i053_raan090` | 0.3570 | 0.2578 | 12.156 |
| `T012_h550_i053_raan180_20261015_d72h_s30` | 550 | 53 | 180 | 20261015 | `h550_i053_raan180` | 0.3655 | 0.2648 | 12.451 |
| `T013_h500_i070_raan000_20260115_d72h_s30` | 500 | 70 | 0 | 20260115 | `h500_i070_raan000` | 0.3414 | 0.2447 | 11.462 |
| `T014_h500_i070_raan090_20260115_d72h_s30` | 500 | 70 | 90 | 20260115 | `h500_i070_raan090` | 0.3669 | 0.2653 | 12.417 |
| `T015_h500_i070_raan180_20260115_d72h_s30` | 500 | 70 | 180 | 20260115 | `h500_i070_raan180` | 0.2222 | 0.1321 | 6.173 |
| `T016_h500_i070_raan000_20260415_d72h_s30` | 500 | 70 | 0 | 20260415 | `h500_i070_raan000` | 0.3636 | 0.2611 | 12.255 |
| `T017_h500_i070_raan090_20260415_d72h_s30` | 500 | 70 | 90 | 20260415 | `h500_i070_raan090` | 0.2117 | 0.1127 | 5.288 |
| `T018_h500_i070_raan180_20260415_d72h_s30` | 500 | 70 | 180 | 20260415 | `h500_i070_raan180` | 0.3517 | 0.2536 | 11.899 |
| `T019_h500_i070_raan000_20260715_d72h_s30` | 500 | 70 | 0 | 20260715 | `h500_i070_raan000` | 0.3318 | 0.2351 | 10.931 |
| `T020_h500_i070_raan090_20260715_d72h_s30` | 500 | 70 | 90 | 20260715 | `h500_i070_raan090` | 0.3676 | 0.2639 | 12.363 |
| `T021_h500_i070_raan180_20260715_d72h_s30` | 500 | 70 | 180 | 20260715 | `h500_i070_raan180` | 0.1910 | 0.1010 | 4.749 |
| `T022_h500_i070_raan000_20261015_d72h_s30` | 500 | 70 | 0 | 20261015 | `h500_i070_raan000` | 0.3658 | 0.2640 | 12.383 |
| `T023_h500_i070_raan090_20261015_d72h_s30` | 500 | 70 | 90 | 20261015 | `h500_i070_raan090` | 0.1699 | 0.0924 | 4.235 |
| `T024_h500_i070_raan180_20261015_d72h_s30` | 500 | 70 | 180 | 20261015 | `h500_i070_raan180` | 0.3593 | 0.2579 | 12.092 |

### Scenario -> orbit family -> split

`environment_family_id` is the **strict target-domain isolation key**. All four
seasonal templates of a family, and every trajectory derived from them, land in
exactly one split.

| family | altitude | inclination | RAAN | scenarios | split |
|---|---|---|---|---|---|
| `h500_i070_raan000` | 500 km | 70 deg | 0 deg | `T013`, `T016`, `T019`, `T022` | `target_test` |
| `h500_i070_raan090` | 500 km | 70 deg | 90 deg | `T014`, `T017`, `T020`, `T023` | `target_train` |
| `h500_i070_raan180` | 500 km | 70 deg | 180 deg | `T015`, `T018`, `T021`, `T024` | `target_calibration` |
| `h550_i053_raan000` | 550 km | 53 deg | 0 deg | `T001`, `T004`, `T007`, `T010` | `target_train` |
| `h550_i053_raan090` | 550 km | 53 deg | 90 deg | `T002`, `T005`, `T008`, `T011` | `target_calibration` |
| `h550_i053_raan180` | 550 km | 53 deg | 180 deg | `T003`, `T006`, `T009`, `T012` | `target_test` |

> **The orbit triplet (altitude, inclination, RAAN) maps one-to-one onto the 6
> families.** Each is therefore a family-identity proxy and is permanently excluded
> from every model input — by measurement, not by taste. Any near-constant
> re-encoding is equally barred.

---

## 3. Exported files per scenario

Each scenario directory under `data/stk_report/<scenario_id>/` carries the STK
report exports; the aligned per-scenario battery input is under
`data/afterstk/stk_external_data/<scenario_id>_Battery_ExternalData.csv`.

| export | file | what it supplies |
|---|---|---|
| solar panel power | `LEO_BatterySat_Solar_Panel_Power.csv` | charge availability |
| solar panel area | `LEO_BatterySat_Solar_Panel_Area.csv` | illuminated area |
| beta angle | `LEO_BatterySat_Beta_Angle.csv` | sun-orbit geometry |
| lighting times | `LEO_BatterySat_Lighting_Times.csv` | eclipse entry/exit |
| access intervals | `access_intervals_*.csv` / `Satellite-...-Access.csv` | ground-station contact, driving comm load |
| battery environment | `LEO_BatterySat_Battery_Environment_TimeSeries.csv` | position/temperature series |

### Export quality flags, recorded as measured

* **Duplicate export rows removed:** 207,384 from the power series and the same
  count from the area series. STK's report export duplicated rows; de-duplication is
  recorded per scenario in `power_duplicate_rows_removed` / `area_duplicate_rows_removed`.
* **Environment position file missing for 1 scenario(s):** `T001_h550_i053_raan000_20260115_d72h_s30`. Handled as
  `MISSING_ENV_POSITION_FILLED_NOMINAL` with `position_coverage_ratio = 0.0`, and
  flagged rather than silently imputed.
* **Concatenated environment reports rebuilt for 10 scenario(s).**
* Per-scenario flags live in `scenario_quality_flags`; every one is retained in the
  audit manifest rather than cleaned away.

---

## 4. Raw export integrity

`data/afterstk/sha256_manifest.csv` carries 44 entries. The four layer
files and their hashes:

| file | size (bytes) | sha256 |
|---|---|---|
| `L0_multi_scenario_stk_environment.csv` | 80,769,719 | `954427072702f5a77f06e6e01d76b7e3a6a5d192367968f5cbc065b79929bff7` |
| `L1_multi_scenario_battery_telemetry_72h.csv` | 103,929,919 | `053b6cc06e6f5e48a542458201a032c3e6d5159f3d7077a08c4cd710cfe8a975` |
| `L2_multi_scenario_reference_points.csv` | 7,347,988 | `c19c3ff8144b19164bd4da05093fd3e890478e9435a0b302541149fea9343268` |
| `L3_multi_scenario_training_windows.csv` | 42,664,107 | `d4db52049afb25539da6a201d2113bc629d7a30af7a1b8b4e5a5a088e2be5678` |

Manifest sha256: `e2676b5dce103cfc8deac3ef7c7825b68ffaec538a6404b7dd0319ae36c3a8f1`
Scenario audit sha256: `f33aa0c9c6d46d2cd0065d5c3a437ab5a6d26cbe8b8f5d7b5d8c72aff601f71d`

---

## 5. Automation / export procedure

The scenarios were produced in STK and exported as the report CSVs listed in section 3.
The export step itself is **not** scripted inside this project: no `.sc` file, no
STK Object Model / Connect automation script, and no `stk` Python binding appears
anywhere in the tree. That is recorded here as a gap rather than described as an
existing capability.

To regenerate the scenarios, a practitioner would need to rebuild them in STK from
the orbit parameters in section 2 (altitude, inclination, RAAN, epoch, 72 h span,
30 s step) and re-export the six report types. **The results in this package do not
depend on that being done** — they are reproduced from the exported data.

