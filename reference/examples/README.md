# Examples

## `example_input.csv`

Two contrasting batteries, each with exactly **20** observed-SOH reference points on the
14-day grid — the context length the frozen ARC route requires.

| battery_id | anchor SOH | why it is here |
|---|---|---|
| `BATT_NEAR_EOL` | 0.714894 | near end of life; its forecast path crosses 0.70 inside 112 days |
| `BATT_MID_LIFE` | 0.976992 | mid life; its path does **not** reach 0.70, so the RUL read-out returns a STATUS rather than a number |

Both histories are real rows from the shipped reference slice, so the example is not
synthetic.

Required columns: `battery_id`, `reference_index`, `soh_observed`.
Optional: `timestamp_utc` (carried through to the output only).

## `example_expected_output.csv`

Produced by actually running the CLI, not hand-written:

```
python -m battery_entry predict \
    --input examples/example_input.csv \
    --output examples/example_expected_output.csv \
    --intervals --rul
```

Observed exit code: **3**.

Exit code 3 is the correct answer here, not a failure: 1 of the 2
batteries has a forecast path that never reaches the 0.70 EOL threshold inside 112 days, so
the read-out returns `NO_CROSSING_WITHIN_FORECAST_HORIZON`. **Nothing is extrapolated past
day 112.**

| exit code | meaning |
|---|---|
| 0 | every battery predicted, and every requested RUL crossed inside 112 days |
| 2 | the input was refused by the schema validator |
| 3 | predicted successfully, but at least one path did not reach EOL within 112 days |

### What the output columns mean

| column group | note |
|---|---|
| `predicted_soh_day{28,56,112}` | the point prediction — the **default** output |
| `baseline_predicted_soh_day*` | the non-learning `observable_local_trend_extrapolation` baseline, always reported alongside |
| `interval_lo/hi_day*` | only with `--intervals`; labelled `EXPERIMENTAL_SIMULATION_INTERVAL`, carrying `CONFORMAL_NOT_VALIDATED` |
| `rul_days`, `rul_status` | only with `--rul`; labelled `FINITE_HORIZON_EVIDENCE_INSUFFICIENT` |
| `warning_is_forecast_not_restatement` | `False` when the anchor is already at or below 0.80 — then "the warning fires" restates the present observation rather than forecasting |

### Intervals are per-horizon MARGINAL

The three horizons are calibrated **separately**. Three intervals that each cover with
probability 0.90 have *joint* coverage no greater than 0.90 and possibly far less. They are
**not** a 112-day band. See `../LIMITATIONS.md`.

### Every number here is simulation

The deployment head was fitted on the STK-derived simulated twin. These are **not**
real-satellite predictions and no on-orbit guarantee follows from them.

## Reproducing the point-only output

```
python -m battery_entry predict \
    --input examples/example_input.csv --output /tmp/point.csv
```

That path emits no interval and no RUL, which is the intended default: the interval and the
RUL each carry a caveat that a caller must opt into seeing.
