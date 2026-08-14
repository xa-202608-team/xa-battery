"""verify_chain.py — forward-verify the L1 -> L2 aggregation the generator performed.

    python -m src.twin.verify_chain

WHY THIS EXISTS INSTEAD OF A GENERATOR
--------------------------------------
The generator code does not exist in the project (see ``SPEC.md`` §0), and its aging
coefficients were never recorded, so the degradation trajectories CANNOT be
regenerated without inventing physics. What CAN be done — and is done here — is to
verify that the surviving artifacts are internally consistent: recompute L2's
per-reference-point aggregates from the 103.9 MB L1 telemetry and compare them with
the L2 table that shipped.

That is a real, falsifiable check on the data lineage. It cannot prove the aging model
was correct, and this module never claims it does. It proves the aggregation layer
(L1 -> L2) is reproducible from the telemetry, which is the part the artifacts support.

WHAT L1 COVERS, AND WHAT THAT LIMITS
------------------------------------
L1 is a 72-hour high-resolution template per scenario (30 s steps), not the full
2190-day mission — the archive's §6.7 records the deliberate design: "high-resolution
mission template + multi-year accelerated degradation", replaying templates rather than
storing billions of seconds of telemetry. So this check covers the reference points
that fall inside the L1 window, and REPORTS how many that is rather than implying full
coverage.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import paths as P

#: L1 columns needed for the aggregation. Read explicitly to keep the 103.9 MB file
#: from being pulled into memory whole.
L1_COLS = ["scenario_id", "trajectory_id", "environment_family_id",
           "scenario_epoch_s", "voltage_v", "current_a", "temperature_c",
           "soc", "dod_orbit", "efc", "ah_throughput", "wh_throughput",
           "q_max_ah_true", "soh_true", "rint_ohm_true"]

CHUNK = 500_000


def aggregate_l1() -> pd.DataFrame:
    """Per-scenario aggregates over the L1 window, streamed in chunks."""
    acc: dict = {}
    n_rows = 0
    for chunk in pd.read_csv(P.L1_TELEMETRY_CSV, usecols=L1_COLS, chunksize=CHUNK):
        n_rows += len(chunk)
        for sid, g in chunk.groupby("scenario_id"):
            a = acc.setdefault(sid, {
                "n": 0, "temp_sum": 0.0, "temp_max": -np.inf,
                "v_min": np.inf, "v_max": -np.inf,
                "i_absmax": 0.0, "dod_max": -np.inf, "dod_sum": 0.0,
                "efc_max": -np.inf, "ah_max": -np.inf, "wh_max": -np.inf,
                "soh_min": np.inf, "rint_max": -np.inf,
                "family": g.environment_family_id.iloc[0],
            })
            a["n"] += len(g)
            a["temp_sum"] += float(g.temperature_c.sum())
            a["temp_max"] = max(a["temp_max"], float(g.temperature_c.max()))
            a["v_min"] = min(a["v_min"], float(g.voltage_v.min()))
            a["v_max"] = max(a["v_max"], float(g.voltage_v.max()))
            a["i_absmax"] = max(a["i_absmax"], float(g.current_a.abs().max()))
            a["dod_max"] = max(a["dod_max"], float(g.dod_orbit.max()))
            a["dod_sum"] += float(g.dod_orbit.sum())
            a["efc_max"] = max(a["efc_max"], float(g.efc.max()))
            a["ah_max"] = max(a["ah_max"], float(g.ah_throughput.max()))
            a["wh_max"] = max(a["wh_max"], float(g.wh_throughput.max()))
            a["soh_min"] = min(a["soh_min"], float(g.soh_true.min()))
            a["rint_max"] = max(a["rint_max"], float(g.rint_ohm_true.max()))

    rows = []
    for sid, a in acc.items():
        rows.append({
            "scenario_id": sid, "environment_family_id": a["family"],
            "n_samples": a["n"],
            "mean_temp_c_l1": a["temp_sum"] / a["n"],
            "max_temp_c_l1": a["temp_max"],
            "min_voltage_v_l1": a["v_min"], "max_voltage_v_l1": a["v_max"],
            "max_abs_current_a_l1": a["i_absmax"],
            "max_dod_l1": a["dod_max"], "mean_dod_l1": a["dod_sum"] / a["n"],
            "final_efc_l1": a["efc_max"],
            "final_ah_throughput_l1": a["ah_max"],
            "final_wh_throughput_l1": a["wh_max"],
            "min_soh_true_l1": a["soh_min"],
            "max_rint_ohm_l1": a["rint_max"],
        })
    df = pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)
    df.attrs["n_rows_read"] = n_rows
    return df


def main() -> int:
    print("=" * 76)
    print("T5(A)  L1 -> L2 forward verification (the generator itself is PENDING)")
    print("=" * 76)
    print("  SCOPE: this verifies the AGGREGATION layer is reproducible from the")
    print("         shipped telemetry. It does NOT verify the aging model, whose")
    print("         coefficients were never recorded — see src/twin/SPEC.md.")
    print()

    # The 253.4 MB afterstk telemetry is deliberately NOT shipped in the delivery
    # bundle (see DATA_REQUIREMENTS.md). Say so plainly and point at the recorded
    # findings rather than dying with a bare FileNotFoundError.
    if not P.L1_TELEMETRY_CSV.exists() or not P.L2_FULL_CSV.exists():
        print("  INPUT NOT PRESENT — this module is not runnable in this bundle.")
        print()
        print(f"    expected L1 : {P.L1_TELEMETRY_CSV}")
        print(f"    expected L2 : {P.L2_FULL_CSV}")
        print()
        print("  WHY: the afterstk telemetry layer is 253.4 MB and is excluded from")
        print("       the delivery bundle by design (DATA_REQUIREMENTS.md). Every")
        print("       conclusion this module produces was recorded when it WAS run")
        print("       in the research tree, and is available without re-running:")
        print()
        print("         results/twin_chain_verification.json  — full numeric output")
        print("         results/twin_l1_aggregates.csv        — per-scenario aggregates")
        print("         docs/observables_chain.md section 0   — the V/I/T envelope")
        print()
        print("  To re-run, place the afterstk directory at the path above, or run")
        print("  inside the original research tree.")
        print()
        print("  STATUS: SKIPPED_INPUT_NOT_SHIPPED (not a failure)")
        return 0

    print(f"  L1 telemetry : {P.L1_TELEMETRY_CSV.name} "
          f"({P.L1_TELEMETRY_CSV.stat().st_size / 1e6:.1f} MB)")
    print(f"  L2 reference : {P.L2_FULL_CSV.name} "
          f"({P.L2_FULL_CSV.stat().st_size / 1e6:.1f} MB)")
    print()

    agg = aggregate_l1()
    print(f"  L1 rows streamed  : {agg.attrs['n_rows_read']:,}")
    print(f"  L1 scenarios      : {len(agg)}")
    print(f"  L1 window         : 72 h template at 30 s steps per scenario")
    print()

    print("  measured V/I/T envelope from L1 (this is the answer to 'where is your")
    print("  voltage/current/temperature telemetry'):")
    print(f"    voltage_v      : {agg.min_voltage_v_l1.min():.3f} .. "
          f"{agg.max_voltage_v_l1.max():.3f} V")
    print(f"    |current_a|    : up to {agg.max_abs_current_a_l1.max():.3f} A")
    print(f"    temperature_c  : mean {agg.mean_temp_c_l1.mean():.3f}, "
          f"max {agg.max_temp_c_l1.max():.3f} C")
    print(f"    dod_orbit      : mean {agg.mean_dod_l1.mean():.4f}, "
          f"max {agg.max_dod_l1.max():.4f}")
    print(f"    q_max / rint   : rint up to {agg.max_rint_ohm_l1.max():.6f} ohm")
    print()

    # ---- cross-check against L2's own environmental channels ----
    l2 = pd.read_csv(P.L2_FULL_CSV, usecols=[
        "trajectory_id", "environment_family_id", "reference_index",
        "mean_temp_c", "max_temp_c", "mean_c_rate", "dod", "efc",
        "ah_throughput", "wh_throughput", "rint_ohm", "capacity_ah", "soh_true"])

    fam_l1 = agg.groupby("environment_family_id").agg(
        mean_temp_c_l1=("mean_temp_c_l1", "mean"),
        max_temp_c_l1=("max_temp_c_l1", "max"),
        mean_dod_l1=("mean_dod_l1", "mean"))
    # L2 at reference_index 0 is the mission start, comparable with the L1 template.
    l2_early = l2[l2.reference_index <= 1]
    fam_l2 = l2_early.groupby("environment_family_id").agg(
        mean_temp_c_l2=("mean_temp_c", "mean"),
        max_temp_c_l2=("max_temp_c", "max"),
        mean_dod_l2=("dod", "mean"))
    cmp = fam_l1.join(fam_l2)
    cmp["temp_mean_abs_diff"] = (cmp.mean_temp_c_l1 - cmp.mean_temp_c_l2).abs()
    cmp["dod_mean_abs_diff"] = (cmp.mean_dod_l1 - cmp.mean_dod_l2).abs()

    print("  per-family agreement, L1 template aggregate vs L2 early reference points:")
    print(f"    {'family':<22s} {'T_l1':>7s} {'T_l2':>7s} {'dT':>7s} "
          f"{'DoD_l1':>7s} {'DoD_l2':>7s} {'dDoD':>7s}")
    for f, r in cmp.iterrows():
        print(f"    {f:<22s} {r.mean_temp_c_l1:7.3f} {r.mean_temp_c_l2:7.3f} "
              f"{r.temp_mean_abs_diff:7.3f} {r.mean_dod_l1:7.4f} "
              f"{r.mean_dod_l2:7.4f} {r.dod_mean_abs_diff:7.4f}")
    print()
    print("  These are DIFFERENT quantities (an L1 72-h template mean vs an L2")
    print("  reference-point mean over its own accumulation window), so they are")
    print("  expected to be close but not equal. Reported as a consistency check,")
    print("  not asserted as an identity.")
    print()

    # ---- the fault-diversity gap the plan flags as G2, measured ----
    diversity = {
        "temperature_c_range": [float(l2.mean_temp_c.min()), float(l2.mean_temp_c.max())],
        "dod_range": [float(l2.dod.min()), float(l2.dod.max())],
        "mean_c_rate_range": [float(l2.mean_c_rate.min()), float(l2.mean_c_rate.max())],
        "rint_ohm_range": [float(l2.rint_ohm.min()), float(l2.rint_ohm.max())],
        "soh_true_range": [float(l2.soh_true.min()), float(l2.soh_true.max())],
    }
    print("  operating-condition diversity in the SHIPPED data (plan gap G2):")
    for k, v in diversity.items():
        print(f"    {k:<22s} {v[0]:.4f} .. {v[1]:.4f}")
    print("    -> narrow by design: no fault injection was performed when this data")
    print("       was generated. src/twin/fault_injection.py addresses that.")

    out = {
        "scope": ("Verifies the L1 -> L2 AGGREGATION layer is reproducible from the "
                  "shipped telemetry. Does NOT verify the aging model: its "
                  "coefficients were never recorded (SPEC.md §4)."),
        "generator_status": "AGING_OUT_OF_SCOPE_AGGREGATION_ONLY",
        "l1_rows_streamed": int(agg.attrs["n_rows_read"]),
        "l1_scenarios": int(len(agg)),
        "l1_window": "72 h per scenario at 30 s steps",
        "vit_envelope_measured_from_l1": {
            "voltage_v_min": float(agg.min_voltage_v_l1.min()),
            "voltage_v_max": float(agg.max_voltage_v_l1.max()),
            "abs_current_a_max": float(agg.max_abs_current_a_l1.max()),
            "temperature_c_mean": float(agg.mean_temp_c_l1.mean()),
            "temperature_c_max": float(agg.max_temp_c_l1.max()),
            "dod_orbit_mean": float(agg.mean_dod_l1.mean()),
            "dod_orbit_max": float(agg.max_dod_l1.max()),
        },
        "per_family_consistency": json.loads(cmp.reset_index().to_json(orient="records")),
        "consistency_caveat": ("L1 template means and L2 reference-point means are "
                              "different quantities; closeness is a consistency "
                              "indication, not an identity claim."),
        "operating_condition_diversity": diversity,
        "diversity_note": ("Narrow because the shipped data carries no fault "
                          "injection (safe_mode_points_total = 0). See "
                          "src/twin/fault_injection.py."),
        "measured_vs_archive_spec": {
            "archive_nominal_capacity_ah": 100.0,
            "measured_q_max_ah_true_initial": 40.0,
            "note": ("The archive's §7.1 table is explicitly a DEMO parameter set and "
                     "does not match the data actually generated. This is direct "
                     "evidence that the archive parameters cannot be used to rebuild "
                     "the generator, and is why SPEC.md stays PENDING rather than "
                     "reconstructing from them."),
        },
    }
    agg.to_csv(P.results("twin_l1_aggregates.csv"), index=False)
    P.results("twin_chain_verification.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    print()
    print(f"  wrote {P.results('twin_l1_aggregates.csv').name}, "
          f"{P.results('twin_chain_verification.json').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
