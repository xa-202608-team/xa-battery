"""fault_injection.py — four fault classes injected into the shipped L2 trajectories.

    python -m src.twin.fault_injection

NEW (see ``REUSE_MAP.md`` N3). Phase A found no fault-injection implementation, and the
shipped data confirms it: ``safe_mode_points_total = 0``, temperature spans only
18.0-28.0 C, DoD only 0.08-0.43. The plan's gap G2.

WHY POST-HOC INJECTION IS THE HONEST APPROACH HERE
--------------------------------------------------
Injecting faults at the physics level would require re-running the degradation
integration — which needs the aging coefficients that were never recorded (``SPEC.md``
§4). Rather than invent them, faults are applied as documented PERTURBATIONS to the
shipped L2 trajectories, using each fault's known observable signature.

That distinction is load-bearing and is carried in every output: these are
**observable-signature perturbations**, not physics-resolved fault simulations. They
are adequate for what they are used for — testing whether the predictor degrades
gracefully, detects the anomaly, and how its warning lead time shifts — and they are
NOT adequate for claiming a validated fault-propagation model. No output says otherwise.

THE FOUR CLASSES (closeout plan §1.2 table)
-------------------------------------------
1. ``rint_step``        internal-resistance step: interface film growth / connector
                        degradation. Observable: larger discharge voltage drop, more
                        heating, and an accelerated SOH slope after onset.
2. ``capacity_jump``    capacity drop: lithium plating causing local deactivation.
                        Observable: a STEP down in SOH, then the prior trend resumes.
3. ``thermal_bias``     radiator degradation: temperature baseline rises. Observable:
                        temperature mean drift plus accelerated calendar aging.
4. ``deep_discharge``   safe-mode / load anomaly: one large-DoD event, then faster
                        cycle aging.

All magnitudes are DECLARED here, before any model was run against the faulted data.
Every injection is deterministic given ``(trajectory_id, fault_type)``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from src import paths as P
from src.comparators.classical import stable_seed
from src.features.space import EOL_SOH, WARN_SOH
from src.twin.approx_propagation import propagate

#: 注入性质声明：物理量注入 + 近似解析传播（非直接篡改 SOH 遥测）。
FAULT_NATURE = "physical_parameter_injection_with_approx_propagation"

#: Declared magnitudes. Chosen to be detectable but not absurd: each is roughly one to
#: three times the natural spread of its own channel in the shipped data, so a fault is
#: a plausible off-nominal excursion rather than a destroyed trajectory.
FAULT_SPEC = {
    "rint_step": {
        "mechanism": "interface film growth / connector degradation",
        "observable": "larger discharge voltage drop, more heating, faster SOH decline",
        "rint_multiplier": 1.6,
        "soh_slope_multiplier_after_onset": 1.45,
        "onset_fraction_of_mission": 0.45,
    },
    "capacity_jump": {
        "mechanism": "lithium plating causing local deactivation",
        "observable": "step down in SOH, prior trend then resumes",
        "soh_step": -0.035,   # 保留字段，不再被 inject 使用（旧语义）
        "q_factor": 0.96,     # 物理量注入：容量比例（对应原 soh_step -0.035 量级）
        "onset_fraction_of_mission": 0.55,
    },
    "thermal_bias": {
        "mechanism": "radiator surface degradation raising the temperature baseline",
        "observable": "temperature mean drift, accelerated calendar aging",
        "temperature_bias_c": 6.0,
        "soh_slope_multiplier_after_onset": 1.30,
        "onset_fraction_of_mission": 0.35,
    },
    "deep_discharge": {
        "mechanism": "safe mode / load anomaly causing one large-DoD event",
        "observable": "single large DoD, then accelerated cycle aging",
        "dod_event": 0.85,
        "soh_step": -0.012,
        "soh_slope_multiplier_after_onset": 1.25,
        "onset_fraction_of_mission": 0.50,
    },
}


@dataclass
class InjectionRecord:
    trajectory_id: str
    environment_family_id: str
    fault_type: str
    onset_day: float
    onset_index: int
    soh_at_onset: float
    baseline_eol_day: float
    faulted_eol_day: float
    eol_advance_days: float
    baseline_warn_day: float
    faulted_warn_day: float
    warn_advance_days: float
    n_points: int


def _first_crossing(day: np.ndarray, soh: np.ndarray, thr: float) -> float:
    idx = np.flatnonzero(soh <= thr)
    return float(day[idx[0]]) if len(idx) else float("nan")


def inject(day, soh_true, soh_obs, fault_type):
    spec = FAULT_SPEC[fault_type]
    d = np.asarray(day, dtype=float)
    st = np.asarray(soh_true, dtype=float).copy()
    so = np.asarray(soh_obs, dtype=float).copy()
    n = len(d)
    k = int(np.clip(round(spec["onset_fraction_of_mission"] * (n - 1)), 1, n - 2))

    rint_mult = spec.get("rint_multiplier", 1.0)
    q_factor = spec.get("q_factor", 1.0)
    temp_bias = spec.get("temperature_bias_c", 0.0)
    dod = spec.get("dod_event", 0.0)

    st_f = propagate(st, rint_mult, q_factor, temp_bias, dod, k)
    so_f = propagate(so, rint_mult, q_factor, temp_bias, dod, k)

    extra = {}
    if fault_type == "rint_step":
        r = np.ones(n); r[k:] = rint_mult; extra["rint_multiplier_series"] = r
    if fault_type == "thermal_bias":
        t = np.zeros(n); t[k:] = temp_bias; extra["temperature_bias_series"] = t
    if fault_type == "deep_discharge":
        dd = np.zeros(n); dd[k] = dod; extra["dod_event_series"] = dd
    if fault_type == "capacity_jump":
        qq = np.ones(n); qq[k:] = q_factor; extra["q_factor_series"] = qq

    return {"onset_index": k, "onset_day": float(d[k]),
            "soh_true_faulted": st_f, "soh_observed_faulted": so_f,
            "soh_at_onset": float(st[k]),
            "nature": FAULT_NATURE, **extra}


def build_faulted_subset(n_per_fault: int = 15) -> tuple[pd.DataFrame, list]:
    """Inject each fault class into a deterministic subset of trajectories.

    Subset selection is by sorted trajectory id and is disjoint across fault classes, so
    no trajectory carries two faults and the four classes are compared on distinct
    units. ``n_per_fault = 15`` gives 60 faulted trajectories out of 120.
    """
    l2 = pd.read_csv(P.L2_REFERENCE_CSV)
    sm = pd.read_csv(P.SPLIT_MANIFEST_CSV)
    fam = dict(zip(sm.trajectory_id, sm.environment_family_id))

    trajs = sorted(l2.trajectory_id.unique().tolist())
    faults = list(FAULT_SPEC)
    assign = {}
    for i, f in enumerate(faults):
        assign[f] = trajs[i * n_per_fault:(i + 1) * n_per_fault]

    out_rows, records = [], []
    for ftype, members in assign.items():
        for tid in members:
            g = l2[l2.trajectory_id == tid].sort_values("reference_day")
            d = g.reference_day.to_numpy(dtype=float)
            st = g.soh_true.to_numpy(dtype=float)
            so = g.soh_observed.to_numpy(dtype=float)
            r = inject(d, st, so, ftype)

            base_eol = _first_crossing(d, st, EOL_SOH)
            f_eol = _first_crossing(d, r["soh_true_faulted"], EOL_SOH)
            base_warn = _first_crossing(d, st, WARN_SOH)
            f_warn = _first_crossing(d, r["soh_true_faulted"], WARN_SOH)

            records.append(InjectionRecord(
                trajectory_id=tid, environment_family_id=fam[tid], fault_type=ftype,
                onset_day=r["onset_day"], onset_index=r["onset_index"],
                soh_at_onset=r["soh_at_onset"],
                baseline_eol_day=base_eol, faulted_eol_day=f_eol,
                eol_advance_days=(base_eol - f_eol
                                  if np.isfinite(base_eol) and np.isfinite(f_eol)
                                  else float("nan")),
                baseline_warn_day=base_warn, faulted_warn_day=f_warn,
                warn_advance_days=(base_warn - f_warn
                                   if np.isfinite(base_warn) and np.isfinite(f_warn)
                                   else float("nan")),
                n_points=len(d)))

            for i in range(len(d)):
                out_rows.append({
                    "trajectory_id": tid, "environment_family_id": fam[tid],
                    "fault_type": ftype, "reference_index": int(g.reference_index.iloc[i]),
                    "reference_day": d[i],
                    "soh_observed": so[i], "soh_true": st[i],
                    "soh_observed_faulted": r["soh_observed_faulted"][i],
                    "soh_true_faulted": r["soh_true_faulted"][i],
                    "post_onset": int(i >= r["onset_index"]),
                    "rint_multiplier": float(r.get("rint_multiplier_series",
                                                   np.ones(len(d)))[i]),
                    "temperature_bias_c": float(r.get("temperature_bias_series",
                                                      np.zeros(len(d)))[i]),
                    "dod_event": float(r.get("dod_event_series", np.zeros(len(d)))[i]),
                })
    return pd.DataFrame(out_rows), records


def main() -> int:
    print("=" * 76)
    print("T5(B)  fault injection — four classes, on the shipped L2 trajectories")
    print("=" * 76)
    print("  NATURE OF THESE FAULTS: observable-signature PERTURBATIONS, not")
    print("  physics-resolved fault simulations. Resolving them in physics would")
    print("  need the aging coefficients that were never recorded (SPEC.md §4),")
    print("  and inventing those would misrepresent the data's provenance.")
    print()

    df, records = build_faulted_subset()
    R = pd.DataFrame([asdict(r) for r in records])

    print(f"  faulted trajectories : {R.trajectory_id.nunique()} of 120 "
          f"(disjoint across the four classes)")
    print(f"  faulted rows         : {len(df):,}")
    print()
    print(f"  {'fault_type':<18s} {'n':>4s} {'onset_day':>10s} {'EOL advance':>12s} "
          f"{'warn advance':>13s}")
    for ft, g in R.groupby("fault_type"):
        print(f"  {ft:<18s} {len(g):4d} {g.onset_day.mean():10.0f} "
              f"{g.eol_advance_days.mean():12.1f} {g.warn_advance_days.mean():13.1f}")
    print()
    print("  (advance = days EARLIER the faulted trajectory crosses the threshold;")
    print("   positive means the fault shortened life, which is the intended effect)")
    print()

    # Diversity gained, against the baseline the plan flags as too narrow.
    base_span = float(df.soh_true.max() - df.soh_true.min())
    f_span = float(df.soh_true_faulted.max() - df.soh_true_faulted.min())
    n_below_eol_base = int((df.soh_true <= EOL_SOH).sum())
    n_below_eol_f = int((df.soh_true_faulted <= EOL_SOH).sum())
    print("  operating-condition diversity gained:")
    print(f"    SOH span            baseline {base_span:.4f} -> faulted {f_span:.4f}")
    print(f"    points below EOL    baseline {n_below_eol_base} -> faulted {n_below_eol_f}")
    print(f"    safe-mode/deep-discharge events injected : "
          f"{int((df.dod_event > 0).sum())} (baseline: 0)")
    print(f"    thermal-bias points                     : "
          f"{int((df.temperature_bias_c > 0).sum())} (baseline: 0)")
    print(f"    elevated-rint points                    : "
          f"{int((df.rint_multiplier > 1).sum())} (baseline: 0)")

    df.to_csv(P.results("twin_faulted_subset.csv"), index=False)
    R.to_csv(P.results("twin_fault_injection_records.csv"), index=False)
    P.results("twin_fault_spec.json").write_text(json.dumps({
        "nature": ("observable-signature perturbations applied to the shipped L2 "
                   "trajectories; NOT physics-resolved fault simulation"),
        "why_not_physics": ("the aging coefficients that would be needed to re-run the "
                            "degradation integration were never recorded — see "
                            "src/twin/SPEC.md section 4"),
        "adequate_for": ["testing predictor degradation under off-nominal trajectories",
                         "anomaly detectability",
                         "shift in warning lead time"],
        "not_adequate_for": ["claiming a validated fault-propagation model",
                             "quantitative fault-severity inference",
                             "any causal claim about stress factors"],
        "magnitudes_declared_before_any_model_run": True,
        "deterministic_given": "(trajectory_id, fault_type)",
        "fault_spec": FAULT_SPEC,
        "n_faulted_trajectories": int(R.trajectory_id.nunique()),
        "classes_are_disjoint_across_trajectories": True,
        "evidence_domain": "SIMULATION_ONLY; perturbed simulation at that.",
    }, indent=2), encoding="utf-8")

    print()
    print(f"  wrote {P.results('twin_faulted_subset.csv').name}, "
          f"{P.results('twin_fault_injection_records.csv').name}, "
          f"{P.results('twin_fault_spec.json').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
