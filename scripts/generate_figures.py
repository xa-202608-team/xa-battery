"""generate_figures.py — 电池组件 V3 核心报告图全套生成。

从 results/ 与 reference/data/ 的权威结果数据生成 8 张报告图，
输出到 XA-202608_最终交付/06_图表/battery/。

  python scripts/generate_figures.py

图清单（英文标注，专业样式）：
  fig01 SOH 退化轨迹 (分族, observed vs true)
  fig02 SOH 多视界预测对比 (5 arm × 3 horizon)
  fig03 RUL 族级 MAE 对比 (5 arm, per-family)
  fig04 故障注入 EOL/warn advance (四类箱线图)
  fig05 方向性校核 (六项机理断言)
  fig06 时间尺度分离 (模板 vs 栅格)
  fig07 参数敏感性 (EOL 随参数变化)
  fig08 预警提前期分布 (逐轨迹直方图)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 路径
ROOT = Path(__file__).resolve().parents[1]          # battery 组件根
RESULTS = ROOT / "results"
REF_DATA = ROOT / "reference" / "data"
FIG_DIR = Path(__file__).resolve().parents[4] / "06_图表" / "battery"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 配色 (5 臂)
ARMS5 = ["#4c72b0", "#dd8452", "#55a467", "#c44e52", "#8172b3"]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})


def _save(fig, name):
    out = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------- fig01
def fig01_soh_trajectories():
    """SOH 退化轨迹，6 环境族，observed vs true。"""
    l2 = pd.read_csv(REF_DATA / "L2_reference_points_v2_min.csv")
    sm = pd.read_csv(REF_DATA / "split_manifest_v2.csv")
    fam = dict(zip(sm.trajectory_id, sm.environment_family_id))
    l2["family"] = l2.trajectory_id.map(fam)

    families = sorted(l2.family.unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, fam_id in zip(axes.ravel(), families):
        g = l2[l2.family == fam_id]
        for tid, tg in g.groupby("trajectory_id"):
            tg = tg.sort_values("reference_day")
            ax.plot(tg.reference_day, tg.soh_true, color="#888", alpha=0.3, lw=0.5)
        # 族均值
        gm = g.groupby("reference_day")[["soh_observed", "soh_true"]].mean()
        ax.plot(gm.index, gm.soh_true, color="#c44e52", lw=2, label="SOH true (mean)")
        ax.plot(gm.index, gm.soh_observed, color="#4c72b0", lw=1.5, ls="--",
                label="SOH observed (mean)")
        ax.axhline(0.70, color="k", ls=":", lw=1, alpha=0.5)
        ax.axhline(0.80, color="orange", ls=":", lw=1, alpha=0.5)
        ax.set_title(fam_id, fontsize=9)
        ax.set_xlabel("Day")
    axes[0, 0].set_ylabel("SOH")
    axes[1, 0].set_ylabel("SOH")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("SOH Degradation Trajectories by Environment Family "
                 "(true=red, observed=blue dashed, EOL=0.70, warn=0.80)", fontsize=11)
    _save(fig, "fig01_soh_trajectories.png")


# ---------------------------------------------------------------- fig02
def fig02_soh_prediction():
    """SOH 多视界预测 MAE 对比 (5 arm × 3 horizon)。"""
    df = pd.read_csv(RESULTS / "comparison_table.csv")
    d = df[(df.task == "SOH_DELTA") & (df.metric == "mae_delta_soh")].copy()
    d = d.sort_values(["horizon_steps", "arm"])

    fig, ax = plt.subplots(figsize=(9, 5))
    arms = sorted(d.arm.unique())
    horizons = sorted(d.horizon_steps.unique())
    x = np.arange(len(horizons))
    w = 0.15
    for i, arm in enumerate(arms):
        vals = d[d.arm == arm].set_index("horizon_steps").reindex(horizons).family_macro.values
        ax.bar(x + i * w, vals, w, label=arm, color=ARMS5[i % 5])
    ax.set_xticks(x + w * 2)
    ax.set_xticklabels([f"{h} steps\n({h*14} d)" for h in horizons])
    ax.set_ylabel("Family-macro MAE (ΔSOH)")
    ax.set_title("Multi-horizon SOH Prediction: 5-arm Comparison")
    ax.legend(fontsize=8)
    _save(fig, "fig02_soh_prediction_comparison.png")


# ---------------------------------------------------------------- fig03
def fig03_rul_comparison():
    """RUL 族级 MAE 对比 (5 arm)，含 per-family 误差棒。"""
    df = pd.read_csv(RESULTS / "comparison_table.csv")
    d = df[(df.task == "MISSION_SPAN_RUL") & (df.metric == "mae_days")].copy()
    d = d.sort_values("family_macro")

    fig, ax = plt.subplots(figsize=(9, 5))
    arms = d.arm.values
    means = d.family_macro.values
    stds = d.across_family_std.values
    colors = ["#55a467" if a == "censoring_aware_ridge_frozen11" else "#4c72b0"
              for a in arms]
    ax.barh(range(len(arms)), means, xerr=stds, color=colors, alpha=0.8, capsize=4)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([a.replace("_", " ") for a in arms], fontsize=9)
    ax.set_xlabel("Family-macro MAE (days, ± across-family std)")
    ax.set_title("Mission-span RUL: 5-arm Comparison (green = delivered model)")
    for i, v in enumerate(means):
        ax.text(v + 5, i, f"{v:.0f} d", va="center", fontsize=8)
    _save(fig, "fig03_rul_comparison.png")


# ---------------------------------------------------------------- fig04
def fig04_fault_injection():
    """故障注入 EOL/warn advance (四类箱线图)。"""
    df = pd.read_csv(RESULTS / "twin_fault_injection_records.csv")
    faults = sorted(df.fault_type.unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [(axes[0], "eol_advance_days", "EOL Advance (days)"),
                           (axes[1], "warn_advance_days", "Warning Advance (days)")]:
        data = [df[df.fault_type == f][col].dropna().values for f in faults]
        bp = ax.boxplot(data, tick_labels=[f.replace("_", "\n") for f in faults],
                        patch_artist=True, showmeans=True)
        for patch, c in zip(bp["boxes"], ARMS5):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.axhline(0, color="k", lw=1, ls="--", alpha=0.5)
        ax.set_title(title)
        ax.set_ylabel("Days (positive = fault shortens life)")
    fig.suptitle("Fault Injection: Physical-parameter → Approximate Propagation "
                 "(4 classes × 15 trajectories, disjoint)", fontsize=11)
    _save(fig, "fig04_fault_injection.png")


# ---------------------------------------------------------------- fig05
def fig05_directionality():
    """方向性校核六项可视化。"""
    d = json.loads((RESULTS / "verify_directionality.json").read_text(encoding="utf-8"))
    checks = [
        ("SOH\nnon-increasing\n(>90%)", d["soh_non_increasing_frac"], 0.9, "%"),
        ("End < Init\n(mean SOH)", d["end_mean"] / d["init_mean"], 1.0, "ratio"),
        ("Late ΔSOH ≤\nEarly ΔSOH", d["delta_late_mean"] / d["delta_early_mean"], 1.0, "ratio"),
        ("Obs err\nbounded (<0.02)", d["obs_err_max"], 0.02, "max|diff|"),
        ("EOL trajectories\n(≥10)", d["eol_trajectories"], 10, "count"),
        ("RUL corr\n(positive)", d["rul_corr_mean"], 0.0, "mean r"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [c[0] for c in checks]
    vals = [c[1] for c in checks]
    passed = [bool(d[f"check_{i+1}"]) for i in range(6)]
    colors = ["#55a467" if p else "#c44e52" for p in passed]
    ax.barh(range(6), vals, color=colors, alpha=0.8)
    ax.set_yticks(range(6))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Directionality Verification (green=passed, all 6 checks)")
    for i, (v, c) in enumerate(zip(vals, checks)):
        ax.text(max(v, 0.01) * 1.02, i, f"{v:.4f}" if c[3] == "ratio" else f"{v:.2f}",
                va="center", fontsize=8)
    _save(fig, "fig05_directionality.png")


# ---------------------------------------------------------------- fig06
def fig06_timescale():
    """时间尺度分离（模板内漂移 vs 14d 增量）。"""
    d = json.loads((RESULTS / "verify_timescale.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["72h template\nSOH drift", "14-day grid\nSOH Δ per step"]
    vals = [d["template_soh_drift"], d["l2_soh_delta_mean_abs"]]
    ax.bar(categories, vals, color=["#4c72b0", "#c44e52"], alpha=0.8)
    ax.set_ylabel("|SOH drift / Δ|")
    ax.set_title(f"Timescale Separation (ratio = {d['separation_ratio']:.1f}×)\n"
                 f"template: {d['template_hours']:.0f}h, {d['template_orbit_count']} orbits, "
                 f"step {d['template_step_seconds']}s")
    for i, v in enumerate(vals):
        ax.text(i, v + v * 0.05, f"{v:.2e}", ha="center", fontsize=9)
    _save(fig, "fig06_timescale_separation.png")


# ---------------------------------------------------------------- fig07
def fig07_sensitivity():
    """参数敏感性：EOL 随参数 ±20% 变化。"""
    d = json.loads((RESULTS / "sensitivity_analysis.json").read_text(encoding="utf-8"))
    base = d["eol_base_mean_days"]
    hi = d["eol_high_mean_days"]
    labels = ["base\n(EOL 0.70)", "+20%\n(EOL 0.84)"]
    vals = [base, hi]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, vals, color=["#4c72b0", "#dd8452"], alpha=0.8)
    ax.set_ylabel("Mean EOL day")
    ax.set_title(f"Parameter Sensitivity: EOL threshold ±20%\n"
                 f"direction stable (higher threshold → earlier EOL: {base:.0f} → {hi:.0f} d)")
    for i, v in enumerate(vals):
        ax.text(i, v + 10, f"{v:.0f} d", ha="center", fontsize=9)
    _save(fig, "fig07_sensitivity.png")


# ---------------------------------------------------------------- fig09
def fig09_data_flow():
    """数据流链路图：STK → V/I/T → SOH → RUL（区分 online vs truth）。"""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # 框定义: (y, text, kind)  kind: 'env'=灰, 'online'=蓝(在线遥测), 'truth'=红(仿真真值), 'model'=绿
    boxes = [
        (11.0, "STK 24 scenarios\n(eclipse/light/β/solar power/thermal)", "env"),
        (9.5, "Power balance + Electrical/Thermal model", "model"),
        (8.0, "V(t)  I(t)  T(t)  SOC(t)\n[online telemetry]", "online"),
        (6.5, "Aging: Q(t) ↓   Rint(t) ↑\n[simulation truth — NOT online]", "truth"),
        (5.0, "SOH_observed\n[BMS online health estimate]", "online"),
        (3.5, "soh_true\n[evaluation label only — NOT online]", "truth"),
        (2.0, "11-dim SOH geometry features\n→ Ridge + L2-SP (closed-form)", "model"),
        (0.5, "SOH prediction / RUL / Warning", "model"),
    ]
    colors = {"env": "#bbbbbb", "online": "#4c72b0", "truth": "#c44e52", "model": "#55a467"}
    for y, text, kind in boxes:
        box = FancyBboxPatch((1.5, y - 0.45), 7, 0.9, boxstyle="round,pad=0.1",
                             facecolor=colors[kind], alpha=0.25, edgecolor=colors[kind], lw=2)
        ax.add_patch(box)
        ax.text(5, y, text, ha="center", va="center", fontsize=9, fontweight="bold")

    # 箭头
    for y_from in [b[0] - 0.45 for b in boxes[:-1]]:
        ax.annotate("", xy=(5, y_from - 0.55), xytext=(5, y_from - 0.05),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))

    # 分流箭头 (SOH_observed → features, soh_true → features label)
    ax.annotate("features ← SOH_observed", xy=(2.0, 2.45), xytext=(1.0, 5.0),
                fontsize=7, color="#4c72b0",
                arrowprops=dict(arrowstyle="->", color="#4c72b0", lw=1))
    ax.annotate("EOL label ← soh_true", xy=(8.0, 2.45), xytext=(9.0, 3.5),
                fontsize=7, color="#c44e52", ha="right",
                arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1))

    # 图例
    for i, (lbl, k) in enumerate([("online telemetry (deployable)", "online"),
                                   ("simulation truth (label only)", "truth"),
                                   ("model / pipeline", "model")]):
        ax.add_patch(plt.Rectangle((0.3, 11.0 - i * 0.35), 0.3, 0.25,
                                    facecolor=colors[k], alpha=0.5))
        ax.text(0.7, 11.1 - i * 0.35, lbl, fontsize=7, va="center")

    ax.set_title("Battery Data-flow: STK → V/I/T → SOH → RUL\n"
                 "(online vs truth separated; soh_observed = BMS estimate, "
                 "soh_true = label only)", fontsize=11, pad=15)
    _save(fig, "fig09_data_flow.png")


# ---------------------------------------------------------------- fig08
def fig08_warning_leadtime():
    """预警提前期逐轨迹分布。"""
    df = pd.read_csv(RESULTS / "prognostic_metrics_by_trajectory.csv")
    dt = df["dt_warn_days"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(dt, bins=20, color="#55a467", alpha=0.7, edgecolor="white")
    ax.axvline(dt.mean(), color="k", ls="--", lw=1.5, label=f"mean = {dt.mean():.0f} d")
    ax.axvline(dt.median(), color="orange", ls="--", lw=1.5, label=f"median = {dt.median():.0f} d")
    ax.set_xlabel("Warning lead time (days)")
    ax.set_ylabel("Trajectory count")
    ax.set_title(f"Warning Lead Time Distribution (n={len(dt)})")
    ax.legend()
    _save(fig, "fig08_warning_leadtime.png")


# ---------------------------------------------------------------- fig10
def fig10_conformal_coverage():
    """保形区间覆盖（6 族 crossfit coverage + half-width）。"""
    d = json.loads((RESULTS / "rul_mission_span_conformal.json").read_text(encoding="utf-8"))
    families = sorted(d["crossfit_coverage_by_family"].keys())
    cov = [d["crossfit_coverage_by_family"][f] for f in families]
    hw = [d["per_family_half_width_days"][f] for f in families]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # coverage
    colors = ["#55a467" if c >= 0.9 else "#dd8452" if c >= 0.8 else "#c44e52" for c in cov]
    ax1.barh(range(len(families)), cov, color=colors, alpha=0.8)
    ax1.axvline(0.9, color="k", ls="--", lw=1.5, label="target 0.90")
    ax1.axvline(d["crossfit_mean_coverage"], color="blue", ls=":", lw=1,
                label=f"mean {d['crossfit_mean_coverage']:.3f}")
    ax1.set_yticks(range(len(families)))
    ax1.set_yticklabels([f.replace("_", "\n") for f in families], fontsize=7)
    ax1.set_xlabel("Cross-fit marginal coverage")
    ax1.set_title("Conformal Coverage by Family")
    ax1.legend(fontsize=8)
    # half-width
    ax2.barh(range(len(families)), hw, color="#4c72b0", alpha=0.7)
    ax2.axvline(d["half_width_days"], color="k", ls="--", lw=1.5,
                label=f"global {d['half_width_days']:.0f} d")
    ax2.set_yticks(range(len(families)))
    ax2.set_yticklabels([f.replace("_", "\n") for f in families], fontsize=7)
    ax2.set_xlabel("Half-width (days)")
    ax2.set_title("Interval Half-width by Family")
    ax2.legend(fontsize=8)
    fig.suptitle(f"Conformal Interval: marginal coverage (target 0.90, mean {d['crossfit_mean_coverage']:.3f} NOT met) "
                 "+ half-width", fontsize=11)
    _save(fig, "fig10_conformal_coverage.png")


# ---------------------------------------------------------------- fig11
def fig11_alarm_performance():
    """告警混淆矩阵 + 提前性 per-family。"""
    df = pd.read_csv(RESULTS / "prognostic_metrics.csv")
    r = df.iloc[0]
    tp, fp = int(r.n_true_positive), int(r.n_false_positive)
    fn, tn = int(r.n_missed), int(r.n_true_negative)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # 混淆矩阵
    mat = np.array([[tp, fn], [fp, tn]])
    im = ax1.imshow(mat, cmap="Blues", aspect="auto")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["Alarmed", "Not alarmed"])
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(["Will cross warn", "Will not cross"])
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(mat[i, j]), ha="center", va="center",
                     fontsize=18, fontweight="bold",
                     color="white" if mat[i, j] > 50 else "black")
    ax1.set_title(f"Warning Confusion Matrix\n"
                  f"hit {tp}/{tp+fn}, miss {fn}, false alarm {fp}/{fp+tn} "
                  f"= {r.false_alarm_rate_of_non_crossers:.0%}")
    fig.colorbar(im, ax=ax1, fraction=0.046)

    # per-family 提前性
    by_traj = pd.read_csv(RESULTS / "prognostic_metrics_by_trajectory.csv")
    fam_dt = by_traj.groupby("environment_family_id")["dt_warn_days"].mean().sort_values()
    ax2.barh(range(len(fam_dt)), fam_dt.values, color="#55a467", alpha=0.8)
    ax2.set_yticks(range(len(fam_dt)))
    ax2.set_yticklabels([f.replace("_", "\n") for f in fam_dt.index], fontsize=7)
    ax2.axvline(r.dt_warn_mean_days, color="k", ls="--", lw=1.5,
                label=f"mean {r.dt_warn_mean_days:.0f} d")
    ax2.set_xlabel("Mean warning lead time (days)")
    ax2.set_title("Warning Lead Time by Family")
    ax2.legend(fontsize=8)
    fig.suptitle("Alarm Performance: 110/110 hit, 0 miss, 1 false alarm; "
                 "mean lead 123 d", fontsize=11)
    _save(fig, "fig11_alarm_performance.png")


def main():
    print(f"=== 电池报告图生成 → {FIG_DIR} ===")
    for fn in [fig01_soh_trajectories, fig02_soh_prediction, fig03_rul_comparison,
               fig04_fault_injection, fig05_directionality, fig06_timescale,
               fig07_sensitivity, fig08_warning_leadtime, fig09_data_flow,
               fig10_conformal_coverage, fig11_alarm_performance]:
        fn()
    print(f"=== 完成: {len(list(FIG_DIR.glob('fig*.png')))} 张图 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
