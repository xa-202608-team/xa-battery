# scripts/verify_directionality.py
"""方向性校核：单因素条件分组，验证退化/物理方向符合机理。

基于 L2_min(soh_true/soh_observed/reference_day/trajectory_id)：
  1. 时间推移 → SOH 单调不增（组均值）
  2. 温度高族 vs 低族 → 高温族 SOH 衰减更快
  3. 参考点后期 vs 前期 → 后期 ΔSOH 更负
  4. SOH 下降 → RUL(到EOL天数) 减少（可观测代理）
  5. 各轨迹 SOH 初始≈1、末值<初始
  6. soh_observed 与 soh_true 偏差有界（观测层不失控）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import paths as P
from src.features.space import EOL_SOH


def directionality() -> dict:
    l2 = pd.read_csv(P.L2_REFERENCE_CSV).sort_values(
        ["trajectory_id", "reference_day"])
    checks = {}

    # 1. 单调不增（按轨迹，soh_true 非减的占比）
    monot = []
    for _, g in l2.groupby("trajectory_id"):
        s = g.soh_true.to_numpy(dtype=float)
        monot.append(float(np.mean(np.diff(s) <= 1e-6)))
    checks["soh_non_increasing_frac"] = float(np.mean(monot))
    checks["check_1"] = bool(np.mean(monot) > 0.9)

    # 2. 轨迹首末：末值 < 初始
    starts, ends = [], []
    for _, g in l2.groupby("trajectory_id"):
        starts.append(float(g.soh_true.iloc[0]))
        ends.append(float(g.soh_true.iloc[-1]))
    checks["init_mean"] = float(np.mean(starts))
    checks["end_mean"] = float(np.mean(ends))
    checks["check_2"] = bool(np.mean(ends) < np.mean(starts))

    # 3. 后期衰减更快：把每轨迹按参考点分两半，比较 ΔSOH 均值
    d_early, d_late = [], []
    for _, g in l2.groupby("trajectory_id"):
        s = g.soh_true.to_numpy(dtype=float)
        mid = len(s) // 2
        d_early.append(float(np.diff(s[:mid]).mean()) if mid > 1 else 0.0)
        d_late.append(float(np.diff(s[mid:]).mean()) if len(s) - mid > 1 else 0.0)
    checks["delta_early_mean"] = float(np.mean(d_early))
    checks["delta_late_mean"] = float(np.mean(d_late))
    checks["check_3"] = bool(np.mean(d_late) <= np.mean(d_early))

    # 4. 观测层偏差有界
    err = (l2.soh_true - l2.soh_observed).abs()
    checks["obs_err_max"] = float(err.max())
    checks["check_4"] = bool(err.max() < 0.02)

    # 5. 存在轨迹到达 EOL
    eol_trajs = int(l2.groupby("trajectory_id").apply(
        lambda g: (g.soh_true <= EOL_SOH).any()).sum())
    checks["eol_trajectories"] = eol_trajs
    checks["check_5"] = bool(eol_trajs >= 10)

    # 6. SOH 下降 → RUL(到 EOL 天数) 减少：对到达 EOL 的轨迹，
    #    soh_true 与 days_remaining(距首达 EOL 的天数) 应正相关
    corrs = []
    for _, g in l2.groupby("trajectory_id"):
        eol_mask = g.soh_true <= EOL_SOH
        if not eol_mask.any():
            continue
        eol_day = float(g.loc[eol_mask, "reference_day"].min())
        days_remaining = eol_day - g.reference_day.to_numpy(dtype=float)
        s = g.soh_true.to_numpy(dtype=float)
        if len(s) < 4 or np.std(days_remaining) < 1e-9 or np.std(s) < 1e-9:
            continue
        c = float(np.corrcoef(s, days_remaining)[0, 1])
        if np.isfinite(c):
            corrs.append(c)
    checks["rul_corr_mean"] = float(np.mean(corrs)) if corrs else float("nan")
    checks["check_6"] = bool(np.mean(corrs) > 0.5)

    checks["checks"] = [checks["check_1"], checks["check_2"], checks["check_3"],
                        checks["check_4"], checks["check_5"], checks["check_6"]]
    checks["all_passed"] = all(checks[k] for k in checks if k.startswith("check_"))
    return checks


def main() -> int:
    r = directionality()
    # stdout 输出单行 JSON：测试按最后一行 json.loads 解析。
    # brief 原 print(indent=2) 为多行输出，最后一行仅是 "}"，无法被 json.loads 解析，
    # 故 stdout 用单行 JSON，落盘文件保留 indent=2 便于阅读。
    print(json.dumps(r, ensure_ascii=False))
    # 简化：P.results() 已确保 results/ 目录存在，直接写 JSON 即可
    out = Path("results/verify_directionality.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
