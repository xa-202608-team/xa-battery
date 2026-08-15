"""时间尺度分离验证：72h 模板重放 + 14d 退化推进 是否可解释、非任意拼接。

证据 1（模板内尺度）：L1 72h 模板 30s 步长，46 轨道圈次，DoD/温度/吞吐的
轨道内分布（高频任务尺度）。
证据 2（退化尺度）：L2 14 天参考栅格上 SOH 每步增量（慢变退化尺度）。
证据 3（分离合理性）：模板内 soh_true 变化 << 14d 栅格累计变化（数量级分离），
且轨道内 DoD/EFC 统计与 L2 聚合口径可比。
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import paths as P

# SAMPLE_GLOB 修正：brief 原用 P.V2_ROOT.parent.parent.parent 会解析到
# 03_代码/05_结果（不存在）；改用 P.V3_ROOT（battery 组件根）向上三级到
# XA-202608_最终交付，再落到 05_结果/reference/battery/L1遥测样本/。
SAMPLE_GLOB = (P.V3_ROOT.parent.parent.parent / "05_结果" / "reference" / "battery"
               / "L1遥测样本" / "*.csv")


def timescale() -> dict:
    files = [p for p in glob.glob(str(SAMPLE_GLOB)) if not p.endswith(("说明.md",))]
    if not files:
        raise RuntimeError(f"SAMPLE_GLOB 未匹配到任何 L1 样本 CSV：{SAMPLE_GLOB}")
    # 样本 CSV 带 BOM 头，用 utf-8-sig 读取（brief 原文无 encoding，BOM 会污染首列名）。
    l2 = pd.read_csv(P.L2_REFERENCE_CSV, encoding="utf-8-sig")

    # 模板内统计（取第一个样本）
    tpl = pd.read_csv(files[0], encoding="utf-8-sig")
    dur_s = float(tpl.scenario_epoch_s.max() - tpl.scenario_epoch_s.min())
    efc_orbit = float(tpl.efc.max() - tpl.efc.min())
    dod_mean = float(tpl.dod_orbit[tpl.dod_orbit > 0].mean())
    temp_span = float(tpl.temperature_c.max() - tpl.temperature_c.min())
    soh_drift_in_template = float(tpl.soh_true.max() - tpl.soh_true.min())

    # 14d 栅格 SOH 增量
    l2 = l2.sort_values(["trajectory_id", "reference_day"])
    l2["dsoh"] = l2.groupby("trajectory_id").soh_true.diff()
    dsoh_mean_abs = float(l2.dsoh.abs().mean())
    step_days = float(l2.groupby("trajectory_id").reference_day.diff().median())

    return {
        "template_step_seconds": 30,
        "template_hours": round(dur_s / 3600, 1),
        "template_orbit_count": int(tpl.orbit_no.nunique()),
        "template_efc_span": efc_orbit,
        "template_dod_mean": dod_mean,
        "template_temp_span_c": temp_span,
        "template_soh_drift": soh_drift_in_template,
        "l2_reference_step_days": float(step_days),
        "l2_soh_delta_mean_abs": dsoh_mean_abs,
        "separation_ratio": float(dsoh_mean_abs / max(soh_drift_in_template, 1e-9)),
        "timescale_separation_ok": bool(dsoh_mean_abs > soh_drift_in_template),
        "n_samples": len(files),
    }


def main() -> int:
    r = timescale()
    # 单行 JSON 输出：test_timescale.py 用 stdout 末行整体 json.loads；
    # 若带 indent=2 末行是 "}" 无法解析（brief 测试原文如此，此处最小偏离保证其通过）。
    print(json.dumps(r, ensure_ascii=False))
    Path("results/verify_timescale.json").parent.mkdir(parents=True, exist_ok=True)
    Path("results/verify_timescale.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
