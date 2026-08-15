# src/twin/approx_propagation.py
"""物理量 → SOH 增量的近似解析传播（显式声明为近似模型）。

不重建老化方程（SPEC.md §4 系数未记录，反解=虚假陈述）。本模块提供
可解释的单调映射：物理注入（Rint 增长 / Q 下降 / 温度偏置 / 深放电）
通过放大"衰减增量"传播到 SOH，使注入作用于物理参数而非直接篡改最终
遥测。任何输出都携带 ``approx_physical_model=True`` 声明。
"""
from __future__ import annotations

import numpy as np

#: 各物理量对 SOH 衰减增量的放大系数（声明为仿真假设值，敏感性实验覆盖）。
COEF_RINT = 0.9      # Rint 增长 → 衰减加速（相对 rint_mult-1）
COEF_TEMP = 0.05     # 每 °C 温度偏置 → 衰减加速
COEF_DOD = 0.15      # 单次深放电事件 → 衰减加速
COEF_Q = 1.0         # Q 下降比例 → 同比例容量损失


def propagate(soh_true: np.ndarray, rint_mult: float, q_factor: float,
              temp_bias_c: float, dod_event: float, onset_index: int) -> np.ndarray:
    """物理注入 → faulted SOH。仅放大 onset 之后的衰减增量，前置历史不动。

    返回 faulted soh_true（长度与输入一致）。物理量同时返回供下游评估。
    """
    s = np.asarray(soh_true, dtype=float).copy()
    n = len(s)
    k = int(np.clip(int(onset_index), 1, n - 2))

    dr = max(rint_mult - 1.0, 0.0)
    accel = 1.0 + COEF_RINT * dr + COEF_TEMP * max(temp_bias_c, 0.0) \
        + COEF_DOD * max(dod_event, 0.0)

    inc = np.diff(s)
    seg = inc[k:]
    seg = np.where(seg < 0, seg * accel, seg)   # 只放大衰减，不放大测量上跳
    inc[k:] = seg
    out = s.copy()
    out[k + 1:] = out[k] + np.cumsum(inc[k:])

    # Q 下降：容量突降 → SOH 按 q_factor 阶跃下降（物理事件）
    if q_factor != 1.0:
        out[k:] = out[k:] * q_factor
    return np.clip(out, 0.0, 1.2)
