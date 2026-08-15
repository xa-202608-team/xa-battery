# src/metrics/trend.py
"""趋势一致性指标：direction accuracy + slope error（覆盖 FAQ 的趋势一致性）。"""
from __future__ import annotations

import numpy as np


def direction_accuracy(truth: np.ndarray, pred: np.ndarray) -> float:
    """逐相邻段符号一致率：sign(Δpred)==sign(Δtruth) 的占比。

    平坦段（|Δ|<1e-9）不计入分母，避免无意义比较。
    """
    t = np.asarray(truth, dtype=float)
    p = np.asarray(pred, dtype=float)
    dt = np.diff(t)
    dp = np.diff(p)
    valid = np.abs(dt) > 1e-9
    if not valid.any():
        return 1.0
    return float(np.mean(np.sign(dt[valid]) == np.sign(dp[valid])))


def slope_error(truth: np.ndarray, pred: np.ndarray) -> float:
    """预测斜率与真值斜率的相对偏差（无符号，越小越好）。"""
    t = np.asarray(truth, dtype=float)
    p = np.asarray(pred, dtype=float)
    x = np.arange(len(t), dtype=float)
    k_t = float(np.polyfit(x, t, 1)[0])
    k_p = float(np.polyfit(x, p, 1)[0])
    if abs(k_t) < 1e-9:
        return float(abs(k_p)) if abs(k_p) > 1e-9 else 0.0
    return float(abs(k_p - k_t) / abs(k_t))
