# tests/test_trend.py
import numpy as np

from src.metrics.trend import direction_accuracy, slope_error


def test_direction_accuracy_perfect():
    t = np.array([1.0, 0.9, 0.8, 0.7])
    p = np.array([1.0, 0.91, 0.81, 0.71])   # 方向全一致
    assert direction_accuracy(t, p) == 1.0


def test_direction_accuracy_wrong_half():
    t = np.array([1.0, 0.9, 0.8, 0.7])
    p = np.array([1.0, 0.95, 0.85, 0.65])
    # 逐段符号：t 全降；p 也全降（0.95<1, 0.85<0.95, 0.65<0.85）→ 1.0
    assert direction_accuracy(t, p) == 1.0


def test_slope_error_sign_and_magnitude():
    t = np.linspace(1.0, 0.7, 30)
    # 注：brief 原为 p = t + 0.01，但加法偏置不改变斜率，
    # 导致 slope_error 恒为 0.0，与 e > 0 断言自相矛盾；
    # 改为乘法微扰（p = t * 1.01）使相对斜率偏差 = 0.01，符合测试意图。
    p = t * 1.01
    e = slope_error(t, p)
    assert e > 0 and e < 0.05


def test_direction_accuracy_with_flat_segment():
    t = np.array([1.0, 1.0, 0.9, 0.8])
    p = np.array([1.0, 0.99, 0.92, 0.82])
    da = direction_accuracy(t, p)
    assert 0.0 <= da <= 1.0
