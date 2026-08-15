# tests/test_fault_injection.py
import numpy as np
import pytest

from src.twin.approx_propagation import propagate
from src.twin.fault_injection import inject


def test_propagate_rint_increase_accelerates_decline():
    soh = np.linspace(1.0, 0.80, 21)
    base = propagate(soh, rint_mult=1.0, q_factor=1.0, temp_bias_c=0.0,
                     dod_event=0.0, onset_index=10)
    faulted = propagate(soh, rint_mult=1.6, q_factor=1.0, temp_bias_c=0.0,
                        dod_event=0.0, onset_index=10)
    assert faulted[-1] < base[-1], "内阻增大应加速 SOH 衰减"


def test_propagate_preserves_pre_onset():
    soh = np.linspace(1.0, 0.80, 21)
    out = propagate(soh, rint_mult=1.6, q_factor=1.0, temp_bias_c=0.0,
                    dod_event=0.0, onset_index=10)
    np.testing.assert_allclose(out[:10], soh[:10], atol=1e-12)


def test_propagate_no_fault_identity():
    soh = np.linspace(1.0, 0.80, 21)
    out = propagate(soh, rint_mult=1.0, q_factor=1.0, temp_bias_c=0.0,
                    dod_event=0.0, onset_index=10)
    np.testing.assert_allclose(out, soh, atol=1e-9)


def test_propagate_capacity_jump_steps_down():
    soh = np.linspace(1.0, 0.80, 21)
    base = propagate(soh, rint_mult=1.0, q_factor=1.0, temp_bias_c=0.0,
                     dod_event=0.0, onset_index=10)
    faulted = propagate(soh, rint_mult=1.0, q_factor=0.96, temp_bias_c=0.0,
                        dod_event=0.0, onset_index=10)
    # 阶跃：onset 后首点立即下降（faulted[k] 已按 q_factor 下移）
    assert faulted[11] < faulted[10]
    # 整体更早到达低值
    assert faulted[-1] < base[-1]


def test_inject_returns_physical_channels():
    d = np.arange(0, 2100, 100, dtype=float)
    st = np.linspace(1.0, 0.75, len(d))
    so = st + 0.002
    r = inject(d, st, so, "rint_step")
    assert "rint_multiplier_series" in r
    assert "soh_true_faulted" in r
    assert len(r["soh_true_faulted"]) == len(d)
    # 物理量注入而非直接改 SOH 的声明
    assert r["nature"] == "physical_parameter_injection_with_approx_propagation"
