# tests/test_soh_online_definition.py
from pathlib import Path


def test_soh_online_definition_doc_exists():
    txt = Path("docs/soh_online_definition.md").read_text(encoding="utf-8")
    assert "BMS" in txt and "soh_observed" in txt
    assert "0.0006" in txt or "MAE" in txt       # 观测误差证据
    assert "soh_true" in txt                      # truth 通道区分


def test_full_l2_copied():
    from src import paths as P
    assert P.FULL_L2_CSV.exists(), "完整 L2 未复制进交付包"
    import pandas as pd
    df = pd.read_csv(P.FULL_L2_CSV, nrows=3)
    assert "capacity_ah" in df.columns and "soh_true" in df.columns
