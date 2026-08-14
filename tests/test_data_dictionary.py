# tests/test_data_dictionary.py
import csv
from pathlib import Path


def test_data_dictionary_covers_required_fields():
    rows = list(csv.DictReader(
        open("docs/battery_data_dictionary.csv", encoding="utf-8")))
    names = {r["field"] for r in rows}
    for req in ("voltage_v", "current_a", "temperature_c", "soc",
                "soh_observed", "soh_true", "rul_days"):
        assert req in names, f"缺少 {req}"
    # 每行必须有在线可用/Truth 标注
    for r in rows:
        assert r["online_available"] in {"是", "否", "条件性"}
        assert r["is_truth"] in {"是", "否"}
