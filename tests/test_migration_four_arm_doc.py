# tests/test_migration_four_arm_doc.py
from pathlib import Path


def test_four_arm_doc_exists_and_covers_arms():
    txt = Path("docs/migration_four_arm.md").read_text(encoding="utf-8")
    for arm in ("Target-only", "Feature-transfer", "Parameter-transfer", "Full L2-SP"):
        assert arm in txt, f"缺少 {arm}"
    assert "Adaptive Transfer Guard" in txt or "负迁移保护" in txt
