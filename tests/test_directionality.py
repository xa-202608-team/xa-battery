# tests/test_directionality.py
import json
import subprocess
import sys


def test_directionality_script_runs_and_passes():
    """方向性六项断言全部通过，输出 JSON 且 exit 0。"""
    out = subprocess.run([sys.executable, "scripts/verify_directionality.py"],
                         capture_output=True, text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["all_passed"] is True
    assert len(r["checks"]) >= 6
