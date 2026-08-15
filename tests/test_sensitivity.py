# tests/test_sensitivity.py
import json
import subprocess
import sys


def test_sensitivity_script_runs():
    out = subprocess.run([sys.executable, "scripts/sensitivity_analysis.py"],
                         capture_output=True, text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert "parameters" in r and len(r["parameters"]) >= 3
