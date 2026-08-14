import json
import subprocess
import sys


def test_timescale_script_runs():
    out = subprocess.run([sys.executable, "scripts/verify_timescale.py"],
                         capture_output=True, text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["template_step_seconds"] == 30
    assert r["template_hours"] == 72
    assert r["l2_reference_step_days"] == 14
    assert r["timescale_separation_ok"] is True
