import subprocess
from pathlib import Path


def test_no_forbidden_tracked_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root
    ).decode().split("\0")
    forbidden = [
        path for path in tracked if path
        and (
            path.startswith(("data/", "results/", "checkpoints/"))
            and path not in {
                "data/README.md", "data/data_manifest.json",
                "results/README.md", "results/public_summary.json",
                "results/expected_metrics.json",
                "checkpoints/README.md", "checkpoints/checkpoint_manifest.json",
            }
        )
    ]
    assert forbidden == []


def test_reference_payload_dirs_have_no_tracked_files() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = [
        path for path in subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=root
        ).decode().split("\0") if path
    ]
    forbidden = [
        path for path in tracked
        if path.startswith(("reference/data/", "reference/models/", "reference/results/"))
    ]
    assert forbidden == []
