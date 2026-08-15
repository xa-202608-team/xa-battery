"""paths.py — the ONLY path resolver in the package. No parent-project paths exist.

Every path is derived from ``Path(__file__).resolve().parent.parent``, which is the
package root (``battery_release_v2/``). Nothing here ever mentions the research
project root, ``D:\\SUFA``, ``parents[3]``, or any absolute host path — and
``battery_entry.verify`` greps the whole package to prove that mechanically rather
than trusting this docstring.

Write policy: this package writes ONLY under a caller-supplied output directory.
:func:`guard_output_write` enforces it, so a ``reproduce`` or ``predict`` run cannot
touch the package's own reference results even by an accidental path join.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

#: battery_release_v2/battery_entry/paths.py -> battery_release_v2/
PACKAGE_ROOT: Path = Path(__file__).resolve().parent.parent

CONFIG_DIR: Path = PACKAGE_ROOT / "configs"
MODEL_DIR: Path = PACKAGE_ROOT / "models"
DATA_DIR: Path = PACKAGE_ROOT / "data"
SCHEMA_DIR: Path = PACKAGE_ROOT / "schemas"
REFERENCE_DIR: Path = PACKAGE_ROOT / "results" / "reference"
DOCS_DIR: Path = PACKAGE_ROOT / "docs"
EXAMPLES_DIR: Path = PACKAGE_ROOT / "examples"
TESTS_DIR: Path = PACKAGE_ROOT / "tests"

RELEASE_MANIFEST: Path = PACKAGE_ROOT / "release_manifest.json"
SHA256SUMS: Path = PACKAGE_ROOT / "SHA256SUMS"
CLAIMS_MANIFEST: Path = PACKAGE_ROOT / "CLAIMS_MANIFEST.json"

#: The frozen ARC source space. Carried as a plain .npz inside the package so no
#: parent artifact directory is consulted at runtime.
ARC_SPACE_NPZ: Path = MODEL_DIR / "arc_clean_fixed_space.npz"
#: The frozen alpha table (LOFO per (fold, horizon) and fixed-holdout per horizon).
FROZEN_ALPHA_JSON: Path = MODEL_DIR / "frozen_alpha.json"

#: The reference data slice: the formal route's L3 v2 rows, restricted to the columns
#: the package actually needs. Enough for a full refit of the frozen configuration.
#: Human-readable, and what a reader inspects.
L3_REFERENCE_CSV: Path = DATA_DIR / "L3_windows_v2_arc_clean_fixed_min.csv"
#: The SAME numbers as float64 binary. Measured: a CSV round-trip perturbs feature
#: values by up to 1.11e-16, which the ridge solve amplifies to ~2.6e-13 in the
#: prediction. Loading numerics from here instead makes the golden replay bit-exact
#: against the frozen Phase 4 predictions. Costs ~1.1 MB — less than the CSV.
REFERENCE_MATRIX_NPZ: Path = DATA_DIR / "reference_matrix.npz"
SPLIT_MANIFEST_CSV: Path = DATA_DIR / "split_manifest_v2.csv"
L2_REFERENCE_CSV: Path = DATA_DIR / "L2_reference_points_v2_min.csv"


class RefusedWrite(RuntimeError):
    """Raised when a write target is outside the caller-supplied output directory."""


def _normalise(path: str | Path) -> Path:
    """Resolve without requiring existence, so ``..`` cannot escape."""
    p = Path(path)
    p = p if p.is_absolute() else (Path.cwd() / p)
    tail: list[str] = []
    probe = p
    while not probe.exists() and probe != probe.parent:
        tail.append(probe.name)
        probe = probe.parent
    return probe.resolve().joinpath(*reversed(tail))


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def guard_output_write(path: str | Path, output_root: str | Path) -> Path:
    """Return ``path`` resolved, or raise :class:`RefusedWrite`.

    Two refusals, both load bearing:

    1. the target must be inside ``output_root`` — the directory the caller named;
    2. the target must NOT be inside the package root, so a run can never overwrite
       the shipped reference results, models or data even if ``output_root`` was
       pointed at the package by mistake.
    """
    p = _normalise(path)
    root = _normalise(output_root)
    if not _is_within(p, root):
        raise RefusedWrite(f"write target {p} is outside the output directory {root}")
    if _is_within(p, PACKAGE_ROOT):
        raise RefusedWrite(
            f"write target {p} is inside the package root {PACKAGE_ROOT}; the shipped "
            f"reference results, models and data are read-only")
    return p


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def package_files(include_generated: bool = True) -> list[Path]:
    """Every file in the package, sorted, excluding caches and the checksum file.

    ``SHA256SUMS`` is excluded because it cannot contain its own hash.
    """
    skip_names = {"SHA256SUMS"}
    out: list[Path] = []
    for p in sorted(PACKAGE_ROOT.rglob("*")):
        if not p.is_file():
            continue
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        if p.name in skip_names:
            continue
        if not include_generated and p.name == "release_manifest.json":
            continue
        out.append(p)
    return out


def relpath(p: str | Path) -> str:
    """POSIX-style path relative to the package root, for manifests."""
    return str(Path(p).resolve().relative_to(PACKAGE_ROOT)).replace("\\", "/")
