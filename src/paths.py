"""paths.py — every path this package reads, in one place.

v2 IS READ-ONLY. Nothing in this package ever writes inside ``V2_ROOT``, and the
guard :func:`assert_v2_untouched` is called by the test suite so a stray write is
caught rather than discovered at submission time.

Historical trees (``SUFA_ROOT``, ``ARCHIVE_ROOT``) are read-only too: implementations
were COPIED into this package (see ``REUSE_MAP.md``), never modified in place.
"""
from __future__ import annotations

from pathlib import Path

#: This package. ``paths.py`` lives in ``<V3_ROOT>/src/``, so one level up.
V3_ROOT = Path(__file__).resolve().parents[1]

#: The frozen delivery package. READ-ONLY — see module docstring.
V2_ROOT = V3_ROOT / "reference"

#: The historical research tree. READ-ONLY.
SUFA_ROOT = V3_ROOT.parent

#: The pre-Phase-0 archive. READ-ONLY, accessed by exact path only (never scanned).
ARCHIVE_ROOT = Path(r"D:\SUFAcopy_archive_pre_phase0_20260804_160617\archived_files")

# ------------------------------------------------------------------ v2 (read-only)
ARC_SPACE_NPZ = V2_ROOT / "models" / "arc_clean_fixed_space.npz"
FROZEN_ALPHA_JSON = V2_ROOT / "models" / "frozen_alpha.json"
CONFORMAL_QUANTILES_JSON = V2_ROOT / "models" / "conformal_quantiles.json"

L2_REFERENCE_CSV = V2_ROOT / "data" / "L2_reference_points_v2_min.csv"
L3_WINDOWS_CSV = V2_ROOT / "data" / "L3_windows_v2_arc_clean_fixed_min.csv"
SPLIT_MANIFEST_CSV = V2_ROOT / "data" / "split_manifest_v2.csv"

GENERATION_CONFIG_JSON = V2_ROOT / "docs" / "stk_metadata" / "generation_config.json"
DATASET_SUMMARY_JSON = V2_ROOT / "docs" / "stk_metadata" / "dataset_summary.json"

# ------------------------------------------- historical research tree (read-only)
#
# RESOLUTION ORDER, and why it is this way round:
#
# Each of these inputs is looked up IN-PACKAGE FIRST, then in the research tree. The
# in-package copy is what ships (see DATA_REQUIREMENTS.md); the research-tree path is
# the fallback so the package still runs unchanged inside the original working tree,
# which is where every number in results/ was produced.
#
# The fallback is NOT a silent convenience: :func:`input_provenance` reports which
# location each input actually resolved to, and the self-containment test asserts that a
# shipped bundle resolves everything in-package. So "it worked on my machine because the
# research tree was there" cannot go unnoticed.

def _first_existing(*candidates: Path) -> Path:
    """First path that exists; otherwise the first candidate, so errors name the
    in-package location the bundle is supposed to carry rather than a machine-specific
    research-tree path."""
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


#: The authoritative source model. Its SHA-256 equals the ``model_npz_sha256``
#: recorded inside ARC_SPACE_NPZ, which is how it was identified as THE model that
#: produced the frozen ARC space. v2 deliberately ships the clip/scaler WITHOUT the
#: source coefficients, so L2-SP must read beta_src from here.
SOURCE_MODEL_NPZ = _first_existing(
    V3_ROOT / "data" / "source_prior" / "arc_clean_fixed_alpha10_model.npz",
    SUFA_ROOT / "artifacts" / "experimental_arc_sensitivity"
    / "arc_clean_fixed_alpha10" / "model.npz")

#: NASA PCoE public degradation dataset, cleaned to one npz per cell.
PUBLIC_SOURCE_DIR = _first_existing(
    V3_ROOT / "data" / "source_public",
    SUFA_ROOT / "data" / "processed" / "clean" / "arc")

#: The generator's OUTPUTS (the generator code itself does not exist — see SPEC.md).
#: NOT shipped: 253.4 MB, and every conclusion drawn from it is already recorded in
#: results/twin_chain_verification.json and docs/observables_chain.md. Only
#: src/twin/verify_chain.py needs it, and it fails with a clear message when absent.
AFTERSTK_DIR = SUFA_ROOT / "data" / "afterstk"

#: 完整 L2（34 列，逐参考点遥测/工况/真值）。从归档电池交付复制进交付包，
#: 供可信度验证脚本（方向性/时间尺度/敏感性）与在线量证据使用。
FULL_L2_CSV = V3_ROOT / "data" / "afterstk" / "L2_multi_scenario_reference_points.csv"

L1_TELEMETRY_CSV = AFTERSTK_DIR / "L1_multi_scenario_battery_telemetry_72h.csv"
L2_FULL_CSV = AFTERSTK_DIR / "L2_multi_scenario_reference_points.csv"
L0_ENVIRONMENT_CSV = AFTERSTK_DIR / "L0_multi_scenario_stk_environment.csv"

#: Phase 4's 29-field stress observability contract. NOT shipped either: its findings
#: are restated in full in docs/observables_chain.md section 3, which is what the
#: report cites.
STRESS_CONTRACT_JSON = (SUFA_ROOT / "reports" / "stk_transfer_v2"
                        / "04_exposure_stress" / "stress_observability_contract.json")
STRESS_ABLATION_CSV = (SUFA_ROOT / "reports" / "stk_transfer_v2"
                       / "04_exposure_stress" / "feature_ablation_results.csv")

# ------------------------------------------------------------------ v3 (writable)
RESULTS_DIR = V3_ROOT / "results"
DOCS_DIR = V3_ROOT / "docs"
DOCKER_DIR = V3_ROOT / "docker"


def results(name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name


# -------------------------------------------------------- twin L2 reconstruction
#
# Self-contained ageing reconstruction inputs (twin_code_r3 integration).
# All files live under ``data/twin_reconstruction/`` inside this package.
TWIN_RECON_DIR = V3_ROOT / "data" / "twin_reconstruction"

TWIN_STRESS_DRIVER_CSV = TWIN_RECON_DIR / "stress_driver_v1.csv"
TWIN_TRAJECTORY_PARAMETERS_CSV = (
    TWIN_RECON_DIR / "trajectory_parameters_v1.csv"
)
TWIN_ANNUAL_CALENDAR_CSV = (
    TWIN_RECON_DIR / "annual_environment_family_calendar.csv"
)
TWIN_CALIBRATION_REFERENCE_CSV = (
    TWIN_RECON_DIR / "calibration_reference_truth.csv"
)
TWIN_RECONSTRUCTION_CONFIG_JSON = (
    TWIN_RECON_DIR / "reconstruction_config_v1.json"
)


#: Files whose SHA-256 the guard pins. Chosen as the ones a careless refit would
#: touch first: the frozen space, the frozen alpha, and the three data slices.
_GUARDED = ("models/arc_clean_fixed_space.npz", "models/frozen_alpha.json",
            "models/conformal_quantiles.json",
            "data/L2_reference_points_v2_min.csv",
            "data/L3_windows_v2_arc_clean_fixed_min.csv",
            "data/split_manifest_v2.csv")


def v2_sha256() -> dict[str, str]:
    """SHA-256 of each guarded v2 file, for the read-only assertion."""
    import hashlib
    out = {}
    for rel in _GUARDED:
        p = V2_ROOT / rel
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        out[rel] = h.hexdigest()
    return out


def assert_v2_untouched(expected: dict[str, str]) -> None:
    """Raise if any guarded v2 file changed. Called by the test suite."""
    now = v2_sha256()
    drift = {k: (expected[k], now[k]) for k in expected if expected.get(k) != now.get(k)}
    if drift:
        raise RuntimeError(f"battery_release_v2 was modified: {drift}")


# --------------------------------------------------------------- provenance report

def input_provenance() -> dict:
    """Where each input actually resolved, and whether it is inside this package.

    Exists so a shipped bundle can PROVE it is self-contained rather than quietly
    falling back to a research tree that happens to be present. Consumed by
    ``tests/test_self_contained.py`` and printable via ``python -m src.paths``.
    """
    def entry(p: Path, required: str) -> dict:
        rp = p.resolve()
        try:
            in_pkg = V3_ROOT.resolve() in rp.parents or rp == V3_ROOT.resolve()
        except Exception:
            in_pkg = False
        in_v2 = False
        try:
            in_v2 = V2_ROOT.resolve() in rp.parents
        except Exception:
            pass
        return {"path": str(p), "exists": p.exists(),
                "in_package": bool(in_pkg), "in_frozen_v2": bool(in_v2),
                "required_for": required}

    return {
        "arc_space_npz": entry(ARC_SPACE_NPZ, "all entry points"),
        "l2_reference_csv": entry(L2_REFERENCE_CSV, "rul, prognostic, twin, comparators"),
        "l3_windows_csv": entry(L3_WINDOWS_CSV, "l2sp, comparators"),
        "split_manifest_csv": entry(SPLIT_MANIFEST_CSV, "rul, prognostic, twin"),
        "generation_config_json": entry(GENERATION_CONFIG_JSON, "l2sp"),
        "source_model_npz": entry(SOURCE_MODEL_NPZ, "l2sp, pretrain, comparators"),
        "public_source_dir": entry(PUBLIC_SOURCE_DIR, "pretrain"),
        "afterstk_l1_csv": entry(L1_TELEMETRY_CSV, "twin.verify_chain ONLY (not shipped)"),
        "stress_contract_json": entry(STRESS_CONTRACT_JSON,
                                      "docs only, already restated (not shipped)"),
    }


#: Inputs a shipped bundle MUST resolve locally (in v3 or in the frozen v2 beside it).
#: ``afterstk`` and the stress contract are deliberately excluded — see the comments on
#: those constants.
SHIPPED_REQUIRED = ("arc_space_npz", "l2_reference_csv", "l3_windows_csv",
                    "split_manifest_csv", "generation_config_json",
                    "source_model_npz", "public_source_dir")


def main() -> int:
    """``python -m src.paths`` — print where every input resolved."""
    prov = input_provenance()
    print("input resolution report")
    print("=" * 76)
    for k, v in prov.items():
        loc = ("in-package" if v["in_package"]
               else "frozen-v2" if v["in_frozen_v2"]
               else "research-tree")
        flag = "OK " if v["exists"] else "MISSING"
        star = "*" if k in SHIPPED_REQUIRED else " "
        print(f" {star}{flag:8s} {loc:14s} {k}")
        print(f"           {v['path']}")
    missing = [k for k in SHIPPED_REQUIRED if not prov[k]["exists"]]
    external = [k for k in SHIPPED_REQUIRED
                if prov[k]["exists"] and not (prov[k]["in_package"]
                                              or prov[k]["in_frozen_v2"])]
    print("=" * 76)
    print(f"  (*) required for a shipped bundle")
    print(f"  missing required        : {missing or 'none'}")
    print(f"  resolved OUTSIDE bundle : {external or 'none'}")
    if not missing and not external:
        print("  RESULT : SELF_CONTAINED (bundle carries every required input)")
    else:
        print("  RESULT : NOT_SELF_CONTAINED")
    return 0 if (not missing and not external) else 1


if __name__ == "__main__":
    raise SystemExit(main())
