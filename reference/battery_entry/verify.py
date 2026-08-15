"""verify.py — the ``verify`` subcommand's checks.

Six groups of checks, each answering a question a reader would otherwise have to take
on trust:

1. **manifest present and well formed** — ``release_manifest.json`` declares every
   shipped file, its role and its sha256;
2. **every SHA256 matches** — recomputed from disk, not read from a cache;
3. **no parent-project dependency** — greps every shipped text file for the parent
   project's paths, for ``parents[3]``-style escapes, and for absolute host paths. This
   is the check that makes "self-contained" a measurement rather than a claim;
4. **model / space / feature-order consistency** — the 11 feature names, their order,
   the clip/scaler array lengths and their sha256 all agree with the shipped schema and
   with the frozen source manifest values;
5. **schemas parse and agree with the code** — the declared input contract matches what
   ``schema.py`` actually enforces;
6. **the frozen verdicts are present and unmodified** — the gate strings ship as data,
   so a report that omitted or softened one would fail here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from battery_entry import FROZEN_STATUS, OUTPUT_LABELS
from battery_entry import paths as P
from battery_entry.conformal import load_quantiles
from battery_entry.features import (
    FEATURE_NAMES, FROZEN_COLUMNS, feature_names_sha256)
from battery_entry.model import load_frozen_space

#: Text extensions that are grepped for forbidden path references.
TEXT_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".csv", ".cfg",
                 ".toml", ".lock", ".dockerfile", ".dockerignore", ""}


def _patterns() -> dict[str, Any]:
    """Load the pattern definitions from data.

    They live in ``schemas/forbidden_patterns.json`` rather than in this module for a
    concrete reason: a checker whose own source contains its own patterns matches itself,
    which makes the check circular and impossible to satisfy honestly. Keeping them in
    data means this file has nothing for them to match.
    """
    return json.loads(
        (P.SCHEMA_DIR / "forbidden_patterns.json").read_text(encoding="utf-8"))


def _strip_python_prose(src: str) -> str:
    """Blank out docstrings and comments, leaving executable code in place.

    A docstring that says "copied from src/production_model/features.py" is provenance;
    an import of that module is a dependency. Only the latter can affect behaviour, so
    only the latter is scanned by the code-level checks. Line numbering is preserved so
    reported line numbers still point at the real source.
    """
    import io
    import tokenize

    out = src.splitlines(keepends=True)
    blanked = list(out)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src

    prev_type = tokenize.NEWLINE
    for tok in toks:
        drop = False
        if tok.type == tokenize.COMMENT:
            drop = True
        elif tok.type == tokenize.STRING and prev_type in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.DEDENT,
                tokenize.ENCODING):
            drop = True          # a bare string statement == a docstring
        if drop:
            (r0, c0), (r1, c1) = tok.start, tok.end
            for r in range(r0 - 1, min(r1, len(blanked))):
                line = blanked[r]
                start = c0 if r == r0 - 1 else 0
                end = c1 if r == r1 - 1 else len(line.rstrip("\r\n"))
                keep_nl = line[len(line.rstrip("\r\n")):]
                body = line.rstrip("\r\n")
                body = body[:start] + " " * max(0, end - start) + body[end:]
                blanked[r] = body + keep_nl
        if tok.type not in (tokenize.NL, tokenize.COMMENT):
            prev_type = tok.type
    return "".join(blanked)


def _forbidden_imports(tree: Any, roots: set[str]) -> list[tuple[int, str]]:
    """Every import of a forbidden top-level package, found at the AST level.

    AST rather than regex, so a conditional, aliased, nested or ``importlib`` import is
    caught the same as a plain one.
    """
    import ast

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in roots:
                    hits.append((node.lineno, f"import {a.name}"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.split(".")[0] in roots:
                hits.append((node.lineno, f"from {mod} import ..."))
        elif isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name in ("import_module", "__import__"):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if arg.value.split(".")[0] in roots:
                            hits.append((node.lineno, f"dynamic import {arg.value!r}"))
    return hits


def _parent_path_used_in_io(tree: Any, rxs: list[Any]) -> list[tuple[int, str]]:
    """Any file-opening call whose argument embeds a parent-tree path literal.

    This is the check that actually matters. A parent path in a docstring is
    documentation; a parent path handed to ``open`` or ``pd.read_csv`` is a dependency.
    Walking the AST catches the second while ignoring the first.
    """
    import ast

    pats = _patterns()
    io_names = set(pats["file_opening_calls"])
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "attr", None) or getattr(fn, "id", None)
        if name not in io_names:
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    for rx in rxs:
                        if rx.search(sub.value):
                            hits.append((node.lineno,
                                         f"{name}(... {sub.value[:60]!r} ...)"))
    return hits


def _is_text(p: Path) -> bool:
    return p.suffix.lower() in TEXT_SUFFIXES


def check_manifest() -> dict[str, Any]:
    """The manifest exists, parses, and covers exactly the shipped files."""
    out: dict[str, Any] = {"name": "release_manifest"}
    if not P.RELEASE_MANIFEST.exists():
        out.update(ok=False, error="release_manifest.json is missing")
        return out
    m = json.loads(P.RELEASE_MANIFEST.read_text(encoding="utf-8"))
    declared = {f["path"] for f in m.get("files", [])}
    on_disk = {P.relpath(p) for p in P.package_files()}
    # the manifest cannot list itself with its own hash
    on_disk.discard("release_manifest.json")
    declared.discard("release_manifest.json")
    out["n_declared"] = len(declared)
    out["n_on_disk"] = len(on_disk)
    out["missing_from_disk"] = sorted(declared - on_disk)
    out["not_declared"] = sorted(on_disk - declared)
    out["ok"] = not out["missing_from_disk"] and not out["not_declared"]
    for k in ("package", "version", "status", "point_model", "transfer_mechanism"):
        out[k] = m.get(k)
    return out


def check_sha256sums() -> dict[str, Any]:
    """Every shipped file's sha256 matches ``SHA256SUMS``, recomputed from disk."""
    out: dict[str, Any] = {"name": "sha256sums"}
    if not P.SHA256SUMS.exists():
        out.update(ok=False, error="SHA256SUMS is missing")
        return out
    want: dict[str, str] = {}
    for line in P.SHA256SUMS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            out.update(ok=False, error=f"malformed SHA256SUMS line: {line!r}")
            return out
        h, rel = parts[0], parts[1].lstrip("*").strip()
        want[rel] = h

    mismatched, missing = [], []
    for rel, h in sorted(want.items()):
        p = P.PACKAGE_ROOT / rel
        if not p.exists():
            missing.append(rel)
        elif P.sha256_file(p) != h:
            mismatched.append(rel)
    on_disk = {P.relpath(p) for p in P.package_files()}
    out["n_checked"] = len(want)
    out["mismatched"] = mismatched
    out["missing"] = missing
    out["unlisted_on_disk"] = sorted(on_disk - set(want))
    out["ok"] = not mismatched and not missing and not out["unlisted_on_disk"]
    return out


def check_no_parent_dependency() -> dict[str, Any]:
    """No shipped file *depends on* the parent project, a host path or an STK install.

    Four layers, in increasing order of what they prove:

    1. **hard failures** — an absolute host path, an STK install path, a ``parents[N>=1]``
       escape or a ``sys.path`` manipulation reaching outside. Scanned on code with
       docstrings and comments stripped, because these are only harmful when executed.
    2. **forbidden imports** — any import of the research project's packages, found at the
       AST level so conditional, aliased and dynamic imports are caught too.
    3. **parent paths actually USED for I/O** — a file-opening call whose argument embeds a
       parent-tree path literal. This has no exemptions of any kind.
    4. **prose mentions** — a relative parent path in a docstring, comment or markdown
       file. These are *counted and reported*, not failed: such a path cannot resolve from
       inside this package, so it is manifestly documentation recording where a frozen
       input came from. Naming provenance is a virtue, not a dependency.

    The distinction matters. Failing layer 4 would force the package to lie about its own
    lineage; failing layers 1-3 is what makes "self-contained" true.
    """
    import ast

    pats = _patterns()
    out: dict[str, Any] = {"name": "no_parent_dependency"}

    hard = [(re.compile(p["pattern"]), p["why"], p.get("scan", "raw"))
            for p in pats["hard_failure_patterns"]]
    parent_rxs = [re.compile(p["pattern"])
                  for p in pats["parent_tree_path_patterns"]]
    parent_why = {p["pattern"]: p["why"] for p in pats["parent_tree_path_patterns"]}
    import_roots = set(pats["forbidden_import_roots"])
    self_exempt = set(pats["self_exempt_files"])
    doc_exempt = set(pats["documentation_files_exempt_from_prose_scan"])

    hard_hits: list[dict[str, Any]] = []
    import_hits: list[dict[str, Any]] = []
    io_hits: list[dict[str, Any]] = []
    prose_mentions: list[dict[str, Any]] = []
    n_py = n_text = 0

    for p in P.package_files():
        rel = P.relpath(p)
        if rel in self_exempt:
            continue
        if not _is_text(p):
            continue
        try:
            raw = p.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        n_text += 1
        is_py = p.suffix.lower() == ".py"
        code = _strip_python_prose(raw) if is_py else raw

        # ---- layer 1: hard failures. `scan` says whether prose counts for this one.
        for rx, why, scope in hard:
            target = code if scope == "code" else raw
            for m in rx.finditer(target):
                hard_hits.append({"file": rel,
                                  "line": target[:m.start()].count("\n") + 1,
                                  "match": m.group(0)[:80], "why": why})

        if is_py:
            n_py += 1
            try:
                tree = ast.parse(raw)
            except SyntaxError as e:
                hard_hits.append({"file": rel, "line": e.lineno or 0,
                                  "match": "<syntax error>", "why": str(e)})
                continue
            # ---- layer 2: forbidden imports
            for line, what in _forbidden_imports(tree, import_roots):
                import_hits.append({"file": rel, "line": line, "match": what,
                                    "why": pats["forbidden_import_reason"]})
            # ---- layer 3: parent paths used for I/O
            for line, what in _parent_path_used_in_io(tree, parent_rxs):
                io_hits.append({"file": rel, "line": line, "match": what,
                                "why": "a parent-tree path handed to a file-opening call"})

        # ---- layer 4: prose mentions, counted only
        if rel not in doc_exempt:
            scan = code if is_py else raw
            for rx in parent_rxs:
                for m in rx.finditer(scan):
                    prose_mentions.append({
                        "file": rel, "line": scan[:m.start()].count("\n") + 1,
                        "match": m.group(0)[:80],
                        "why": parent_why.get(rx.pattern, "parent-tree path mention")})

    out["n_files_scanned"] = n_text
    out["n_python_files_scanned"] = n_py
    out["hard_failures"] = hard_hits
    out["forbidden_imports"] = import_hits
    out["parent_paths_used_for_io"] = io_hits
    out["prose_mentions_counted_not_failed"] = len(prose_mentions)
    out["prose_mention_examples"] = prose_mentions[:10]
    out["policy"] = pats["parent_tree_path_policy"]
    out["policy_detail"] = pats["parent_tree_path_policy_detail"]
    out["hits"] = hard_hits + import_hits + io_hits
    out["n_hits"] = len(out["hits"])
    out["ok"] = not out["hits"]
    return out


def check_model_consistency() -> dict[str, Any]:
    """Feature names, order, clip/scaler shapes and the frozen hashes all agree."""
    out: dict[str, Any] = {"name": "model_consistency"}
    space = load_frozen_space()
    schema = json.loads(
        (P.SCHEMA_DIR / "feature_schema_v2.json").read_text(encoding="utf-8"))

    checks: dict[str, bool] = {}
    checks["feature_count_is_11"] = len(FEATURE_NAMES) == 11
    checks["code_order_matches_space"] = tuple(space.feature_names) == tuple(FEATURE_NAMES)
    checks["code_order_matches_schema"] = (
        tuple(schema["feature_names"]) == tuple(FEATURE_NAMES))
    checks["frozen_column_order_matches"] = (
        tuple(schema["frozen_columns"]) == tuple(FROZEN_COLUMNS))
    checks["feature_names_sha256_matches_schema"] = (
        schema["feature_names_sha256"] == feature_names_sha256())

    for nm, arr in (("clip_lo", space.clip_lo), ("clip_hi", space.clip_hi),
                    ("scaler_mean", space.scaler_mean),
                    ("scaler_std", space.scaler_std)):
        checks[f"{nm}_length_11"] = int(np.asarray(arr).size) == 11
        checks[f"{nm}_finite"] = bool(np.isfinite(np.asarray(arr)).all())
    checks["clip_lo_below_clip_hi"] = bool(np.all(space.clip_lo <= space.clip_hi))
    checks["scaler_std_positive"] = bool(np.all(space.scaler_std > 0))

    import hashlib
    clip_sha = hashlib.sha256(
        np.asarray(space.clip_lo, dtype=float).tobytes()
        + np.asarray(space.clip_hi, dtype=float).tobytes()).hexdigest()
    scaler_sha = hashlib.sha256(
        np.asarray(space.scaler_mean, dtype=float).tobytes()
        + np.asarray(space.scaler_std, dtype=float).tobytes()).hexdigest()
    checks["clip_sha256_matches_frozen"] = clip_sha == space.clip_sha256
    checks["scaler_sha256_matches_frozen"] = scaler_sha == space.scaler_sha256
    checks["space_is_arc_frozen"] = space.name == "ARC_FROZEN_SPACE"
    checks["source_model_is_arc_clean_fixed"] = space.source_model_id == "arc_clean_fixed"

    # frozen alphas present for every cell the package claims to reproduce
    alpha = json.loads(P.FROZEN_ALPHA_JSON.read_text(encoding="utf-8"))
    checks["frozen_alpha_lofo_18_cells"] = len(alpha["lofo"]) == 18
    checks["frozen_alpha_fixed_3_horizons"] = len(alpha["fixed_holdout"]) == 3
    checks["alpha_not_selected_here"] = alpha["alpha_selected_in_this_package"] is False

    # conformal half-widths present for every (protocol, arm, coverage, horizon)
    q = load_quantiles()
    checks["conformal_gate_is_not_validated"] = q.gate_status == "CONFORMAL_NOT_VALIDATED"
    checks["conformal_per_horizon_marginal_only"] = q.per_horizon_marginal_only
    checks["conformal_no_simultaneous_claim"] = not q.simultaneous_coverage_claimed
    checks["conformal_half_widths_24_cells"] = len(q.table) == 24

    out["checks"] = checks
    out["failed"] = sorted(k for k, v in checks.items() if not v)
    out["ok"] = not out["failed"]
    out["feature_names"] = list(FEATURE_NAMES)
    out["feature_names_sha256"] = feature_names_sha256()
    out["space_model_npz_sha256"] = space.model_npz_sha256
    return out


def check_schemas() -> dict[str, Any]:
    """Every shipped schema parses, and the input schema matches what code enforces."""
    from battery_entry.schema import (
        FORBIDDEN_INPUT_COLUMNS, OPTIONAL_COLUMNS, REQUIRED_COLUMNS)

    out: dict[str, Any] = {"name": "schemas"}
    parsed, bad = [], []
    for p in sorted(P.SCHEMA_DIR.glob("*.json")):
        try:
            json.loads(p.read_text(encoding="utf-8"))
            parsed.append(P.relpath(p))
        except json.JSONDecodeError as e:
            bad.append({"file": P.relpath(p), "error": str(e)})
    out["parsed"] = parsed
    out["unparseable"] = bad

    s = json.loads(
        (P.SCHEMA_DIR / "predict_input_schema.json").read_text(encoding="utf-8"))
    checks = {
        "required_columns_match_code": (
            tuple(s["required_columns"]) == tuple(REQUIRED_COLUMNS)),
        "optional_columns_match_code": (
            tuple(s["optional_columns"]) == tuple(OPTIONAL_COLUMNS)),
        "forbidden_columns_match_code": (
            tuple(s["forbidden_columns"]) == tuple(FORBIDDEN_INPUT_COLUMNS)),
        "context_length_is_20": s["context_length"] == 20,
        "soh_upper_bound_declared": s["soh_range"]["max_inclusive"] == 1.5,
    }
    out["checks"] = checks
    out["failed"] = sorted(k for k, v in checks.items() if not v)
    out["ok"] = not bad and not out["failed"]
    return out


def check_frozen_verdicts() -> dict[str, Any]:
    """The Phase 3/4/5/6 verdicts ship verbatim and are unmodified."""
    out: dict[str, Any] = {"name": "frozen_verdicts"}
    claims = json.loads(P.CLAIMS_MANIFEST.read_text(encoding="utf-8"))
    want = {
        "point_model": "target_only_arc_space",
        "high_stability_baseline": "observable_local_trend_extrapolation",
        "transfer_mechanism": "FEATURE_SPACE_TRANSFER_WITH_TARGET_DOMAIN_HEAD_REFIT",
        "phase3_verdict": "SOURCE_PRIOR_NOT_VALIDATED",
        "TIME_NORMALIZATION_GATE": "RATE_DAY_REPARAMETERIZATION_ONLY",
        "EXPOSURE_GATE": "EXPOSURE_INPUT_NOT_AVAILABLE",
        "STRESS_GATE_LOW_DATA": "STRESS_NOT_VALIDATED",
        "STRESS_GATE_FULL_DATA": "STRESS_NOT_VALIDATED",
        "CONFORMAL_GATE": "CONFORMAL_NOT_VALIDATED",
        "rul_evidence": "RUL_EVIDENCE_INSUFFICIENT",
        "phase6": "PHASE6_CLOSED_WITHOUT_TRAINING",
        "patchtst": "PATCHTST_SKIPPED_BY_PREDEFINED_GATE",
        "candidate_status": "CANDIDATE_NOT_PROMOTED",
        "evidence_domain": "SIMULATION_ONLY",
    }
    checks = {f"code_{k}": FROZEN_STATUS.get(k) == v for k, v in want.items()}
    for k, v in want.items():
        checks[f"manifest_{k}"] = claims["frozen_status"].get(k) == v
    checks["interval_label"] = (
        OUTPUT_LABELS["interval"] == "EXPERIMENTAL_SIMULATION_INTERVAL")
    checks["rul_label"] = (
        OUTPUT_LABELS["rul"] == "FINITE_HORIZON_EVIDENCE_INSUFFICIENT")
    checks["patchtst_not_run"] = claims["patchtst"]["was_run"] is False
    checks["patchtst_not_implemented"] = claims["patchtst"]["was_implemented"] is False
    checks["patchtst_not_a_failure"] = (
        claims["patchtst"]["is_experiment_failure"] is False)
    checks["no_promotion_claimed"] = claims["is_promoted"] is False
    checks["no_production_v2_claimed"] = claims["creates_production_v2"] is False

    out["checks"] = checks
    out["failed"] = sorted(k for k, v in checks.items() if not v)
    out["ok"] = not out["failed"]
    return out


def check_no_neural_implementation() -> dict[str, Any]:
    """PatchTST / TimesFM / any neural sequence model is neither shipped nor imported.

    Phase 6 was closed WITHOUT training, so the absence of a model is CHECKED, not
    asserted. Scanned on executable code with docstrings and comments stripped: a
    docstring stating "PatchTST was not run" must not itself trip the detector, while an
    actual instantiation, layer, autograd call or optimiser does.
    """
    pats = _patterns()
    out: dict[str, Any] = {"name": "no_neural_implementation"}
    compiled = [(re.compile(p["pattern"]), p["why"]) for p in pats["neural_patterns"]]
    self_exempt = set(pats["self_exempt_files"])

    hits = []
    n_scanned = 0
    for p in sorted(P.PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = P.relpath(p)
        if rel in self_exempt:
            continue
        raw = p.read_text(encoding="utf-8", errors="ignore")
        code = _strip_python_prose(raw)
        n_scanned += 1
        for rx, why in compiled:
            for m in rx.finditer(code):
                hits.append({"file": rel, "match": m.group(0), "why": why,
                             "line": code[:m.start()].count("\n") + 1})
    out["n_python_files_scanned"] = n_scanned
    out["scan_scope"] = pats["neural_scan_scope"]
    out["hits"] = hits
    out["n_hits"] = len(hits)
    out["note"] = pats["neural_permitted"]
    out["ok"] = not hits
    return out


def run_all() -> dict[str, Any]:
    """Run every verification group and return the combined result."""
    groups = [
        check_manifest(),
        check_sha256sums(),
        check_no_parent_dependency(),
        check_model_consistency(),
        check_schemas(),
        check_frozen_verdicts(),
        check_no_neural_implementation(),
    ]
    return {
        "groups": groups,
        "n_groups": len(groups),
        "n_passed": sum(1 for g in groups if g.get("ok")),
        "failed_groups": [g["name"] for g in groups if not g.get("ok")],
        "ok": all(g.get("ok") for g in groups),
    }
