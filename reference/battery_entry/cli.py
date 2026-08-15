"""cli.py — the unified command line entry point.

    python -m battery_entry verify
    python -m battery_entry reproduce --mode quick --output <dir>
    python -m battery_entry reproduce --mode full  --output <dir>
    python -m battery_entry predict --input <csv> --output <csv>

Exit codes:

  0  success
  1  a check failed / a run did not complete
  2  the input was refused by the schema validator
  3  predictions succeeded but a path did not reach EOL inside 112 days
     (NO_CROSSING_WITHIN_FORECAST_HORIZON — a real answer, not a failure)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from battery_entry import FROZEN_STATUS, __version__

BANNER = (
    "battery_release_v2 — STK-transfer battery SOH prediction\n"
    "  point model        : target_only_arc_space\n"
    "  transfer mechanism : FEATURE_SPACE_TRANSFER_WITH_TARGET_DOMAIN_HEAD_REFIT\n"
    "  evidence domain    : SIMULATION_ONLY (STK-derived twin; NOT satellite accuracy)\n"
    "  status             : CANDIDATE_NOT_PROMOTED\n")


def cmd_verify(args: argparse.Namespace) -> int:
    from battery_entry.verify import run_all

    res = run_all()
    print(BANNER)
    print("verify")
    print("=" * 72)
    for g in res["groups"]:
        mark = "PASS" if g.get("ok") else "FAIL"
        print(f"  [{mark}] {g['name']}")
        if not g.get("ok"):
            if g.get("error"):
                print(f"         error: {g['error']}")
            for k in ("failed", "mismatched", "missing", "not_declared",
                      "missing_from_disk", "unlisted_on_disk", "unparseable"):
                v = g.get(k)
                if v:
                    print(f"         {k}: {v[:10]}")
            if g.get("hits"):
                for h in g["hits"][:10]:
                    print(f"         hit: {h}")
    print("=" * 72)
    print(f"  groups passed : {res['n_passed']}/{res['n_groups']}")
    if res["failed_groups"]:
        print(f"  failed        : {res['failed_groups']}")
    print(f"  RESULT        : {'VERIFY_OK' if res['ok'] else 'VERIFY_FAILED'}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(res, indent=2, default=str), encoding="utf-8")
        print(f"  json          : {args.json_out}")
    return 0 if res["ok"] else 1


def cmd_reproduce(args: argparse.Namespace) -> int:
    from battery_entry.reproduce import run_full, run_quick

    out = Path(args.output).resolve()
    print(BANNER)
    print(f"reproduce --mode {args.mode} --output {out}")
    print("=" * 72)

    if args.mode == "quick":
        res = run_quick(out)
        steps = res["steps"]
        f = steps["01_feature_recomputation"]
        print(f"  features recomputed   : {f['n_rows']} rows x {f['n_features']} dims, "
              f"max|diff| = {f['max_abs_diff']:.3e} "
              f"(atol {f['atol']:.0e}) -> {'OK' if f['ok'] else 'FAIL'}")
        g = steps["02_golden_point_prediction_replay"]
        print(f"  golden replay         : {g['n_cells']} cells, "
              f"{g['n_bit_exact']} bit-exact, worst |diff| = {g['worst_abs_diff']:.3e} "
              f"(rtol 0, atol {g['atol']:.0e}) -> {'OK' if g['ok'] else 'FAIL'}")
        c = steps["04_conformal_coverage_reapplied"]
        print(f"  coverage re-applied   : {len(c['rows'])} (coverage, horizon) cells "
              f"[{FROZEN_STATUS['CONFORMAL_GATE']}]")
        print(f"  RUL                   : {FROZEN_STATUS['rul_evidence']}")
        print(f"  elapsed               : {res['elapsed_seconds']}s "
              f"(target <= {res['target_seconds']}s, "
              f"within = {res['within_target_runtime']})")
        print("=" * 72)
        print(f"  RESULT : {'QUICK_OK' if res['ok'] else 'QUICK_FAILED'}")
        return 0 if res["ok"] else 1

    res = run_full(out, solver=args.solver)
    print(f"  refit                 : {res['refit']}")
    print(f"  arms                  : {res['arms_run']}")
    print(f"  outer cells           : {res['n_outer_cells']}")
    print(f"  point predictions     : {res['n_point_predictions']}")
    print(f"  interval cells        : {res['n_interval_cells']}")
    for r in res["matched_horizon_summary"]:
        print(f"    {r['arm']:<20} family_macro_mae = {r['family_macro_mae']:.6f} "
              f"({r['n_cells']} cells)")
    print(f"  vs reference          : max |d family_macro_mae| = "
          f"{res['max_abs_family_macro_mae_deviation_vs_reference']:.3e}")
    print(f"  elapsed               : {res['elapsed_seconds']}s")
    print("=" * 72)
    print(f"  RESULT : {'FULL_OK' if res['ok'] else 'FULL_FAILED'}")
    return 0 if res["ok"] else 1


def cmd_predict(args: argparse.Namespace) -> int:
    from battery_entry.predict import run_predict

    code, summary = run_predict(
        args.input, args.output,
        with_intervals=args.intervals, with_rul=args.rul,
        coverage=args.coverage, solver=args.solver)

    print(BANNER)
    print("predict")
    print("=" * 72)
    if summary["status"] != "OK":
        print(f"  status : {summary['status']}")
        print(f"  error  : {summary['error']}")
        print("=" * 72)
        return code

    print(f"  batteries             : {summary['n_batteries']}")
    print(f"  output                : {summary['output']}")
    if summary["with_intervals"]:
        print(f"  intervals             : {summary['labels']['interval']} "
              f"@ nominal {summary['coverage_nominal']} "
              f"[{FROZEN_STATUS['CONFORMAL_GATE']}, per-horizon MARGINAL]")
    if summary["with_rul"]:
        print(f"  RUL                   : {summary['labels']['rul']}")
        print(f"  no crossing in 112 d  : {summary['n_no_crossing_within_112d']} "
              f"-> exit code 3 if > 0 (nothing extrapolated)")
    print(f"  evidence domain       : {summary['evidence_domain']}")
    print("=" * 72)
    print(f"  RESULT : {'PREDICT_OK' if code == 0 else f'PREDICT_EXIT_{code}'}")
    return code


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m battery_entry",
        description="STK-transfer battery SOH prediction — self-contained release.")
    ap.add_argument("--version", action="version", version=f"battery_entry {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verify", help="check manifest, SHA256, isolation, model config")
    v.add_argument("--json-out", default=None, help="also write the result as JSON")
    v.set_defaults(func=cmd_verify)

    r = sub.add_parser("reproduce", help="quick or full reproduction")
    r.add_argument("--mode", choices=("quick", "full"), required=True)
    r.add_argument("--output", required=True, help="output directory (created if absent)")
    r.add_argument("--solver", choices=("numpy", "torch"), default="numpy",
                   help="ridge solve path; both solve the same normal equations")
    r.set_defaults(func=cmd_reproduce)

    p = sub.add_parser("predict", help="predict from an observed-SOH history CSV")
    p.add_argument("--input", required=True, help="input CSV (see examples/)")
    p.add_argument("--output", required=True, help="output CSV")
    p.add_argument("--intervals", action="store_true",
                   help="also emit EXPERIMENTAL_SIMULATION_INTERVAL bounds")
    p.add_argument("--rul", action="store_true",
                   help="also emit the FINITE_HORIZON_EVIDENCE_INSUFFICIENT RUL read-out")
    p.add_argument("--coverage", type=float, default=0.90, choices=(0.90, 0.95),
                   help="nominal coverage; only calibrated levels are accepted")
    p.add_argument("--solver", choices=("numpy", "torch"), default="numpy")
    p.set_defaults(func=cmd_predict)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
