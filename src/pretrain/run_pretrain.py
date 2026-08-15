"""run_pretrain.py — T3(A): source-domain pretraining on the public dataset.

    python -m src.pretrain.run_pretrain

Trains a ``LinearHead(nn.Module)`` with AdamW + early stopping on the NASA PCoE
public degradation data, and reports the clip/scaler it fits alongside the frozen
ones v2 ships.

WHAT IS DELIVERED VS WHAT IS DEMONSTRATED — read this before quoting anything
----------------------------------------------------------------------------
The DELIVERED transfer chain uses:
  * the FROZEN clip/scaler from ``battery_release_v2/models/arc_clean_fixed_space.npz``
  * the FROZEN source coefficients from the artifact whose SHA-256 v2 records

Both were produced by the original audited source fit. This script does NOT replace
either. It re-runs the source-domain training in PyTorch so that the "public dataset
-> PyTorch pretraining" step is executable and inspectable rather than asserted, and
it reports how close a fresh fit lands to the frozen one.

A fresh fit will NOT reproduce the frozen numbers bit-for-bit, and that is expected:
the frozen fit used the full original preprocessing chain (its own cell selection,
cycle cleaning, and horizon layout), which the archive does not fully record. Any
gap is reported as a measured difference, never papered over. The frozen artifacts
remain the ones used downstream.
"""
from __future__ import annotations

import json

import numpy as np

from src import paths as P
from src.data.source_dataset import build_windows, summarise
from src.features.space import load_frozen_space, load_source_prior
from src.finetune import _runtime as RT
from src.finetune.l2sp import (LinearHead, solve_l2sp_closed_form,
                               train_l2sp_gradient)

#: Quantile clip bounds. 1%-99% is the rule the archive's tech scheme records for the
#: source model ("冻结公开数据源模型和 1%–99% 裁剪边界").
CLIP_Q = (1.0, 99.0)

#: Source-domain ridge strength. Taken from the frozen source model's own recorded
#: ``alpha`` so the fresh fit is solving the comparable problem, not a retuned one.
HORIZON_CYCLES = 8


def fit_clip_scaler(X: np.ndarray) -> dict:
    """Fit clip bounds and a z-scaler on the SOURCE design — train-only by construction.

    This is what "the coordinate system is fitted on the source domain" means
    concretely. Downstream it is APPLIED to target rows, never refitted there.
    """
    lo = np.percentile(X, CLIP_Q[0], axis=0)
    hi = np.percentile(X, CLIP_Q[1], axis=0)
    Xc = np.clip(X, lo, hi)
    mean = Xc.mean(axis=0)
    std = Xc.std(axis=0)
    std[std == 0] = 1.0
    return {"clip_lo": lo, "clip_hi": hi, "scaler_mean": mean, "scaler_std": std}


def main() -> int:
    RT.seed_everything(0)
    space = load_frozen_space()
    prior = load_source_prior()

    print("=" * 76)
    print("T3(A)  source-domain pretraining, PyTorch, on the PUBLIC dataset")
    print("=" * 76)
    print(f"  runtime            : {json.dumps(RT.runtime_report())}")
    print()

    d = build_windows(horizon=HORIZON_CYCLES)
    s = summarise(d)
    print(f"  dataset            : {s['dataset']}")
    print(f"  source             : {d['provenance']}")
    print(f"  windows            : {s['n_windows']}  from {s['n_cells_used']} cells "
          f"({s['n_cells_skipped']} skipped: fewer than "
          f"{d['context_length'] + d['horizon_cycles']} usable cycles)")
    print(f"  context length     : {s['context_length']} points (frozen route context)")
    print(f"  horizon            : {s['horizon_cycles']} CYCLES (target domain uses "
          f"14-day steps — different units, see module docstring)")
    print(f"  label              : {d['label']}")
    print(f"  label mean/std     : {s['label_mean']:.6f} / {s['label_std']:.6f}")
    print()

    X, y = d["X"], d["y"]

    # ---- the coordinate system, fitted on source ----
    fitted = fit_clip_scaler(X)
    Zf = (np.clip(X, fitted["clip_lo"], fitted["clip_hi"])
          - fitted["scaler_mean"]) / fitted["scaler_std"]

    print("  clip/scaler fitted on source (1%-99% quantile clip, then z-score)")
    print("  comparison with the FROZEN space v2 ships (which stays the delivered one):")
    for k in ("clip_lo", "clip_hi", "scaler_mean", "scaler_std"):
        a, b = fitted[k], getattr(space, k)
        print(f"    {k:<12s} max|fresh - frozen| = {np.max(np.abs(a - b)):.6g}")
    print("    (differences expected: the frozen fit used the original full")
    print("     preprocessing chain, which the archive does not fully record.")
    print("     The FROZEN arrays remain the ones used downstream.)")
    print()

    # ---- PyTorch training, with the closed form as the known target ----
    w = np.ones(len(y))
    alpha = float(prior.alpha)
    run = train_l2sp_gradient(Zf, y, w, beta_src=np.zeros(X.shape[1]),
                              alpha=alpha, lambda_sp=0.0,
                              max_epochs=20000, patience=800)
    closed = solve_l2sp_closed_form(Zf, y, w, np.zeros(X.shape[1]), alpha, 0.0)
    gap = float(np.max(np.abs(run.beta - closed)))

    print(f"  PyTorch head       : LinearHead(nn.Module), AdamW, early stopping")
    print(f"    alpha            : {alpha} (from the frozen source model's own record)")
    print(f"    epochs run       : {run.n_epochs}  best epoch {run.best_epoch}  "
          f"early_stopped={run.early_stopped}")
    print(f"    objective        : {run.best_objective:.12f}")
    print(f"    max|gradient - closed form| : {gap:.3e}   "
          f"({'PASS' if gap < 1e-6 else 'FAIL'} at 1e-6)")
    print()

    # ---- how the fresh head compares with the frozen source head ----
    beta_frozen, b_frozen = prior.for_horizon(8)
    print("  fresh source head vs FROZEN source head (beta_src used by L2-SP):")
    print(f"    ||beta_fresh||   = {np.linalg.norm(run.beta):.6f}")
    print(f"    ||beta_frozen||  = {np.linalg.norm(beta_frozen):.6f}")
    print(f"    cosine similarity = "
          f"{float(run.beta @ beta_frozen / (np.linalg.norm(run.beta) * np.linalg.norm(beta_frozen))):.6f}")
    print("    NOT expected to match: different preprocessing chain and different")
    print("    horizon units (cycles vs 14-day steps). Reported, not reconciled.")
    print()

    train_mae = float(np.abs(Zf @ run.beta - y).mean())
    print(f"  in-sample source MAE (delta scale) : {train_mae:.6f}")
    print("  NOTE: in-sample. The source domain has no held-out claim attached to it;")
    print("        every generalisation number in this package is target-domain LOFO.")
    print()

    # ------------------------------------------------------------------ outputs
    out = P.RESULTS_DIR / "pretrain_source"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "source_pretrained.npz",
        beta_pytorch=run.beta, beta_closed_form=closed,
        clip_lo=fitted["clip_lo"], clip_hi=fitted["clip_hi"],
        scaler_mean=fitted["scaler_mean"], scaler_std=fitted["scaler_std"],
        feature_names=np.array(space.feature_names, dtype=object),
        alpha=alpha, horizon_cycles=HORIZON_CYCLES,
        context_length=d["context_length"])

    (out / "pretrain_report.json").write_text(json.dumps({
        **s,
        "runtime": RT.runtime_report(),
        "alpha": alpha,
        "epochs_run": run.n_epochs, "best_epoch": run.best_epoch,
        "early_stopped": run.early_stopped,
        "objective": run.best_objective,
        "max_abs_gradient_minus_closed_form": gap,
        "equivalence_pass_at_1e-6": bool(gap < 1e-6),
        "in_sample_source_mae_delta": train_mae,
        "clip_rule": f"{CLIP_Q[0]}%-{CLIP_Q[1]}% quantile clip then z-score, fitted on source only",
        "clip_max_abs_diff_vs_frozen": {
            k: float(np.max(np.abs(fitted[k] - getattr(space, k))))
            for k in ("clip_lo", "clip_hi", "scaler_mean", "scaler_std")},
        "delivered_artifacts_are_the_frozen_ones": True,
        "why": ("This run demonstrates the public-dataset -> PyTorch pretraining step "
                "is executable. The DELIVERED chain uses v2's frozen clip/scaler and "
                "the frozen source coefficients (SHA-256 verified). A fresh fit is not "
                "expected to reproduce them bit-for-bit because the original full "
                "preprocessing chain is not recorded in the archive."),
        "skipped_cells": d["skipped"],
        "evidence_domain": ("PUBLIC SOURCE DOMAIN (NASA PCoE laboratory cells). "
                            "Not spaceflight data; not satellite accuracy."),
    }, indent=2), encoding="utf-8")

    print(f"  wrote {out / 'source_pretrained.npz'}")
    print(f"  wrote {out / 'pretrain_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
