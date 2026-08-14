"""deep_seq.py — the deep sequence comparator arm. NOT the delivered model.

SCOPE, STATED UP FRONT
----------------------
Phase 6's frozen verdict ``PATCHTST_SKIPPED_BY_PREDEFINED_GATE`` closed the promotion
of deep models to DELIVERED PREDICTOR. It did not, and could not, bar running one as a
COMPARISON BASELINE — which is what the competition's "compare against at least one
existing method" asks for. See ``GATES_scope_amendment.md``.

So: whatever this arm scores, the delivered predictor stays ``target_only_arc_space``.
If it wins, that is recorded as "to be re-evaluated under a new pre-registered
protocol", not promoted here. That asymmetry is deliberate and is the whole reason the
gate can stay intact while the comparison still happens.

WHY GRU AND NOT PatchTST
------------------------
Both were available (``src/models/patchtst.py`` is a reusable PatchTST, see
``REUSE_MAP.md`` 1.2). GRU is used because the input is a 20-point univariate SOH
history: PatchTST's advantage is patching long multivariate series, and at length 20
its patch embedding would see 2-5 patches, so the comparison would be dominated by
architecture overhead rather than modelling capacity. A GRU is the honest strong
sequence baseline at this sequence length. The PatchTST path remains available for a
longer-context study.

THIS ARM LIVES IN v3 BECAUSE IT MUST
------------------------------------
``verify::no_neural_implementation`` scans all of ``battery_release_v2``. A single
``nn.Module`` there would drop verify from 7/7 to 6/7 and invalidate SHA256SUMS.
"""
from __future__ import annotations

from dataclasses import dataclass

# IMPORT FIRST: float64 / CPU / single-thread determinism.
from src.finetune import _runtime as RT

import numpy as np
import torch
import torch.nn as nn

#: Declared before any run. Small because the training set is a few thousand windows of
#: a 20-point series; a larger net would overfit and the comparison would be unfair in
#: the other direction.
HIDDEN = 32
N_LAYERS = 1
LR = 5e-3
MAX_EPOCHS = 300
PATIENCE = 30
BATCH = 256


class GRURegressor(nn.Module):
    """GRU over the 20-point SOH history -> one scalar (the delta label).

    Input is the RAW SOH history, not the 11 geometry features: the point of a
    sequence model is that it derives its own representation. Handing it the frozen 11
    would make it a re-parameterised ridge and the comparison would be vacuous.
    """

    def __init__(self, hidden: int = HIDDEN, n_layers: int = N_LAYERS):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=n_layers,
                          batch_first=True, dtype=RT.DTYPE)
        self.head = nn.Linear(hidden, 1, dtype=RT.DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, L) -> (n, L, 1)
        out, _ = self.gru(x.unsqueeze(-1))
        return self.head(out[:, -1, :]).squeeze(-1)


@dataclass
class DeepRun:
    pred: np.ndarray
    n_epochs: int
    best_epoch: int
    best_val_loss: float
    n_params: int


def train_gru(ctx_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
              ctx_val: np.ndarray, y_val: np.ndarray,
              ctx_test: np.ndarray,
              seed: int = 0, max_epochs: int = MAX_EPOCHS) -> DeepRun:
    """Train with AdamW + early stopping on a held-out INNER family, predict on test.

    The validation split is an inner FAMILY, not a random row split: random rows would
    leak, because windows from one trajectory overlap heavily and a random split would
    put near-duplicate rows on both sides. This mirrors the protocol every other arm
    uses, which is the only way the comparison means anything.

    Standardisation of the SOH history uses TRAIN statistics only.
    """
    RT.seed_everything(seed)

    mu = float(ctx_train.mean())
    sd = float(ctx_train.std()) or 1.0

    def prep(a):
        return RT.as_tensor((np.asarray(a, dtype=float) - mu) / sd)

    Xtr, Xva, Xte = prep(ctx_train), prep(ctx_val), prep(ctx_test)
    ytr, yva = RT.as_tensor(y_train), RT.as_tensor(y_val)
    wtr = RT.as_tensor(w_train)
    wtr = wtr / wtr.mean()

    model = GRURegressor()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())

    n = Xtr.shape[0]
    g = torch.Generator().manual_seed(seed)
    best, best_ep, best_state, since = float("inf"), 0, None, 0
    ep = 0
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = (wtr[idx] * (model(Xtr[idx]) - ytr[idx]) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            v = float(((model(Xva) - yva) ** 2).mean().item())
        if v < best - 1e-12:
            best, best_ep, since = v, ep, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).cpu().numpy().astype(float)

    return DeepRun(pred=pred, n_epochs=ep, best_epoch=best_ep,
                   best_val_loss=best, n_params=n_params)
