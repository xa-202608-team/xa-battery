"""censored_loss.py — right-censoring-aware ridge, solved as a convex QP.

THE OBJECTIVE
-------------
Uncensored anchors get ordinary weighted squared error. Censored anchors get a
ONE-SIDED penalty: predicting a RUL *shorter* than the already-observed follow-up
is provably wrong, but predicting *longer* is not — the cell simply had not failed
yet when the mission ended::

    L(b) = sum_{i in U} w_i (y_i - z_i.b)^2
         + lam_c * sum_{j in C} w_j * max(0, c_j - z_j.b)^2
         + alpha * ||b||^2

where ``U`` is uncensored, ``C`` is censored, and ``c_j`` is the observed follow-up
duration. This is the "hinge that only penalises predicting RUL shorter than the
observed follow-up" the closeout plan asks for.

WHY AN ACTIVE-SET SOLVER, NOT GRADIENT DESCENT
----------------------------------------------
``max(0, c - p)^2`` is convex and C^1, so ``L`` is a convex piecewise-quadratic. On
any fixed active set ``A = {j in C : z_j.b < c_j}`` the objective IS an ordinary
weighted ridge in which the active censored rows behave exactly like observations
with target ``c_j``. So we alternate:

  1. solve the weighted ridge closed form on ``U + A``
  2. recompute ``A``

and stop when ``A`` stops changing. Each step solves the true objective restricted
to a face, the objective decreases monotonically, and there are finitely many active
sets — so this terminates, at the global optimum of a convex problem. It reuses the
frozen ``solve_ridge_weighted`` path rather than introducing a learning rate, a
schedule, and a convergence tolerance that would all need auditing.

:func:`fit_censored_ridge_torch` solves the SAME objective by AdamW gradient descent
and exists to prove the two agree (``tests/test_torch_matches_closed_form.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.features.protocol import solve_ridge_weighted, weighted_intercept

#: Weight on the censored hinge relative to an observed residual. 1.0 means a
#: violated censored bound costs the same as an equally-sized observed error.
#: Declared here, not tuned on results.
LAMBDA_CENSORED = 1.0

#: Active-set iteration cap. Reaching it is reported, never silently ignored.
MAX_ACTIVE_SET_ITERS = 50


@dataclass
class CensoredFit:
    coef: np.ndarray
    intercept: float
    alpha: float
    n_uncensored: int
    n_censored: int
    n_censored_active: int
    n_iters: int
    converged: bool
    objective: float


def _augment(Z: np.ndarray, y: np.ndarray, w: np.ndarray,
             Zc: np.ndarray, c: np.ndarray, wc: np.ndarray,
             active: np.ndarray, lam_c: float
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack uncensored rows with the ACTIVE censored rows retargeted to their bound."""
    if not active.any():
        return Z, y, w
    return (np.vstack([Z, Zc[active]]),
            np.concatenate([y, c[active]]),
            np.concatenate([w, lam_c * wc[active]]))


def objective(Z: np.ndarray, y: np.ndarray, w: np.ndarray,
              Zc: np.ndarray, c: np.ndarray, wc: np.ndarray,
              coef: np.ndarray, intercept: float,
              alpha: float, lam_c: float = LAMBDA_CENSORED) -> float:
    """The censoring-aware objective, for monotonicity checks and reporting."""
    r = y - (Z @ coef + intercept)
    obs = float(np.sum(w * r ** 2))
    viol = np.maximum(0.0, c - (Zc @ coef + intercept)) if len(c) else np.zeros(0)
    cen = float(lam_c * np.sum(wc * viol ** 2))
    return obs + cen + alpha * float(coef @ coef)


def _solve_ridge_with_intercept(Z: np.ndarray, y: np.ndarray, w: np.ndarray,
                                alpha: float) -> tuple[np.ndarray, float]:
    """加权岭含截距的**精确联合解**（加权中心化消去截距，coef+intercept 联立）。

    与 ``solve_ridge_weighted`` 的区别：后者注释明示 "no intercept column"——
    用未中心化 Z 解 coef，再由 ``weighted_intercept`` 后补截距。这两步**不等于**
    (coef, intercept) 联合优化：后补截距时 coef 已固定在不含截距的最优点上，
    而真联合最优要求两者同时取最优。``test_08`` 正是抓住这点——active-set 的
    目标值被 torch 联合梯度下降超过。

    本函数用加权中心化（概率权重 w/Σw 求均值，原始 w 进法方程使 α 可比）消去
    截距，所得 (coef, intercept) 是含截距加权岭的全局闭式最优。
    """
    wsum = w.sum()
    wp = w / wsum                                   # 概率权重（中心化用）
    z_bar = (Z * wp[:, None]).sum(axis=0)
    y_bar = float((y * wp).sum())
    Zc = Z - z_bar
    yc = y - y_bar
    A = Zc.T @ (Zc * w[:, None]) + alpha * np.eye(Z.shape[1])  # 原始 w（mean≈1, α 可比）
    coef = np.linalg.solve(A, Zc.T @ (yc * w))
    intercept = y_bar - float(z_bar @ coef)
    return coef, intercept


def fit_censored_ridge(Z: np.ndarray, y: np.ndarray, w: np.ndarray,
                       Zc: np.ndarray, c: np.ndarray, wc: np.ndarray,
                       alpha: float,
                       lam_c: float = LAMBDA_CENSORED) -> CensoredFit:
    """Solve the censoring-aware ridge by active-set iteration.

    ``Z, y, w``   — uncensored design, labels, weights
    ``Zc, c, wc`` — censored design, observed-follow-up lower bounds, weights
    """
    Z = np.ascontiguousarray(Z, dtype=float)
    Zc = np.ascontiguousarray(Zc, dtype=float) if len(c) else np.zeros((0, Z.shape[1]))
    active = np.zeros(len(c), dtype=bool)

    coef = np.zeros(Z.shape[1])
    intercept = 0.0
    converged = False
    it = 0
    for it in range(1, MAX_ACTIVE_SET_ITERS + 1):
        Za, ya, wa = _augment(Z, y, w, Zc, c, wc, active, lam_c)
        # 加权中心化联合解 coef+intercept（不再用 solve_ridge_weighted 无截距 + 后补）。
        coef, intercept = _solve_ridge_with_intercept(Za, ya, wa, alpha)

        new_active = ((Zc @ coef + intercept) < c) if len(c) else active
        if np.array_equal(new_active, active):
            converged = True
            break
        active = new_active

    return CensoredFit(
        coef=coef, intercept=float(intercept), alpha=float(alpha),
        n_uncensored=int(len(y)), n_censored=int(len(c)),
        n_censored_active=int(active.sum()), n_iters=it, converged=converged,
        objective=objective(Z, y, w, Zc, c, wc, coef, intercept, alpha, lam_c))


def predict(fit: CensoredFit, Z: np.ndarray) -> np.ndarray:
    return Z @ fit.coef + fit.intercept


# --------------------------------------------------------------- torch reference

def fit_censored_ridge_torch(Z: np.ndarray, y: np.ndarray, w: np.ndarray,
                             Zc: np.ndarray, c: np.ndarray, wc: np.ndarray,
                             alpha: float, lam_c: float = LAMBDA_CENSORED,
                             steps: int = 20000, lr: float = 0.05,
                             seed: int = 0) -> tuple[np.ndarray, float]:
    """The SAME objective, minimised by AdamW. Used only to validate the QP solver.

    Deterministic: float64 CPU, zero init, no data shuffling, no dropout. The seed is
    set for form only — nothing here is stochastic.
    """
    import torch

    torch.manual_seed(seed)
    dt, dev = torch.float64, torch.device("cpu")
    tZ = torch.tensor(Z, dtype=dt, device=dev)
    ty = torch.tensor(y, dtype=dt, device=dev)
    tw = torch.tensor(w, dtype=dt, device=dev)
    tZc = torch.tensor(Zc, dtype=dt, device=dev)
    tc = torch.tensor(c, dtype=dt, device=dev)
    twc = torch.tensor(wc, dtype=dt, device=dev)

    b = torch.zeros(Z.shape[1], dtype=dt, device=dev, requires_grad=True)
    b0 = torch.zeros(1, dtype=dt, device=dev, requires_grad=True)
    opt = torch.optim.AdamW([b, b0], lr=lr, weight_decay=0.0)

    for _ in range(steps):
        opt.zero_grad()
        r = ty - (tZ @ b + b0)
        loss = (tw * r ** 2).sum() + alpha * (b @ b)
        if tc.numel():
            v = torch.clamp(tc - (tZc @ b + b0), min=0.0)
            loss = loss + lam_c * (twc * v ** 2).sum()
        loss.backward()
        opt.step()

    return b.detach().numpy(), float(b0.detach().item())
