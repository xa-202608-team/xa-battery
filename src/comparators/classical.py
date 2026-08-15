"""classical.py — classical stochastic degradation models. NEW (see REUSE_MAP.md N1).

Phase A found NO Wiener / particle-filter / double-exponential implementation anywhere
in the historical tree or the archive — the keyword family's only >=3-hit file was the
closeout plan itself. These are the standard battery-PHM comparators a reviewer expects,
so they are written here from the literature definitions.

WIENER PROCESS WITH DRIFT
-------------------------
SOH degradation as a drifting Brownian motion::

    S(t) = S(0) - mu*t + sigma*W(t),    mu > 0

First passage to the EOL threshold has an inverse-Gaussian distribution, so RUL from a
state ``s`` at time ``t`` has closed-form mean ``(s - L) / mu``. ``mu`` and ``sigma``
are estimated per trajectory from its observed increments, so this is a per-unit
adaptive model — it needs no training set at all, which is precisely why it is a fair
"existing method" baseline rather than a straw man.

DOUBLE-EXPONENTIAL CAPACITY FADE + PARTICLE FILTER
--------------------------------------------------
The standard empirical battery capacity model::

    Q(t) = a*exp(b*t) + c*exp(d*t)

with the four parameters tracked by a bootstrap particle filter (sequential importance
resampling). The filter carries a particle cloud over ``(a,b,c,d)``, weights particles
by their likelihood against each new observation, resamples when the effective sample
size drops, and projects each particle forward to its EOL crossing — so RUL comes out
as a distribution, which is the usual reason this method is chosen in PHM.

DETERMINISM
-----------
The particle filter is stochastic. Every call takes an explicit seed and the seed is
derived from the trajectory id, so a rerun reproduces the numbers exactly. The RNG is
never seeded from the clock.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EOL_SOH = 0.70


# ------------------------------------------------------------------ Wiener

@dataclass
class WienerFit:
    mu: float                     # drift per day (positive = degrading)
    sigma: float                  # diffusion per sqrt(day)
    n_increments: int
    degrading: bool


def fit_wiener(day: np.ndarray, soh: np.ndarray) -> WienerFit:
    """Maximum-likelihood drift and diffusion from observed increments.

    For a Wiener process the increments over ``dt`` are independent
    ``N(-mu*dt, sigma^2*dt)``, so the MLEs are the mean and variance of the
    increments normalised by ``dt``.
    """
    d = np.asarray(day, dtype=float)
    s = np.asarray(soh, dtype=float)
    dt = np.diff(d)
    ds = np.diff(s)
    ok = dt > 0
    dt, ds = dt[ok], ds[ok]
    if len(dt) < 2:
        return WienerFit(mu=0.0, sigma=1e-9, n_increments=len(dt), degrading=False)

    mu = float(-np.sum(ds) / np.sum(dt))                 # positive when SOH falls
    resid = ds + mu * dt
    var = float(np.sum(resid ** 2 / dt) / len(dt))
    return WienerFit(mu=mu, sigma=float(np.sqrt(max(var, 1e-18))),
                     n_increments=len(dt), degrading=mu > 0)


def wiener_rul(fit: WienerFit, current_soh: float,
               threshold: float = EOL_SOH,
               max_days: float = 3000.0) -> float:
    """Expected first-passage time to ``threshold`` — the inverse-Gaussian mean.

    A non-degrading fit (mu <= 0) has infinite expected passage time; it is censored
    at ``max_days`` rather than returned as inf, so the MAE stays finite and the
    censoring is uniform across arms.
    """
    gap = float(current_soh) - float(threshold)
    if gap <= 0:
        return 0.0
    if fit.mu <= 0:
        return max_days
    return float(min(gap / fit.mu, max_days))


# ------------------------------------------- double exponential + particle filter

@dataclass
class ParticleFilterResult:
    rul_mean: float
    rul_median: float
    rul_p10: float
    rul_p90: float
    n_effective: float
    n_resample: int


def _double_exp(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``a*exp(b*t) + c*exp(d*t)`` evaluated for a whole particle cloud.

    ``p`` is ``(n_particles, 4)``, ``t`` is ``(n_times,)``, result is
    ``(n_particles, n_times)``. Exponents are clipped before exponentiation so a wild
    particle cannot overflow to inf and poison the weight normalisation.
    """
    p = np.atleast_2d(np.asarray(p, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    a, b, c, d = p[:, 0:1], p[:, 1:2], p[:, 2:3], p[:, 3:4]
    bt = np.clip(b * t[None, :], -50.0, 50.0)
    dt_ = np.clip(d * t[None, :], -50.0, 50.0)
    return a * np.exp(bt) + c * np.exp(dt_)


def particle_filter_rul(day: np.ndarray, soh: np.ndarray,
                        threshold: float = EOL_SOH,
                        n_particles: int = 2000,
                        obs_sigma: float = 0.005,
                        seed: int = 0,
                        max_days: float = 3000.0,
                        horizon_days: float = 3000.0) -> ParticleFilterResult:
    """Bootstrap particle filter over the double-exponential fade model.

    ``obs_sigma`` is the assumed SOH observation noise. It is DECLARED, not tuned on
    results: 0.005 is the order of the anchor noise residual v2's own L3 table carries.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(day, dtype=float)
    s = np.asarray(soh, dtype=float)
    if len(d) < 3:
        return ParticleFilterResult(max_days, max_days, max_days, max_days, 0.0, 0)

    t0 = d[0]
    t = d - t0

    # Prior: a near the initial SOH with a slow decay, plus a small fast term.
    # Ranges are wide and fixed in advance; they are not fitted to the target.
    a = rng.uniform(0.7, 1.1, n_particles)
    b = -np.abs(rng.normal(0.0, 2e-4, n_particles))
    c = rng.uniform(-0.15, 0.15, n_particles)
    dd = -np.abs(rng.normal(0.0, 5e-3, n_particles))
    P = np.column_stack([a, b, c, dd])
    w = np.full(n_particles, 1.0 / n_particles)

    n_resample = 0
    for k in range(len(t)):
        pred = _double_exp(P, t[k:k + 1])[:, 0]
        ll = -0.5 * ((s[k] - pred) / obs_sigma) ** 2
        ll -= ll.max()
        w = w * np.exp(ll)
        tot = w.sum()
        if tot <= 0 or not np.isfinite(tot):
            w = np.full(n_particles, 1.0 / n_particles)
        else:
            w = w / tot

        n_eff = 1.0 / np.sum(w ** 2)
        if n_eff < n_particles / 2:
            idx = rng.choice(n_particles, size=n_particles, p=w)
            P = np.atleast_2d(P[idx])
            # Roughening: jitter proportional to each parameter's spread, so the
            # cloud does not collapse to a single point after repeated resampling.
            # A floor on the spread keeps the jitter non-degenerate even when every
            # surviving particle happens to share a parameter value.
            sd = np.maximum(P.std(axis=0) * 0.05, 1e-9)
            P = P + rng.normal(0.0, 1.0, size=P.shape) * sd[None, :]
            P[:, 1] = -np.abs(P[:, 1])
            P[:, 3] = -np.abs(P[:, 3])
            w = np.full(n_particles, 1.0 / n_particles)
            n_resample += 1

    # Project each particle forward to its own EOL crossing.
    grid = np.arange(t[-1], t[-1] + horizon_days + 14.0, 14.0)
    traj = _double_exp(P, grid)
    below = traj <= threshold
    first = np.where(below.any(axis=1), below.argmax(axis=1), len(grid) - 1)
    rul = grid[first] - t[-1]
    rul = np.clip(rul, 0.0, max_days)

    # Weighted quantiles over the particle cloud.
    order = np.argsort(rul)
    rs, ws = rul[order], w[order]
    cw = np.cumsum(ws)

    def wq(q):
        i = int(np.searchsorted(cw, q))
        return float(rs[min(i, len(rs) - 1)])

    return ParticleFilterResult(
        rul_mean=float(np.sum(w * rul)), rul_median=wq(0.5),
        rul_p10=wq(0.1), rul_p90=wq(0.9),
        n_effective=float(1.0 / np.sum(w ** 2)), n_resample=n_resample)


def stable_seed(text: str) -> int:
    """Deterministic per-trajectory seed, so the filter is reproducible."""
    import hashlib
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:8], 16)
