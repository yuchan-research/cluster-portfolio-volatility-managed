"""
Covariance estimation and long-only mean-variance optimization (NumPy only).

  covariance      time-weighted covariance (equal or exponential weights) [K24 4.6]
  ledoit_wolf     shrinkage toward a scaled identity (Ledoit & Wolf 2004): for the full-universe benchmark
  gmvp            min wᵀΣw  s.t. Σw = 1, 0 ≤ w (≤ cap)
  msrp            max μᵀw / √(wᵀΣw)  s.t. Σw = 1, w ≥ 0

The projection functions (project_simplex, _project_budget) are called at every FISTA step, about 25,000
times per month. The first implementation used 100 bisection steps, which took most of the run time
(about 170 of 190 seconds per month), so it was replaced by an exact closed-form solution of the same problem.
Strategy weights from the two methods differ by about 1e-17 (checked on 2005-02 and 2009-05).
Full backtest time: several hours -> about 40 minutes.
"""

import numpy as np


def covariance(X, w):                                                           # weighted sample covariance
    """Time-weighted covariance of (T, n) returns (w sums to 1; equal weights give the 1/T sample covariance)."""
    Xc = X - w @ X                                                    # remove weighted mean
    return (Xc * w[:, None]).T @ Xc                                   # weighted covariance (n, n)


def ledoit_wolf(X, w):                                                          # Ledoit-Wolf shrinkage covariance
    """Ledoit & Wolf (2004) shrinkage toward a scaled identity. Time weights are approximated by rescaling observations by √(T·w_t)."""
    T, N = X.shape
    Xc = (X - w @ X) * np.sqrt(T * w)[:, None]                        # demean and apply weights (unchanged under equal weights)
    S = Xc.T @ Xc / T                                                 # weighted covariance
    mu = np.trace(S) / N                                              # scale of the target matrix
    d2 = ((S - mu * np.eye(N)) ** 2).sum()                            # ‖S − μI‖²
    b2_bar = ((Xc ** 2).sum(1) ** 2).sum() / T**2 - (S ** 2).sum() / T  # (1/T²) Σ_t ‖x_t x_tᵀ − S‖²
    b2 = min(b2_bar, d2)                                              # apply upper bound
    delta = b2 / d2 if d2 > 0 else 1.0
    return delta * mu * np.eye(N) + (1.0 - delta) * S


def project_simplex(v, cap=None):                                               # exact projection onto the (capped) simplex
    """Euclidean projection onto {w ≥ 0, Σw = 1, (w ≤ cap)} (exact solution).

    The solution has the form w_i = clip(v_i − τ, 0, cap). g(τ) = Σ clip(v_i − τ, 0, cap) is a piecewise linear,
    decreasing function of τ whose slope changes only at v_i and v_i − cap. We find the segment between these
    knots that contains g(τ) = 1 and solve the linear equation inside it, which gives τ exactly.
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    if cap is None:                                                   # no cap
        u = np.sort(v)[::-1]
        css = np.cumsum(u) - 1.0                                      # cumulative sum − 1
        rho = np.nonzero(u - css / np.arange(1, n + 1) > 0)[0][-1]    # number of positive components − 1 (Duchi et al. 2008)
        tau = css[rho] / (rho + 1.0)                                  # shift τ
        return np.maximum(v - tau, 0.0)
    if cap * n <= 1.0:                                                # cap too small for the weights to sum to 1
        return np.full(n, 1.0 / n)                                    # fall back to equal weights
    g = lambda t: np.minimum(np.maximum(v - t, 0.0), cap).sum()       # sum of weights at τ (decreasing)
    knots = np.sort(np.concatenate([v, v - cap]))[::-1]               # knots where the slope changes (largest τ first)
    lo, hi = 0, len(knots) - 1                                        # invariant: g(knots[lo]) < 1 ≤ g(knots[hi])
    while hi - lo > 1:                                                # until two adjacent knots remain
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if g(knots[mid]) >= 1.0 else (mid, hi)     # keep the half that contains the solution
    t_lo, t_hi = knots[hi], knots[lo]                                 # segment [t_lo, t_hi] containing the solution
    mid_tau = 0.5 * (t_lo + t_hi)
    active = (v - mid_tau > 0.0) & (v - mid_tau < cap)                # components strictly between 0 and cap (fixed in the segment)
    n_act = int(active.sum())
    if n_act == 0:                                                    # degenerate segment or no such component
        return np.minimum(np.maximum(v - t_lo, 0.0), cap)
    n_cap = float((v - mid_tau >= cap).sum())                         # number of components at the cap
    tau = (v[active].sum() + n_cap * cap - 1.0) / n_act               # solve Σ(v_i − τ) + cap·n_cap = 1
    return np.minimum(np.maximum(v - tau, 0.0), cap)


def _fista(grad, project, x0, L, n_iter=3000, tol=1e-10):
    """Accelerated projected gradient descent (FISTA)."""
    x, y, t = x0.copy(), x0.copy(), 1.0
    for _ in range(n_iter):
        x_new = project(y - grad(y) / L)                              # gradient step, then project onto the constraint set
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        if np.abs(x_new - x).max() < tol:
            return x_new
        x, t = x_new, t_new
    return x                                                          # result at the iteration limit


def gmvp(Sigma, cap=None):                                                      # long-only minimum variance
    """Long-only global minimum-variance portfolio."""
    n = len(Sigma)
    if cap is not None and cap * n < 1.0:                             # cap too small to be feasible
        cap = None                                                    # drop the cap
    L = 2.0 * np.linalg.eigvalsh(Sigma)[-1] + 1e-12                   # Lipschitz constant of the gradient
    grad = lambda w: 2.0 * Sigma @ w
    proj = lambda v: project_simplex(v, cap)                          # projection onto the constraint set
    return _fista(grad, proj, np.full(n, 1.0 / n), L)                 # start from equal weights


def _project_budget(v, mu):                                                     # projection onto {y >= 0, mu'y = 1}
    """Projection onto {y ≥ 0, μᵀy = 1} (exact solution): y = max(v + νμ, 0).

    g(ν) = μᵀ max(v + νμ, 0) is a piecewise linear, increasing function of ν with knots at −v_i/μ_i.
    We find the segment that contains g(ν) = 1 and solve the linear equation inside it to get ν exactly.
    """
    v = np.asarray(v, dtype=float)
    mu = np.asarray(mu, dtype=float)
    g = lambda nu: float(mu @ np.maximum(v + nu * mu, 0.0))           # increasing in ν
    nz = mu != 0.0                                                    # only components with μ ≠ 0 create knots
    knots = np.sort(-v[nz] / mu[nz])                                  # knots (ascending)
    if len(knots) == 0:                                               # all μ are 0 (ruled out earlier in msrp)
        return np.maximum(v, 0.0)
    if g(knots[0]) >= 1.0:                                            # solution lies left of the first knot
        lo_t, hi_t, probe = -np.inf, knots[0], knots[0] - 1.0         # open segment on the left
    elif g(knots[-1]) <= 1.0:                                         # solution lies right of the last knot
        lo_t, hi_t, probe = knots[-1], np.inf, knots[-1] + 1.0        # open segment on the right
    else:                                                             # solution lies between two knots
        lo, hi = 0, len(knots) - 1
        while hi - lo > 1:                                            # until two adjacent knots remain
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if g(knots[mid]) <= 1.0 else (lo, mid)  # keep the half that contains the solution
        lo_t, hi_t = knots[lo], knots[hi]
        probe = 0.5 * (lo_t + hi_t)
    active = (v + probe * mu) > 0.0                                   # components positive in this segment (fixed)
    denom = float(mu[active] @ mu[active])                            # Σ_A μ_i²
    if denom <= 0.0:                                                  # zero slope (rare)
        return np.maximum(v + (lo_t if np.isfinite(lo_t) else hi_t) * mu, 0.0)
    nu = (1.0 - float(mu[active] @ v[active])) / denom                # solve Σ_A μ_i(v_i + νμ_i) = 1
    return np.maximum(v + nu * mu, 0.0)


def msrp(mu, Sigma):                                                            # long-only maximum Sharpe
    """Long-only maximum Sharpe ratio portfolio. min yᵀΣy s.t. μᵀy = 1, y ≥ 0 -> w = y / Σy."""
    if mu.max() <= 0:                                                 # no asset with positive expected return
        return gmvp(Sigma), False                                     # undefined -> fall back to GMVP
    n = len(Sigma)
    L = 2.0 * np.linalg.eigvalsh(Sigma)[-1] + 1e-12
    y0 = _project_budget(np.full(n, 1.0 / n), mu)
    y = _fista(lambda z: 2.0 * Sigma @ z, lambda z: _project_budget(z, mu), y0, L)
    return y / y.sum(), True
