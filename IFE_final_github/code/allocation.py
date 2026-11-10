"""
Within-cluster weights β (sum = 1 inside each cluster). All statistics use the time weights w.

  gaussian  ② [K24 4.2] stocks closer to the cluster center (mean return series) get larger weights
  ivp       ③ [López de Prado 2016] proportional to inverse variance
  periphery ⑥ [Pozzi, Di Matteo, Aste 2013] stocks with lower average within-cluster correlation (periphery) get larger weights
"""

import numpy as np


def _normalize_within(raw, labels, K):                                          # scale scores so each cluster sums to 1
    """Normalize so that each cluster sums to 1."""
    beta = np.zeros_like(raw)
    for k in range(K):
        m = labels == k
        total = raw[m].sum()
        beta[m] = raw[m] / total if total > 0 else 1.0 / m.sum()      # normalize (equal weights if the sum is 0)
    return beta                                                       # β


def beta_gaussian(X, w, labels, K, sigma_scale):                                # Gaussian-kernel weights [K24]
    """β_i ∝ exp(−d_i² / 2σ²), d_i² = Σ_t w_t (r_it − μ_kt)², σ² = scale × mean within-cluster d²."""
    raw = np.ones(X.shape[1])                                         # default (for single-stock clusters)
    for k in range(K):
        m = labels == k
        if m.sum() < 2:
            continue                                                  # keep 1
        R = X[:, m]                                                   # member returns (T, n_k)
        mu = R.mean(axis=1, keepdims=True)                            # cluster center series (T, 1)
        d2 = w @ (R - mu) ** 2                                        # weighted squared distance per stock
        sigma2 = sigma_scale * d2.mean()
        raw[m] = np.exp(-d2 / (2.0 * sigma2)) if sigma2 > 0 else 1.0  # Gaussian score
    return _normalize_within(raw, labels, K)


def beta_ivp(X, w, labels, K):                                                  # inverse-variance weights
    """β_i ∝ 1 / σ_i² (weighted variance)."""
    var = w @ (X - w @ X) ** 2                                        # weighted variance per stock
    raw = 1.0 / np.maximum(var, 1e-12)                                # inverse variance (avoid division by 0)
    return _normalize_within(raw, labels, K)


def beta_periphery(C_raw, labels, K):                                           # periphery weights
    """β_i ∝ 1 − c̄_i, c̄_i = average (weighted) correlation with the other stocks in the same cluster."""
    raw = np.ones(C_raw.shape[0])                                     # default (for single-stock clusters)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) < 2:
            continue                                                  # keep 1
        sub = C_raw[np.ix_(idx, idx)]
        c_bar = (sub.sum(axis=1) - 1.0) / (len(idx) - 1)              # average correlation excluding the diagonal
        raw[idx] = np.maximum(1.0 - c_bar, 1e-6)                      # larger score for peripheral stocks
    return _normalize_within(raw, labels, K)


def intra_cluster_beta(method, X, w, C_raw, labels, K, cfg):                    # pick the beta rule by name
    """Pick the β function by its configured name."""
    if method == "gaussian":                                          # ② Gaussian kernel
        return beta_gaussian(X, w, labels, K, cfg.gauss_sigma_scale)
    if method == "ivp":                                               # ③ inverse variance
        return beta_ivp(X, w, labels, K)                              # inversely proportional to variance
    if method == "periphery":                                         # ⑥ periphery weighting
        return beta_periphery(C_raw, labels, K)                       # inversely related to average correlation
    raise ValueError(f"unknown intra method: {method}")               # guard against a wrong setting
