"""
Choosing the number of clusters K [C23 2.3] and signed-graph clustering [K24 Ch. 3].

The correlation matrix Γ is treated as the adjacency matrix of a signed graph (zero diagonal);
a spectral embedding is built and k-means++ finds K groups.
"""

import numpy as np


def correlation(X, w):                                                          # weighted correlation matrix
    """N×N weighted correlation matrix from (T, N) returns and time weights w."""
    X_c = X - w @ X                                                   # remove weighted mean per stock
    S = (X_c * w[:, None]).T @ X_c                                    # weighted covariance
    sd = np.sqrt(np.clip(np.diag(S), 1e-18, None))                    # weighted standard deviation per stock
    C = S / np.outer(sd, sd)
    return np.clip(C, -1.0, 1.0)                                      # fix values outside [−1, 1] from rounding


def choose_k(C, T_eff, cfg):                                                    # number of clusters for this month
    """Compute all three K candidates and return the one for the configured method. T_eff = effective observations."""
    N = C.shape[0]
    eig = np.sort(np.linalg.eigvalsh(C))[::-1]
    lam_plus = (1.0 + np.sqrt(N / T_eff)) ** 2                        # Marchenko–Pastur upper edge λ+ (approximate under weights)
    k_mp = int((eig > lam_plus).sum())                                # number of eigenvalues above λ+ [C23 eq. (5)]
    share = np.cumsum(eig) / eig.sum()
    k_var = int(np.searchsorted(share, cfg.k_var_threshold) + 1)      # smallest count reaching threshold P [C23 eq. (6)]
    candidates = {"fixed": cfg.k_fixed, "mp": k_mp, "var": k_var}     # keep all three for diagnostics
    k = candidates[cfg.k_method]                                      # K for the configured method
    if cfg.k_method != "fixed":                                       # only for dynamic K
        k = int(np.clip(k, *cfg.k_bounds))                            # apply lower/upper bounds
    diag = {"K_fixed": cfg.k_fixed, "K_mp": k_mp, "K_var": k_var,
            "T_eff": float(T_eff),                                    # effective observations
            "top_eig_share": float(eig[0] / eig.sum())}               # share of the first eigenvalue
    return k, diag                                                    # K to use and diagnostics


def _inv_sqrt_degree(d):                                                        # D^(-1/2), zero for isolated nodes
    """Degree vector to the power −1/2 (0 where degree is 0)."""
    out = np.zeros_like(d)
    pos = d > 1e-12
    out[pos] = 1.0 / np.sqrt(d[pos])                                  # diagonal of D^{-1/2}
    return out


def _sym_laplacian(A):
    """Symmetric normalized Laplacian I − D^{-1/2} A D^{-1/2} of a non-negative adjacency matrix A."""
    s = _inv_sqrt_degree(A.sum(axis=1))                               # D^{-1/2}
    return np.eye(len(A)) - s[:, None] * A * s[None, :]


def embedding_signed_laplacian(C, K):                                           # signed Laplacian embedding (alternative)
    """[K24 3.2] Signed Laplacian: degree = sum of |Γ|; smallest K eigenvectors, row-normalized."""
    A = C.copy()
    np.fill_diagonal(A, 0.0)
    s = _inv_sqrt_degree(np.abs(A).sum(axis=1))                       # D̄^{-1/2}, D̄ = Σ|Γ_ij|
    L = np.eye(len(A)) - s[:, None] * A * s[None, :]
    _, vecs = np.linalg.eigh(L)
    U = vecs[:, :K]
    return U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)


def embedding_sponge_sym(C, K, tau_pos, tau_neg):
    """[K24 3.4] SPONGE_sym: smallest K generalized eigenvectors of (L+_sym + τ−I, L−_sym + τ+I)."""
    A = C.copy()
    np.fill_diagonal(A, 0.0)
    A_pos = np.maximum(A, 0.0)
    A_neg = np.maximum(-A, 0.0)                                       # negative-correlation graph Γ−
    M1 = _sym_laplacian(A_pos) + tau_neg * np.eye(len(A))
    M2 = _sym_laplacian(A_neg) + tau_pos * np.eye(len(A))             # denominator matrix (positive definite)
    Lc = np.linalg.cholesky(M2)                                       # M2 = Lc Lcᵀ
    Lc_inv = np.linalg.inv(Lc)                                        # inverse of Lc
    S = Lc_inv @ M1 @ Lc_inv.T                                        # turn the generalized problem into a standard symmetric one
    S = (S + S.T) / 2.0
    _, Y = np.linalg.eigh(S)
    return Lc_inv.T @ Y[:, :K]                                        # K-dimensional embedding in original coordinates


def kmeans_pp(Z, K, rng, n_init=3, max_iter=200):                               # k-means with k-means++ seeding
    """Run k-means++ initialization n_init times and return the labels with the lowest inertia."""
    n = len(Z)
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        centers = [Z[rng.integers(n)]]                                # first center chosen at random
        for _ in range(K - 1):
            d2 = ((Z[:, None, :] - np.array(centers)[None]) ** 2).sum(-1).min(1)  # squared distance to nearest center
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
            centers.append(Z[rng.choice(n, p=p)])
        centers = np.array(centers)
        labels = np.full(n, -1)
        for _ in range(max_iter):
            d2 = ((Z[:, None, :] - centers[None]) ** 2).sum(-1)       # point-to-center squared distances (n, K)
            new = d2.argmin(1)
            for k in range(K):
                if not (new == k).any():                              # if no point is assigned
                    far = d2[np.arange(n), new].argmax()              # take the currently farthest point
                    new[far] = k                                      # and move it to this cluster
            if np.array_equal(new, labels):                           # no change
                break
            labels = new
            centers = np.array([Z[labels == k].mean(0) for k in range(K)])
        inertia = ((Z - centers[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_labels, best_inertia = labels.copy(), inertia        # keep it
    return best_labels


def cluster_embedding(C, K, cfg):
    """Spectral embedding with the configured algorithm (reused across repeats)."""
    if cfg.cluster_algo == "sponge_sym":
        return embedding_sponge_sym(C, K, cfg.tau_pos, cfg.tau_neg)
    if cfg.cluster_algo == "signed_laplacian":
        return embedding_signed_laplacian(C, K)
    raise ValueError(f"unknown cluster_algo: {cfg.cluster_algo}")     # guard against a wrong setting
