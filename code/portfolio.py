"""
Portfolio construction at a single date t [K24 4.1 (i)~(iii)].

  ① prepare estimation returns (winsorization, market residuals if enabled)
  ② correlation matrix -> choose K -> spectral embedding
  ③ R repeats: k-means++ labels -> for each β method
        synthetic ETF returns r*_k = Σ β_i r_i -> Σ*, μ* -> α (GMVP/MSRP) -> w_i = α_k β_i
  ④ average w over the R repeats (separately for each β method)

The three β methods share the same cluster labels within a repeat (so only the effect of β is compared).
"""

import numpy as np
from data import winsorize_cross_section, market_residuals, time_weights, effective_n
from clustering import correlation, choose_k, cluster_embedding, kmeans_pp
from allocation import intra_cluster_beta
from optimization import covariance, gmvp, msrp, project_simplex


def build_portfolios(window, cfg, rng):                                         # stock weights for every beta/alpha pair
    """window: (lookback × N) returns DataFrame. Returns stock weights per β method, estimation returns, time weights, diagnostics."""
    X = winsorize_cross_section(window, cfg.winsor_q).to_numpy()      # estimation returns (T, N)
    T, N = X.shape
    w = time_weights(T, cfg.halflife)                                 # time weights (larger for recent months)
    X_clu = market_residuals(X, w) if cfg.use_residual else X
    C_clu = correlation(X_clu, w)                                     # weighted correlation for clustering
    C_raw = correlation(X, w)                                         # weighted correlation of raw returns (for periphery β)
    K, diag = choose_k(C_clu, effective_n(w), cfg)                    # number of clusters (uses effective observations)
    Z = cluster_embedding(C_clu, K, cfg)                              # spectral embedding (shared across repeats)

    keys = [(m, o) for m in cfg.intra_methods for o in cfg.objectives]  # (β method, objective) combinations
    sums = {k: np.zeros(N) for k in keys}
    msrp_fallback = 0                                                 # times MSRP fell back to GMVP
    sizes = []
    first_labels = None                                               # labels from the first repeat (saved for inspection)

    for rep in range(cfg.n_repeats):                                  # R repeats [K24 4.1-iii]
        labels = kmeans_pp(Z, K, rng, n_init=cfg.kmeans_n_init)
        if rep == 0:
            first_labels = labels.copy()                              # keep labels
        sizes.append(np.bincount(labels, minlength=K))
        for method in cfg.intra_methods:                              # per β method
            beta = intra_cluster_beta(method, X, w, C_raw, labels, K, cfg)
            B = np.zeros((N, K))                                      # stock -> cluster weight matrix
            B[np.arange(N), labels] = beta                            # B[i, label_i] = β_i
            X_syn = X @ B                                             # synthetic ETF returns (T, K)
            Sigma = covariance(X_syn, w)                              # K×K weighted covariance Σ*
            for obj in cfg.objectives:
                if obj == "gmvp":
                    alpha = gmvp(Sigma)                               # cluster weights α
                else:
                    alpha, ok = msrp(w @ X_syn, Sigma)                # μ* = past weighted mean (no future information)
                    msrp_fallback += int(not ok)
                sums[(method, obj)] += B @ alpha                      # accumulate w_i = α_{label_i} × β_i

    weights = {}
    for method, obj in keys:
        port_w = sums[(method, obj)] / cfg.n_repeats                  # average over R (named apart from time weights w)
        if cfg.max_weight is not None:                                # stock cap
            port_w = project_simplex(port_w, cfg.max_weight)          # project onto the capped simplex
        weights[f"{method}-{obj}"] = port_w / port_w.sum()            # normalize to 1, name e.g. "ivp-gmvp"

    sizes = np.concatenate(sizes)                                     # cluster sizes of all repeats
    diag.update({"K": K, "N": N,
                 "cluster_size_min": int(sizes.min()),
                 "cluster_size_max": int(sizes.max()),
                 "msrp_fallback": msrp_fallback})                     # MSRP fallbacks
    return weights, X, w, diag, first_labels                          # weights, estimation returns, time weights, diagnostics, labels
