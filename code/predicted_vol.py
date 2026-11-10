"""
Does each optimizer's own risk forecast match what the portfolio actually did?
    python predicted_vol.py   (step [6-4] of main.py)

Goal 1 of the project is that a 500 × 500 covariance matrix cannot be estimated from 60 months of data.
This script measures that directly. Every month we record the volatility each optimizer predicts for the
portfolio it just chose, and compare the average with the volatility the portfolio actually realized.

  cluster portfolio    √(α'Σ*α)      the K × K covariance is the only risk estimate the optimizer sees
  no-cluster LW-GMVP   √(w'Σ_LW w)   the full 500 × 500 shrinkage covariance

A ratio near 1 means the risk estimate was honest; a small ratio means the optimizer believed it was
holding a much safer portfolio than it really was.

The monthly loop has to be rerun because the covariance matrices are not saved by the backtest
(about 4 minutes). Settings come from config.Config, exactly as in backtest.py.

Output: output/predicted_vol.csv (monthly predictions), summary printed to the screen
"""

import time
import numpy as np
import pandas as pd
from config import Config
from data import (load_panel, select_universe, winsorize_cross_section,
                  time_weights, effective_n, market_residuals)
from clustering import correlation, choose_k, cluster_embedding, kmeans_pp
from allocation import intra_cluster_beta
from optimization import covariance, ledoit_wolf, gmvp
from metrics import net_returns
from report import load_results, main_strategy


def main(cfg=None):                                                             # predicted vs realised volatility
    """Rerun the monthly loop recording predicted volatility, then compare with realized volatility."""
    cfg = cfg or Config()
    out = cfg.output_dir
    gross, turns, bench = load_results(cfg)
    name, _ = main_strategy(gross, turns, bench, cfg)                  # main strategy, e.g. Cluster-ivp-gmvp
    beta_method, objective = name.replace("Cluster-", "").split("-")   # its β method and objective
    print(f"Main strategy {name} -> β={beta_method}, α={objective}")
    if objective != "gmvp":                                            # μ* would also be needed for MSRP
        print("    [note] predicted volatility is reported for the GMVP objective")

    rng = np.random.default_rng(cfg.seed)                              # same generator as the backtest
    panel = load_panel(cfg)
    dates = panel.returns.index
    rows, t0 = [], time.time()

    for t_idx in range(cfg.lookback, len(dates) - 1):                  # same range as backtest.run_backtest
        window = select_universe(panel, t_idx, cfg)
        X = winsorize_cross_section(window, cfg.winsor_q).to_numpy()   # estimation returns (T, N)
        tw = time_weights(len(X), cfg.halflife)
        X_clu = market_residuals(X, tw) if cfg.use_residual else X
        C_clu = correlation(X_clu, tw)
        C_raw = correlation(X, tw)                                     # raw correlation (periphery β)
        K, _ = choose_k(C_clu, effective_n(tw), cfg)
        Z = cluster_embedding(C_clu, K, cfg)

        pred = []
        for _ in range(cfg.n_repeats):                                 # same R repeats as the backtest
            labels = kmeans_pp(Z, K, rng, n_init=cfg.kmeans_n_init)
            beta = intra_cluster_beta(beta_method, X, tw, C_raw, labels, K, cfg)
            B = np.zeros((len(beta), K))                               # stock -> cluster weight matrix
            B[np.arange(len(beta)), labels] = beta                     # B[i, label_i] = β_i
            Sigma_star = covariance(X @ B, tw)                         # K × K covariance of the synthetic assets
            alpha = gmvp(Sigma_star)
            pred.append(np.sqrt(max(alpha @ Sigma_star @ alpha, 0.0)))
        cluster_pred = float(np.mean(pred)) * np.sqrt(12.0)            # annualized, averaged over repeats

        Sigma_lw = ledoit_wolf(X, tw)                                  # full 500 × 500 shrinkage covariance
        w_lw = gmvp(Sigma_lw, cfg.max_weight)
        lw_pred = float(np.sqrt(max(w_lw @ Sigma_lw @ w_lw, 0.0))) * np.sqrt(12.0)

        rows.append({"date": dates[t_idx + 1], "K": K,                 # month the prediction applies to
                     "cluster_pred_vol": cluster_pred,
                     "lw_pred_vol": lw_pred})
        print(f"\r  {dates[t_idx]:%Y-%m}  K={K}  cluster {cluster_pred:.3f}  LW {lw_pred:.3f}   ", end="")

    print()
    pred = pd.DataFrame(rows).set_index("date")
    pred.to_csv(out / "predicted_vol.csv")

    net = net_returns(gross, turns, cfg.cost_bps_main)                 # realized net returns at the main cost
    print(f"\n{'Portfolio':24s} {'predicted':>10s} {'realized':>10s} {'ratio':>7s}")
    for label, col, strat in [("Cluster " + name.replace("Cluster-", ""), "cluster_pred_vol", name),
                              ("No-cluster LW-GMVP", "lw_pred_vol", "NoCluster-LW-gmvp")]:
        if strat not in net.columns:                                   # benchmark not computed
            continue
        p = pred[col].mean()
        r = net[strat].std(ddof=1) * np.sqrt(12.0)                     # realized volatility
        print(f"{label:24s} {p:9.1%} {r:9.1%} {p / r:7.2f}")
    print(f"\nDone ({time.time() - t0:.0f}s) -> {out / 'predicted_vol.csv'}")


if __name__ == "__main__":
    main()
