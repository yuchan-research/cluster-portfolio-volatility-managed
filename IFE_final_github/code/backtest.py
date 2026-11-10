"""
Monthly rolling walk-forward backtest [K24 4.1-iv, monthly adj.].

End of month t: compute weights from the past lookback months (including t) -> record realized return in t+1 -> move one month.
Execution assumption: weights are computed at the month-t closing price and the portfolio is rebalanced at that same price.
"""

import numpy as np
import pandas as pd
from data import select_universe, realized_next_returns
from optimization import ledoit_wolf, gmvp, msrp, project_simplex
from portfolio import build_portfolios


def _turnover(w_new, w_prev, r_prev):
    """Trading needed to move from last month's weights (after drifting with returns) to the new weights: Σ|Δw|."""
    if w_prev is None:                                                # first month
        return 1.0
    grown = w_prev * (1.0 + r_prev.reindex(w_prev.index).fillna(0.0))  # holdings after one month of returns
    drifted = grown / grown.sum()                                     # weights just before rebalancing
    idx = w_new.index.union(drifted.index)                            # union of old and new stocks
    return float((w_new.reindex(idx, fill_value=0.0) - drifted.reindex(idx, fill_value=0.0)).abs().sum())


def run_backtest(panel, cfg, holdings=None):
    """Run the full backtest and return monthly gross returns (before costs), turnover, diagnostics and cluster membership.

    If a list is passed as `holdings`, the stock weights of every strategy and the returns they earned are
    appended to it (one DataFrame per month). volatility.py needs them to net the layer's trades per stock.
    """
    rng = np.random.default_rng(cfg.seed)
    prev_w = {}                                                       # last month's weights per strategy (none in the first month)
    prev_r = None                                                     # last month's realized returns
    rows, turn_rows, diag_rows, member_rows = [], [], [], []
    dates = panel.returns.index

    for t_idx in range(cfg.lookback, len(dates) - 1):                 # from the first full window to the second-to-last month
        window = select_universe(panel, t_idx, cfg)                   # universe and estimation window at t
        weights, X, tw, diag, labels = build_portfolios(window, cfg, rng)  # weights, time weights, diagnostics, cluster labels
        members = window.columns
        member_rows.append(pd.DataFrame({"date": dates[t_idx], "PERMNO": members, "cluster": labels}))
        port = {f"Cluster-{m}": pd.Series(w, index=members) for m, w in weights.items()}
        if cfg.run_benchmarks:                                        # benchmarks (same universe, time weights and cap)
            port["EW-Universe"] = pd.Series(1.0 / len(members), index=members)  # equal weight
            Sigma_lw = ledoit_wolf(X, tw)
            if "gmvp" in cfg.objectives:
                port["NoCluster-LW-gmvp"] = pd.Series(gmvp(Sigma_lw, cfg.max_weight), index=members)  # optimization with cap
            if "msrp" in cfg.objectives:
                w_ms, _ = msrp(tw @ X, Sigma_lw)                       # full-universe MSRP
                if cfg.max_weight is not None:                        # if there is a cap
                    w_ms = project_simplex(w_ms, cfg.max_weight)      # apply it the same way as the cluster strategies
                port["NoCluster-LW-msrp"] = pd.Series(w_ms, index=members)

        nxt, miss = realized_next_returns(panel, t_idx, members, cfg)  # realized t+1 returns (delist-adjusted)

        row, trow = {"date": dates[t_idx + 1]}, {"date": dates[t_idx + 1]}
        for name, w in port.items():
            trow[name] = _turnover(w, prev_w.get(name), prev_r)       # turnover (costs are subtracted per level in the report)
            row[name] = float((w * nxt).sum())                        # gross return before costs
            prev_w[name] = w                                          # keep for next month
            if holdings is not None:                                  # keep the holdings for the layer costs
                held = w[w.abs() > 1e-12]
                holdings.append(pd.DataFrame({"date": dates[t_idx + 1], "strategy": name,  # month the return is earned
                                              "PERMNO": held.index, "weight": held.to_numpy(),
                                              "ret": nxt.reindex(held.index).fillna(0.0).to_numpy()}))  # return earned in t+1
        prev_r = nxt                                                  # realized returns for the next turnover

        eff_n = {f"effN_{m}": float(1.0 / (w ** 2).sum()) for m, w in weights.items()}  # effective number of holdings
        eff_n.update({f"maxW_{m}": float(w.max()) for m, w in weights.items()})
        exch = panel.exchange.iloc[t_idx].reindex(members)            # exchange codes of universe stocks
        mix = {"share_nyse": float((exch == 1).mean()),
               "share_amex": float((exch == 2).mean()),
               "share_nasdaq": float((exch == 3).mean())}             # NASDAQ share
        diag_rows.append({"date": dates[t_idx], **miss, **mix, **diag, **eff_n})
        rows.append(row)
        turn_rows.append(trow)
        print(f"\r  {dates[t_idx]:%Y-%m}  N={diag['N']}  K={diag['K']}  "
              f"size[{diag['cluster_size_min']},{diag['cluster_size_max']}]   ", end="")

    print()
    rets = pd.DataFrame(rows).set_index("date")                       # monthly gross returns
    turns = pd.DataFrame(turn_rows).set_index("date")
    diags = pd.DataFrame(diag_rows).set_index("date")
    membership = pd.concat(member_rows, ignore_index=True)            # monthly PERMNO -> cluster table
    return rets, turns, diags, membership
