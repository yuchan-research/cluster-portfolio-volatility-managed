"""
Performance metrics and main strategy selection.

  Required: annualized return, annualized standard deviation, Sharpe (excess over Ken French RF), MDD
  Additional: Calmar, Sortino (excess over RF), beta vs S&P 500, average monthly turnover

The backtest saves gross returns (before costs) and turnover; here, for any cost level,
net return = gross return − (bp / 10,000) × turnover is computed (no need to rerun the backtest).

sharpe / max_drawdown / ann_return are single-metric functions shared by the layer analysis scripts
(validation.py, cost_sensitivity.py, c_analysis.py). They use the same definitions as performance().
"""

import numpy as np
import pandas as pd

SP500_NAME = "S&P 500 TR"


def net_returns(gross, turns, bps):                                             # returns after costs
    """Monthly net returns after a cost of bps basis points."""
    return gross - bps / 1e4 * turns.reindex_like(gross)              # subtract traded amount × cost rate


def sharpe(r, rf=None):
    """Arithmetic Sharpe = (mean monthly excess return / monthly standard deviation) × √12. Without rf, r is treated as excess return."""
    ex = r if rf is None else r - rf.reindex(r.index)                 # excess return
    ex = ex.dropna()
    return ex.mean() / ex.std(ddof=1) * np.sqrt(12.0)


def max_drawdown(r):
    """Maximum drawdown (negative, e.g. −0.25 = −25%)."""
    wealth = (1.0 + r.dropna()).cumprod()
    peak = wealth.cummax().clip(lower=1.0)                            # the starting value of 1 also counts as a peak
    return float((wealth / peak - 1.0).min())                         # lowest point relative to the running peak


def ann_return(r):                                                              # geometric annual return (CAGR)
    """Geometric annualized return."""
    r = r.dropna()
    return (1.0 + r).prod() ** (12.0 / len(r)) - 1.0


def performance(r, turnover, rf, mkt):                                          # all metrics for one return series
    """Dict of metrics for monthly returns r. rf and mkt are the risk-free and S&P 500 returns on the same dates."""
    r = r.dropna()
    rf = rf.reindex(r.index)
    ex = r - rf
    wealth = (1.0 + r).cumprod()                                      # cumulative wealth ($1 invested)
    years = len(r) / 12.0
    ann_ret = wealth.iloc[-1] ** (1.0 / years) - 1.0                  # geometric annualized return
    mdd = max_drawdown(r)                                             # maximum drawdown (negative)
    downside = np.sqrt((np.minimum(ex, 0.0) ** 2).mean())             # downside deviation of excess returns (monthly)
    mkt_ex = mkt.reindex(r.index) - rf                                # S&P 500 excess returns
    beta = ex.cov(mkt_ex) / mkt_ex.var()
    return {
        "Annualized Return": ann_ret,                                 # annualized return (required)
        "Annualized Std": r.std(ddof=1) * np.sqrt(12.0),              # annualized standard deviation (required)
        "Sharpe": ex.mean() / ex.std(ddof=1) * np.sqrt(12.0),         # annualized Sharpe ratio, excess over RF (required)
        "Max Drawdown": mdd,                                          # maximum drawdown (required)
        "Calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,          # annual return ÷ |MDD| (additional)
        "Sortino": ex.mean() / downside * np.sqrt(12.0) if downside > 0 else np.nan,  # excess return per downside risk (additional)
        "Beta vs S&P 500": beta,                                      # market sensitivity (additional)
        "Avg Monthly Turnover": turnover.reindex(r.index).iloc[1:].mean(),  # average turnover excluding the first month (initial purchase)
        "Months": len(r),
    }


def periods_of(index, split_dates):
    """List of (name, dates): the full period and the sub-periods split at split_dates."""
    out = [("Full", index)]
    edges = [None, *pd.to_datetime(list(split_dates)), None]
    if split_dates:                                                   # if there are sub-periods
        for start, end in zip(edges[:-1], edges[1:]):
            idx = index
            if start is not None:
                idx = idx[idx > start]                                # after the boundary
            if end is not None:
                idx = idx[idx <= end]
            out.append((f"{idx[0]:%Y-%m}~{idx[-1]:%Y-%m}", idx))
    return out


def performance_table(gross, turns, cost_bps_list, split_dates, bench):         # metrics for every strategy and cost
    """Performance table: cost level × period × strategy (+ S&P 500 TR). bench is a DataFrame with RF and SP500TR columns."""
    rows = []
    for bps in cost_bps_list:
        net = net_returns(gross, turns, bps)
        net[SP500_NAME] = bench["SP500TR"]                            # add benchmark (no trading costs)
        tn = turns.copy()
        tn[SP500_NAME] = 0.0                                          # holding the index has zero turnover
        for label, idx in periods_of(gross.index, split_dates):
            for col in net.columns:
                m = performance(net.loc[idx, col], tn[col], bench["RF"], bench["SP500TR"])
                rows.append({"Cost (bp)": bps, "Period": label, "Strategy": col, **m})
    return pd.DataFrame(rows)


def select_main_strategy(table, cfg):
    """Pick the main cluster strategy with a rule fixed in advance.

    Rule: among cluster strategies, the highest Sharpe over the whole backtest at the main cost (cost_bps_main).
    Ties are broken by the smaller MDD. No model is trained, so there is no train / test split; every strategy is
    reported in full and the rule only labels which one the report follows.
    (If config.split_dates is set, the first sub-period is used instead, which is how earlier versions selected.)
    """
    labels = [p for p in table["Period"].unique() if p != "Full"]     # sub-period names (empty by default)
    sel_period = labels[0] if labels else "Full"                      # whole backtest unless sub-periods are set
    t = table[(table["Period"] == sel_period) & (table["Cost (bp)"] == cfg.cost_bps_main)]
    t = t[t["Strategy"].str.startswith("Cluster-")]                   # cluster strategies only
    best = t.sort_values(["Sharpe", "Max Drawdown"], ascending=[False, False]).iloc[0]  # highest Sharpe, then smaller drawdown
    return best["Strategy"], sel_period                               # main strategy name and the period it was chosen on
