"""
Low-volatility benchmarks without clusters:  python ivp_benchmark.py   (step [6-7] of main.py)

The main strategy weights stocks by inverse variance inside each cluster (ivp), which is a low-volatility
tilt in itself. To see whether the clusters add anything beyond that tilt, the same rule is applied to the
whole universe with no clusters:

  NoCluster-ivp   w_i ∝ 1 / σ_i²   (the main strategy's β rule over all 500 stocks)
  NoCluster-ivol  w_i ∝ 1 / σ_i    (plain inverse volatility, a common low-volatility portfolio)

Everything else is the same as the backtest: universe, 60-month window with 18-month half-life weights,
winsorized estimation returns, 5% stock cap, delisting returns, 20bp costs, and the layer charged on the
netted order per stock. No random step is involved. The loop takes about two minutes.

Output: output/ivp_benchmark/ (monthly returns, turnover, holdings), output/ivp_benchmark.csv
"""

import numpy as np
import pandas as pd
from config import Config, VMConfig
from data import load_panel, select_universe, realized_next_returns, winsorize_cross_section, time_weights
from optimization import project_simplex
from backtest import _turnover
from metrics import net_returns, sharpe, ann_return, max_drawdown as mdd
from report import load_results, main_strategy
from volatility import apply_layer, load_holdings, HOLDINGS_FILE
from qa_analysis import jk_memmel

NAMES = ("NoCluster-ivp", "NoCluster-ivol")
HALVES = [("2005-2012", "2005-02", "2012-12"), ("2013-2020", "2013-01", "2020-12")]


def run_loop(cfg, out):                                                         # monthly loop for the two benchmarks
    """Build both portfolios every month and save returns, turnover and holdings to out."""
    panel = load_panel(cfg)                                            # same data as the backtest
    dates = panel.returns.index
    prev_w, prev_r, rows, turns, held = {}, None, [], [], []
    for t_idx in range(cfg.lookback, len(dates) - 1):                  # same range as the backtest
        window = select_universe(panel, t_idx, cfg)
        X = winsorize_cross_section(window, cfg.winsor_q).to_numpy()
        tw = time_weights(X.shape[0], cfg.halflife)
        var = np.maximum(tw @ (X - tw @ X) ** 2, 1e-12)                # weighted variance per stock
        port = {"NoCluster-ivp": 1.0 / var, "NoCluster-ivol": 1.0 / np.sqrt(var)}
        nxt, _ = realized_next_returns(panel, t_idx, window.columns, cfg)  # realized t+1 returns
        d1 = dates[t_idx + 1]                                          # month the return is earned
        row, trow = {"date": d1}, {"date": d1}
        for name, raw in port.items():
            w = project_simplex(raw / raw.sum(), cfg.max_weight)       # normalize and cap at 5%
            w = pd.Series(w, index=window.columns)
            trow[name] = _turnover(w, prev_w.get(name), prev_r)
            row[name] = float((w * nxt).sum())                         # gross return
            prev_w[name] = w                                           # keep for next month
            h = w[w > 1e-12]
            held.append(pd.DataFrame({"date": d1, "strategy": name, "PERMNO": h.index,
                                      "weight": h.to_numpy(), "ret": nxt.reindex(h.index).fillna(0.0).to_numpy()}))
        prev_r = nxt
        rows.append(row)
        turns.append(trow)
    out.mkdir(parents=True, exist_ok=True)
    gross = pd.DataFrame(rows).set_index("date")                       # monthly gross returns
    turn = pd.DataFrame(turns).set_index("date")
    gross.to_csv(out / "monthly_returns_gross.csv")
    turn.to_csv(out / "turnover.csv")
    pd.concat(held, ignore_index=True).to_csv(out / HOLDINGS_FILE, index=False, float_format="%.10g")
    return gross, turn


def main(cfg=None, vmcfg=None):                                                 # compare with the main strategy
    """Run the loop if needed, then compare the benchmarks with the main strategy, with and without the layer."""
    cfg = cfg or Config()
    vmcfg = vmcfg or VMConfig()
    sub = cfg.output_dir / "ivp_benchmark"
    if (sub / "turnover.csv").exists():                                # done before
        gross_b = pd.read_csv(sub / "monthly_returns_gross.csv", index_col="date", parse_dates=True)
        turn_b = pd.read_csv(sub / "turnover.csv", index_col="date", parse_dates=True)
    else:
        gross_b, turn_b = run_loop(cfg, sub)
    gross, turns, bench = load_results(cfg)
    rf, bps = bench["RF"], cfg.cost_bps_main
    S, _ = main_strategy(gross, turns, bench, cfg)
    cols = [S, *NAMES, "NoCluster-LW-gmvp"]
    g = pd.concat([gross[[S, "NoCluster-LW-gmvp"]], gross_b], axis=1)[cols]
    t = pd.concat([turns[[S, "NoCluster-LW-gmvp"]], turn_b], axis=1)[cols]
    holdings = {**load_holdings(cfg.output_dir), **load_holdings(sub)}  # stock holdings for the netted costs
    vg, vt, _, _ = apply_layer(g, t, rf, vmcfg, holdings)
    base, layer = net_returns(g, t, bps), net_returns(vg, vt, bps)     # after costs

    eff_n = {}                                                         # effective number of stocks
    for name in NAMES:
        h = pd.read_csv(sub / HOLDINGS_FILE, parse_dates=["date"])
        by = h[h.strategy == name].groupby("date")["weight"]
        eff_n[name] = ((1.0 / by.apply(lambda v: (v ** 2).sum())).mean(), by.max().mean())  # effective N, largest weight

    rows = []
    print(f"{'':20s} {'no layer: Sharpe  MDD    CAGR':>32s}   {'layer: Sharpe  MDD    CAGR':>30s}  turnover")
    for c in cols:
        r0, r1 = base[c], layer[c]                                     # without / with the layer
        row = {"Portfolio": c, "Sharpe": sharpe(r0, rf), "MDD": mdd(r0), "CAGR": ann_return(r0),
               "Sharpe layer": sharpe(r1, rf), "MDD layer": mdd(r1), "CAGR layer": ann_return(r1),
               "Turnover": t[c].iloc[1:].mean(), "Turnover layer": vt[c].iloc[1:].mean()}
        for lab, a, z in HALVES:                                       # sub-periods (no layer)
            row[f"Sharpe {lab}"] = sharpe(r0.loc[a:z], rf.loc[a:z])
        if c in eff_n:
            row["Effective N"], row["Largest weight"] = eff_n[c]
        rows.append(row)
        print(f"{c:20s} {row['Sharpe']:8.3f} {row['MDD']:7.1%} {row['CAGR']:6.2%}   "
              f"{row['Sharpe layer']:8.3f} {row['MDD layer']:7.1%} {row['CAGR layer']:6.2%}  "
              f"{row['Turnover']:5.1%} -> {row['Turnover layer']:5.1%}")
    table = pd.DataFrame(rows).set_index("Portfolio")
    print("\nSharpe by half (no layer):")
    print(table[[f"Sharpe {lab}" for lab, _, _ in HALVES]].round(3).to_string())
    print("\nEqual Sharpe tests against the main strategy (Jobson-Korkie, Memmel):")
    for name in NAMES:
        for lab, a, b in (("no layer", base[S], base[name]), ("layer", layer[S], layer[name])):
            z, p = jk_memmel(a, b, rf)
            table.loc[name, f"p vs main ({lab})"] = p
            print(f"    {name:15s} {lab:9s} z {z:+.2f}  p {p:.2f}")
        print(f"    {name:15s} correlation with the main strategy {base[S].corr(base[name]):.3f}, "
              f"effective N {eff_n[name][0]:.0f}, largest weight {eff_n[name][1]:.1%}")
    table.to_csv(cfg.output_dir / "ivp_benchmark.csv", encoding="utf-8-sig", float_format="%.4f")
    print(f"\nDone -> {cfg.output_dir / 'ivp_benchmark.csv'}")


if __name__ == "__main__":
    main()
