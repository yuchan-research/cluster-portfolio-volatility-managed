"""
Stock holdings for the layer's trading costs:  python netted_turnover.py   (step [4] of main.py)

The layer scales the whole portfolio by w_t. Each month the rebalance to the new base weights and the
exposure change go out as one order per stock, so a stock the rebalance buys and the exposure cut sells
is only traded for the difference:

    target today      w_t · a_it                       a_it = base weights at t (sum 1)
    held after drift  E_{t−1} · d_it                   d_it = last month's base weights after drifting
                      E_{t−1} = w_{t−1} (1 + r_base,t) / (1 + r_layer,t)   share in stocks before rebalancing
    turnover          Σ_i | w_t · a_it − E_{t−1} · d_it |

This is the only way the layer's costs are charged in the project (volatility.layer_turnover). It needs
the stock weights of every month, which the backtest saves in output/holdings.csv.gz. If that file is
missing (results from an older run, or a seed folder), this script reruns the monthly loop to rebuild it
and checks that the rebuilt holdings give back the saved monthly returns.

Output: output/holdings.csv.gz, output/netted_turnover.csv (layer turnover per strategy, default settings)
"""

import time
from dataclasses import replace
import pandas as pd
from config import Config, VMConfig
from data import load_panel
from backtest import run_backtest
from report import load_results
from volatility import apply_layer, load_holdings, HOLDINGS_FILE


def build_holdings(cfg, out_dir, strategy=None):                                # rerun the loop for the holdings
    """Rerun the monthly loop and save the stock holdings to out_dir; strategy limits it to one cluster strategy."""
    run = replace(cfg, run_benchmarks=strategy is None)                # benchmarks only for the full set
    if strategy is not None:                                           # one cluster strategy is enough (seed runs)
        beta, objective = strategy.replace("Cluster-", "").split("-")  # its β method and objective
        run = replace(run, intra_methods=(beta,), objectives=(objective,))  # only that combination
    t0 = time.time()
    holdings = []
    run_backtest(load_panel(run), run, holdings=holdings)              # same loop as the backtest
    h = pd.concat(holdings, ignore_index=True)
    saved = pd.read_csv(out_dir / "monthly_returns_gross.csv", index_col="date", parse_dates=True)
    rebuilt = (h.weight * h.ret).groupby([h.date, h.strategy]).sum().unstack()
    gap = (rebuilt - saved[rebuilt.columns].reindex(rebuilt.index)).abs().max().max()
    if gap > 1e-9:                                                     # the rerun did not reproduce the backtest
        raise RuntimeError(f"rebuilt holdings do not match the saved returns (max diff {gap:.2e})")
    h.to_csv(out_dir / HOLDINGS_FILE, index=False, float_format="%.10g")
    print(f"    holdings rebuilt in {time.time() - t0:.0f}s, returns match the backtest (max diff {gap:.1e})")
    return h


def main(cfg=None, vmcfg=None):
    """Rebuild the holdings if needed, then save the layer's netted turnover at the default settings."""
    cfg = cfg or Config()
    vmcfg = vmcfg or VMConfig()
    out = cfg.output_dir
    if not (out / HOLDINGS_FILE).exists():                             # older results without holdings
        print(f"    {HOLDINGS_FILE} not found -> rerunning the monthly loop (about 20 minutes)")
        build_holdings(cfg, out)
    gross, turns, bench = load_results(cfg)
    _, vm_turn, _, _ = apply_layer(gross, turns, bench["RF"], vmcfg, load_holdings(out))
    vm_turn.to_csv(out / "netted_turnover.csv")
    print(f"    {'strategy':22s} {'base':>6s} {'layer':>6s}   (average monthly turnover)")
    for col in gross.columns:
        print(f"    {col:22s} {turns[col].iloc[1:].mean():6.1%} {vm_turn[col].iloc[1:].mean():6.1%}")


if __name__ == "__main__":
    main()
