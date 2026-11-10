"""
Comparison across transaction cost levels:  python cost_sensitivity.py   (step [6-2] of main.py)

No backtest rerun is needed: net return = gross return − (bp/10,000) × turnover, so any cost level
can be computed from the saved gross returns and turnover.

  [1] Performance by cost level (main strategy: base vs layer)
  [2] Break-even cost: the cost at which the layer's Sharpe and return advantage disappears
  [3] At the main cost (20bp): by 4-year block and by strategy

Output: output/cost_comparison.csv, output/cost_comparison.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import Config, VMConfig
from metrics import net_returns, sharpe, ann_return, max_drawdown as mdd
from report import plot_style, load_results, main_strategy
from volatility import apply_layer

COSTS = (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0)                        # one-way costs compared (bp)
SUBPERIODS = [("2005-02~2008-12", "2005-02", "2008-12"),               # 4-year block 1 (includes the GFC)
              ("2009-01~2012-12", "2009-01", "2012-12"),               # 4-year block 2
              ("2013-01~2016-12", "2013-01", "2016-12"),               # 4-year block 3
              ("2017-01~2020-12", "2017-01", "2020-12")]               # 4-year block 4 (includes COVID)


def main(cfg=None, vmcfg=None):
    """Run comparisons [1]–[3] and save the table and figure."""
    cfg = cfg or Config()
    vmcfg = vmcfg or VMConfig()
    plot_style()
    out = cfg.output_dir
    gross, turns, bench = load_results(cfg)                            # backtest results, RF/S&P 500
    rf = bench["RF"]
    main_name, _ = main_strategy(gross, turns, bench, cfg)
    main_bps = cfg.cost_bps_main
    vm_gross, vm_turn, _, _ = apply_layer(gross, turns, rf, vmcfg)
    print(f"Main strategy: {main_name} | average monthly turnover base {turns[main_name].mean():.2f} -> "
          f"layer {vm_turn[main_name].mean():.2f}")                     # layer turnover (netted)

    rows = []
    print("\n[1] By cost level (full period, main strategy)")
    print(f"{'Cost':>6s} | {'Base Sharpe':>11s} {'Layer':>8s} {'Diff':>7s} | "
          f"{'Base ret.':>11s} {'Layer':>8s} | {'Base MDD':>9s} {'Layer':>8s}")
    for bps in COSTS:
        b = net_returns(gross, turns, bps)[main_name]
        v = net_returns(vm_gross, vm_turn, bps)[main_name]             # layer net returns
        sb, sv = sharpe(b, rf), sharpe(v, rf)
        rows.append({"Cost (bp)": bps, "Base Sharpe": sb, "Layer Sharpe": sv, "dSharpe": sv - sb,
                     "Base ann. return": ann_return(b), "Layer ann. return": ann_return(v),  # annualized returns
                     "Base MDD": mdd(b), "Layer MDD": mdd(v)})
        print(f"{bps:5.0f}bp | {sb:11.3f} {sv:8.3f} {sv-sb:+7.3f} | "
              f"{ann_return(b):10.2%} {ann_return(v):8.2%} | {mdd(b):9.1%} {mdd(v):8.1%}")

    print("\n[2] Break-even cost (where the layer's advantage disappears)")
    grid = np.arange(0.0, 300.0, 0.5)                                  # 0–300bp in 0.5bp steps
    d_sharpe, d_ret = [], []
    for bps in grid:
        b = net_returns(gross, turns, bps)[main_name]
        v = net_returns(vm_gross, vm_turn, bps)[main_name]
        d_sharpe.append(sharpe(v, rf) - sharpe(b, rf))
        d_ret.append(ann_return(v) - ann_return(b))                    # annualized return difference
    d_sharpe, d_ret = np.array(d_sharpe), np.array(d_ret)
    be_s = grid[np.argmax(d_sharpe <= 0)] if (d_sharpe <= 0).any() else np.inf
    be_r = grid[np.argmax(d_ret <= 0)] if (d_ret <= 0).any() else np.inf
    print(f"    Sharpe advantage disappears at: {be_s:.0f}bp" if np.isfinite(be_s) else
          "    Sharpe advantage remains up to 300bp")                  # if none
    print(f"    Return advantage disappears at: {be_r:.0f}bp" if np.isfinite(be_r) else
          "    Return advantage remains up to 300bp")                  # if none
    rows.append({"Cost (bp)": "Break-even (Sharpe)", "Base Sharpe": be_s})  # record (value in bp)
    rows.append({"Cost (bp)": "Break-even (return)", "Base Sharpe": be_r})  # record (value in bp)

    print(f"\n[3] At {main_bps:g}bp")
    b_m = net_returns(gross, turns, main_bps)
    v_m = net_returns(vm_gross, vm_turn, main_bps)
    print("  By 4-year block (main strategy)")
    for label, a, z in SUBPERIODS:                                     # per 4-year block
        bb, vv = b_m[main_name].loc[a:z], v_m[main_name].loc[a:z]      # block returns
        rf_s = rf.loc[a:z]                                             # block risk-free returns
        rows.append({"Cost (bp)": main_bps, "Period": label, "Strategy": main_name,
                     "Base Sharpe": sharpe(bb, rf_s), "Layer Sharpe": sharpe(vv, rf_s),
                     "dSharpe": sharpe(vv, rf_s) - sharpe(bb, rf_s),
                     "Base MDD": mdd(bb), "Layer MDD": mdd(vv)})
        print(f"    {label:16s} Sharpe {sharpe(bb, rf_s):5.2f} -> {sharpe(vv, rf_s):5.2f}   "
              f"MDD {mdd(bb):6.1%} -> {mdd(vv):6.1%}")
    print("  By strategy (full period)")
    for col in gross.columns:
        sb, sv = sharpe(b_m[col], rf), sharpe(v_m[col], rf)
        rows.append({"Cost (bp)": main_bps, "Period": "Full", "Strategy": col,
                     "Base Sharpe": sb, "Layer Sharpe": sv, "dSharpe": sv - sb,
                     "Base MDD": mdd(b_m[col]), "Layer MDD": mdd(v_m[col])})
        print(f"    {col:24s} {sb:5.2f} -> {sv:5.2f} ({sv-sb:+.2f})   "
              f"MDD {mdd(b_m[col]):6.1%} -> {mdd(v_m[col]):6.1%}")

    pd.DataFrame(rows).to_csv(out / "cost_comparison.csv", index=False,
                              encoding="utf-8-sig", float_format="%.4f")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))                  # 2 panels
    fig.suptitle(f"Comparison across cost levels: {main_name}", fontsize=13, fontweight="bold", y=1.02)
    ax = axes[0]                                                       # (a) Sharpe by cost level
    costs = np.array(COSTS)
    sb = [r["Base Sharpe"] for r in rows[:len(COSTS)]]
    sv = [r["Layer Sharpe"] for r in rows[:len(COSTS)]]
    ax.plot(costs, sb, "o-", color="#d62728", lw=1.8, label="Base")
    ax.plot(costs, sv, "o-", color="#7b2cbf", lw=1.8, label="+ Volatility management")
    ax.axvline(main_bps, color="#333333", ls="--", lw=1.2)
    ax.text(main_bps + 0.5, min(sb) + 0.02, f"{main_bps:g}bp", fontsize=9)
    ax.set_xlabel("One-way transaction cost (bp)")
    ax.set_ylabel("Sharpe")
    ax.set_title("(a) Sharpe by cost level", fontsize=11, loc="left")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax = axes[1]                                                       # (b) size of the advantage
    ax.plot(grid, d_sharpe, color="#7b2cbf", lw=1.8)
    ax.axhline(0, color="black", lw=1.0)
    if np.isfinite(be_s):                                              # if there is a break-even
        ax.axvline(be_s, color="#d62728", ls="--", lw=1.2)
        ax.text(be_s + 3, max(d_sharpe) * 0.8, f"Break-even {be_s:.0f}bp",
                fontsize=10, color="#d62728")
    ax.set_xlabel("One-way transaction cost (bp)")
    ax.set_ylabel("ΔSharpe (layer - base)")
    ax.set_title("(b) When does the advantage disappear as costs rise?", fontsize=11, loc="left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cost_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDone -> {out / 'cost_comparison.csv'}, {out / 'cost_comparison.png'}")


if __name__ == "__main__":
    main()
