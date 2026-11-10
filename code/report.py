"""
Builds performance tables and figures from the saved backtest results (output/). Does not rerun the backtest.

  make_report      performance tables and figures for the cluster strategies and benchmarks (step ⑨)
  make_vm_report   applies the volatility-managed layer (⑧), then base vs layer tables, alpha regression and figures

Run:     python report.py   (rebuilds both)
Input:   output/monthly_returns_gross.csv, output/turnover.csv, output/diagnostics.csv, data/benchmarks_monthly.csv
Output:  output/metrics.csv, output/main_summary.csv, output/monthly_returns_net.csv, output/results.png
         output/vm_monthly_returns_gross.csv  gross returns after the layer
         output/vm_turnover.csv               turnover after the layer (netted order per stock)
         output/vm_weights.csv                monthly exposure w_t and the previous-month market volatility signal
         output/vm_metrics.csv                cost × period × strategy × (Base/Layer) metrics
         output/vm_alpha.csv                  [MM17] managed vs unmanaged alpha regression
         output/vm_summary.csv                summary for the main strategy and benchmarks
         output/vm_results.png                cumulative wealth, drawdown and exposure figure

The main strategy is selected with the team rule (metrics.select_main_strategy) from the **base strategy table**.
Layer results are not used for the selection.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import Config, VMConfig
from benchmarks import load_benchmarks
from metrics import (performance, performance_table, periods_of, net_returns,
                     select_main_strategy, SP500_NAME)                          # main strategy rule, benchmark name
from volatility import apply_layer, mm_alpha


def plot_style():
    """Common figure style for the layer and analysis figures."""
    plt.rcParams["axes.unicode_minus"] = False                        # render minus signs with a plain hyphen


def load_results(cfg):
    """Read backtest results (gross returns, turnover) and RF/S&P 500 on the same dates."""
    out = cfg.output_dir
    gross = pd.read_csv(out / "monthly_returns_gross.csv", index_col="date", parse_dates=True)
    turns = pd.read_csv(out / "turnover.csv", index_col="date", parse_dates=True)
    bench = load_benchmarks(gross.index)                              # RF, S&P 500
    return gross, turns, bench


def main_strategy(gross, turns, bench, cfg):
    """Select the main strategy from the base table with the team rule (layer results are not used)."""
    table = performance_table(gross, turns, cfg.cost_bps_list, cfg.split_dates, bench)
    return select_main_strategy(table, cfg)                           # (main strategy name, period it was chosen on)


def plot_results(net, bench, diags, main_name, path):                           # cumulative wealth and drawdown figure
    """Cumulative wealth (log), drawdown and K over time, at the main cost level."""
    curves = net.copy()
    curves[SP500_NAME] = bench["SP500TR"]                             # add S&P 500 Total Return
    wealth = (1.0 + curves).cumprod()
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    for col in wealth:
        lw = 2.4 if col in (main_name, SP500_NAME) else 1.0           # thicker lines for the main strategy and S&P 500
        axes[0].plot(wealth.index, wealth[col], lw=lw, label=col)     # 1) cumulative wealth
        axes[1].plot(wealth.index, wealth[col] / wealth[col].cummax() - 1.0, lw=lw, label=col)  # 2) drawdown
    axes[0].set_yscale("log")
    axes[0].set_title("Portfolio value of $1 (net of costs, log scale)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_title("Drawdown")
    axes[2].plot(diags.index, diags["K"], lw=1.4, label="K used (Marchenko–Pastur, bounded)")  # 3) K used
    axes[2].set_title("Number of clusters per month")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_report(cfg=None):                                                      # base strategy tables and figure
    """Performance tables, main strategy selection and figures for the base strategies."""
    cfg = cfg or Config()
    out = cfg.output_dir
    gross = pd.read_csv(out / "monthly_returns_gross.csv", index_col="date", parse_dates=True)
    turns = pd.read_csv(out / "turnover.csv", index_col="date", parse_dates=True)
    diags = pd.read_csv(out / "diagnostics.csv", index_col="date", parse_dates=True)
    bench = load_benchmarks(gross.index)                              # RF, S&P 500 on the same dates
    table = performance_table(gross, turns, cfg.cost_bps_list, cfg.split_dates, bench)
    table.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    main_name, sel_period = select_main_strategy(table, cfg)          # main strategy by the predefined rule
    summary = table[table["Strategy"].isin([main_name, SP500_NAME, "EW-Universe", "NoCluster-LW-gmvp", "NoCluster-LW-msrp"])]
    summary.to_csv(out / "main_summary.csv", index=False, encoding="utf-8-sig")
    net = net_returns(gross, turns, cfg.cost_bps_main)
    net.to_csv(out / "monthly_returns_net.csv")
    plot_results(net, bench, diags, main_name, out / "results.png")
    view = table[(table["Cost (bp)"] == cfg.cost_bps_main)]
    cols = ["Strategy", "Annualized Return", "Annualized Std", "Sharpe", "Max Drawdown", "Calmar", "Beta vs S&P 500", "Avg Monthly Turnover"]
    for period in view["Period"].unique():
        print(f"\n=== {period} | cost {cfg.cost_bps_main:g}bp ===")
        print(view[view["Period"] == period][cols].round(3).to_string(index=False))
    print(f"\nMain strategy (highest Sharpe over {sel_period}): {main_name}")
    return table, main_name


# ── Volatility-managed layer (⑧) ───────────────────────────────────
def vm_performance_table(gross, turns, vm_gross, vm_turn, bench, cfg):
    """Performance table: cost × period × strategy × (Base/Layer)."""
    rows = []
    for bps in cfg.cost_bps_list:
        base_net = net_returns(gross, turns, bps)
        vm_net = net_returns(vm_gross, vm_turn, bps)                  # layer net returns (netted turnover)
        for label, idx in periods_of(gross.index, cfg.split_dates):
            for col in gross.columns:
                for kind, net, tn in (("Base", base_net, turns), ("Layer", vm_net, vm_turn)):
                    m = performance(net.loc[idx, col], tn[col], bench["RF"], bench["SP500TR"])
                    rows.append({"Cost (bp)": bps, "Period": label, "Strategy": col,
                                 "Layer": kind, **m})
            m = performance(bench["SP500TR"].loc[idx], pd.Series(0.0, index=gross.index),  # S&P 500 (zero turnover)
                            bench["RF"], bench["SP500TR"])            # no costs, no layer
            rows.append({"Cost (bp)": bps, "Period": label, "Strategy": SP500_NAME,
                         "Layer": "Base", **m})
    return pd.DataFrame(rows)


def plot_vm(base_net, vm_net, weights, signal, bench, main_name, path):
    """Cumulative wealth, drawdown and exposure for the main strategy."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 13))
    curves = {"Base " + main_name: base_net[main_name],
              "Layer " + main_name: vm_net[main_name],
              SP500_NAME: bench["SP500TR"]}
    colors = {"Base " + main_name: "#d62728", "Layer " + main_name: "#7b2cbf", SP500_NAME: "#1f77b4"}
    for name, r in curves.items():
        wealth = (1.0 + r).cumprod()
        axes[0].plot(wealth.index, wealth, lw=1.8, color=colors[name], label=name)
        axes[1].plot(wealth.index, (wealth / wealth.cummax() - 1.0) * 100, lw=1.4,  # drawdown (%)
                     color=colors[name], label=name)
    axes[0].set_yscale("log")
    axes[0].set_title("(1) Cumulative value of $1, net of costs (log scale)")
    axes[1].set_title("(2) Drawdown (%)")
    axes[2].plot(weights.index, weights[main_name], lw=1.5, color="#7b2cbf", label="Exposure w")
    axes[2].axhline(1.0, color="black", lw=0.8, ls=":")               # 1x reference line
    ax2 = axes[2].twinx()
    ann_vol = np.sqrt(signal * 12.0) * 100.0                          # signal converted to annualized volatility (approx.)
    ax2.fill_between(signal.index, 0, ann_vol, color="#4363d8", alpha=0.15,
                     label="Previous-month market volatility (ann., %)")
    ax2.set_ylabel("Market volatility (%)")
    axes[2].set_title("(3) Exposure and previous-month market volatility")
    h1, l1 = axes[2].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
    for ax in axes[:2]:
        ax.legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_vm_report(cfg=None, vmcfg=None):
    """Apply the layer to the saved backtest results and build tables, alpha regression and figures."""
    cfg = cfg or Config()
    vmcfg = vmcfg or VMConfig()
    plot_style()
    out = cfg.output_dir
    gross, turns, bench = load_results(cfg)                           # backtest results and RF/S&P 500

    print(f"    Layer settings: cap {vmcfg.cap}x, burn-in {vmcfg.burn_in} months, "
          f"c={vmcfg.c_mode}, signal={vmcfg.signal}")
    vm_gross, vm_turn, weights, signal = apply_layer(gross, turns, bench["RF"], vmcfg)
    vm_gross.to_csv(out / "vm_monthly_returns_gross.csv")
    vm_turn.to_csv(out / "vm_turnover.csv")
    weights.assign(market_signal=signal).to_csv(out / "vm_weights.csv")

    table = vm_performance_table(gross, turns, vm_gross, vm_turn, bench, cfg)
    table.to_csv(out / "vm_metrics.csv", index=False, encoding="utf-8-sig")
    main_name, sel_period = main_strategy(gross, turns, bench, cfg)

    base_net = net_returns(gross, turns, cfg.cost_bps_main)
    vm_net = net_returns(vm_gross, vm_turn, cfg.cost_bps_main)
    alpha = pd.DataFrame({col: mm_alpha(vm_net[col] - bench["RF"], base_net[col] - bench["RF"])
                          for col in gross.columns}).T
    alpha.to_csv(out / "vm_alpha.csv", encoding="utf-8-sig")

    keep = [main_name, "EW-Universe", "NoCluster-LW-gmvp", "NoCluster-LW-msrp", SP500_NAME]
    table[table["Strategy"].isin(keep)].to_csv(out / "vm_summary.csv", index=False, encoding="utf-8-sig")
    plot_vm(base_net, vm_net, weights, signal, bench, main_name, out / "vm_results.png")

    cols = ["Strategy", "Layer", "Annualized Return", "Annualized Std", "Sharpe",
            "Max Drawdown", "Calmar", "Avg Monthly Turnover"]
    view = table[(table["Cost (bp)"] == cfg.cost_bps_main) & (table["Strategy"].isin(keep))]
    for period in view["Period"].unique():
        print(f"\n=== [Layer] {period} | cost {cfg.cost_bps_main:g}bp ===")
        print(view[view["Period"] == period][cols].round(3).to_string(index=False))
    print("\n=== Layer alpha (main cost, annualized) ===")
    print(alpha.round(3).to_string())
    w = weights[main_name]
    print(f"\nMain strategy {main_name} exposure: mean {w.mean():.2f}, "
          f"min {w.min():.2f}, max {w.max():.2f}, at cap in {int((w >= vmcfg.cap - 1e-9).sum())} months")
    return table, main_name


if __name__ == "__main__":
    make_report()                                                     # cluster strategy tables and figures
    make_vm_report()
