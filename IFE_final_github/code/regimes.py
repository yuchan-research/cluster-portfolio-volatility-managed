"""
Performance by market regime:  python regimes.py   (step [6-5] of main.py)

The full-period table hides the fact that the volatility layer helps in one kind of market and costs
money in another. This splits the 191 months into five regimes at dates chosen from market history,
not from our results, and reports the main model, the same portfolio without the layer, and the S&P 500.

  Before crisis    2005-02 ~ 2007-10   calm bull market, up to the S&P 500 peak (2007-10-09)
  Financial crisis 2007-11 ~ 2009-03   peak to trough (2009-03-09)
  Recovery         2009-04 ~ 2014-12   QE era, until QE3 purchases ended (2014-10)
  Late bull market 2015-01 ~ 2019-12   calm years before COVID
  COVID            2020-01 ~ 2020-12   crash and rebound

Uses only the monthly results saved in output/ (no backtest rerun).

Output: output/regimes.csv, output/regimes.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import Config, VMConfig
from metrics import net_returns, sharpe, ann_return, max_drawdown as mdd, SP500_NAME
from report import plot_style, load_results, main_strategy
from volatility import apply_layer

REGIMES = [("Before crisis", "2005-02", "2007-10"),
           ("Financial crisis", "2007-11", "2009-03"),  # S&P 500 peak to trough
           ("Recovery", "2009-04", "2014-12"),
           ("Late bull market", "2015-01", "2019-12"),  # calm years before COVID
           ("COVID", "2020-01", "2020-12")]


def main(cfg=None, vmcfg=None):
    """Report each regime and save the table and figure."""
    cfg = cfg or Config()
    vmcfg = vmcfg or VMConfig()
    plot_style()
    out = cfg.output_dir
    gross, turns, bench = load_results(cfg)
    rf, spx = bench["RF"], bench["SP500TR"]
    name, _ = main_strategy(gross, turns, bench, cfg)
    bps = cfg.cost_bps_main
    vm_gross, vm_turn, weights, _ = apply_layer(gross, turns, rf, vmcfg)

    base = net_returns(gross, turns, bps)[name]                        # portfolio without the layer
    main_net = net_returns(vm_gross, vm_turn, bps)[name]               # main model (with the layer)
    series = {"Main": main_net, "No layer": base, SP500_NAME: spx}
    w = weights[name]
    print(f"Main strategy {name}, cost {bps:g}bp, layer cap {vmcfg.cap}x\n")

    rows = []
    head = (f"{'Regime':18s} {'months':>6s} | " + " ".join(f"{k:>9s}" for k in series)
            + " | " + " ".join(f"{k + ' ret':>11s}" for k in series) + f" | {'exposure':>8s}")
    print(head)
    for label, a, b in REGIMES:
        m = (gross.index >= a) & (gross.index <= pd.Period(b, "M").end_time)
        row = {"Regime": label, "From": a, "To": b, "Months": int(m.sum()),
               "Exposure": float(w[m].mean())}                         # average exposure set by the layer
        for k, r in series.items():
            row[f"{k} Sharpe"] = sharpe(r[m], rf)
            row[f"{k} Return"] = ann_return(r[m])
            row[f"{k} MDD"] = mdd(r[m])
        rows.append(row)
        print(f"{label:18s} {int(m.sum()):6d} | "
              + " ".join(f"{row[k + ' Sharpe']:9.2f}" for k in series)
              + " | " + " ".join(f"{row[k + ' Return']:10.1%}" for k in series)
              + f" | {row['Exposure']:7.2f}x")

    print(f"\n{'Regime':18s} " + " ".join(f"{k + ' MDD':>12s}" for k in series))
    for row in rows:
        print(f"{row['Regime']:18s} " + " ".join(f"{row[k + ' MDD']:11.1%}" for k in series))

    pd.DataFrame(rows).to_csv(out / "regimes.csv", index=False,
                              encoding="utf-8-sig", float_format="%.4f")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                   gridspec_kw={"height_ratios": [2.4, 1]})
    fig.suptitle(f"Performance by regime: {name} (cost {bps:g}bp)",
                 fontsize=13, fontweight="bold", y=0.98)
    xs, width = np.arange(len(rows)), 0.26
    for i, (k, colour) in enumerate(zip(series, ("#7b2cbf", "#d62728", "#8c8c8c"))):
        vals = [r[f"{k} Sharpe"] for r in rows]
        bars = ax1.bar(xs + (i - 1) * width, vals, width, color=colour, label=k)
        for bar, v in zip(bars, vals):
            ax1.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v),
                         ha="center", va="bottom" if v >= 0 else "top",
                         fontsize=8, xytext=(0, 2 if v >= 0 else -2), textcoords="offset points")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylabel("Sharpe ratio")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([])                                            # labels only on the lower panel
    ax1.legend(fontsize=9, ncol=3, loc="upper center")
    ax1.grid(alpha=0.3, axis="y")
    bars = ax2.bar(xs, [r["Exposure"] for r in rows], 0.5, color="#7b2cbf")
    for bar, r in zip(bars, rows):
        ax2.annotate(f"{r['Exposure']:.2f}x", (bar.get_x() + bar.get_width() / 2, r["Exposure"]),
                     ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax2.axhline(1.0, color="#555555", lw=0.8, ls="--")                 # 1x reference line
    ax2.set_ylabel("average exposure")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{r['Regime']}\n{r['From']}~{r['To']}" for r in rows], fontsize=8)
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "regimes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDone -> {out / 'regimes.csv'}, {out / 'regimes.png'}")


if __name__ == "__main__":
    main()
