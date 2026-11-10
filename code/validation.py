"""
Robustness checks for the volatility-managed layer:  python validation.py   (step [6-1] of main.py)

Checks that the layer's results are not driven by (1) a particular setting, (2) a particular period or event,
or (3) simply holding less exposure. Uses only the monthly results saved in output/ (no backtest rerun).

  [1] Parameter sensitivity   leverage cap × c burn-in × signal definition × c estimation method
                              (every setting is charged on the netted order, like the reported model)
  [2] Periods                 the full sample, plus any sub-periods set in config.split_dates (none by default)
  [3] Crisis exclusion        does the Sharpe gain remain without the GFC and COVID months?
  [4] Control                 a portfolio scaled down by a *fixed* ratio to the layer's volatility
  [5] Block bootstrap         distribution of ΔSharpe and ΔMDD (12-month blocks, paths resampled)
  [6] Borrowing spread        the reported layer (netted costs) when leverage above 1x pays rf + a spread
  [7] Own-volatility signal   the portfolio's own past monthly variance as the signal instead of market RV
  [8] Levered months          months at high exposure where the portfolio still fell (signal mismatch)

Output: output/vm_validation.csv, output/vm_validation.png
Sharpe is arithmetic: (mean monthly excess return × 12) / (monthly std × √12): same definition as metrics.performance.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import Config, VMConfig
from metrics import net_returns, periods_of, sharpe, ann_return, max_drawdown as mdd
from report import plot_style, load_results, main_strategy
from volatility import apply_layer, layer_turnover, load_holdings, vm_weights

CRISES = {"GFC": ("2007-11", "2009-03"), "COVID": ("2020-02", "2020-04")}
#   GFC runs from the month after the S&P 500 peak (2007-10) to its trough (2009-03), the same window as the report
BOOT_N, BOOT_BLOCK = 2000, 12                                          # bootstrap repetitions, block length (months)


def block_bootstrap(base_ex, vm_ex, rf, seed=0):                                # 12-month block bootstrap
    """Resample (base, layer) pairs in 12-month blocks to get the distributions of ΔSharpe and ΔMDD."""
    rs = np.random.RandomState(seed)                                   # generator separate from the global one
    a, b, f = base_ex.to_numpy(), vm_ex.to_numpy(), rf.to_numpy()
    n, nb = len(a), int(np.ceil(len(a) / BOOT_BLOCK))                  # sample length, number of blocks
    d_sharpe, d_mdd = np.empty(BOOT_N), np.empty(BOOT_N)
    for i in range(BOOT_N):
        st = rs.randint(0, n, nb)                                      # random block starts
        idx = ((st[:, None] + np.arange(BOOT_BLOCK)[None, :]).ravel()[:n]) % n  # circular block indices
        aa, bb, ff = a[idx], b[idx], f[idx]                            # resample both series in the same order
        d_sharpe[i] = (bb.mean() / bb.std(ddof=1) - aa.mean() / aa.std(ddof=1)) * np.sqrt(12.0)
        ca, cb = np.cumprod(1 + aa + ff), np.cumprod(1 + bb + ff)      # cumulative wealth (total return)
        d_mdd[i] = ((cb / np.maximum.accumulate(cb) - 1).min()
                    - (ca / np.maximum.accumulate(ca) - 1).min()) * 100.0  # minus base MDD (%p, positive = improvement)
    return d_sharpe, d_mdd


def layer_net(gross, turns, rf, name, vmcfg, bps):                              # net returns of the layer for one setting
    """Net returns of one strategy with the layer applied."""
    vg, vt, _, _ = apply_layer(gross[[name]], turns[[name]], rf, vmcfg)
    return net_returns(vg, vt, bps)[name]


def main(cfg=None):
    """Run checks [1]–[5] and save the table and figure."""
    cfg = cfg or Config()
    plot_style()
    out = cfg.output_dir
    gross, turns, bench = load_results(cfg)                            # backtest results, RF/S&P 500
    rf = bench["RF"]
    main_name, sel_period = main_strategy(gross, turns, bench, cfg)
    bps = cfg.cost_bps_main
    print(f"Main strategy: {main_name} (chosen on {sel_period}), main cost {bps:g}bp")

    base_net = net_returns(gross, turns, bps)[main_name]
    base_ex = base_net - rf                                            # base excess returns
    rows = {}

    def add(tag, vm_net):
        """Record the result of one setting and return its row."""
        ex = vm_net - rf                                               # layer excess returns
        rows[tag] = {"Sharpe": sharpe(ex), "MDD": mdd(vm_net),
                     "dSharpe": sharpe(ex) - sharpe(base_ex),          # Sharpe difference vs base
                     "dMDD(%p)": (mdd(vm_net) - mdd(base_net)) * 100.0}  # MDD difference vs base (positive = improvement)
        return rows[tag]

    add("Base strategy (no layer)", base_net)

    print("\n[1] Parameter sensitivity (ΔSharpe / ΔMDD %p)")
    print(f"{'':12s}" + "".join(f"{'burn ' + str(b):>22s}" for b in (24, 36, 60)))
    for cap in (1.0, 1.25, 1.5, 2.0, 3.0, np.inf):                     # leverage cap
        cells = []
        for burn in (24, 36, 60):                                      # c burn-in
            r = add(f"[1] cap={cap} burn={burn}",
                    layer_net(gross, turns, rf, main_name, VMConfig(cap=cap, burn_in=burn), bps))  # net returns for this setting
            cells.append(f"{r['dSharpe']:+.2f} / {r['dMDD(%p)']:+5.1f}")
        print(f"cap {cap:<8}" + "".join(f"{c:>22s}" for c in cells))
    for sig in ("rv", "var", "sd", "rv3"):
        add(f"[1] signal={sig}", layer_net(gross, turns, rf, main_name, VMConfig(signal=sig), bps))
    for mode in ("realtime", "full"):                                  # c estimation method (full has no cap, as in the paper)
        vmcfg = VMConfig(c_mode=mode, cap=np.inf if mode == "full" else 1.5)
        add(f"[1] c={mode}", layer_net(gross, turns, rf, main_name, vmcfg, bps))

    vg, vt, w, signal = apply_layer(gross[[main_name]], turns[[main_name]], rf, VMConfig())
    vm_net = net_returns(vg, vt, bps)[main_name]                       # net returns with default settings
    vm_ex = vm_net - rf

    print("\n[2] Sub-periods (default settings)")
    for label, idx in periods_of(gross.index, cfg.split_dates):
        s0, s1 = sharpe(base_ex.loc[idx]), sharpe(vm_ex.loc[idx])      # Sharpe (base, layer)
        m0, m1 = mdd(base_net.loc[idx]), mdd(vm_net.loc[idx])          # MDD (base, layer)
        rows[f"[2] {label}"] = {"Sharpe": s1, "MDD": m1, "dSharpe": s1 - s0,
                                "dMDD(%p)": (m1 - m0) * 100.0}
        print(f"    {label:16s} base {s0:5.2f} / {m0*100:6.1f}%   ->   layer {s1:5.2f} / {m1*100:6.1f}%"
              f"   (ΔSharpe {s1-s0:+.2f}, ΔMDD {(m1-m0)*100:+.1f}%p)")

    print("\n[3] Excluding crisis months")
    P = gross.index.to_period("M")
    combos = [("Full sample", []), *[(f"Excl. {k}", [v]) for k, v in CRISES.items()],
              ("Excl. both", list(CRISES.values()))]
    for name, wins in combos:
        keep = pd.Series(True, index=P)                                # keep mask
        for a, b in wins:
            keep.loc[a:b] = False
        k = keep.to_numpy()
        s0, s1 = sharpe(base_ex[k]), sharpe(vm_ex[k])                  # Sharpe (base, layer)
        rows[f"[3] {name}"] = {"Sharpe": s1, "dSharpe": s1 - s0, "n": int(k.sum())}
        print(f"    {name:12s} (n={k.sum():3d})  base {s0:5.2f}  ->  layer {s1:5.2f}   (ΔSharpe {s1-s0:+.2f})")

    print("\n[4] Control (fixed scaling to the same volatility)")
    scale = vm_ex.std(ddof=1) / base_ex.std(ddof=1)                    # fixed ratio matching the layer's volatility
    ctrl_net = rf + scale * base_ex
    add(f"[4] Control ({scale:.2f}x fixed)", ctrl_net)
    print(f"    ratio {scale:.2f} | control MDD {mdd(ctrl_net)*100:6.1f}%  vs  "
          f"layer {mdd(vm_net)*100:6.1f}%  (base {mdd(base_net)*100:6.1f}%)")

    print("\n[5] Block bootstrap (12-month blocks, 2,000 draws)")
    d_sharpe, d_mdd = block_bootstrap(base_ex, vm_ex, rf)
    rows["[5] Bootstrap"] = {"P(dSharpe>0)": float((d_sharpe > 0).mean()),
                             "dSharpe median": float(np.median(d_sharpe)),
                             "P(dMDD>0)": float((d_mdd > 0).mean()),
                             "dMDD median(%p)": float(np.median(d_mdd))}
    print(f"    ΔSharpe > 0 in {np.mean(d_sharpe > 0):.1%} (median {np.median(d_sharpe):+.2f}), "
          f"MDD improved in {np.mean(d_mdd > 0):.1%} (median {np.median(d_mdd):+.1f}%p)")

    print("\n[6] Borrowing spread on the levered part (reported layer, netted costs)")
    rep_net = vm_net
    borrowed = (w[main_name] - 1.0).clip(lower=0.0)                    # share financed above 1x
    print(f"    average borrowed share {borrowed.mean():.1%}, months above 1x: {(borrowed > 0).sum()}")
    for spread in (0, 50, 100, 200):                                   # annual spread over rf (bp)
        r = rep_net - borrowed * spread / 1e4 / 12.0                   # pay the spread on the borrowed part
        row = add(f"[6] spread={spread}bp", r)
        row["CAGR"] = ann_return(r)                                    # annual return too
        print(f"    {spread:3d}bp  Sharpe {row['Sharpe']:.3f}  CAGR {row['CAGR']:.2%}  MDD {row['MDD']*100:.1f}%")

    print("\n[7] Signal from the portfolio's own past monthly variance (compare with [1] signal=rv)")
    x = gross[main_name] - rf                                          # gross excess returns, as in apply_layer
    for months in (3, 6, 12):                                          # look-back for the own variance
        own = x.rolling(months).var().shift(1)                         # variance known at the start of the month
        w_own = vm_weights(x, own, VMConfig())                         # same exposure rule, other signal
        turn = layer_turnover(w_own, gross[main_name], rf, load_holdings(cfg.output_dir)[main_name])
        r = rf + w_own * x - bps / 1e4 * turn
        row = add(f"[7] own variance {months}m", r)
        print(f"    {months:2d} months  Sharpe {row['Sharpe']:.3f}  MDD {row['MDD']*100:.1f}%")
    print(f"    market RV (default)  Sharpe {rows['[1] signal=rv']['Sharpe']:.3f}  MDD {rows['[1] signal=rv']['MDD']*100:.1f}%")

    print("\n[8] High-exposure months (w > 1.2) where the layer still lost more than 3%")
    lev = pd.DataFrame({"w": w[main_name], "layer": rep_net, "base": base_net,
                        "mkt vol": np.sqrt(signal * 12.0)})            # previous-month market vol (annual)
    lev = lev[lev.w > 1.2]
    hurt = lev[lev.layer < 0]                                          # levered months with a loss
    bad = lev[lev.layer < -0.03].sort_values("layer")
    rows["[8] Levered months"] = {"n": len(lev), "n loss > 3%": len(bad),
                                  "loss sum layer": hurt.layer.sum(), "loss sum base": hurt.base.sum()}
    print(f"    {len(lev)} months with w > 1.2, {len(bad)} of them lost more than 3%")
    print(f"    summed losses in levered losing months: layer {hurt.layer.sum():.1%} vs base {hurt.base.sum():.1%}")
    print(bad.head(5).to_string(float_format=lambda v: f"{v:.3f}"))

    pd.DataFrame(rows).T.to_csv(out / "vm_validation.csv", encoding="utf-8-sig", float_format="%.4f")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"Volatility-managed layer robustness: {main_name} (cost {bps:g}bp)",
                 fontsize=13, fontweight="bold", y=1.02)
    ax = axes[0]                                                       # (a) drawdown curves
    for name, r, c in [("Base", base_net, "#d62728"), ("Layer", vm_net, "#7b2cbf"),
                       (f"Control ({scale:.2f}x)", ctrl_net, "#8c8c8c")]:
        wealth = (1.0 + r).cumprod()
        ax.plot(wealth.index, (wealth / wealth.cummax() - 1.0) * 100, lw=1.5, color=c,
                label=f"{name} (MDD {mdd(r)*100:.1f}%)")
    ax.set_title("(a) Drawdown (%)", fontsize=11, loc="left")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    ax = axes[1]                                                       # (b) exposure
    ax.plot(w.index, w[main_name], lw=1.4, color="#7b2cbf", label="Exposure w")
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax2 = ax.twinx()
    ax2.fill_between(signal.index, 0, np.sqrt(signal * 12.0) * 100.0, color="#4363d8",
                     alpha=0.15, label="Previous-month market vol (ann., %)")
    ax2.set_ylabel("Market volatility (%)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
    ax.set_title("(b) Exposure and market volatility", fontsize=11, loc="left")
    ax.grid(alpha=0.3)
    ax = axes[2]                                                       # (c) bootstrap
    ax.hist(d_sharpe, bins=45, color="#7b2cbf", alpha=0.75)
    ax.axvline(0, color="black", lw=1.2)
    ax.text(0.03, 0.95, f"ΔSharpe > 0: {np.mean(d_sharpe > 0):.0%}\n"
                        f"median {np.median(d_sharpe):+.2f}\n"
                        f"MDD improved: {np.mean(d_mdd > 0):.0%}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    ax.set_title("(c) Block bootstrap ΔSharpe", fontsize=11, loc="left")
    ax.set_xlabel("ΔSharpe (layer - base)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "vm_validation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDone -> {out / 'vm_validation.csv'}, {out / 'vm_validation.png'}")


if __name__ == "__main__":
    main()
