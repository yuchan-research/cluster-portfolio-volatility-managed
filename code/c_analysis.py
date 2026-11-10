"""
Analysis of the layer coefficient c:  python c_analysis.py   (step [6-3] of main.py)

The layer's exposure is w_t = clip(c_t / RV_{t-1}, 0, cap). This script reproduces, in one place, what c is
and how much the results change with the way c is estimated.

  [1] Meaning of c          c is essentially "the usual market variance so far" (median of past RV)
  [2] Exposure examples     c / RV and the final exposure in calm and crisis months
  [3] Constant-c invariance without cap and costs, doubling c leaves Sharpe unchanged; only MDD and return change
  [4] Full-sample vs real-time c   the paper's version (fixed over the full sample) vs this implementation (re-estimated monthly)
  [5] Burn-in               24 / 36 / 60 months: was the layer switched on before the GFC?
  [6] Initialization alternatives  c set from long market data (since 1990, 1970, 1926) instead of strategy returns, applied from the first month
  [7] Crisis entry          monthly c and exposure for 2007-02 ~ 2008-10
  [8] Exposure distribution how often w sits at the 1.5x cap, below 0.5x etc., cap months per year
  [9] c window and scale    expanding (used) vs rolling 36/60/120-month c; c multiplied by 0.5 to 1.5

Output: output/c_analysis.csv, output/c_analysis.png
The main strategy follows the team rule (metrics.select_main_strategy); the cost is config.cost_bps_main.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import Config, VMConfig
from metrics import net_returns, sharpe, max_drawdown, ann_return
from report import plot_style, load_results, main_strategy
from volatility import apply_layer, signal_series, market_rv_monthly, layer_turnover, load_holdings

EXAMPLE_MONTHS = ["2017-06", "2013-06", "2009-06", "2008-11", "2020-04"]  # [2] example months (calm -> crisis)
C_PATH_MONTHS = ["2007-02", "2008-01", "2008-09", "2009-06",              # [1] dates at which c is shown
                 "2011-01", "2014-01", "2017-01", "2020-01"]
ALT_STARTS = ["1990-01", "1970-01", "1926-07"]                           # [6] start months of the long market data
PRE_CRISIS = ("2007-03", "2008-09")                                      # window for the pre-crisis average exposure
CRISIS_SPAN = (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-31"))   # figure shading (GFC)


def perf(r, rf):                                                                # Sharpe, MDD and annual return
    """(arithmetic Sharpe, MDD, geometric annualized return)."""
    return sharpe(r, rf), max_drawdown(r), ann_return(r)               # same definitions as metrics


def realtime_c(x, s, burn):                                                     # c re-estimated each month from past data
    """Real-time c_t computed the same way as volatility.vm_weights (past data only, NaN before burn months)."""
    c = pd.Series(np.nan, index=x.index)                               # result (NaN initially)
    for j in range(burn, len(x)):                                      # every month after the burn-in
        c.iloc[j] = x.iloc[:j].std(ddof=1) / (x.iloc[:j] / s.iloc[:j]).std(ddof=1)  # c that equalizes volatility before/after management
    return c


def market_c(rv_all, P, start):                                                 # c from a long market history
    """Alternative c: median market RV from the start month through month t−1 (available from the first month)."""
    return [rv_all.loc[pd.Period(start, "M"):p - 1].median() for p in P]


def layer_net(x, rf, parts, w, bps):                                           # net layer returns for a given exposure path
    """Net returns with exposure w, charged on the netted order as in volatility.apply_layer."""
    gross = rf + w * x                                                 # managed gross returns
    vt = layer_turnover(w, x + rf, rf, parts)
    return gross - bps / 1e4 * vt


def main(cfg=None, vmcfg=None):
    """Run analyses [1]–[9] and save the table and figure."""
    cfg = cfg or Config()
    vmcfg = vmcfg or VMConfig()
    plot_style()
    out = cfg.output_dir
    gross, turns, bench = load_results(cfg)                            # backtest results, RF/S&P 500
    rf = bench["RF"]
    S, _ = main_strategy(gross, turns, bench, cfg)
    bps = cfg.cost_bps_main
    g, t = gross[[S]], turns[[S]]                                      # main strategy only
    x = g[S] - rf
    parts = load_holdings(cfg.output_dir)[S]                           # stock holdings for the netted costs
    s = signal_series(g.index, vmcfg)                                  # previous-month market RV
    rv_all = market_rv_monthly()["RV2"]                                # monthly market RV since 1926
    P = g.index.to_period("M")
    c = realtime_c(x, s, vmcfg.burn_in)
    w_rt = apply_layer(g, t, rf, vmcfg)[2][S]
    rows = []
    print(f"Main strategy {S}, cost {bps:g}bp, cap {vmcfg.cap}x, burn-in {vmcfg.burn_in} months\n")

    print("[1] Meaning of c: c ≈ median of past market RV so far")
    print(f"    median RV {s.median():.5f} -> median 1/RV {1 / s.median():.0f}x (cannot be used as exposure directly)")
    print(f"    median c {c.median():.5f}, range {c.min():.5f} ~ {c.max():.5f}, lowest in {c.idxmin():%Y-%m}")
    print(f"    {'Date':8s} {'Months':>8s} {'c':>9s} {'Past RV median':>15s} {'As ann. vol':>12s}")
    for d in C_PATH_MONTHS:
        i = int(np.flatnonzero(g.index.strftime("%Y-%m") == d)[0])
        med = s.iloc[:i].median()
        print(f"    {d:8s} {i:8d} {c.iloc[i]:9.5f} {med:15.5f} {np.sqrt(med * 12) * 100:11.1f}%")
        rows.append({"Section": "[1] c path", "Item": d, "Months used": i, "c": c.iloc[i], "Past RV median": med})

    print("\n[2] Exposure examples")
    for d in EXAMPLE_MONTHS:
        i = g.index[g.index.strftime("%Y-%m") == d][0]
        print(f"    {d}  last month's market vol {np.sqrt(s[i] * 12) * 100:5.1f}%  c/RV {c[i] / s[i]:5.2f}  "
              f"final exposure {w_rt[i]:.2f}")                         # after the cap
        rows.append({"Section": "[2] Example", "Item": d, "Last month vol (ann.)": np.sqrt(s[i] * 12),
                     "c/RV": c[i] / s[i], "Exposure": w_rt[i]})

    print("\n[3] Constant-c invariance (full-sample c, no cap, no costs)")
    w_full = apply_layer(g, t, rf, VMConfig(c_mode="full", cap=np.inf))[2][S]
    for mult in (1.0, 2.0):                                            # c × 1 and c × 2
        sh, md, ar = perf(rf + w_full * mult * x, rf)
        print(f"    c × {mult:.0f}   Sharpe {sh:.3f}   MDD {md:6.1%}   ann. return {ar:5.1%}")
        rows.append({"Section": "[3] Constant-c invariance", "Item": f"c×{mult:.0f}", "Sharpe": sh, "MDD": md, "Ann. return": ar})

    print("\n[4] Full-sample c vs real-time c")
    for cap, cost in ((np.inf, 0.0), (vmcfg.cap, bps)):                # (no cap, no cost), (default cap, main cost)
        cells = []
        for mode in ("full", "realtime"):
            vg, vt, _, _ = apply_layer(g, t, rf, VMConfig(c_mode=mode, cap=cap, burn_in=vmcfg.burn_in))
            sh, md, ar = perf(net_returns(vg, vt, cost)[S], rf)
            cells.append(f"{sh:.3f} / {md:6.1%}")
            rows.append({"Section": "[4] c method", "Item": f"{mode}, cap {cap}, {cost:g}bp",
                         "Sharpe": sh, "MDD": md, "Ann. return": ar})
        label = "no cap, no cost" if cap == np.inf else f"cap {cap}x, {bps:g}bp"
        print(f"    {label:16s} full-sample {cells[0]}   real-time {cells[1]}")

    sb, mb, ab = perf(net_returns(g, t, bps)[S], rf)
    print(f"\n[5] Burn-in (base Sharpe {sb:.3f}, MDD {mb:.1%})")
    for burn in (24, 36, 60):                                          # burn-in months
        vg, vt, _, _ = apply_layer(g, t, rf, VMConfig(burn_in=burn))
        sh, md, ar = perf(net_returns(vg, vt, bps)[S], rf)
        start = g.index[burn].strftime("%Y-%m")                        # month the layer switches on
        print(f"    {burn} months -> layer active from {start}   ΔSharpe {sh - sb:+.2f}   ΔMDD {(md - mb) * 100:+.1f}%p")
        rows.append({"Section": "[5] Burn-in", "Item": f"{burn} months", "Active from": start,
                     "dSharpe": sh - sb, "dMDD(%p)": (md - mb) * 100})

    print("\n[6] Initialization alternatives: median RV of long market data as c, applied from the first month")
    alt_w = {}
    rt_sh, rt_md, rt_ar = perf(layer_net(x, rf, parts, w_rt, bps), rf)
    print(f"    {'Method':44s} {'Sharpe':>7s} {'MDD':>8s} {'Return':>7s} | pre-crisis avg. exposure")
    baseline = [("No layer (base)", sb, mb, ab, pd.Series(1.0, index=g.index)),  # reference 1
                (f"This implementation (strategy returns, from month {vmcfg.burn_in})", rt_sh, rt_md, rt_ar, w_rt)]  # reference 2
    for nm, sh, md, ar, w in baseline:
        pre = w.loc[PRE_CRISIS[0]:PRE_CRISIS[1]].mean()
        print(f"    {nm:44s} {sh:7.3f} {md:7.1%} {ar:6.2%} | {pre:.2f}")
        rows.append({"Section": "[6] Initialization", "Item": nm, "Sharpe": sh, "MDD": md, "Ann. return": ar,
                     "Pre-crisis avg. exposure": pre})
    for start in ALT_STARTS:
        c_alt = pd.Series(market_c(rv_all, P, start), index=g.index)   # median market RV through t−1
        w = (c_alt / s).clip(vmcfg.floor, vmcfg.cap)
        sh, md, ar = perf(layer_net(x, rf, parts, w, bps), rf)
        nm = f"Alt.: market RV median ({start[:4]}~), from month 1"
        alt_w[start[:4]] = w                                           # keep
        pre = w.loc[PRE_CRISIS[0]:PRE_CRISIS[1]].mean()
        print(f"    {nm:44s} {sh:7.3f} {md:7.1%} {ar:6.2%} | {pre:.2f}")
        rows.append({"Section": "[6] Initialization", "Item": nm, "Sharpe": sh, "MDD": md, "Ann. return": ar,
                     "Pre-crisis avg. exposure": pre, "First-month c": c_alt.iloc[0]})

    print("\n[7] Crisis entry (2007-02 ~ 2008-10)")
    crisis = pd.DataFrame({"Last month mkt vol (%)": np.sqrt(s * 12) * 100, "c": c, "Exposure": w_rt,
                           "Alt. (1926~) exposure": alt_w["1926"]}).loc["2007-02":"2008-10"]
    print(crisis.round({"Last month mkt vol (%)": 1, "c": 5, "Exposure": 2, "Alt. (1926~) exposure": 2}).to_string())

    print("\n[8] Exposure distribution after the burn-in")
    act = w_rt.iloc[vmcfg.burn_in:]                                    # months where the layer is on
    at_cap = act >= vmcfg.cap - 1e-9                                   # months at the cap
    bands = {"at cap": int(at_cap.sum()),                              # w = cap
             "1.0 to cap": int(((act >= 1.0) & ~at_cap).sum()),
             "0.5 to 1.0": int(((act >= 0.5) & (act < 1.0)).sum()),
             "below 0.5": int((act < 0.5).sum()),
             "below 0.1": int((act < 0.1).sum())}                      # almost out of the market
    for k, v in bands.items():
        print(f"    {k:11s} {v:4d} of {len(act)} months")
        rows.append({"Section": "[8] Exposure", "Item": k, "Months used": v})
    raw = (c / s).iloc[vmcfg.burn_in:]                                 # exposure before the cap
    thr = np.sqrt(c / vmcfg.cap * 12).iloc[vmcfg.burn_in:]             # market vol (ann.) below which the cap binds
    print(f"    c/RV in capped months: median {raw[at_cap].median():.2f}, max {raw.max():.2f}")  # how far above the cap
    print(f"    cap binds when last month's market vol < {thr.median():.1%} (range {thr.min():.1%} ~ {thr.max():.1%})")
    rows.append({"Section": "[8] Exposure", "Item": "c/RV median when capped", "c/RV": raw[at_cap].median()})
    rows.append({"Section": "[8] Exposure", "Item": "cap threshold vol (median)", "Last month vol (ann.)": thr.median()})
    per_year = at_cap.groupby(act.index.year).sum().astype(int)        # cap months per year
    print("    cap months per year: " + ", ".join(f"{y}: {n}" for y, n in per_year.items()))
    for y, n in per_year.items():
        rows.append({"Section": "[8] Cap months by year", "Item": str(y), "Months used": int(n)})

    print("\n[9] c estimation window and scale (cap and costs as in the main run)")
    def from_c(cs):                                                    # exposure for a given c path
        return (cs / s).clip(vmcfg.floor, vmcfg.cap).where(cs.notna(), 1.0)  # 1x before the burn-in
    variants = [("expanding (used)", c)]
    for win in (36, 60, 120):
        cw = pd.Series(np.nan, index=x.index)
        for j in range(vmcfg.burn_in, len(x)):                         # every month after the burn-in
            lo = max(0, j - win)
            cw.iloc[j] = x.iloc[lo:j].std(ddof=1) / (x.iloc[lo:j] / s.iloc[lo:j]).std(ddof=1)  # same ratio on the window
        variants.append((f"rolling {win} months", cw))                 # keep
    for k in (0.5, 0.75, 1.25, 1.5):
        variants.append((f"c x {k:g}", c * k))                         # keep
    for nm, cs in variants:
        w = from_c(cs)
        sh, md, ar = perf(layer_net(x, rf, parts, w, bps), rf)
        n_cap = int((w.iloc[vmcfg.burn_in:] >= vmcfg.cap - 1e-9).sum())  # cap months
        avg = w.iloc[vmcfg.burn_in:].mean()
        print(f"    {nm:18s} Sharpe {sh:.3f}  MDD {md:6.1%}  return {ar:5.2%}  avg exposure {avg:.2f}  cap months {n_cap}")
        rows.append({"Section": "[9] c window/scale", "Item": nm, "Months used": n_cap, "Exposure": avg,
                     "Sharpe": sh, "MDD": md, "Ann. return": ar})

    pd.DataFrame(rows).to_csv(out / "c_analysis.csv", index=False, encoding="utf-8-sig", float_format="%.5f")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))                  # 2 panels
    fig.suptitle(f"Layer coefficient c and initial conditions: {S}", fontsize=13, fontweight="bold", y=1.02)
    ax = axes[0]                                                       # (a) path of c
    ax.plot(c.index, c.values, color="#7b2cbf", lw=1.8, label="This implementation: real-time c (strategy returns)")
    for start, col in zip(ALT_STARTS, ("#f4a261", "#e76f51", "#264653")):
        ax.plot(g.index, market_c(rv_all, P, start), color=col, lw=1.2, ls="--",
                label=f"Alt.: market RV median ({start[:4]}~)")
    ax.axvspan(*CRISIS_SPAN, color="#999999", alpha=0.2, lw=0)
    ax.set_title("(a) Path of c: shading is the GFC", fontsize=11, loc="left")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]                                                       # (b) exposure around the crisis
    idx = slice("2006-06", "2010-06")
    ax.plot(w_rt.loc[idx].index, w_rt.loc[idx].values, color="#7b2cbf", lw=1.8, label="This implementation")
    ax.plot(alt_w["1926"].loc[idx].index, alt_w["1926"].loc[idx].values, color="#264653", lw=1.4, ls="--",
            label="Alt.: market RV median (1926~)")
    ax.axhline(1.0, color="black", lw=0.8, ls=":")                     # 1x reference line
    ax.axvspan(*CRISIS_SPAN, color="#999999", alpha=0.2, lw=0)
    ax.set_title("(b) Exposure around the GFC", fontsize=11, loc="left")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "c_analysis.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDone -> {out / 'c_analysis.csv'}, {out / 'c_analysis.png'}")


if __name__ == "__main__":
    main()
