"""
Checks run for the questions from other teams:  python qa_analysis.py   (step [6-6] of main.py)

Every check reads the saved backtest in output/ and takes a few seconds. Costs are config.cost_bps_main
(20bp). Every layer series is charged on the netted order per stock (volatility.layer_turnover).

  [1] Layer on / off      the same layer on the main strategy, LW-GMVP, LW-MSRP and the S&P 500 TR
  [2] Without the GFC     Sharpe, MDD and CAGR with Nov 2007 - Mar 2009 left out
  [3] Fast crashes        monthly exposure and returns in late 2018 and early 2020
  [4] Faster signals      market RV over the last 10 / 5 trading days, and a re-entry rule fixed in advance
  [5] Block length        block bootstrap of layer vs base with 3, 6, 12, 24 and 36-month blocks
  [6] Sharpe tests        Jobson-Korkie test with the Memmel (2003) correction
  [7] Cluster stability   adjusted Rand index of the cluster labels between consecutive months
  [8] Regimes             Sharpe of the six cluster strategies (no layer) in the five regimes of regimes.py
  [9] Leave one year out  full-period Sharpe with each calendar year removed in turn
  [10] Concentration      holdings of the main strategy and the no-cluster MSRP: largest weights, effective
                          number of stocks, stocks at the 5% cap, and the 2020 return without its top two stocks
  [11] Survivorship       equal weight on the 500 most traded stocks that survive the whole sample, chosen once
                          with hindsight, against the point-in-time universe used everywhere else (report 8.3)
  [12] Index futures      the layer's exposure changes made with S&P 500 futures instead of trading the stocks:
                          the portfolio stays fully invested and (w - 1) x (S&P 500 - rf) is added (report 8.4)

Output: output/qa_analysis.csv (one row per number reported)
"""

import numpy as np
import pandas as pd
from math import erf, sqrt
from config import Config, VMConfig
from metrics import net_returns, sharpe, ann_return, max_drawdown as mdd, SP500_NAME
from report import load_results, main_strategy
from regimes import REGIMES
from volatility import apply_layer, layer_turnover, load_holdings, signal_series, vm_weights
from volatility import FF_DAILY_CSV, HOLDINGS_FILE
from data import load_panel

GFC = ("2007-11", "2009-03")                           # left out in [2]
CRASH_MONTHS = [("2018-09", "2018-12"), ("2020-01", "2020-05")]  # windows shown in [3]
BLOCKS = (3, 6, 12, 24, 36)                            # bootstrap block lengths in [5]
BOOT_N = 2000                                          # bootstrap draws (as validation.py)
FUTURES_BPS = 2.0                                      # one-way cost of an index futures trade, per unit of notional


def layer_net(gross, turns, rf, name, vmcfg, bps):                              # layer returns after costs
    """Net returns and exposure of one series with the layer (netted costs)."""
    vg, vt, w, _ = apply_layer(gross[[name]], turns[[name]], rf, vmcfg)
    return net_returns(vg, vt, bps)[name], w[name]                     # net returns, exposure


def short_rv(days):                                                             # market RV from the last few days
    """Monthly market variance from the last `days` trading days, scaled to a full month."""
    ff = pd.read_csv(FF_DAILY_CSV, index_col=0, parse_dates=True)["Mkt-RF"]  # daily excess market return
    by_month = ff.groupby(ff.index.to_period("M"))
    return by_month.apply(lambda r: (r.iloc[-days:] ** 2).sum() * len(r) / min(days, len(r)))


def block_bootstrap(base_ex, vm_ex, rf, block, seed=0):                         # same as validation.py, any block
    """Distributions of the Sharpe and MDD differences (layer minus base) from circular block resampling."""
    rs = np.random.RandomState(seed)
    a, b, f = base_ex.to_numpy(), vm_ex.to_numpy(), rf.to_numpy()
    n, nb = len(a), int(np.ceil(len(a) / block))                       # sample length, number of blocks
    d_sharpe, d_mdd = np.empty(BOOT_N), np.empty(BOOT_N)
    for i in range(BOOT_N):
        st = rs.randint(0, n, nb)
        idx = ((st[:, None] + np.arange(block)[None, :]).ravel()[:n]) % n  # circular block indices
        aa, bb, ff = a[idx], b[idx], f[idx]                            # same draw for both series
        d_sharpe[i] = (bb.mean() / bb.std(ddof=1) - aa.mean() / aa.std(ddof=1)) * np.sqrt(12.0)
        ca, cb = np.cumprod(1 + aa + ff), np.cumprod(1 + bb + ff)
        d_mdd[i] = ((cb / np.maximum.accumulate(cb) - 1).min()
                    - (ca / np.maximum.accumulate(ca) - 1).min()) * 100.0  # minus base MDD (%p)
    return d_sharpe, d_mdd


def jk_memmel(r1, r2, rf):                                                      # test of equal Sharpe ratios
    """z and two-sided p of Jobson-Korkie (1981) with the Memmel (2003) correction, monthly excess returns."""
    a, b = r1 - rf, r2 - rf
    T = len(a)
    m1, m2, s1, s2 = a.mean(), b.mean(), a.std(), b.std()
    s12 = a.cov(b)
    theta = (2 * s1**2 * s2**2 - 2 * s1 * s2 * s12 + 0.5 * m1**2 * s2**2 + 0.5 * m2**2 * s1**2
             - m1 * m2 / (2 * s1 * s2) * (s12**2 + s1**2 * s2**2)) / T
    z = (s2 * m1 - s1 * m2) / np.sqrt(theta)
    return z, 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def adjusted_rand(a, b):
    """Adjusted Rand index of two label series on the same stocks (1 = same partition, 0 = chance)."""
    ct = pd.crosstab(a, b).to_numpy()
    pairs = lambda v: v * (v - 1) / 2
    s, sa, sb, n = pairs(ct).sum(), pairs(ct.sum(1)).sum(), pairs(ct.sum(0)).sum(), pairs(ct.sum())
    expected = sa * sb / n
    return (s - expected) / ((sa + sb) / 2 - expected)


def main(cfg=None, vmcfg=None):
    """Run [1]-[12] and save output/qa_analysis.csv."""
    cfg = cfg or Config()
    vmcfg = vmcfg or VMConfig()
    gross, turns, bench = load_results(cfg)
    rf, bps = bench["RF"], cfg.cost_bps_main
    S, _ = main_strategy(gross, turns, bench, cfg)
    base = net_returns(gross, turns, bps)                              # all strategies after costs, no layer
    final, w = layer_net(gross, turns, rf, S, vmcfg, bps)
    gross = gross.assign(**{SP500_NAME: bench["SP500TR"]})             # S&P 500 TR as one more series
    turns = turns.assign(**{SP500_NAME: 0.0})                          # buy and hold, no turnover
    base[SP500_NAME] = bench["SP500TR"]
    rows = []

    def add(section, item, **vals):
        rows.append({"Section": section, "Item": item, **vals})

    print("[1] Layer on / off")
    for name in (S, "NoCluster-LW-gmvp", "NoCluster-LW-msrp", SP500_NAME):
        lay = final if name == S else layer_net(gross, turns, rf, name, vmcfg, bps)[0]
        b = base[name]
        print(f"    {name:20s} no layer {sharpe(b, rf):.3f} / {mdd(b):6.1%}   layer {sharpe(lay, rf):.3f} / "
              f"{mdd(lay):6.1%} / CAGR {ann_return(lay):5.2%}")
        add("[1] Layer on/off", name, sharpe_base=sharpe(b, rf), mdd_base=mdd(b),
            sharpe_layer=sharpe(lay, rf), mdd_layer=mdd(lay), cagr_layer=ann_return(lay))

    print("\n[2] Without the GFC months")
    P = gross.index.to_period("M")
    keep = ~((P >= pd.Period(GFC[0])) & (P <= pd.Period(GFC[1])))
    for nm, r in (("no layer", base[S]), ("final", final)):
        r = r[keep]                                                    # without the GFC
        print(f"    {nm:9s} Sharpe {sharpe(r, rf[keep]):.3f}  MDD {mdd(r):6.1%}  CAGR {ann_return(r):5.2%}  ({keep.sum()} months)")
        add("[2] Without GFC", nm, sharpe=sharpe(r, rf[keep]), mdd=mdd(r), cagr=ann_return(r), months=int(keep.sum()))

    print("\n[3] Fast crashes (monthly, %)")
    table = pd.DataFrame({"exposure": w, "final": final * 100, "no layer": base[S] * 100,
                          "S&P 500 TR": bench["SP500TR"] * 100})
    for a, z in CRASH_MONTHS:
        part = table.loc[a:z]
        print(part.round(2).to_string())
        for d, r in part.iterrows():
            add("[3] Fast crashes", f"{d:%Y-%m}", **r.to_dict())

    print("\n[4] Faster signals")
    x = gross[S] - rf
    monthly = signal_series(gross.index, vmcfg)                        # signal used in the report
    prev = gross.index.to_period("M") - 1                              # month the signal comes from
    last10 = pd.Series(short_rv(10).reindex(prev).to_numpy(), index=gross.index)  # last 10 trading days
    last5 = pd.Series(short_rv(5).reindex(prev).to_numpy(), index=gross.index)  # last 5 trading days
    falling = monthly < monthly.shift(1)                               # market RV fell last month
    signals = {"monthly RV (used)": monthly, "last 10 days": last10, "last 5 days": last5,
               "last 5 days when RV fell": monthly.where(~falling, last5)}
    for nm, sig in signals.items():
        ws = vm_weights(x, sig, vmcfg)                                 # exposure (c re-estimated for this signal)
        to = layer_turnover(ws, gross[S], rf, load_holdings(cfg.output_dir)[S])
        r = rf + ws * x - bps / 1e4 * to
        n_cap = int((ws >= vmcfg.cap - 1e-9).sum())                    # months at the cap
        print(f"    {nm:26s} Sharpe {sharpe(r, rf):.3f}  CAGR {ann_return(r):5.2%}  MDD {mdd(r):6.1%}  "
              f"turnover {to.iloc[1:].mean():5.1%}  w 2020-04 {ws.loc['2020-04'].iloc[0]:.3f}  cap months {n_cap}")
        add("[4] Faster signals", nm, sharpe=sharpe(r, rf), cagr=ann_return(r), mdd=mdd(r),
            turnover=to.iloc[1:].mean(), w_2020_04=ws.loc["2020-04"].iloc[0], cap_months=n_cap)

    print("\n[5] Block length of the bootstrap (layer vs no layer, 2,000 draws)")
    for block in BLOCKS:
        ds, dm = block_bootstrap(base[S] - rf, final - rf, rf, block)
        print(f"    {block:2d} months: Sharpe up in {np.mean(ds > 0):5.1%} (median {np.median(ds):+.2f}), "
              f"MDD better in {np.mean(dm > 0):5.1%} (median {np.median(dm):+.1f}%p)")
        add("[5] Block length", f"{block} months", p_sharpe_up=float(np.mean(ds > 0)), sharpe_median=float(np.median(ds)),
            p_mdd_better=float(np.mean(dm > 0)), mdd_median=float(np.median(dm)))

    print("\n[6] Equal Sharpe tests (Jobson-Korkie, Memmel correction)")
    sp_layer = layer_net(gross, turns, rf, SP500_NAME, vmcfg, bps)[0]  # S&P 500 with the layer
    pairs = [("final vs S&P 500 TR", final, base[SP500_NAME]), ("no layer vs S&P 500 TR", base[S], base[SP500_NAME]),
             ("no layer vs LW-GMVP", base[S], base["NoCluster-LW-gmvp"]), ("no layer vs LW-MSRP", base[S], base["NoCluster-LW-msrp"]),
             ("final vs no layer", final, base[S]), ("final vs S&P 500 + layer", final, sp_layer)]
    for nm, r1, r2 in pairs:
        z, p = jk_memmel(r1, r2, rf)
        print(f"    {nm:26s} Sharpe {sharpe(r1, rf):.3f} vs {sharpe(r2, rf):.3f}   z {z:+.2f}   p {p:.2f}")
        add("[6] Sharpe test", nm, sharpe_1=sharpe(r1, rf), sharpe_2=sharpe(r2, rf), z=z, p=p)

    print("\n[7] Cluster stability between consecutive months (first of the 10 repeats)")
    cm = pd.read_csv(cfg.output_dir / "cluster_membership.csv", parse_dates=["date"])
    by_date = {d: g.set_index("PERMNO")["cluster"] for d, g in cm.groupby("date")}
    dates = sorted(by_date)
    ari, overlap = [], []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        a, b = by_date[d0], by_date[d1]
        common = a.index.intersection(b.index)
        ari.append(adjusted_rand(a[common], b[common]))
        overlap.append(len(common) / len(a))                           # share of the universe kept
    ari = np.array(ari)
    print(f"    {len(ari)} pairs: ARI mean {ari.mean():.2f}, median {np.median(ari):.2f}, "
          f"min {ari.min():.2f}; universe kept month to month {np.mean(overlap):.1%}")
    add("[7] Cluster stability", "ARI", mean=ari.mean(), median=float(np.median(ari)), min=ari.min(),
        universe_kept=float(np.mean(overlap)))

    print("\n[8] Sharpe of the six cluster strategies by regime (no layer)")
    cluster = [c for c in base.columns if c.startswith("Cluster-")]
    reg = pd.DataFrame({lab: {c.replace("Cluster-", ""): sharpe(base[c].loc[a:z], rf.loc[a:z]) for c in cluster}
                        for lab, a, z in REGIMES})
    print(reg.round(2).to_string())
    print("    best: " + ", ".join(f"{k}: {v}" for k, v in reg.idxmax().items()))
    for strat, r in reg.iterrows():
        add("[8] Regimes", strat, **r.to_dict())

    print("\n[9] Sharpe with one calendar year removed")
    lw_ms = "NoCluster-LW-msrp"
    series = {"final": final, "no layer": base[S], "S&P 500 TR": base[SP500_NAME],
              "S&P 500 + layer": sp_layer, "LW-MSRP": base[lw_ms],
              "LW-MSRP + layer": layer_net(gross, turns, rf, lw_ms, vmcfg, bps)[0]}
    years = sorted(set(gross.index.year))                              # 2005 .. 2020
    loo = pd.DataFrame({nm: {str(y): sharpe(r[r.index.year != y], rf[rf.index.year != y]) for y in years}
                        for nm, r in series.items()})
    loo.loc["none"] = {nm: sharpe(r, rf) for nm, r in series.items()}
    print(loo.round(3).to_string())
    beat = int((loo["final"] > loo["S&P 500 + layer"]).sum())          # samples where the final model is higher
    print(f"    final above S&P 500 + layer in {beat} of {len(loo)} samples")
    for yr, r in loo.iterrows():
        add("[9] Leave one year out", yr, **r.to_dict())

    print("\n[10] Concentration of the holdings")
    h = pd.read_csv(cfg.output_dir / HOLDINGS_FILE, parse_dates=["date"])
    names = (pd.read_csv(cfg.data_path, usecols=["PERMNO", "TICKER", "COMNAM"])
             .drop_duplicates("PERMNO", keep="last").set_index("PERMNO"))
    for strat in (S, lw_ms):                                           # main strategy, no-cluster MSRP
        g = h[h.strategy == strat]
        by_month = g.groupby("date")["weight"]
        eff_n = (1.0 / by_month.apply(lambda v: (v ** 2).sum())).mean()  # effective number of stocks
        at_cap = by_month.apply(lambda v: (v >= 0.05 - 1e-6).sum()).mean()  # stocks at the 5% cap
        print(f"    {strat:20s} largest weight mean {by_month.max().mean():.1%} (max {by_month.max().max():.1%}), "
              f"effective N {eff_n:.0f}, stocks at the cap {at_cap:.1f}")
        add("[10] Concentration", strat, max_weight_mean=by_month.max().mean(), max_weight_max=by_month.max().max(),
            effective_n=eff_n, at_cap=at_cap)
    y20 = h[(h.strategy == lw_ms) & (h.date.dt.year == 2020)].copy()   # MSRP holdings in 2020
    y20["contrib"] = y20.weight * y20.ret
    top = y20.groupby("PERMNO").agg(avg_weight=("weight", lambda v: v.sum() / 12), contrib=("contrib", "sum"))
    top = top.sort_values("contrib", ascending=False).head(5).join(names)
    print("    no-cluster MSRP, 2020: average weight and summed monthly contribution")
    print(top.round(4).to_string())
    two = top.index[:2]
    monthly = y20.groupby("date").apply(lambda m: pd.Series({
        "all": m.contrib.sum(),                                        # actual gross return
        "without": (m.contrib.sum() - m[m.PERMNO.isin(two)].contrib.sum())  # without the two stocks,
                   / (1.0 - m[m.PERMNO.isin(two)].weight.sum())}))     # rest scaled back to 100%
    r_all, r_wo = (1 + monthly["all"]).prod() - 1, (1 + monthly["without"]).prod() - 1  # 2020 gross returns
    r_main = (1 + gross[S].loc["2020"]).prod() - 1                     # main strategy, 2020 gross
    print(f"    2020 gross return: no-cluster MSRP {r_all:.1%}, without {', '.join(top.TICKER[:2])} {r_wo:.1%}; "
          f"main strategy {r_main:.1%}")
    for permno, r in top.iterrows():
        add("[10] MSRP 2020 contributors", str(r.TICKER), avg_weight=r.avg_weight, contrib=r.contrib)
    add("[10] MSRP 2020", "gross return", all=r_all, without_top2=r_wo, main_strategy=r_main)

    print("\n[11] Survivorship: stocks chosen once with hindsight vs the point-in-time universe (gross)")
    panel = load_panel(cfg)
    rets = panel.returns.set_axis(panel.returns.index.to_period("M"))
    survived = rets.iloc[1:].notna().all(axis=0)                       # a return in every month of the sample
    top = panel.volume.mean(axis=0)[survived].dropna().nlargest(cfg.universe_size).index
    ew_surv = pd.Series(rets.loc[gross.index.to_period("M"), top].mean(axis=1).to_numpy(), index=gross.index)
    ew_pit = gross["EW-Universe"]                                      # the universe used in the report
    print(f"    {int(survived.sum())} stocks survive the whole sample; equal weight, top {cfg.universe_size} by volume: "
          f"CAGR {ann_return(ew_surv):.2%} against {ann_return(ew_pit):.2%} for the point-in-time universe")
    add("[11] Survivorship", "equal weight, gross", cagr_survivors=ann_return(ew_surv), cagr_point_in_time=ann_return(ew_pit),
        sharpe_survivors=sharpe(ew_surv, rf), sharpe_point_in_time=sharpe(ew_pit, rf))

    print(f"\n[12] Exposure changed with S&P 500 futures ({FUTURES_BPS:g}bp per unit traded) instead of the stocks")
    hedge = (w - 1.0) * (bench["SP500TR"] - rf) - FUTURES_BPS / 1e4 * (w - w.shift(1)).abs().fillna(0.0)
    fut = base[S] + hedge                                              # stocks fully invested, futures on top
    for nm, r in (("stocks traded (report)", final), ("index futures", fut)):
        print(f"    {nm:24s} Sharpe {sharpe(r, rf):.3f}  CAGR {ann_return(r):.2%}  MDD {mdd(r):.1%}")
        add("[12] Index futures", nm, sharpe=sharpe(r, rf), cagr=ann_return(r), mdd=mdd(r))

    pd.DataFrame(rows).to_csv(cfg.output_dir / "qa_analysis.csv", index=False, encoding="utf-8-sig",
                              float_format="%.4f")
    print(f"\nDone -> {cfg.output_dir / 'qa_analysis.csv'}")


if __name__ == "__main__":
    main()
