"""
Seed sensitivity:  python seed_sensitivity.py [--workers 4]

The only random step in the pipeline is the k-means++ initialisation (Config.seed). The submitted
results use seed 42, fixed before any result was seen. Here the whole backtest is rerun with seeds
1 to 10 and nothing else changed, to see how much of each result is just the luck of the seed.

For every seed we report, at the main cost (20bp):
  - Sharpe, CAGR and MDD of the six cluster strategies (no layer)
  - which strategy the main-strategy rule would pick
  - the main strategy with the volatility layer (netted costs, from the stock holdings of that seed)

Backtests and holdings are saved in output/seeds/seed<n>/ and reused, so a second run only rebuilds the
tables. A seed folder from before holdings were saved gets them rebuilt for the main strategy (a few minutes).
One seed takes about 20 minutes; --workers runs several seeds side by side.

Output: output/seed_sensitivity.csv (one row per seed and strategy)
        output/seed_summary.csv     (mean, spread and how often each strategy ranks first)
        output/seed_sensitivity.png
"""

import argparse
from dataclasses import replace
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")                                  # write figures to file only
import matplotlib.pyplot as plt
import pandas as pd
from config import Config, VMConfig
from data import load_panel
from backtest import run_backtest
from benchmarks import load_benchmarks
from metrics import net_returns, sharpe, ann_return, max_drawdown as mdd
from report import plot_style
from volatility import apply_layer, load_holdings, HOLDINGS_FILE
from netted_turnover import build_holdings

SEEDS = tuple(range(1, 11))                            # extra seeds; 42 is the main run
MAIN = "Cluster-ivp-gmvp"                              # strategy the report is built on


def seed_dir(cfg, seed):                                                        # output folder for a seed
    """Folder holding one seed's backtest (the main run for seed 42)."""
    if seed == Config().seed:                                          # submitted seed
        return cfg.output_dir
    return cfg.output_dir / "seeds" / f"seed{seed}"                    # extra seed


def run_one(seed):                                                              # backtest for one seed
    """Backtest one seed unless its results are already saved."""
    cfg = Config()
    out = seed_dir(cfg, seed)                                          # where this seed lives
    if (out / "turnover.csv").exists():                                # backtest done before
        if not (out / HOLDINGS_FILE).exists():                         # but no holdings yet
            build_holdings(replace(cfg, seed=seed, output_dir=out), out, strategy=MAIN)  # main strategy only
        return seed
    out.mkdir(parents=True, exist_ok=True)
    run = replace(cfg, seed=seed, run_benchmarks=False, output_dir=out)  # same settings, other seed
    holdings = []                                                      # stock weights, for the layer's costs
    rets, turns, diags, _ = run_backtest(load_panel(run), run, holdings)
    rets.to_csv(out / "monthly_returns_gross.csv")
    turns.to_csv(out / "turnover.csv")
    diags.to_csv(out / "diagnostics.csv")
    pd.concat(holdings, ignore_index=True).to_csv(out / HOLDINGS_FILE, index=False, float_format="%.10g")
    return seed


def seed_rows(cfg, vmcfg, seed):                                                # metrics for one seed
    """Metrics of the six cluster strategies (and the main one with the layer) for one seed."""
    out = seed_dir(cfg, seed)
    gross = pd.read_csv(out / "monthly_returns_gross.csv", index_col="date", parse_dates=True)
    turns = pd.read_csv(out / "turnover.csv", index_col="date", parse_dates=True)
    gross = gross[[c for c in gross.columns if c.startswith("Cluster-")]]  # cluster strategies only
    turns = turns[gross.columns]                                       # same columns
    rf = load_benchmarks(gross.index)["RF"]
    bps = cfg.cost_bps_main                                            # 20bp
    net = net_returns(gross, turns, bps)                               # after costs
    vm_gross, vm_turn, _, _ = apply_layer(gross[[MAIN]], turns[[MAIN]], rf, vmcfg, load_holdings(out))  # layer on the main strategy
    layer = net_returns(vm_gross, vm_turn, bps)[MAIN]                  # layer after netted costs
    sharpes = {c: sharpe(net[c], rf) for c in net.columns}
    best = max(sharpes, key=sharpes.get)                               # what the selection rule would pick
    rows = [{"seed": seed, "strategy": c, "layer": "no", "sharpe": sharpes[c],
             "cagr": ann_return(net[c]), "mdd": mdd(net[c]), "picked": c == best}  # CAGR, MDD, picked or not
            for c in net.columns]
    rows.append({"seed": seed, "strategy": MAIN, "layer": "yes", "sharpe": sharpe(layer, rf),
                 "cagr": ann_return(layer), "mdd": mdd(layer), "picked": False})
    return rows                                                        # rows for this seed


def main(cfg=None, vmcfg=None, workers=1):
    """Run the missing seeds, then build the tables and the figure."""
    cfg = cfg or Config()
    vmcfg = vmcfg or VMConfig()
    todo = [s for s in SEEDS if not (seed_dir(cfg, s) / HOLDINGS_FILE).exists()]  # seeds still to run
    if todo:
        print(f"Running seeds {todo} with {workers} worker(s), about 20 minutes per seed")
        if workers > 1:
            with Pool(workers) as pool:
                for s in pool.imap_unordered(run_one, todo):
                    print(f"  seed {s} done")
        else:
            for s in todo:                                             # each seed
                run_one(s)
                print(f"  seed {s} done")

    seeds = (cfg.seed,) + SEEDS                                        # 42 first, then 1..10
    table = pd.DataFrame([r for s in seeds for r in seed_rows(cfg, vmcfg, s)])
    table.to_csv(cfg.output_dir / "seed_sensitivity.csv", index=False, float_format="%.4f")

    groups = table.groupby(["strategy", "layer"])
    summary = groups["sharpe"].agg(["mean", "std", "min", "max"])     # spread of the Sharpe ratio
    summary["sharpe seed 42"] = table[table.seed == cfg.seed].set_index(["strategy", "layer"])["sharpe"]
    summary["mdd mean"] = groups["mdd"].mean()
    summary["picked first"] = groups["picked"].sum().astype(int)       # times the rule picks it
    summary = summary.sort_values("mean", ascending=False)
    summary.to_csv(cfg.output_dir / "seed_summary.csv", float_format="%.4f")
    print(f"\nSharpe at {cfg.cost_bps_main:g}bp over {len(seeds)} seeds")
    print(summary.round(3).to_string())

    plot_style()
    base = table[table.layer == "no"]                                  # strategies without the layer
    order = base.groupby("strategy")["sharpe"].mean().sort_values().index  # lowest mean at the bottom
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, name in enumerate(order):
        s = base[base.strategy == name]
        ax.scatter(s.sharpe, [i] * len(s), color="#9aa5b1", s=28, zorder=2)  # every seed
        ax.scatter(s[s.seed == cfg.seed].sharpe, [i], color="#2a78d6", s=60, zorder=3)  # seed 42
    lay = table[table.layer == "yes"]                                  # main strategy with the layer
    top = len(order)
    ax.scatter(lay.sharpe, [top] * len(lay), color="#9aa5b1", s=28, zorder=2)  # every seed
    ax.scatter(lay[lay.seed == cfg.seed].sharpe, [top], color="#2a78d6", s=60, zorder=3)  # seed 42
    ax.set_yticks(range(top + 1))
    ax.set_yticklabels([n.replace("Cluster-", "") for n in order] + ["ivp-gmvp + layer"])
    ax.set_xlabel(f"Sharpe ratio, {cfg.cost_bps_main:g}bp costs")
    ax.set_title(f"Sharpe over {len(seeds)} k-means seeds (blue = seed 42, used in the report)",
                 loc="left", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.output_dir / "seed_sensitivity.png", dpi=130)
    plt.close(fig)
    print(f"\nDone -> {cfg.output_dir / 'seed_summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerun the backtest with other k-means seeds")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of seeds run at the same time")
    main(workers=parser.parse_args().workers)
