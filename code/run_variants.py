"""
Sensitivity runs that need the backtest repeated:  python run_variants.py <kind> <value>

Everything else in the project reuses the saved monthly results, but the number of clusters and the
random seed change the portfolio itself, so the whole walk-forward has to run again.

    python run_variants.py k 5      hold K fixed at 5 instead of choosing it each month
    python run_variants.py k 10     hold K fixed at 10 (the value used in [K24])
    python run_variants.py seed 7   keep the monthly K rule, change the k-means seed
    python run_variants.py delist none       missing returns after the last price set to 0% (original data)
    python run_variants.py delist category   one fixed return per delisting category instead of CRSP DLRET

For K and the seed the benchmarks are switched off: equal weight and the Ledoit-Wolf portfolios depend on
neither, so they are taken from the main run. The delisting runs keep them, since every portfolio changes.

Output: output/variant_<kind><value>.csv (one row per strategy, at config.cost_bps_main)

Why it matters: the monthly rule and a fixed K = 10 differ by about 0.002 in Sharpe, while changing
only the seed moves it by about 0.014. The K rule is kept because it removes a hand-picked number,
not because it performs better.
"""

import sys
import time
import numpy as np
import pandas as pd
from config import Config
from benchmarks import load_benchmarks
from data import load_panel
from backtest import run_backtest


def main(kind="k", value=10, cfg=None):
    """Rerun the backtest with one setting changed and save the performance of every strategy."""
    cfg = cfg or Config()
    cfg.run_benchmarks = kind == "delist"                              # benchmarks depend only on the delisting rule
    if kind == "k":
        cfg.k_method, cfg.k_fixed = "fixed", int(value)
    elif kind == "seed":                                               # different k-means initialisation
        cfg.seed = int(value)                                          # change the random seed
    elif kind == "delist":
        cfg.delist_mode = str(value)
    else:
        raise ValueError(f"kind must be k, seed or delist: {kind}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"=== {kind} = {value} ===", flush=True)
    panel = load_panel(cfg)
    gross, turns, diags, _ = run_backtest(panel, cfg)
    rf = load_benchmarks(gross.index)["RF"]                            # risk-free returns on the same dates

    rows = []
    for col in gross.columns:
        r = gross[col] - cfg.cost_bps_main / 1e4 * turns[col]
        ex = r - rf
        wealth = (1.0 + r).cumprod()
        rows.append({"strategy": col,
                     "Sharpe": ex.mean() / ex.std(ddof=1) * np.sqrt(12.0),  # annualized Sharpe
                     "MDD": float((wealth / wealth.cummax().clip(lower=1.0) - 1.0).min()),
                     "Return": wealth.iloc[-1] ** (12.0 / len(r)) - 1.0,
                     "Turnover": turns[col].iloc[1:].mean()})
    res = pd.DataFrame(rows).set_index("strategy").round(4)
    res.to_csv(cfg.output_dir / f"variant_{kind}{value}.csv")
    print(res.to_string())
    print(f"mean K {diags['K'].mean():.2f}")
    print(f"=== {kind}={value} done in {time.time() - t0:.0f}s ===", flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "k",
         sys.argv[2] if len(sys.argv) > 2 else 10)
