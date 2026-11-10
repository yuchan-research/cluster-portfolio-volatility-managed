"""
Entry point:  python main.py

Runs the whole pipeline: each month t, weights are set using only data up to t and evaluated on the t+1 return.

  [0] External data check   create data/benchmarks_monthly.csv (RF, S&P 500 TR) with benchmarks.build() if missing
  [1] Load and preprocess   ① universe                                          data.py
  [2] Backtest              ②–⑦ correlation -> K -> SPONGE_sym -> β -> α -> stock weights
                                                                               backtest.py (<- portfolio, clustering,
                                                                               allocation, optimization)
  [3] Tables and figures    ⑨ cluster strategies vs benchmarks, main strategy  report.make_report
  [4] Stock holdings        ⑧ the layer's costs are charged on one netted      netted_turnover.py
                                order per stock; rebuilds holdings if missing
  [5] Volatility layer      ⑧ layer on top of the strategies, base vs layer    report.make_vm_report (<- volatility)
  [6] Robustness checks     evidence for the report appendix                   validation.py, cost_sensitivity.py,
                                                                               c_analysis.py, predicted_vol.py,
                                                                               regimes.py, qa_analysis.py,
                                                                               ivp_benchmark.py

Options
  python main.py --skip-backtest   skip [1][2] and rerun [3]–[6] from the backtest results saved in output/
  python main.py --no-analysis     skip the [6] robustness checks

Run time on a laptop (Python 3.8): about 50 minutes in total, of which the backtest takes 30-40 minutes and
predicted_vol.py about 13; with --skip-backtest about 15 minutes.
All results are saved in output/ (file list in README.md).
"""

import argparse
import time
import pandas as pd
from config import Config, VMConfig
import benchmarks
from data import load_panel
from backtest import run_backtest
from volatility import HOLDINGS_FILE
from report import make_report, make_vm_report
import validation
import cost_sensitivity
import c_analysis
import predicted_vol
import regimes
import netted_turnover
import qa_analysis
import ivp_benchmark


def main(cfg=None, vmcfg=None, skip_backtest=False, analysis=True):
    """Run steps [0]–[6] in order."""
    cfg = cfg or Config()
    vmcfg = vmcfg or VMConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[0] Checking external data")
    if not benchmarks.OUT_CSV.exists():                               # RF/S&P 500 file missing
        benchmarks.build()                                            # build from the Ken French zip and Yahoo Finance
    print(f"    {benchmarks.OUT_CSV.name} ready")

    if not skip_backtest:
        print("[1] Loading and preprocessing data")
        panel = load_panel(cfg)
        print(f"    period {panel.returns.index[0]:%Y-%m} ~ {panel.returns.index[-1]:%Y-%m}, "
              f"{panel.returns.shape[1]:,} stocks")
        for k, v in panel.info.items():
            print(f"    {k}: {v:,}")
        print("[2] Backtest")
        holdings = []                                                 # stock weights, for the layer's costs
        rets, turns, diags, membership = run_backtest(panel, cfg, holdings)
        rets.to_csv(cfg.output_dir / "monthly_returns_gross.csv")
        turns.to_csv(cfg.output_dir / "turnover.csv")
        diags.to_csv(cfg.output_dir / "diagnostics.csv")
        membership.to_csv(cfg.output_dir / "cluster_membership.csv", index=False)
        pd.concat(holdings, ignore_index=True).to_csv(cfg.output_dir / HOLDINGS_FILE, index=False,
                                                      float_format="%.10g")  # compressed by the .gz name
    else:
        print("[1][2] Skipped: using the saved backtest results in output/")

    print("[3] Performance tables and figures (cluster strategies vs benchmarks)")
    make_report(cfg)
    print("\n[4] Stock holdings for the layer's netted costs")
    netted_turnover.main(cfg, vmcfg)                                  # rebuilds output/holdings.csv.gz if missing
    print("\n[5] Volatility-managed layer")
    make_vm_report(cfg, vmcfg)                                        # apply layer, base vs layer tables

    if analysis:
        print("\n[6-1] Layer robustness checks")
        validation.main(cfg)                                          # parameters, periods, crises, control, bootstrap
        print("\n[6-2] Comparison across cost levels")
        cost_sensitivity.main(cfg, vmcfg)                             # 0–50bp, break-even, 4-year blocks
        print("\n[6-3] Layer coefficient c analysis")
        c_analysis.main(cfg, vmcfg)                                   # meaning of c, sensitivity to estimation method and start
        print("\n[6-4] Predicted vs realized risk")
        predicted_vol.main(cfg)                                       # does each optimizer's risk estimate hold up? (goal 1)
        print("\n[6-5] Performance by regime")
        regimes.main(cfg, vmcfg)
        print("\n[6-6] Checks for the Q&A")
        qa_analysis.main(cfg, vmcfg)                                  # layer on/off, block length, signals, tests
        print("\n[6-7] Low-volatility benchmarks without clusters")
        ivp_benchmark.main(cfg, vmcfg)                                # inverse variance / volatility over all stocks

    print(f"\nDone ({time.time() - t0:.0f}s) -> {cfg.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster portfolio + volatility-managed layer backtest")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="rerun tables, layer and checks from the saved backtest results (output/)")
    parser.add_argument("--no-analysis", action="store_true",
                        help="skip the [6] robustness checks")
    args = parser.parse_args()
    main(skip_backtest=args.skip_backtest, analysis=not args.no_analysis)
