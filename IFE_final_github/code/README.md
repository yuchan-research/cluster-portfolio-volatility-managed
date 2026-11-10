# Team 5 — Cluster-Based Portfolio + Volatility-Managed Layer

A monthly rebalanced portfolio is built and backtested on the CRSP monthly data provided for the course.
Each month t, weights are set using only data up to t and evaluated on the t+1 return.

## How to run
```
1. The data is already in data/: the course data IFE_term_stock_data.csv and data/delist_matched.csv
   (CRSP delisting returns matched to the course data). Both are left out of the GitHub repository (.gitignore).
2. cd code
3. python main.py
```
- `python main.py` runs the whole pipeline (backtest -> performance tables -> volatility-managed layer -> robustness checks).
- The whole run takes about 50 minutes: the backtest 30-40 minutes and `predicted_vol.py` about 13 (it reruns the monthly loop to recover the
  covariance matrices). To rebuild only the tables, figures and checks from saved results:
  `python main.py --skip-backtest`.
- To skip the robustness checks: `--no-analysis`.
- Seed check (separate, not run by `main.py`): `python seed_sensitivity.py --workers 4` reruns the backtest with
  k-means seeds 1-10 (about 20 minutes per seed). Saved runs in `output/seeds/` are reused, so with the submitted
  output it only rebuilds the tables.

## Reproducibility
- The only random step is the k-means++ initialisation. It is seeded with `Config.seed = 42`, fixed before any result
  was seen; the block bootstrap in `validation.py` uses its own fixed seed (0).
- External data (RF, S&P 500 TR, daily market returns) is included in `data/`, so nothing is downloaded.
- Both CRSP files are included in data/. If `data/delist_matched.csv` is removed, a warning is printed, delisting
  returns are set to 0% and the numbers differ from the report.
- Rerunning the monthly loop reproduces the saved returns (largest difference about 1e-15). A different numpy/BLAS
  build can still shift results in the third decimal if a k-means assignment flips.
- Required packages: numpy, pandas, matplotlib (tested on Python 3.8 with numpy 1.24, pandas 2.0, matplotlib 3.7).
  `yfinance` is needed only to rebuild `data/benchmarks_monthly.csv` (included in the submission, so usually not needed).
- All settings are in `config.py` (`Config`: cluster portfolio and backtest, `VMConfig`: volatility-managed layer).

## Folder layout
See `../code_structure.png` for how the files call each other.
```
code/                       <- run from here
  main.py                   entry point (whole pipeline)
  config.py                 all settings (reasons in SETTINGS.md)
  data.py                   ① data loading and preprocessing, monthly universe selection
  clustering.py             ②③④ weighted correlation, number of clusters K (Marchenko–Pastur), SPONGE_sym + k-means++
  allocation.py             ⑤ within-cluster weights β (gaussian / ivp / periphery)
  optimization.py           ⑥ across-cluster weights α (long-only GMVP / MSRP), covariance estimation
  portfolio.py              ⑦ stock weights w = α × β, average of 10 repeats, 5% cap per stock
  backtest.py               monthly walk-forward loop, turnover (calls ②–⑦ every month)
  volatility.py             ⑧ volatility-managed layer [Moreira & Muir 2017]
  benchmarks.py, metrics.py ⑨ RF and S&P 500 TR data, performance metrics, main strategy selection
  report.py                 ⑨ performance tables and figures (base strategies, after the layer)
  validation.py             layer robustness checks (parameters, crises, control, bootstrap, borrowing spread, own-vol signal)
  cost_sensitivity.py       comparison across cost levels, break-even cost
  c_analysis.py             layer coefficient c: meaning, estimation method, start, window and scale; months at the cap
  predicted_vol.py          predicted vs realized risk of each optimizer (evidence for goal 1)
  regimes.py                performance in five market regimes
  qa_analysis.py            checks run for the questions from other teams (layer on/off, block length, faster signals, tests)
  ivp_benchmark.py          inverse-variance and inverse-volatility portfolios without clusters (low-volatility tilt check)
  netted_turnover.py        stock holdings for the layer's netted costs (rebuilds output/holdings.csv.gz if missing)
  run_variants.py           reruns the backtest with a fixed K, another random seed or another delisting rule
  seed_sensitivity.py       reruns the backtest with seeds 1-10 and summarises the spread (seed check)
  SETTINGS.md               reasons for each setting and limitations
data/                       external data (Ken French RF and daily market returns, S&P 500 TR) + place for the course data
output/                     results (list below)
```

## Monthly pipeline and files
| Step | Content | File |
|---|---|---|
| ① Universe | top 500 stocks by volume each month (complete 60-month returns) | `data.py` |
| ② Correlation | market factor removed (residuals), half-life 18 months | `data.py`, `clustering.py` |
| ③ Number of clusters K | Marchenko–Pastur, 3–8 per month | `clustering.py` |
| ④ Clustering | SPONGE_sym (signed graph) + k-means++ | `clustering.py` |
| ⑤ Within-cluster β | gaussian / ivp / periphery | `allocation.py` |
| ⑥ Across-cluster α | long-only GMVP or MSRP (K-dimensional) | `optimization.py` |
| ⑦ Stock weights | w = α × β, average of 10 repeats, 5% cap per stock | `portfolio.py` |
| ⑧ Volatility-managed layer | total exposure 0–1.5x based on last month's market volatility | `volatility.py` |
| ⑨ Evaluation | monthly rebalancing, 20bp one-way cost, vs S&P 500 TR | `backtest.py`, `metrics.py`, `report.py` |

①–⑦ and ⑨ are the team framework; ⑧ and the robustness checks (`validation.py`, `cost_sensitivity.py`, `c_analysis.py`,
`predicted_vol.py`, `regimes.py`, `netted_turnover.py`, `qa_analysis.py`, `ivp_benchmark.py`, `run_variants.py`,
`seed_sensitivity.py`) were added.
⑧ is an overlay applied to the monthly returns of the ①–⑦ portfolio, so the backtest is not rerun.

## Output files (output/)
| File | Content | Created by |
|---|---|---|
| `monthly_returns_gross.csv`, `turnover.csv` | monthly gross returns (before costs) and turnover per strategy | `main.py` [2] |
| `diagnostics.csv`, `cluster_membership.csv` | monthly universe, K, cluster sizes etc.; cluster of each stock | `main.py` [2] |
| `metrics.csv`, `main_summary.csv`, `monthly_returns_net.csv`, `results.png` | cost × period × strategy table, main strategy summary, net returns, figure | `report.make_report` |
| `vm_metrics.csv`, `vm_summary.csv`, `vm_alpha.csv`, `vm_results.png` | base vs layer table, summary, alpha regression, figure | `report.make_vm_report` |
| `vm_monthly_returns_gross.csv`, `vm_turnover.csv`, `vm_weights.csv` | gross returns and netted turnover after the layer, monthly exposure | `report.make_vm_report` |
| `vm_validation.csv/.png` | robustness checks | `validation.py` |
| `cost_comparison.csv/.png` | comparison across cost levels, break-even | `cost_sensitivity.py` |
| `c_analysis.csv/.png` | analysis of the coefficient c | `c_analysis.py` |
| `predicted_vol.csv` | monthly volatility each optimizer predicted for its own portfolio | `predicted_vol.py` |
| `regimes.csv/.png` | Sharpe, return, drawdown and exposure in five market regimes | `regimes.py` |
| `qa_analysis.csv` | numbers used in the answers to other teams' questions | `qa_analysis.py` |
| `ivp_benchmark.csv`, `ivp_benchmark/` | no-cluster inverse-variance / inverse-volatility portfolios vs the main strategy | `ivp_benchmark.py` |
| `holdings.csv.gz` | stock weights of every strategy each month and the returns they earned | `main.py` [2] / `netted_turnover.py` |
| `netted_turnover.csv` | layer turnover per month and strategy (netted order per stock) | `netted_turnover.py` |
| `variant_k5.csv`, `variant_k10.csv`, `variant_seed7.csv`, `variant_seed123.csv` | fixed-K and single-seed reruns | `run_variants.py` |
| `variant_delistnone.csv`, `variant_delistcategory.csv` | reruns with delisting returns set to 0% or to fixed category returns | `run_variants.py` |
| `seed_sensitivity.csv`, `seed_summary.csv`, `seed_sensitivity.png`, `seeds/` | results for seeds 1-10 and their spread | `seed_sensitivity.py` |

## Volatility-managed layer (⑧) summary
```
x_t      = r_t − rf_t                        monthly excess return of the strategy without the layer (before costs)
RV_{t-1} = Σ_{d ∈ t-1} (Mkt-RF_d)²           sum of squared daily market excess returns last month (Ken French)
w_t      = clip(c_t / RV_{t-1}, 0, 1.5)      exposure for the next month
r^VM_t   = rf_t + w_t · x_t                  idle cash earns rf, any shortfall is borrowed at rf
```
- c_t is re-estimated every month **from past data only** (from month 24; w = 1 before that).
  The paper fixes c over the full sample, which uses hindsight, so this is **a real-time variant**, not an exact replication of the paper.
- Extra trading from the layer is charged on the netted order: the rebalance and the exposure change go out as one
  order per stock, so opposite trades in the same stock cancel (`volatility.layer_turnover`). This is the only cost
  method in the project; every layer result, including the robustness checks and the seed runs, uses it. The stock
  weights it needs are saved with the backtest in `output/holdings.csv.gz` (`netted_turnover.py` rebuilds them).
- The course data has no daily stock returns, so the realized variance of the **whole market**, not of the portfolio, is used as the signal.

---

# Team framework details (①–⑦, ⑨)

> All settings and their reasons, the main strategy selection rule, the sensitivity plan and the limitations are in [SETTINGS.md](SETTINGS.md).

A monthly rebalancing backtest that runs only on the course data (`data/IFE_term_stock_data.csv`).

## Pipelines of the reference papers

### [K24] Khelifa, Allier & Cucuringu (ICAIF 2024)
1. **Graph**: the Pearson correlation matrix Γ of returns is read as a signed-graph adjacency matrix (§2)
2. **Clustering**: Spectral / Signed Laplacian / SPONGE / SPONGE_sym -> k-means++ (§3)
3. **Within-cluster weights β**: Gaussian kernel, larger closer to the center (mean return vector) (§4.2)
4. **Synthetic assets**: r*_k = Σ β_i r_i, compute μ* and Σ* (K×K) (§4.1-ii)
5. **Across-cluster weights α**: K-dimensional Markowitz (minimum variance under a target return) (§1.1, eq. 1)
6. **Stock weights**: w_i = α_k β_i (§4.1-ii)
7. **Stabilization**: repeat 1–6 R times and average the weights (§4.1-iii)
8. **Evaluation**: hold, move one window, repeat (§4.1-iv), EWA covariance (§4.6), K = 10/17/24 compared (§5.3)

### [C23] Cartea, Cucuringu & Jin (ICAIF 2023)
1. **Market residuals**: R_res = R − β R_mkt (§2.2.1)
2. **Residual correlation matrix** -> signed-graph clustering (§2.2.2)
3. **Choosing K** (§2.3)
   - Marchenko–Pastur: number of eigenvalues above λ₊ = (1 + √(N/T))²
   - Cumulative variance: number of top eigenvalues whose cumulative share reaches P (90%)

## Preprocessing (once before the backtest, data.py)
| Step | Rule | Result |
|---|---|---|
| Exchange | EXCHCD ∈ {1, 2, 3} | 1,207,153 -> 1,206,669 rows |
| Price level | no filter: a high ADJ_PRC is the trace of later reverse splits, not an error, and screening on it would use future information | 0 dropped |
| Returns | computed only from two consecutive monthly prices | missing after a skipped month |
| Spike-reversal errors | if r_t > 300% and (1+r_t)(1+r_{t+1}) − 1 < 50%, both months are missing (Ince & Porter 2006) | 2 cases, 4 returns |
| NASDAQ volume | ÷2.0 before 2001-01, ÷1.8 for 2001-02~12, ÷1.6 for 2002~03 (Gao & Ritter 2010) | NASDAQ share of the top 500 by volume about 32% |
| Minimum price | **not applied**: ADJ_PRC is retroactively adjusted for splits/dividends (e.g. AAPL 2010-12 = 9.97), which leaks future information | – |

**Missing next-month returns** (`realized_next_returns`)
- Delisted stocks get the actual CRSP delisting return (DLRET: last trading price -> final delisting value) in the month after the last price (`data/delist_matched.csv`, Shumway 1997). Missing DLRETs (10 of 8,360 records) use Shumway (1997) / Shumway & Warther (1999): performance-based delisting −30% (NYSE/AMEX) or −55% (NASDAQ), otherwise 0%
- A price that appears again later (data gap) or a removed error -> last price kept (0%); the stock drops out at the next rebalance
- Stocks in `delist_matched.csv` without a usable CRSP record (61 with no record, 18 whose delisting date does not
  match the last price) get 0%. A stock missing from the file entirely would get `Config.delist_fallback` (−2%)
- In the 2005–2020 universe 391 next-month returns were missing (about 2 a month): 370 got the actual DLRET
  (280 mergers, 86 performance-based, 4 other), 6 had no CRSP record (0%), 15 were data gaps (0%). The −2%
  fallback was never used
- `Config.delist_mode` = "dlret" (default) | "category" (fixed return per category, for comparison) | "none" (0%, previous version)

**Nature of returns**: ADJ_PRC includes dividends, so returns are close to total returns -> compared with the S&P 500 **Total Return** index.

## Detailed monthly pipeline
(The numbers below follow the implementation order and differ from ①–⑨ in the "Monthly pipeline and files" table above.)
```
Each month t (using data up to t only)
① Universe     EXCHCD 1/2/3, complete returns for the past lookback (60) months, top N (500) by average VOL   data.py
② Preprocess   1%/99% cross-sectional winsorization (estimation only), market residuals (vs universe EW average) [C23]  data.py
③ Choose K     decided every month: Marchenko–Pastur (default) or cumulative variance — all candidates logged [C23]  clustering.py
④ Clustering   SPONGE_sym or Signed Laplacian embedding -> k-means++ [K24]                    clustering.py
⑤ β            gaussian ② / ivp ③ / periphery ⑥ (labels shared so only the effect of β is compared)  allocation.py
⑥ α            synthetic ETF returns -> Σ* (time-weighted) -> long-only GMVP or MSRP          optimization.py

※ Every estimate in the window (market beta, correlation, K, β, Σ*) uses exponential weights with an 18-month half-life [K24 §4.6].
   60-month window -> about 43 effective observations, latest month weight about 4.2%, latest 12 months about 41% in total.
   The effective number of observations is used as T in Marchenko–Pastur (an approximation of a theory that assumes equal weights).
⑦ Stock weights  w = α × β, average of R repeats [K24], (optional) projection onto the stock cap   portfolio.py
⑧ Holding        realized t+1 returns (raw), turnover and costs                                backtest.py
```

## Sources by step
| Step | Source | Implementation |
|---|---|---|
| Correlation matrix as a signed graph | [K24] §2 | `clustering.correlation` |
| SPONGE_sym clustering | [K24] §3.4 (original: Cucuringu et al. 2019) | `clustering.embedding_sponge_sym` |
| Signed Laplacian (alternative) | [K24] §3.2 (original: Kunegis et al. 2010) | `clustering.embedding_signed_laplacian` |
| k-means++ | [K24] §3.1 (original: Arthur & Vassilvitskii 2007) | `clustering.kmeans_pp` |
| Gaussian-kernel within-cluster β | [K24] §4.2 | `allocation.beta_gaussian` |
| Synthetic asset returns, w = α × β | [K24] §4.1 | `portfolio.build_portfolios` |
| Exponentially weighted (EWA) covariance | [K24] §4.6 | `data.time_weights`, `optimization.covariance` |
| Averaging weights over R repeats | [K24] §4.1-iii, Table 2 | `portfolio.build_portfolios` |
| Walk-forward evaluation | [K24] §4.1-iv, Fig. 2 | `backtest.run_backtest` |
| Turnover and transaction costs | [K24] §4.5 | `backtest._turnover`, `metrics.net_returns` |
| Market residual returns | [C23] §2.2.1 eq. (2) | `data.market_residuals` |
| K by Marchenko–Pastur | [C23] §2.3 eq. (5) | `clustering.choose_k` |
| K by cumulative variance (alternative) | [C23] §2.3 eq. (6) | `clustering.choose_k` |
| Inverse-variance β | within-cluster allocation of HRP, López de Prado (2016) | `allocation.beta_ivp` |
| Periphery-weighted β | Pozzi, Di Matteo & Aste (2013) | `allocation.beta_periphery` |
| Ledoit–Wolf shrinkage (benchmark) | Ledoit & Wolf (2004) | `optimization.ledoit_wolf` |
| Mean-variance optimization (GMVP/MSRP) | Markowitz (1952) | `optimization.gmvp`, `optimization.msrp` |
| Spike-reversal error filter | Ince & Porter (2006) | `data.load_panel` |
| NASDAQ volume adjustment | Gao & Ritter (2010) | `data._nasdaq_volume_divisor` |
| Risk-free rate | Ken French Data Library | `benchmarks.load_ff_rf` |
| Sortino ratio | Sortino & Price (1994) | `metrics.performance` |

**Choices made by the team without a paper basis**: universe (top 500 by volume), no minimum price filter, 18-month half-life, K range 3–20, 5% stock cap, cost levels, main strategy selection rule -> details in [SETTINGS.md](SETTINGS.md).

## Adjustments relative to the papers
| Item | Paper | This framework | Reason |
|---|---|---|---|
| Data frequency | daily | monthly | course data |
| Estimation window | 75 trading days | 60 months, exponential weights with 18-month half-life | enough observations + recent structural change (half-life chosen by how K tracks market volatility, not by returns) |
| Expected return μ* | future returns + noise (η) [K24] | past average (MSRP only) | avoids look-ahead |
| Positions | long-short [K24] | long-only | implementability |
| Gaussian σ² | cross-validation [K24] | scale × mean within-cluster distance² | team setting |
| Market return | SPY [C23] | universe equal-weight average | uses only the given data |
| Universe | top by market cap [C23] | top by average VOL | no market cap data |
| Price/return errors | – | see preprocessing table | removes data errors (judged on the full sample, stated in the report) |
| Delisting | – | actual CRSP delisting return (DLRET), Shumway values if missing | avoids delisting bias [Shumway 1997] |
| Transaction costs | 1bp [K24 §4.5] | 20bp × turnover (0, 1, 10bp also reported) | conservative assumption for monthly stock rebalancing |
| Main strategy choice | – | highest Sharpe over the whole backtest, all six reported | no model is trained, so there is no train / test split |

## Running with other settings
See the top of this document for how to run. To change settings, edit `config.py` or, in Python:
```python
from config import Config, VMConfig
from main import main
main(Config(halflife=12, output_dir=Config().output_dir.parent / "output_hl12"))   # 12-month half-life
main(vmcfg=VMConfig(cap=2.0), skip_backtest=True)                                  # layer cap 2x (reuses the backtest)
```

## External data (for evaluation, not used in the strategy)
- `data/F-F_Research_Data_Factors_CSV.zip`: Ken French monthly RF
- `data/benchmarks_monthly.csv`: RF, S&P 500 Total Return (^SP500TR) and SPY monthly returns (built by `benchmarks.py`, or automatically by `main.py` if missing)
- `data/ff_daily.csv`, `data/market_rv_monthly.csv`: Ken French daily Mkt-RF and monthly market realized variance (layer signal, downloaded automatically by `volatility.py` if missing)
- An S&P 500 approximation that splits annual returns into 12 equal months is not used because it distorts volatility and MDD.
