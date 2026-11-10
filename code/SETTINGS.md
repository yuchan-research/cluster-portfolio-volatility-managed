# Settings log

**Every setting we chose** for the strategy and backtest, with its reason. Used directly for the "Methodology" and "Limitations" parts of the report.

**Types of reasons**
- **[Paper]** method or value from a reference paper
- **[Practice]** common practice in empirical finance
- **[Data check]** decided by inspecting the data (return performance not looked at)
- **[Team]** decided in team discussion
- **[Arbitrary]** no clear basis -> target of sensitivity analysis

References: [K24] Khelifa, Allier & Cucuringu (ICAIF 2024), [C23] Cartea, Cucuringu & Jin (ICAIF 2023), [MM17] Moreira & Muir (2017, JF)

---

## 1. Data preprocessing
| Setting | Value | Reason |
|---|---|---|
| Exchanges | EXCHCD ∈ {1, 2, 3} | [Assignment] |
| Stock identifier | PERMNO | [Assignment] |
| Returns | ratio of ADJ_PRC in two consecutive months − 1 | [Assignment] uses ADJ_PRC |
| Price level | **no filter** | [Data check] A large ADJ_PRC is not an error. ADJ_PRC is anchored to the latest price, so each reverse split multiplies every earlier price: Top Ships traded near $10 in 2005 but shows $1.2 trillion because of later splits. Returns are ratios and stay normal (TOPS: mean −7.5% a month, never beyond ±100%). The earlier filter dropped 34 stocks, 33 of which fell ~100%, and the 34th was Berkshire Hathaway, genuinely priced above $100,000. Screening on the level needs to know about future splits, the same objection as the minimum-price filter below. Keeping them lowers the equal-weight universe return by 0.46%p a year |
| Spike-reversal errors | if r_t > 300% and (1+r_t)(1+r_{t+1}) − 1 < 50%, both months missing (2 cases) | [Practice] Ince & Porter (2006) |
| NASDAQ volume adjustment | ÷2.0 before 2001-01, ÷1.8 for 2001-02~12, ÷1.6 for 2002~03 | [Practice] Gao & Ritter (2010) |
| Minimum price filter | **not used** | [Data check] ADJ_PRC is retroactively adjusted for splits/dividends (AAPL 2010-12 = 9.97), so a past-price filter leaks future information |
| Delisting | no next-month price -> actual CRSP delisting return (DLRET) in that month; missing DLRET -> −30% (NYSE/AMEX) / −55% (NASDAQ) for performance-based delisting, otherwise 0% | [Paper] Shumway (1997), Shumway & Warther (1999) + [Data check] 391 missing next-month returns in the universe, 370 with an actual DLRET. Main strategy (20bp) Sharpe 0.604 (0%) -> 0.602 (DLRET), 0.647 with fixed category returns (`run_variants.py delist`): conclusions unchanged. Fixed category returns (merger +30%) were rejected: the takeover premium is already in the prices before delisting (median +34% over the prior 12 months, actual merger DLRET median +0.5%), so it is double-counted |

## 2. Universe
| Setting | Value | Reason |
|---|---|---|
| History requirement | stocks with all 60 months of returns in the estimation window | [Team] needed for the correlation matrix. Stocks listed for less than 5 years are excluded |
| Liquidity requirement | top 500 by average (adjusted) volume in the window | [Practice] + [Team] volume used as a proxy because there is no market cap. C23 uses the top 25% by market cap |
| Universe size | 500 | [Team] balance between practice and estimation stability -> sensitivity: 200, 1,000 |

## 3. Estimation window and weights
| Setting | Value | Reason |
|---|---|---|
| Method | rolling walk-forward, re-estimated monthly | [Paper] K24 §4.1 (adapted to monthly) |
| Window | 60 months | [Data check] shorter windows (24, 36 months) have too few observations for 500 stocks |
| Time weights | exponential with 18-month half-life (about 43 effective observations, latest month 4.2%) | [Paper] K24 §4.6 EWA + [Data check] at 18 months K has a correlation of 0.68 with market volatility; not chosen by returns, other half-lives not run |
| Scope | the same weights for market beta, correlation, K, β, Σ* and the benchmark covariance | [Team] consistency |
| Winsorization for estimation | monthly cross-sectional 1% top and bottom | [Practice] estimation only, not applied to realized returns |

## 4. Clustering
| Setting | Value | Reason |
|---|---|---|
| Input | correlation of market-residual returns (market = universe equal-weight average) | [Paper] C23 §2.2.1. Equal-weight average instead of SPY to use only the given data |
| Algorithm | SPONGE_sym | [Paper] best in both K24 and C23 |
| SPONGE τ+, τ− | 1, 1 | [Arbitrary] SigNet defaults |
| Choosing K | Marchenko–Pastur each month (eigenvalues above λ₊), effective observations used as T | [Paper] C23 §2.3 + [Data check] the 90% cumulative-variance method always hits the cap (K of 39–44). Using MP with weighted data is an approximation |
| K range | 3 – 20 | [Team] lower 3: with K ≤ 2 diversification across clusters is meaningless. Upper 20: estimation limit with 43 effective observations. Actual K is about 3–8 |
| Fixed-K comparison | done: K = 5 and K = 10 against the monthly rule | [Team] full period, 20bp: 0.574 / 0.604 / 0.602 (ours). The 0.002 gap to K = 10 is smaller than the spread from changing the random seed alone (§10), so the monthly rule is kept |
| Repeats R | 10, final stock weights averaged | [Paper] K24 Table 2 |
| k-means initialization | 3 tries per repeat, lowest inertia kept | [Arbitrary] |
| Random seed | 42 | [Arbitrary] fixed before any result was seen, so the submitted numbers can be reproduced. Seeds 1-10 are rerun in `seed_sensitivity.py` (§10) |

## 5. Within-cluster weights β
| Setting | Value | Reason |
|---|---|---|
| Methods | ② Gaussian kernel, ③ inverse variance, ⑥ periphery weighting — all three are run | [Team] |
| Fair comparison | the three methods share the same cluster labels in each repeat | [Team] isolates the effect of β |
| Gaussian width | σ² = 1.0 × mean weighted within-cluster distance² | [Arbitrary] K24 uses cross-validation. Middle value fixed so it is not chosen by results -> sensitivity: 0.5, 2.0 |
| Periphery score | β ∝ 1 − (average correlation with the other stocks in the cluster) | [Team] simplified idea of Pozzi, Di Matteo & Aste (2013) |

## 6. Across-cluster weights α and final weights
| Setting | Value | Reason |
|---|---|---|
| Objective | both GMVP and MSRP are run | [Team] K24 uses target-return MVO |
| Positions | long-only | [Team] K24 is long-short |
| Expected return μ* | past weighted average of the synthetic ETFs (MSRP only) | [Team] K24's η setting based on future returns leaks future information, so it is not used |
| MSRP exception | if all expected returns are ≤ 0, fall back to GMVP (count recorded) | [Team] |
| Final weights | w_i = α_k × β_i | [Paper] K24 §4.1 |
| Stock cap | 5%, projected onto the capped simplex after averaging the repeats | [Team] avoids concentration. Same for all strategies |

## 7. Backtest and costs
| Setting | Value | Reason |
|---|---|---|
| Period | 2005-02 ~ 2020-12 (191 months) | starts in 2005 because of the 60-month window. The 2000–2002 dot-com crash is not in the evaluation |
| Execution | weights computed at the month-t close and rebalanced at the same price | [Practice] |
| Turnover | Σ\|new weight − old weight after drifting with returns\|, 1 in the first month | [Practice] |
| Transaction costs | 0 / 1 / 10 / 20bp all reported; figures and main strategy selection use 20bp | [Team] 1bp is the K24 level, 20bp is conservative (0–50bp comparison in `cost_sensitivity.py`) |

## 8. Benchmarks and evaluation
| Setting | Value | Reason |
|---|---|---|
| S&P 500 | S&P 500 **Total Return** index (^SP500TR, yfinance), close on the last trading day of each month | [Assignment] + [Data check] ADJ_PRC includes dividends, so compared with total returns. Validated against SPY adjusted prices: 0.09%p per year difference (fees), correlation 0.9993 |
| Risk-free rate | Ken French Data Library monthly RF (1-month T-bill) | [Practice] standard for CRSP-based research |
| Other benchmarks | universe equal weight, Ledoit–Wolf GMVP and MSRP without clustering (same universe, weights and cap) | [Team] to check the effect of clustering |
| Required metrics | annualized return, annualized standard deviation, Sharpe (excess over RF), MDD | [Assignment] |
| Additional metrics | Calmar, Sortino (excess over RF), beta vs S&P 500, average monthly turnover | [Team] |
| Regimes | five windows set by market history, not by our results (`regimes.py`): before crisis 2005-02~2007-10, financial crisis 2007-11~2009-03 (S&P 500 peak to trough), recovery 2009-04~2014-12 (to the end of QE3), late bull 2015-01~2019-12, COVID 2020 | [Team] shows where the layer helps and where it costs |
| Risk-estimate check | predicted vs realized volatility of each optimizer (`predicted_vol.py`) | [Team] direct evidence for goal 1: K × K 0.84 against 500 × 500 Ledoit-Wolf 0.49 |

## 9. Main strategy selection rule
- **No train / validation / test split.** No model is trained to predict returns: clustering is unsupervised and
  every other setting is fixed in advance, so there is nothing to tune on one part of the sample and test on another.
  `config.split_dates = ()`.
- **Period:** the whole backtest, 2005-02 ~ 2020-12 (191 months)
- **Criterion:** the cluster strategy with the **highest Sharpe (excess over RF)** at 20bp costs; ties broken by the smaller MDD
- **Result:** Cluster-ivp-gmvp: 0.602, ahead of 0.569 for the next one (layer results are not used for the selection)
- **Other strategies:** all six shown in the performance tables, so the label does not hide anything.
  Split in half, ivp-gmvp is first in 2013-2020 (0.952) but second in 2005-2012 (0.294 against 0.314 for
  periphery-msrp), where all six sit between 0.20 and 0.31.
- Implementation: `metrics.select_main_strategy`

## 10. Sensitivity analysis plan (defaults are kept regardless of results)
| Setting | Default | Alternatives |
|---|---|---|
| Number of clusters K | monthly (Marchenko–Pastur) | **done**: fixed K = 5 (0.574), fixed K = 10 (0.604) vs ours (0.602) |
| Random seed | 42 | **done**: seeds 1-10 in `seed_sensitivity.py`, ivp-gmvp Sharpe 0.593 on average over 11 seeds (0.582-0.602, sd 0.006) and first in all 11; with the layer 0.805 (0.798-0.812). Seed 42 happens to sit at the top of the base range, about 0.01 above the average |
| Transaction costs | 20bp | **done**: 0 / 1 / 10 / 20bp reported automatically, 0–50bp and break-even in `cost_sensitivity.py` |
| Layer cap and burn-in | 1.5x, 24 months | **done**: cap 1.0–∞ × burn-in 24/36/60 in `validation.py` |
| Layer signal | sum of squared daily returns | **done**: sample variance, standard deviation, 3-month average in `validation.py` |
| Half-life | 18 | not run |
| Universe size | 500 | not run |
| Gaussian width | 1.0 | not run |

## 11. Volatility-managed layer (⑧) [MM17]
| Setting | Value | Reason |
|---|---|---|
| Signal | sum of squared daily market excess returns (Ken French Mkt-RF) in the previous month | [Paper] RV definition of MM17. The course data has no daily stock returns, so market RV is used as a common proxy |
| Exposure | w_t = clip(c_t / RV_{t-1}, 0, 1.5) | [Paper] the 1.5x cap is the leverage limit used by MM17: borrowing up to 50% of equity, below the 2x allowed by the 50% initial margin of Regulation T |
| Coefficient c | re-estimated monthly from past data only (value that equalizes volatility before and after management), expanding window, from month 24 | [Paper] Moreira and Muir (2017) set c so the managed and unmanaged series have the same volatility; c does not change the Sharpe ratio without a cap (checked: c x 1 and c x 2 both 0.660). [Team] the paper's full-sample c uses hindsight -> real-time variant; expanding window because c is a long-run scale, not a signal. With the cap, c sets the risk level: c x 0.5 to 1.5 gives Sharpe 0.77-0.81 but MDD -10% to -22%. Rolling 36/60/120-month c gives 0.82-0.83, close to the expanding 0.81 (`c_analysis.py` [9]) |
| Costs | netted order per stock × 20bp: the rebalance and the exposure change go out as one order (`volatility.layer_turnover`) | [Team] the only cost method, used for every layer result |
| Applied to | main strategy (computed the same way for the other strategies and reported together) | [Team] |
| Sensitivity | cap 1–∞x × burn-in 24/36/60 months, 4 signals, 2 c methods, 3 c initialization alternatives, c window and scale, faster signals | `validation.py`, `c_analysis.py`, `qa_analysis.py` |
| Low-volatility tilt | same inverse-variance rule over all 500 stocks without clusters, and plain inverse volatility | [Team] `ivp_benchmark.py`: no layer 0.567 / 0.474 against 0.602 (p = 0.63 for inverse variance); with the layer 0.670 / 0.500 against 0.806, MDD -23.5% / -32.4% against -14.3% |
| Index futures | not used; tested in `qa_analysis.py` [12] | [Team] changing exposure with S&P 500 futures (2bp) instead of the stocks: Sharpe 0.766, MDD -20.8% against 0.806, -14.3%, because futures move only the market part |
| Months at the cap | 59 of the 167 months after the burn-in (none in 2007-2009, 12 in 2017); the cap binds when last month's market vol is below about 11% a year | `c_analysis.py` [8] |
| Financing | leverage above 1x borrowed at rf | [Team] simplification. `validation.py` [6]: with a 50 / 100 / 200bp spread on the borrowed part (18% on average) the Sharpe falls from 0.806 to 0.797 / 0.788 / 0.769 |
| Signal choice | market RV, not the portfolio's own variance | [Data check] the portfolio's own past 3 / 6 / 12-month variance from monthly returns gives Sharpe 0.436 / 0.644 / 0.660 against 0.806 for market RV (`validation.py` [7]) |

## 12. Limitations to state in the report
- Delisting returns: applied in the month after the last month-end price; 6 universe cases with no CRSP delisting record are kept at 0%
- Volume (share-based, not split-adjusted) is used as a liquidity proxy because there is no market cap
- Using Marchenko–Pastur with weighted data is an approximation of the theory
- Price errors were judged by looking at the full sample (data cleaning)
- Half-life, window length and K method were chosen by inspecting the data, without looking at returns
- Half-life, universe size and the Gaussian width were not tested against alternatives
- RF and S&P 500 are external data (not used in the strategy)
- With a 191-month sample, the statistical significance of performance differences between strategies is limited
- The k-means seed moves the Sharpe ratio by up to about 0.02 (§10); differences between strategies of that size are noise
- Layer: the signal is the realized variance of the whole market, not of the portfolio: in 72 months with exposure above 1.2x, 9 still lost more than 3% (worst 2019-05, -7.6%) while the market was calm (`validation.py` [8])
- Layer: the Sharpe improvement is sensitive to how c is estimated, when estimation starts and the GFC period (the MDD reduction is relatively consistent) — `c_analysis.py`
