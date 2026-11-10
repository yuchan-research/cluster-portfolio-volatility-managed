# Team 5: cluster-based portfolio with a volatility-managed layer

Final version of the Team 5 term project (Introduction to Financial Engineering).

Each month we pick the 500 most traded stocks in the CRSP course data, group them with signed-graph clustering
(SPONGE_sym on residual correlations), weight stocks inside each cluster by inverse variance and the clusters by a
long-only minimum-variance portfolio. On top of that portfolio a volatility-managed layer (Moreira and Muir, 2017)
scales total exposure between 0 and 1.5x using last month's market volatility. Weights at month t use data up to t
only and are evaluated on the t+1 return.

## Results (Feb 2005 - Dec 2020, 191 months, 20bp one-way costs)

| Portfolio | CAGR | Volatility | Sharpe | Max drawdown |
|---|---|---|---|---|
| Final model (Cluster-ivp-gmvp + layer) | 8.88% | 9.62% | 0.806 | -14.3% |
| Cluster-ivp-gmvp, no layer | 8.56% | 13.07% | 0.602 | -48.9% |
| S&P 500 Total Return | 9.79% | 14.78% | 0.625 | -50.9% |
| Ledoit-Wolf GMVP, no clustering | 7.42% | 12.10% | 0.553 | -42.0% |
| Ledoit-Wolf MSRP, no clustering | 10.78% | 14.49% | 0.696 | -45.3% |

The layer keeps the return and cuts the maximum drawdown to about a third of the S&P 500's. The Sharpe difference
to the S&P 500 is not statistically significant (p = 0.38), and most of the Sharpe gain comes from the 2008 crisis;
the drawdown reduction is the robust part (97.8% of bootstrap samples). The layer's trading is charged on the
netted order per stock (the rebalance and the exposure change go out as one order), for every result. Details are in the report and in
`code/SETTINGS.md`.

## Folders

```
code/                 all code; run python main.py from here (see code/README.md)
data/                 external data (Ken French RF and daily market returns, S&P 500 TR)
output/               final results: tables (csv) and figures (png), seeds/ for the seed check
code_structure.png    how the files call each other
```

## How to run

1. Add the two CRSP files to `data/`: `IFE_term_stock_data.csv` (course data) and `delist_matched.csv`
   (CRSP delisting returns matched to it). They are licensed data, so they are not in this repository.
2. `cd code` and `python main.py` (about 50 minutes in total). The backtest also saves each strategy's stock
   weights to `output/holdings.csv.gz`, which the layer's costs need; they contain CRSP returns, so they are not
   in this repository either. With `python main.py --skip-backtest` they are rebuilt first (about 45 minutes).

Requirements: numpy, pandas, matplotlib (tested with Python 3.8, numpy 1.24, pandas 2.0, matplotlib 3.7). The only
random step, the k-means++ initialisation, is fixed with seed 42, so a rerun reproduces `output/`.
