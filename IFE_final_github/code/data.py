"""
Data loading and preprocessing, monthly universe selection, and estimation helpers.

Preprocessing (done once before the backtest)
  1. Exchange filter: EXCHCD ∈ {1, 2, 3} (assignment rule)
  2. Price level: no filter (cfg.price_error_threshold = None)
     A very large ADJ_PRC is not an error. ADJ_PRC is anchored to the most recent price, so every reverse
     split multiplies all earlier prices; a stock that later split 1:20 ten times shows prices a trillion
     times its traded level. Returns are ratios, so they are unaffected and stay in a normal range.
     Screening on the level would drop valid data using information about future splits - the same
     objection that rules out a minimum-price filter below.
  3. Returns: computed only from two consecutive monthly prices (missing if a month is skipped)
  4. Spike-reversal errors (Ince & Porter 2006):
     if r_t > 300% and (1+r_t)(1+r_{t+1}) − 1 < 50%, r_t and r_{t+1} are treated as errors and set to missing
  5. NASDAQ volume double-counting adjustment (Gao & Ritter 2010):
     ÷2.0 before 2001-01, ÷1.8 for 2001-02~2001-12, ÷1.6 for 2002~2003
  6. Last month with a price per stock (to tell data gaps from true delistings)
  7. Delisting returns from the matched CRSP delisting file (Shumway 1997), applied in the month after the last price

Not applied
  - Minimum price ($5) filter: ADJ_PRC is retroactively adjusted for later splits/dividends, so filtering
    on past ADJ_PRC leaks future information (stocks that later split look like low-priced stocks in the past).

All universe and estimation functions use only information up to time t.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass                                                                      # plain container for the panel
class Panel:
    returns: pd.DataFrame     # month × PERMNO returns (after error removal)
    price: pd.DataFrame       # month × PERMNO adjusted prices
    volume: pd.DataFrame      # month × PERMNO volume (after NASDAQ adjustment)
    exchange: pd.DataFrame    # month × PERMNO exchange code
    error_mask: pd.DataFrame  # positions set to missing as spike-reversal errors (True)
    delist_ret: dict          # (PERMNO, month) -> delisting return applied in that month
    last_date: dict           # PERMNO -> last month with a price (data gap vs. true delisting)
    info: dict


def _nasdaq_volume_divisor(dates):                                              # NASDAQ volume divisor by date
    """NASDAQ volume divisor by date (Gao & Ritter 2010)."""
    d = pd.Series(1.0, index=dates)                                   # default 1 (from 2004)
    d[dates < pd.Timestamp("2004-01-01")] = 1.6                       # 2002~2003
    d[dates < pd.Timestamp("2002-01-01")] = 1.8                       # 2001-02~2001-12
    d[dates < pd.Timestamp("2001-02-01")] = 2.0                       # before 2001-01
    return d


def load_panel(cfg):
    """Read the CSV and build the preprocessed Panel."""
    cols = ["PERMNO", "date", "EXCHCD", "ADJ_PRC", "VOL"]                   # only the columns the strategy needs
    df = pd.read_csv(cfg.data_path, usecols=cols, parse_dates=["date"])
    n_rows_raw = len(df)
    df = df[df["EXCHCD"].isin(cfg.exchanges)]                                # 1. keep NYSE/AMEX/NASDAQ only
    if cfg.price_error_threshold is None:                                    # 2. no price-level filter (default)
        bad = np.array([], dtype=np.int64)
    else:                                                                    # optional, kept for comparison
        bad = df.loc[df["ADJ_PRC"] > cfg.price_error_threshold, "PERMNO"].unique()
        df = df[~df["PERMNO"].isin(bad)]

    price = df.pivot(index="date", columns="PERMNO", values="ADJ_PRC").sort_index()  # month × stock prices
    shape = dict(index=price.index, columns=price.columns)
    volume = df.pivot(index="date", columns="PERMNO", values="VOL").reindex(**shape)
    exchange = df.pivot(index="date", columns="PERMNO", values="EXCHCD").reindex(**shape)

    returns = price / price.shift(1) - 1.0                                   # 3. returns from consecutive months only
    nxt = returns.shift(-1)
    spike = (returns > cfg.reversal_threshold) & ((1 + returns) * (1 + nxt) - 1 < cfg.reversal_cum)  # 4. spike then reversal
    error_mask = spike | spike.shift(1, fill_value=False)                    # mark both the spike and the reversal month
    returns = returns.mask(error_mask)                                       # set error returns to missing

    if cfg.nasdaq_volume_adjust:                                             # 5. NASDAQ volume adjustment
        div = _nasdaq_volume_divisor(volume.index)
        is_nasdaq = exchange == 3                                            # NASDAQ positions
        volume = volume.where(~is_nasdaq, volume.div(div, axis=0))           # divide NASDAQ only

    last_date = price.apply(pd.Series.last_valid_index)                      # 6. last month with a price per stock
    delist_ret = load_delist_returns(cfg)                                    # 7. delisting returns (empty if "none")

    info = {"rows_raw": n_rows_raw,
            "rows_after_exchange_filter": len(df),                           # rows after filters
            "bad_price_permno": len(bad),                                    # stocks dropped for price errors
            "spike_reversal_cases": int(spike.values.sum()),                 # number of spike-reversal errors
            "returns_masked": int(error_mask.values.sum()),                  # number of returns set to missing
            "delist_records": len(delist_ret)}
    return Panel(returns, price, volume, exchange, error_mask,
                 delist_ret, last_date.dropna().to_dict(), info)             # + delisting returns, last price month


def load_delist_returns(cfg):
    """(PERMNO, month) -> return for the month after the last price, from the matched CRSP delisting file.

    "dlret"    : actual CRSP DLRET (last trading price -> final delisting value). Column ret_shumway already fills
                 the few missing DLRETs with Shumway (1997) / Shumway & Warther (1999): performance-based
                 delisting -30% (NYSE/AMEX) or -55% (NASDAQ), otherwise 0%.
    "category" : one fixed return per delisting category (cfg.delist_cat_returns), kept for comparison.
    """
    if cfg.delist_mode == "none":                                            # no delisting adjustment
        return {}
    if not cfg.delist_path.exists():                                         # file not in data/
        print(f"    [warning] {cfg.delist_path.name} not found -> delisting returns not applied (0%)")
        return {}
    dl = pd.read_csv(cfg.delist_path)
    month = pd.to_datetime(dl["event_month"]).dt.to_period("M")             # month the return applies (period: avoids
                                                                             #   trading-day vs. calendar month-end mismatch)
    if cfg.delist_mode == "dlret":
        ret = dl["ret_shumway"]                                              # DLRET, Shumway values where missing
    elif cfg.delist_mode == "category":
        ret = dl["cat"].map(cfg.delist_cat_returns)                          # category -> assumed return
    else:
        raise ValueError(f"delist_mode must be dlret, category or none: {cfg.delist_mode}")
    ret = ret.fillna(cfg.delist_fallback)                                    # no record / no category -> fallback
    return {(int(p), m): float(r) for p, m, r in zip(dl["permno"], month, ret)}


def select_universe(panel, t_idx, cfg):                                         # universe and estimation window at t
    """Universe at t: top N by average volume among stocks with returns for the whole window."""
    start = t_idx - cfg.lookback + 1
    window = panel.returns.iloc[start : t_idx + 1]                    # past lookback months of returns (including t)
    complete = window.notna().all(axis=0)                             # stocks with no missing month in the window
    avg_vol = panel.volume.iloc[start : t_idx + 1].mean(axis=0)       # average volume in the window
    avg_vol = avg_vol[complete].dropna()                              # keep stocks with complete returns
    members = avg_vol.nlargest(cfg.universe_size).index
    return window[members]                                            # (lookback × N) returns DataFrame


def realized_next_returns(panel, t_idx, members, cfg):                          # realised t+1 returns with delistings
    """Realized returns for month t+1. A stock with no t+1 price is filled in this order:

      ① removed as a spike-reversal error            -> 0% (last price kept)
      ② delisting record for t+1 (cfg.delist_mode)   -> delisting return
      ③ price appears again later (data gap)          -> 0%
      ④ data ends with no usable record               -> cfg.delist_fallback
    With delist_mode="none" every missing return is 0% (last price kept), as in the original version.
    The stock then drops out at the next rebalance through the universe rule (complete returns).
    """
    r = panel.returns.iloc[t_idx + 1].reindex(members)                # t+1 returns
    t1 = panel.returns.index[t_idx + 1]                               # month t+1
    missing = r.isna()
    err = panel.error_mask.iloc[t_idx + 1].reindex(members, fill_value=False) & missing
    n_delist = 0                                                      # missing returns filled from delisting records
    if cfg.delist_mode != "none":
        for p in r.index[missing & ~err]:                             # missing for reasons other than errors (①: 0% below)
            key = (int(p), t1.to_period("M"))                         # (PERMNO, month t+1)
            if key in panel.delist_ret:                               # ② delisting record found
                r[p] = panel.delist_ret[key]
                n_delist += 1
            elif panel.last_date.get(p, t1) > t1:                     # ③ price appears again later
                r[p] = 0.0                                            # data gap: last price kept
            else:                                                     # ④ data ends, no record
                r[p] = cfg.delist_fallback                            # fallback return
    counts = {"next_missing_error": int(err.sum()),                   # missing because of error removal
              "next_missing_no_data": int((missing & ~err).sum()),    # missing because data ends (delisting etc.)
              "next_delist_matched": n_delist}                        # of which filled with a delisting return
    return r.fillna(0.0), counts                                      # remaining missing (①, or "none") = 0%


def winsorize_cross_section(window, q):                                         # clip extremes month by month
    """Clip extreme values month by month using cross-sectional quantiles (for estimation)."""
    lo = window.quantile(q, axis=1)                                   # lower quantile per month
    hi = window.quantile(1 - q, axis=1)                               # upper quantile per month
    return window.clip(lower=lo, upper=hi, axis=0)


def time_weights(T, halflife):
    """Observation weights in the window (sum 1). Larger for recent months; equal if halflife=None."""
    if halflife is None:                                              # equal-weight setting
        return np.full(T, 1.0 / T)                                    # 1/T for every month
    age = np.arange(T - 1, -1, -1)                                    # age of each observation in months (latest = 0)
    w = 0.5 ** (age / halflife)                                       # weight halves every half-life
    return w / w.sum()                                                # normalize to sum 1


def effective_n(w):                                                             # Kish effective sample size
    """Effective number of observations (Kish): 1 / Σw²."""
    return 1.0 / np.sum(w ** 2)                                       # equals T under equal weights


def market_residuals(X, w):                                                     # returns net of the market factor
    """[C23 eq. (2)] residual = r_i − β_i × r_mkt. Market = universe equal-weight average, β from weighted regression."""
    mkt = X.mean(axis=1)                                              # market return proxy (T,)
    mkt_c = mkt - w @ mkt                                             # remove weighted mean
    X_c = X - w @ X                                                   # remove weighted mean per stock
    beta = (w * mkt_c) @ X_c / (w @ mkt_c ** 2)                       # weighted market beta per stock (N,)
    return X - np.outer(mkt, beta)                                    # residual returns (T, N)
