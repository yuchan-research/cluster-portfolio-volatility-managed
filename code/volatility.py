"""
Volatility-managed layer [Moreira & Muir 2017, JF]: an overlay applied on top of a strategy.

Idea: when last month's market realized variance is high, reduce next month's exposure; when it is low, increase it.
This is based on the evidence that volatility is persistent in the short run while expected returns do not rise as much.

    x_t      = r_t − rf_t                        monthly excess return of the strategy without the layer (before costs)
    RV_{t-1} = Σ_{d ∈ t-1} (Mkt-RF_d)²           sum of squared daily market excess returns in the previous month
    w_t      = clip(c_t / RV_{t-1}, 0, cap)      exposure for the next month
    r^VM_t   = rf_t + w_t · x_t                  idle cash earns rf, any shortfall is borrowed at rf

  - RV uses information only up to the previous month -> no look-ahead.
  - c_t is re-estimated every month from past data only (real time). Before burn_in months, w = 1.
    The paper's version (c fixed over the full sample) is available with c_mode="full" but uses hindsight.
  - The 1.5x cap is the leverage limit M&M use: borrowing up to 50% of equity, within the 2x that Reg T's
    50% initial margin allows.

The course data (CRSP monthly) has no daily stock returns, so the realized variance of the **whole market**
is used as a common proxy instead of the portfolio's own variance (stated as a limitation in the report).

Turnover: the layer scales the whole portfolio by w_t. The rebalance to the new base weights and the
  exposure change go out as one order per stock, so costs are charged on the netted order
    turnover_VM = Σ_i |w_t a_it − E_{t-1} d_it|      (layer_turnover)
  (a: base target weights, d: last month's weights after drifting, E: share in stocks before rebalancing).
  The stock weights come from the backtest (output/holdings.csv.gz, see netted_turnover.py).

External data: Ken French daily factors (cached in data/ff_daily.csv, data/market_rv_monthly.csv).
Settings: config.VMConfig (cap, floor, burn-in, c mode, signal definition).
"""

import io
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from config import PROJECT_DIR

DATA_DIR = PROJECT_DIR / "data"
FF_DAILY_ZIP_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                    "F-F_Research_Data_Factors_daily_CSV.zip")
FF_DAILY_CSV = DATA_DIR / "ff_daily.csv"                              # cache of daily Mkt-RF, RF
MARKET_RV_CSV = DATA_DIR / "market_rv_monthly.csv"                    # cache of monthly realized variance
HOLDINGS_FILE = "holdings.csv.gz"                                      # stock weights saved with the backtest
_HOLDINGS = {}                                                         # holdings already read, per folder


# ── Market realized variance ─────────────────────────────────────
def _download_ff_daily():
    """Download Ken French daily factors (Mkt-RF, RF) as decimals."""
    raw = urllib.request.urlopen(FF_DAILY_ZIP_URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(",Mkt-RF"))
    body = []
    for line in lines[start + 1:]:                                     # after the header
        if not line.strip() or not line.strip()[0].isdigit():          # blank line or start of another table
            break                                                      # end of the daily section
        body.append(line)
    ff = pd.read_csv(io.StringIO("\n".join([lines[start]] + body)), index_col=0)
    ff.index = pd.to_datetime(ff.index.astype(str).str.strip(), format="%Y%m%d")
    ff = ff[["Mkt-RF", "RF"]] / 100.0                                  # percent -> decimal
    ff.index.name = "date"
    return ff


def market_rv_monthly(refresh=False):                                           # monthly realised variance of the market
    """Monthly market realized variance. Downloaded from Ken French and cached if no cache exists."""
    if refresh or not MARKET_RV_CSV.exists():                          # no cache or refresh requested
        ff = _download_ff_daily()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ff.to_csv(FF_DAILY_CSV)
        g = ff.groupby(ff.index.to_period("M"))["Mkt-RF"]
        rv = pd.DataFrame({                                            # monthly realized variance table
            "RV2": g.apply(lambda x: (x ** 2).sum()),                  # sum of squares (M&M definition)
            "RV2_var": g.var(ddof=1) * g.count(),                      # sample-variance based (for comparison)
            "n_days": g.count(),
        })
        rv["ann_vol"] = np.sqrt(rv["RV2"] / rv["n_days"] * 252)        # annualized volatility (for reporting)
        rv.index.name = "month"
        rv.to_csv(MARKET_RV_CSV)
    rv = pd.read_csv(MARKET_RV_CSV, index_col="month")
    rv.index = pd.PeriodIndex(rv.index, freq="M")
    return rv                                                          # monthly realized variance


def signal_series(dates, vmcfg):                                                # signal known at the start of each month
    """Previous-month volatility signal aligned to backtest dates (weights for month t use the value from t−1)."""
    rv = market_rv_monthly()                                           # monthly realized variance
    raw = {"rv": rv["RV2"],                                            # sum-of-squares variance (default)
           "var": rv["RV2_var"],
           "sd": np.sqrt(rv["RV2"]),                                   # standard deviation (1/σ scaling)
           "rv3": rv["RV2"].rolling(3).mean()}[vmcfg.signal]           # 3-month average
    prev = raw.reindex(dates.to_period("M") - 1)                       # shift to the previous month
    return pd.Series(prev.to_numpy(), index=dates, name="signal")


# ── Layer ───────────────────────────────────────────────────────
def vm_weights(excess, signal, vmcfg):
    """Monthly exposure w_t. c_t is estimated from past data only (realtime) or fixed over the full sample (full)."""
    x, s = excess.to_numpy(float), signal.to_numpy(float)              # excess returns, signal
    w = np.ones(len(x))                                                # default exposure 1x
    if vmcfg.c_mode == "full":                                         # paper's version: c fitted on the full sample
        scaled = x / s
        c = np.nanstd(x, ddof=1) / np.nanstd(scaled, ddof=1)           # c that equalizes volatility before and after management
        return pd.Series(np.clip(c / s, vmcfg.floor, vmcfg.cap), index=excess.index, name="w")  # exposure (with floor and cap)
    for j in range(vmcfg.burn_in, len(x)):                             # real time: past data only, every month
        past, past_s = x[:j], s[:j]
        denom = np.nanstd(past / past_s, ddof=1)                       # volatility of the unscaled managed series
        if denom > 0 and np.isfinite(s[j]):                            # if computable
            w[j] = np.clip(np.nanstd(past, ddof=1) / denom / s[j], vmcfg.floor, vmcfg.cap)
    return pd.Series(w, index=excess.index, name="w")


def load_holdings(folder):                                                       # stock holdings of the backtest
    """Monthly stock weights per strategy (saved with the backtest), split into the pieces the netting needs.

    Returns {strategy: {date: (a, d)}}: a = target base weights for the month the return is earned, d = last
    month's base weights after drifting with returns, both on the same stock index (d is None in the first month).
    Read once per folder and kept in memory.
    """
    key = str(folder)
    if key in _HOLDINGS:                                               # already read
        return _HOLDINGS[key]
    path = Path(folder) / HOLDINGS_FILE                                # file written by the backtest
    if not path.exists():                                              # older results without holdings
        raise FileNotFoundError(f"{path} not found: run netted_turnover.py to rebuild it")
    h = pd.read_csv(path, parse_dates=["date"])                        # long table: date, strategy, PERMNO, weight, ret
    out = {}
    for name, g in h.groupby("strategy"):
        parts, prev = {}, None                                         # pieces per month, last month's holdings
        for d, cur in g.groupby("date"):
            cur = cur.set_index("PERMNO")
            a = cur["weight"]
            if prev is None:                                           # first month: nothing held yet
                parts[d] = (a.to_numpy(), None)                        # only the purchase
            else:
                grown = prev["weight"] * (1.0 + prev["ret"])           # holdings after last month's returns
                drift = grown / grown.sum()                            # weights just before rebalancing
                idx = a.index.union(drift.index)
                parts[d] = (a.reindex(idx, fill_value=0.0).to_numpy(),  # target on the common index
                            drift.reindex(idx, fill_value=0.0).to_numpy())  # held on the common index
            prev = cur                                                 # carry to the next month
        out[name] = parts                                              # keep
    _HOLDINGS[key] = out
    return out


def layer_turnover(w, base_gross, rf, parts=None):                              # trading caused by the layer
    """Turnover with the layer when the rebalance and the exposure change go out as one order per stock.

        target       w_t · a_t                     a_t: base target weights (sum 1)
        held         E_{t-1} · d_t                 d_t: last month's base weights after drifting (sum 1)
        E_{t-1}      = w_{t-1} (1 + r_base) / (1 + r_layer)   share in stocks just before rebalancing
        turnover_t   = Σ_i | w_t a_it − E_{t-1} d_it |

    A stock the rebalance buys and the exposure cut sells is only traded for the difference.
    With parts=None the series is one asset (the S&P 500 index): turnover_t = |w_t − E_{t-1}|, and the
    first purchase is free, as for the index without the layer.
    """
    wv = w.to_numpy(float)
    g = base_gross.reindex(w.index).to_numpy(float)                    # base gross returns
    f = rf.reindex(w.index).to_numpy(float)
    to = np.empty(len(wv))
    to[0] = 0.0 if parts is None else wv[0]                            # first month: the initial purchase
    for k in range(1, len(wv)):
        layer_g = f[k - 1] + wv[k - 1] * (g[k - 1] - f[k - 1])         # layer gross return last month
        equity = wv[k - 1] * (1.0 + g[k - 1]) / (1.0 + layer_g)        # share in stocks before rebalancing
        if parts is None:
            to[k] = abs(wv[k] - equity)                                # only the exposure change
        else:
            a, d = parts[w.index[k]]
            to[k] = np.abs(wv[k] * a - equity * d).sum()               # netted order
    return pd.Series(to, index=w.index, name=w.name)


def apply_layer(gross, turns, rf, vmcfg, holdings=None):
    """Apply the layer to each strategy and return (gross returns, netted turnover, exposure, signal).

    holdings: output of load_holdings (default: the main results in output/). A strategy with zero base
    turnover throughout (the S&P 500 index) is treated as one asset.
    """
    holdings = load_holdings(PROJECT_DIR / "output") if holdings is None else holdings
    dates = gross.index
    signal = signal_series(dates, vmcfg)                               # previous-month volatility signal
    rf = rf.reindex(dates)
    vm_gross, vm_turn, weights = {}, {}, {}
    for col in gross.columns:
        x = gross[col] - rf
        w = vm_weights(x, signal, vmcfg)
        vm_gross[col] = rf + w * x                                     # managed gross returns
        if col in holdings:
            vm_turn[col] = layer_turnover(w, gross[col], rf, holdings[col])  # netted order per stock
        elif (turns[col].reindex(dates) == 0).all():
            vm_turn[col] = layer_turnover(w, gross[col], rf)
        else:                                                          # nothing to net with
            raise KeyError(f"no holdings for {col}: run netted_turnover.py")  # stop rather than mix cost methods
        weights[col] = w                                               # keep exposure
    return (pd.DataFrame(vm_gross), pd.DataFrame(vm_turn),            # gross returns, netted turnover
            pd.DataFrame(weights), signal)


def mm_alpha(managed_excess, base_excess):                                      # Moreira-Muir alpha regression
    """[M&M] Regress managed excess returns on unmanaged excess returns: x^VM = α + β x + ε (White robust standard errors)."""
    y = managed_excess.to_numpy(float)
    x = base_excess.to_numpy(float)
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtX_inv = np.linalg.inv(X.T @ X)                                   # (X'X)^{-1}
    V = XtX_inv @ (X.T * e ** 2) @ X @ XtX_inv                         # White (HC0) covariance
    se = np.sqrt(np.diag(V))
    return {"Alpha (ann.)": b[0] * 12.0,                               # annualized alpha
            "t(Alpha)": b[0] / se[0],
            "Beta": b[1],                                              # beta on the unmanaged strategy
            "R2": 1.0 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()}
