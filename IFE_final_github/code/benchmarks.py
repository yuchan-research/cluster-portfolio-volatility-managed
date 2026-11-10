"""
External data for performance evaluation (not used to compute the strategy).

  RF      monthly RF from Ken French's Data Library F-F_Research_Data_Factors (1-month T-bill, %)
  SP500TR S&P 500 Total Return index (^SP500TR, dividends reinvested): compared with total returns because ADJ_PRC includes dividends
  SPY     SPY ETF dividend-adjusted price (used to validate ^SP500TR)

Monthly returns use the closing price on the last trading day of each month, matching CRSP month-end dates.
Run:  python benchmarks.py   -> data/benchmarks_monthly.csv
"""

import io
import zipfile
from pathlib import Path
import pandas as pd
from config import PROJECT_DIR

DATA_DIR = PROJECT_DIR / "data"
FF_ZIP = DATA_DIR / "F-F_Research_Data_Factors_CSV.zip"               # Ken French factor zip (downloaded in advance)
OUT_CSV = DATA_DIR / "benchmarks_monthly.csv"


def load_ff_rf(path=FF_ZIP):
    """Read Ken French monthly RF as decimals (e.g. 0.0025)."""
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    rows = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5 and len(parts[0]) == 6 and parts[0].isdigit():  # only "YYYYMM, Mkt-RF, SMB, HML, RF" rows
            rows.append((parts[0], float(parts[4]) / 100.0))          # store (year-month, RF as decimal)
        elif rows and "Annual" in line:                               # annual table starts
            break                                                     # end of the monthly section
    rf = pd.Series(dict(rows), name="RF")                             # year-month -> RF
    rf.index = pd.PeriodIndex(rf.index, freq="M")
    return rf


def monthly_from_daily(ticker, start="2004-11-01", end="2021-01-10"):           # month-end returns from daily prices
    """Monthly returns based on the last trading day's close of each month."""
    import yfinance as yf
    px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px = px.squeeze("columns") if isinstance(px, pd.DataFrame) else px  # single column -> Series
    month_end = px.groupby(px.index.to_period("M")).last()            # close on the last trading day of each month
    return month_end.pct_change().rename(ticker)


def build():
    """Combine RF, S&P 500 Total Return and SPY monthly returns and save them."""
    rf = load_ff_rf()
    sp = monthly_from_daily("^SP500TR")                               # S&P 500 Total Return monthly returns
    spy = monthly_from_daily("SPY")
    df = pd.concat([rf, sp.rename("SP500TR"), spy.rename("SPY")], axis=1)
    df = df.loc["2005-01":"2020-12"]
    df.index.name = "month"
    df.to_csv(OUT_CSV)
    return df


def load_benchmarks(dates):                                                     # RF and S&P 500 TR on the backtest dates
    """Return RF, SP500TR and SPY aligned to the CRSP month-end dates (DatetimeIndex)."""
    df = pd.read_csv(OUT_CSV, index_col="month")
    df.index = pd.PeriodIndex(df.index, freq="M")
    out = df.reindex(dates.to_period("M"))                            # align to the backtest year-months
    out.index = dates
    return out


if __name__ == "__main__":
    bench = build()
    print(bench.head(3).round(5).to_string())
    print(bench.tail(3).round(5).to_string())
    print("Missing:", bench.isna().sum().to_dict())
    diff = (bench["SP500TR"] - bench["SPY"])                          # difference between the two S&P 500 proxies
    print(f"SP500TR − SPY mean monthly difference {diff.mean()*1e4:.1f}bp, about {diff.mean()*12*100:.2f}%p per year, correlation {bench['SP500TR'].corr(bench['SPY']):.4f}")
    ann = lambda r: (1 + r).prod() ** (12 / len(r)) - 1
    print(f"2005-2020 annualized: SP500TR {ann(bench['SP500TR']):.2%}, SPY {ann(bench['SPY']):.2%}, RF {ann(bench['RF']):.2%}")
