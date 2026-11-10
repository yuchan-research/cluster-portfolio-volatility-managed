"""
Experiment settings.

All hyperparameters live here; each experiment only changes a Config object.
Places where the original papers (daily data) were adapted to monthly data are marked [monthly adj.].

  Config     settings for the cluster portfolio (steps ①–⑦) and the backtest/evaluation (⑨)
  VMConfig   settings for the volatility-managed layer (⑧)

References
  [K24] Khelifa, Allier & Cucuringu (ICAIF 2024)
  [C23] Cartea, Cucuringu & Jin (ICAIF 2023)
  [MM17] Moreira & Muir (2017, JF) "Volatility-Managed Portfolios"
"""
from __future__ import annotations  # Python 3.8/3.9 compatibility (lazy evaluation of PEP 604 annotations)


from dataclasses import dataclass, field
from pathlib import Path
import sys

for _stream in (sys.stdout, sys.stderr):                              # the scripts print Δ, σ, etc.; without this a
    if hasattr(_stream, "reconfigure"):                               # redirected run (python main.py > log) fails on Windows
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).resolve().parent.parent  # project root (parent of code/)


@dataclass
class Config:
    # ── Data ─────────────────────────────────────────────────
    data_path: Path = PROJECT_DIR / "data" / "IFE_term_stock_data.csv"  # CRSP monthly data provided for the course
    output_dir: Path = PROJECT_DIR / "output"
    exchanges: tuple = (1, 2, 3)            # assignment rule: NYSE(1), AMEX(2), NASDAQ(3) only
    price_error_threshold: float | None = None  # None: keep every stock. A number drops PERMNOs whose ADJ_PRC
                                            #   ever exceeds it. Not used: a high adjusted price is not an error,
                                            #   it is the trace of later reverse splits, so the filter would need
                                            #   future information (see data.load_panel and SETTINGS.md §1)
    reversal_threshold: float = 3.0         # spike-reversal error: monthly return > 300% and
    reversal_cum: float = 0.5               #   2-month cumulative return < 50% -> both months are errors (Ince & Porter 2006)
    nasdaq_volume_adjust: bool = True       # NASDAQ double-counted volume adjustment (Gao & Ritter 2010)
    # No minimum price filter: ADJ_PRC is retroactively adjusted for later splits/dividends, so it contains future information

    # ── Universe ─────────────────────────────────────────────
    lookback: int = 60                      # estimation window length (months) [monthly adj.: K24 uses 75 trading days]
    halflife: float | None = 18.0           # exponential half-life of observations in the window (months); None = equal weights [K24 4.6]
                                            # 60 months with half-life 18 -> ~43 effective observations, latest month weight ~4.2%
    universe_size: int = 500                # top N stocks by average VOL in the window (VOL is used only as an activity measure)
    winsor_q: float = 0.01                  # cross-sectional winsorization quantile for estimation returns (not applied to realized returns)

    # ── Clustering ───────────────────────────────────────────
    use_residual: bool = True               # True: correlation of market-factor residuals [C23 2.2.1]
    cluster_algo: str = "sponge_sym"        # "sponge_sym" [K24 3.4] or "signed_laplacian" [K24 3.2]
    tau_pos: float = 1.0                    # SPONGE regularization τ+ (SigNet default)
    tau_neg: float = 1.0                    # SPONGE regularization τ− (SigNet default)
    k_method: str = "mp"                    # "mp" (Marchenko–Pastur) | "var" (cumulative variance) | "fixed" [C23 2.3]
                                            # MP uses the effective number of observations as T (approximation under exponential weights)
    k_fixed: int = 10                       # fixed K (only when k_method="fixed")
    k_var_threshold: float = 0.9            # cumulative variance threshold P for "var" [C23 Table 1: 90%]
    k_bounds: tuple = (3, 20)               # lower/upper bound for dynamic K (with half-life 18, MP gives K of about 3–8)
    n_repeats: int = 10                     # number of clustering repeats R; final stock weights are averaged [K24 Table 2: R = 10]
    kmeans_n_init: int = 3                  # k-means++ initializations per repeat (lowest inertia kept)

    # ── Within-cluster weights β ─────────────────────────────
    intra_methods: tuple = ("gaussian", "ivp", "periphery")  # β methods compared (②③⑥)
    gauss_sigma_scale: float = 1.0          # σ² = scale × mean within-cluster distance² [monthly adj.: K24 uses cross-validation]

    # ── Across-cluster optimization α ────────────────────────
    objectives: tuple = ("gmvp", "msrp")    # across-cluster optimization (long-only) [monthly adj.: K24 uses target-return MVO, long-short]
    max_weight: float | None = 0.05         # 5% cap per stock (avoids concentration; same for all strategies)

    # ── Delisting (see data.realized_next_returns) ───────────
    delist_mode: str = "dlret"              # "dlret": actual CRSP delisting return [Shumway 1997] | "category": fixed return
                                            #   by category | "none": last price kept (0% that month)
    delist_path: Path = PROJECT_DIR / "data" / "delist_matched.csv"  # CRSP delisting records matched to the data (license: not uploaded)
    delist_cat_returns: dict = field(default_factory=lambda: {  # fixed returns used only when delist_mode="category"
        "MERGER": 0.30,
        "LIQUIDATION": -0.50,
        "PERFORMANCE": -0.50,               # assumed loss for performance-based delisting
        "EXCHANGE": 0.00,                   # move to another exchange (no value change)
        "OTHER": -0.02,                     # residual category
    })                                                                          # end of the category table
    delist_fallback: float = -0.02          # true delisting with no usable record in delist_path

    # ── Backtest ─────────────────────────────────────────────
    cost_bps_list: tuple = (0.0, 1.0, 10.0, 20.0) # one-way costs (bp) reported together: net return = gross return − bp × turnover
    cost_bps_main: float = 20.0             # main cost level (bp) used for figures and for selecting the main strategy
    run_benchmarks: bool = True             # whether to compute equal-weight and full-universe GMVP/MSRP (Ledoit–Wolf) benchmarks
    seed: int = 42                          # random seed (k-means reproducibility)
    split_dates: tuple = ()                 # no sub-period split: clustering is unsupervised, so the whole
                                            #   2005-2020 sample is used and the main strategy is chosen on it


@dataclass
class VMConfig:
    """Volatility-managed layer settings [MM17]. Defaults were fixed before looking at results."""
    cap: float = 1.5            # leverage cap [MM17]: borrow up to 50% of equity (Reg T's 50% initial margin allows 2x)
    floor: float = 0.0          # leverage floor (no short position)
    burn_in: int = 24           # minimum months needed to estimate c_t (exposure is 1 before that)
    c_mode: str = "realtime"    # "realtime": c re-estimated monthly from past data | "full": fixed over the full sample [MM17 original, uses hindsight]
    signal: str = "rv"          # "rv": daily sum-of-squares variance [MM17] | "var": sample variance | "sd": standard deviation | "rv3": 3-month average
