"""
Stock screener — runs multiple strategies across the universe and ranks results.

Strategies:
  1. Golden Cross Trend  — EMA50 > EMA200, price above EMA20, MACD bullish
  2. MACD Momentum       — Fresh MACD bullish crossover with RSI in healthy zone
  3. Volume Breakout     — Price > 20-day high with vol > 1.4x average
  4. Oversold Bounce     — RSI < 38 with MACD turning bullish
  5. Composite Top Picks — Highest composite score across all signals
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

from data_fetcher import fetch_ohlcv, get_52_week_stats
from analyzer import analyze
from stocks_universe import get_sector, get_display_name
from data_fetcher import get_market_cap_tier

logger = logging.getLogger(__name__)

MAX_WORKERS = 8


# ── Single stock screener ──────────────────────────────────────────────────────

def screen_stock(symbol: str) -> Optional[dict]:
    """Fetch data + analyze for one symbol. Returns None on failure."""
    df = fetch_ohlcv(symbol, period="1y")
    if df is None:
        return None

    try:
        result = analyze(df)
        stats_52w = get_52_week_stats(df)

        return {
            "symbol": symbol,
            "name": get_display_name(symbol),
            "sector": get_sector(symbol),
            "cap_tier": get_market_cap_tier(symbol),
            **result,
            **stats_52w,
            # Strategies triggered (for filtering)
            "strategy_golden_cross": (
                result["golden_cross"]
                and result["price_above_ema200"]
                and result["macd_bullish"]
                and result["adx"] > 20
            ),
            "strategy_macd_momentum": (
                result["macd_crossover"]
                and 40 <= result["rsi"] <= 68
                and not result["rsi_overbought"]
            ),
            "strategy_breakout": result["breakout"],
            "strategy_oversold_bounce": (
                result["rsi_oversold"]
                and result["macd_bullish"]
                and result["vol_ratio"] > 1.0
            ),
        }
    except Exception as exc:
        logger.error("Analysis failed for %s: %s", symbol, exc)
        return None


# ── Batch screener ────────────────────────────────────────────────────────────

def run_screener(
    symbols: list[str],
    progress_callback=None,
) -> pd.DataFrame:
    """
    Screen all symbols in parallel.
    progress_callback(done, total) is called after each stock finishes.
    Returns a DataFrame sorted by composite score (descending).
    """
    results = []
    total = len(symbols)
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(screen_stock, sym): sym for sym in symbols}

        for future in as_completed(future_map):
            done += 1
            if progress_callback:
                progress_callback(done, total)
            result = future.result()
            if result is not None:
                results.append(result)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


# ── Strategy filters ──────────────────────────────────────────────────────────

def filter_by_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Filter a screener results DataFrame by strategy name."""
    strategy_col = {
        "Golden Cross Trend": "strategy_golden_cross",
        "MACD Momentum": "strategy_macd_momentum",
        "Volume Breakout": "strategy_breakout",
        "Oversold Bounce": "strategy_oversold_bounce",
    }.get(strategy)

    if strategy_col and strategy_col in df.columns:
        return df[df[strategy_col]].copy()
    return df


def get_top_buys(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top-N buy/strong-buy stocks by composite score."""
    buy_df = df[df["signal"].isin(["BUY", "STRONG BUY"])].copy()
    return buy_df.head(n)


def get_sector_breakdown(df: pd.DataFrame) -> dict[str, int]:
    """Count buy-signal stocks per sector."""
    buy_df = df[df["signal"].isin(["BUY", "STRONG BUY"])]
    return buy_df.groupby("sector").size().sort_values(ascending=False).to_dict()


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Quick statistics from a screener run."""
    if df.empty:
        return {}
    return {
        "total_screened": len(df),
        "strong_buy": int((df["signal"] == "STRONG BUY").sum()),
        "buy": int((df["signal"] == "BUY").sum()),
        "watch": int((df["signal"] == "WATCH").sum()),
        "neutral_avoid": int(df["signal"].isin(["NEUTRAL", "AVOID"]).sum()),
        "avg_score": round(float(df["score"].mean()), 1),
        "golden_cross_count": int(df.get("strategy_golden_cross", pd.Series(False)).sum()),
        "breakout_count": int(df.get("strategy_breakout", pd.Series(False)).sum()),
        "macd_momentum_count": int(df.get("strategy_macd_momentum", pd.Series(False)).sum()),
        "oversold_bounce_count": int(df.get("strategy_oversold_bounce", pd.Series(False)).sum()),
    }
