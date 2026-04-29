"""
Data fetcher — downloads OHLCV price data from Yahoo Finance (NSE stocks).
Caches results in memory for the session to avoid repeated API calls.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# In-memory cache: symbol -> (DataFrame, fetched_at_timestamp)
_cache: dict[str, tuple[pd.DataFrame, float]] = {}
CACHE_TTL_SECONDS = 900  # 15 minutes


def _is_stale(fetched_at: float) -> bool:
    return (time.time() - fetched_at) > CACHE_TTL_SECONDS


def fetch_ohlcv(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Download daily OHLCV data for a single NSE symbol.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    Returns None if the download fails or the data is empty.
    """
    cache_key = f"{symbol}_{period}_{interval}"

    if not force_refresh and cache_key in _cache:
        df, fetched_at = _cache[cache_key]
        if not _is_stale(fetched_at):
            return df

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)

        if df is None or df.empty:
            logger.warning("No data returned for %s", symbol)
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.dropna()

        if len(df) < 50:
            logger.warning("Insufficient data for %s (%d rows)", symbol, len(df))
            return None

        _cache[cache_key] = (df, time.time())
        return df

    except Exception as exc:
        logger.error("Failed to fetch %s: %s", symbol, exc)
        return None


def fetch_multiple(
    symbols: list[str],
    period: str = "1y",
    interval: str = "1d",
    max_retries: int = 2,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for a list of symbols. Returns a dict of symbol -> DataFrame.
    Skips symbols that fail to download.
    """
    results: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        for attempt in range(max_retries):
            df = fetch_ohlcv(symbol, period=period, interval=interval)
            if df is not None:
                results[symbol] = df
                break
            if attempt < max_retries - 1:
                time.sleep(0.3)

    return results


def get_current_price(symbol: str) -> Optional[float]:
    """Return the last closing price for a symbol."""
    df = fetch_ohlcv(symbol, period="5d")
    if df is not None and not df.empty:
        return float(df["Close"].iloc[-1])
    return None


def get_52_week_stats(df: pd.DataFrame) -> dict:
    """Return 52-week high, low, and current price position within that range."""
    high_52w = float(df["High"].max())
    low_52w = float(df["Low"].min())
    current = float(df["Close"].iloc[-1])
    position = (current - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5
    return {
        "high_52w": high_52w,
        "low_52w": low_52w,
        "current": current,
        "position_pct": round(position * 100, 1),
    }


def get_market_cap_tier(symbol: str) -> str:
    """Rough tier classification based on NIFTY index membership."""
    from stocks_universe import NIFTY_50, NIFTY_NEXT_50
    if symbol in NIFTY_50:
        return "Large Cap"
    if symbol in NIFTY_NEXT_50:
        return "Mid-Large Cap"
    return "Mid Cap"
