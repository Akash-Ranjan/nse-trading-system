"""
Data fetcher — downloads OHLCV price data from Yahoo Finance (NSE stocks).
Caches results in memory for the session to avoid repeated API calls.
"""

import os
import ssl
import time
import logging
from typing import Optional

import pandas as pd
import yfinance as yf
from curl_cffi import requests as cffi_requests

# ── SSL bypass for corporate/ISP proxy environments ──────────────────────────
# Many Indian corporate networks and ISPs use SSL inspection proxies that
# inject self-signed certificates. We disable strict verification so data
# fetching works on those networks.
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore

# Shared curl_cffi session with SSL verification off — passed to all yfinance calls
_YF_SESSION = cffi_requests.Session(verify=False)

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
        ticker = yf.Ticker(symbol, session=_YF_SESSION)
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
    batch_size: int = 50,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for a list of symbols using yf.download() batch API.
    Batching reduces requests dramatically (1 request per 50 stocks vs 50 individual).
    Returns a dict of symbol -> DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        cache_hits = []
        to_fetch = []

        for sym in batch:
            key = f"{sym}_{period}_{interval}"
            if key in _cache and not _is_stale(_cache[key][1]):
                results[sym] = _cache[key][0]
                cache_hits.append(sym)
            else:
                to_fetch.append(sym)

        if not to_fetch:
            continue

        try:
            raw = yf.download(
                to_fetch,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                session=_YF_SESSION,
                group_by="ticker",
            )

            if raw is None or raw.empty:
                for sym in to_fetch:
                    df_single = fetch_ohlcv(sym, period=period, interval=interval)
                    if df_single is not None:
                        results[sym] = df_single
                continue

            for sym in to_fetch:
                try:
                    if len(to_fetch) == 1:
                        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                    else:
                        df = raw[sym][["Open", "High", "Low", "Close", "Volume"]].copy()

                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df = df.dropna()

                    if len(df) >= 50:
                        results[sym] = df
                        key = f"{sym}_{period}_{interval}"
                        _cache[key] = (df, time.time())
                except Exception:
                    pass

        except Exception as exc:
            logger.warning("Batch download failed (%s), falling back to individual: %s", batch, exc)
            for sym in to_fetch:
                df = fetch_ohlcv(sym, period=period, interval=interval)
                if df is not None:
                    results[sym] = df

        if i + batch_size < len(symbols):
            time.sleep(0.5)

    return results


def get_current_price(symbol: str) -> Optional[float]:
    """Return the last closing price for a symbol."""
    df = fetch_ohlcv(symbol, period="5d")
    if df is not None and not df.empty:
        return float(df["Close"].iloc[-1])
    return None


def get_ticker_info(symbol: str) -> dict:
    """Fetch fundamental info dict from yfinance (P/E, market cap, etc.)."""
    try:
        info = yf.Ticker(symbol, session=_YF_SESSION).info
        return {
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
            "market_cap": info.get("marketCap"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "name": info.get("longName", symbol),
            "avg_volume": info.get("averageVolume"),
        }
    except Exception:
        return {}


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
