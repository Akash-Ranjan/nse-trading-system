"""
Data fetcher — OHLCV data for NSE stocks.

Primary source : Yahoo Finance query2 v8 API (direct HTTP, no cookie/crumb,
                 no rate-limiting for normal usage, works on corporate networks)
Fallback source: yfinance library (curl_cffi session, verify=False for SSL bypass)

The direct API route bypasses the yfinance rate-limit issue entirely.
"""

import os
import ssl
import time
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── SSL / proxy bypass ────────────────────────────────────────────────────────
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore

# Shared requests session — SSL verification off for corporate proxy networks
_SESSION = requests.Session()
_SESSION.verify = False
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
})

logger = logging.getLogger(__name__)

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: dict[str, tuple[pd.DataFrame, float]] = {}

# Daily candles don't change intraday — cache them for 4 hours.
# Intraday candles (1h/30m/15m/5m) expire after 15 minutes so stale data
# never stacks on top of Yahoo's own 15-minute delay.
_CACHE_TTL: dict[str, int] = {
    "1d":  4 * 3600,   # daily   → 4 hours
    "60m":   15 * 60,  # 1-hour  → 15 min
    "1h":    15 * 60,
    "30m":   15 * 60,  # 30-min  → 15 min
    "15m":   10 * 60,  # 15-min  → 10 min
    "5m":     5 * 60,  # 5-min   → 5 min
}
_DEFAULT_CACHE_TTL = 900  # 15 min fallback

_PERIOD_MAP = {
    "1y": "1y", "2y": "2y", "6mo": "6mo", "3mo": "3mo",
    "1mo": "1mo", "5d": "5d", "1d": "1d",
}

_INTERVAL_MAP = {
    "1d": "1d", "1h": "60m", "30m": "30m", "15m": "15m", "5m": "5m",
    "60m": "60m",
}


def _is_stale(fetched_at: float, interval: str = "1d") -> bool:
    ttl = _CACHE_TTL.get(interval, _DEFAULT_CACHE_TTL)
    return (time.time() - fetched_at) > ttl


# ── Direct Yahoo Finance v8 fetch (primary) ───────────────────────────────────

def _fetch_direct(symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV from Yahoo Finance query2 v8 chart API.
    No cookie, no crumb, no rate-limit headache.
    """
    yf_period = _PERIOD_MAP.get(period, period)
    yf_interval = _INTERVAL_MAP.get(interval, interval)

    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": yf_interval, "range": yf_period}

    try:
        r = _SESSION.get(url, params=params, timeout=15)
        if r.status_code == 429:
            logger.warning("query2 rate-limited for %s — will retry after delay", symbol)
            return None
        if r.status_code != 200:
            return None

        j = r.json()
        result = j.get("chart", {}).get("result")
        if not result:
            return None

        result = result[0]
        timestamps = result.get("timestamp", [])
        quote = result.get("indicators", {}).get("quote", [{}])[0]

        if not timestamps or not quote:
            return None

        df = pd.DataFrame({
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Volume": quote.get("volume", []),
        }, index=pd.to_datetime(timestamps, unit="s"))

        df.index = df.index.tz_localize(None)
        df = df.dropna()

        return df if len(df) >= 20 else None

    except Exception as exc:
        logger.debug("Direct fetch failed for %s: %s", symbol, exc)
        return None


# ── yfinance fallback ─────────────────────────────────────────────────────────

def _fetch_yfinance(symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Fallback: use yfinance library with SSL-disabled curl_cffi session."""
    try:
        from curl_cffi import requests as cffi_requests
        import yfinance as yf
        session = cffi_requests.Session(verify=False)
        ticker = yf.Ticker(symbol, session=session)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.dropna()
        return df if len(df) >= 20 else None
    except Exception as exc:
        logger.debug("yfinance fallback failed for %s: %s", symbol, exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data. Uses direct API first, yfinance as fallback."""
    cache_key = f"{symbol}_{period}_{interval}"

    if not force_refresh and cache_key in _cache:
        df, fetched_at = _cache[cache_key]
        if not _is_stale(fetched_at, interval):
            return df

    df = _fetch_direct(symbol, period, interval)

    if df is None:
        df = _fetch_yfinance(symbol, period, interval)

    if df is not None:
        _cache[cache_key] = (df, time.time())

    return df


def fetch_multiple(
    symbols: list[str],
    period: str = "1y",
    interval: str = "1d",
    max_workers: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for many symbols in parallel using the direct API.
    Much faster than sequential fetching — 10 threads, no batch limit.
    """
    results: dict[str, pd.DataFrame] = {}
    to_fetch = []

    for sym in symbols:
        key = f"{sym}_{period}_{interval}"
        if key in _cache and not _is_stale(_cache[key][1], interval):
            results[sym] = _cache[key][0]
        else:
            to_fetch.append(sym)

    if not to_fetch:
        return results

    def _worker(sym: str):
        # Small random-ish delay to spread load
        time.sleep(0.1)
        return sym, fetch_ohlcv(sym, period=period, interval=interval)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, sym): sym for sym in to_fetch}
        for future in as_completed(futures):
            sym, df = future.result()
            if df is not None:
                results[sym] = df

    return results


def get_current_price(symbol: str) -> Optional[float]:
    """
    Fetch latest market price directly from Yahoo Finance meta field.
    Uses regularMarketPrice so it works regardless of candle count.
    Falls back to last close from a 1-month OHLCV fetch if meta is unavailable.
    """
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    try:
        r = _SESSION.get(url, params={"interval": "1d", "range": "5d"}, timeout=10)
        if r.status_code == 200:
            result = r.json().get("chart", {}).get("result")
            if result:
                price = result[0].get("meta", {}).get("regularMarketPrice")
                if price:
                    return float(price)
    except Exception:
        pass
    # Fallback: 1-month OHLCV (always >=20 candles)
    df = fetch_ohlcv(symbol, period="1mo")
    return float(df["Close"].iloc[-1]) if df is not None and not df.empty else None


def get_ticker_info(symbol: str) -> dict:
    """Fetch fundamental data (P/E, market cap etc.) via direct Yahoo Finance API."""
    try:
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
        params = {"modules": "summaryDetail,defaultKeyStatistics"}
        r = _SESSION.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return {}
        j = r.json()
        summary = j.get("quoteSummary", {}).get("result", [{}])[0]
        sd = summary.get("summaryDetail", {})
        ks = summary.get("defaultKeyStatistics", {})
        return {
            "pe_ratio": sd.get("trailingPE", {}).get("raw"),
            "market_cap": sd.get("marketCap", {}).get("raw"),
            "dividend_yield": sd.get("dividendYield", {}).get("raw"),
            "beta": sd.get("beta", {}).get("raw"),
            "forward_pe": ks.get("forwardPE", {}).get("raw"),
        }
    except Exception:
        return {}


def get_event_risk(symbol: str) -> dict:
    """
    Fetch upcoming corporate events that can blow up a trade:
      - Earnings date (gaps ±10% on results day)
      - Ex-dividend date (price drops by dividend amount)
      - Beta (high beta = wider stop needed, smaller position)

    Returns a dict with:
      earnings_date      : str | None  — next earnings date (YYYY-MM-DD)
      earnings_days_away : int | None  — days until earnings
      ex_div_date        : str | None  — ex-dividend date
      ex_div_days_away   : int | None
      dividend_amount    : float | None
      beta               : float | None
      score_penalty      : int         — points to subtract from score (0–30)
      warnings           : list[str]   — human-readable risk warnings
    """
    import datetime as _dt
    result = {
        "earnings_date": None,
        "earnings_days_away": None,
        "ex_div_date": None,
        "ex_div_days_away": None,
        "dividend_amount": None,
        "beta": None,
        "score_penalty": 0,
        "warnings": [],
    }

    try:
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
        params = {"modules": "calendarEvents,summaryDetail"}
        r = _SESSION.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return result

        j = r.json()
        summary = j.get("quoteSummary", {}).get("result", [{}])[0]
        cal = summary.get("calendarEvents", {})
        sd  = summary.get("summaryDetail", {})
        today = _dt.date.today()

        # ── Earnings date ─────────────────────────────────────────────────────
        earnings_dates = cal.get("earnings", {}).get("earningsDate", [])
        if earnings_dates:
            raw_ts = earnings_dates[0].get("raw")
            if raw_ts:
                edate = _dt.date.fromtimestamp(raw_ts)
                days_away = (edate - today).days
                if days_away >= 0:
                    result["earnings_date"] = str(edate)
                    result["earnings_days_away"] = days_away
                    if days_away <= 3:
                        result["score_penalty"] += 30
                        result["warnings"].append(
                            f"🚨 Earnings in {days_away} day(s) ({edate}) — VERY HIGH RISK. "
                            "Stock can gap ±10%. Avoid new entries."
                        )
                    elif days_away <= 7:
                        result["score_penalty"] += 20
                        result["warnings"].append(
                            f"⚠️ Earnings in {days_away} days ({edate}) — HIGH RISK. "
                            "Consider waiting until after results."
                        )
                    elif days_away <= 14:
                        result["score_penalty"] += 10
                        result["warnings"].append(
                            f"📅 Earnings in {days_away} days ({edate}). "
                            "Use tighter stop-loss or smaller position."
                        )

        # ── Ex-dividend date ──────────────────────────────────────────────────
        ex_div_ts = sd.get("exDividendDate", {}).get("raw")
        div_rate  = sd.get("dividendRate", {}).get("raw")
        if ex_div_ts:
            ex_date = _dt.date.fromtimestamp(ex_div_ts)
            days_away = (ex_date - today).days
            if 0 <= days_away <= 7:
                result["ex_div_date"] = str(ex_date)
                result["ex_div_days_away"] = days_away
                result["dividend_amount"] = div_rate
                result["score_penalty"] += 10
                div_str = f" (₹{div_rate:.2f} will be deducted from price)" if div_rate else ""
                result["warnings"].append(
                    f"💰 Ex-dividend date in {days_away} day(s) ({ex_date}){div_str}. "
                    "Price drops by dividend on ex-date."
                )

        # ── Beta ─────────────────────────────────────────────────────────────
        beta = sd.get("beta", {}).get("raw")
        if beta:
            result["beta"] = round(beta, 2)
            if beta > 2.0:
                result["score_penalty"] += 5
                result["warnings"].append(
                    f"⚡ High Beta ({beta:.1f}) — stock is {beta:.1f}× more volatile than NIFTY. "
                    "Use 50% of normal position size."
                )
            elif beta > 1.5:
                result["warnings"].append(
                    f"📊 Beta {beta:.1f} — moderately volatile. "
                    "Consider 75% of normal position size."
                )

    except Exception as exc:
        logger.debug("Event risk fetch failed for %s: %s", symbol, exc)

    result["score_penalty"] = min(result["score_penalty"], 30)
    return result


def get_52_week_stats(df: pd.DataFrame) -> dict:
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
    from stocks_universe import NIFTY_50, NIFTY_NEXT_50
    if symbol in NIFTY_50:
        return "Large Cap"
    if symbol in NIFTY_NEXT_50:
        return "Mid-Large Cap"
    return "Mid Cap"


def clear_cache() -> None:
    _cache.clear()
