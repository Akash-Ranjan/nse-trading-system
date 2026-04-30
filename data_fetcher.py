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


def _fetch_news_headlines(symbol: str, count: int = 8) -> list[str]:
    """
    Fetch recent news headlines for a symbol from Yahoo Finance search API.
    Returns a list of lowercase headline strings (empty list on failure).
    """
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": symbol, "newsCount": count, "quotesCount": 0}
        r = _SESSION.get(url, params=params, timeout=10)
        if r.status_code == 200:
            items = r.json().get("news", [])
            return [item.get("title", "").lower() for item in items if item.get("title")]
    except Exception:
        pass
    return []


# Keyword groups for news-based risk scoring.
# Each entry: (penalty, label, list-of-trigger-words)
_NEWS_RISK_RULES: list[tuple[int, str, list[str]]] = [
    # ── Catastrophic events ──────────────────────────────────────────────────
    (30, "🚨 Fraud/Legal",      ["fraud", "scam", "embezzlement", "money laundering",
                                  "cbi raid", "ed raid", "enforcement directorate",
                                  "sebi ban", "nse ban", "bse ban", "delisting",
                                  "securities fraud", "insider trading", "ponzi"]),
    (30, "🚨 Insolvency",       ["bankruptcy", "insolvency", "nclt", "liquidation",
                                  "default", "npa", "debt restructuring", "moratorium"]),
    # ── Senior management departure ─────────────────────────────────────────
    (25, "⚠️ CEO/MD Exit",      ["ceo resign", "ceo quit", "ceo steps down", "ceo fired",
                                  "md resign", "md quit", "managing director resign",
                                  "cfo resign", "cfo quit", "cfo steps down",
                                  "chairman resign", "promoter resign",
                                  "key executive", "top executive resign"]),
    # ── Regulatory & legal ───────────────────────────────────────────────────
    (20, "⚠️ Regulatory",       ["sebi notice", "sebi order", "sebi investigation",
                                  "rbi penalty", "rbi action", "income tax raid",
                                  "it raid", "customs raid", "gst evasion",
                                  "show cause notice", "adjudication order",
                                  "market regulator", "penalty imposed"]),
    (20, "⚠️ Whistleblower",    ["whistleblower", "whistle blower", "audit concern",
                                  "accounting irregularity", "financial irregularity",
                                  "restatement", "auditor resign", "auditor quit"]),
    # ── Promoter / ownership risk ────────────────────────────────────────────
    (15, "📉 Promoter Risk",    ["promoter pledge", "promoter sell", "promoter stake",
                                  "promoter offload", "bulk deal sell", "block deal sell",
                                  "fii selling", "institutional selling"]),
    # ── Analyst downgrades ───────────────────────────────────────────────────
    (10, "📊 Downgrade",        ["downgrade", "target cut", "sell rating", "underperform",
                                  "underweight", "reduce rating", "rating downgrade"]),
    # ── Positive news (score boost) ──────────────────────────────────────────
    (-5, "✅ Buyback/M&A",      ["buyback", "share repurchase", "merger", "acquisition",
                                  "takeover bid", "open offer", "strategic stake"]),
]


def get_event_risk(symbol: str) -> dict:
    """
    Checks three risk categories for a symbol:
      1. Scheduled corporate events  — earnings, ex-dividend, beta
      2. News-based risks            — CEO resignation, fraud, regulatory, downgrade
      3. Positive news               — buyback, M&A (reduces penalty)

    Returns a dict with:
      earnings_date      : str | None
      earnings_days_away : int | None
      ex_div_date        : str | None
      ex_div_days_away   : int | None
      dividend_amount    : float | None
      beta               : float | None
      news_headlines     : list[str]   — raw headlines fetched (for display)
      news_flags         : list[str]   — triggered labels e.g. '⚠️ CEO/MD Exit'
      score_penalty      : int         — net points to subtract (capped at 30)
      warnings           : list[str]   — human-readable risk messages
    """
    import datetime as _dt
    result: dict = {
        "earnings_date": None,
        "earnings_days_away": None,
        "ex_div_date": None,
        "ex_div_days_away": None,
        "dividend_amount": None,
        "beta": None,
        "news_headlines": [],
        "news_flags": [],
        "score_penalty": 0,
        "warnings": [],
    }

    # ── 1. Structured corporate events (earnings / ex-div / beta) ─────────────
    try:
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
        params = {"modules": "calendarEvents,summaryDetail"}
        r = _SESSION.get(url, params=params, timeout=15)
        if r.status_code == 200:
            j = r.json()
            summary = j.get("quoteSummary", {}).get("result", [{}])[0]
            cal = summary.get("calendarEvents", {})
            sd  = summary.get("summaryDetail", {})
            today = _dt.date.today()

            # Earnings date
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

            # Ex-dividend date
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
                        f"💰 Ex-dividend in {days_away} day(s) ({ex_date}){div_str}. "
                        "Price drops by dividend on ex-date."
                    )

            # Beta
            beta = sd.get("beta", {}).get("raw")
            if beta:
                result["beta"] = round(beta, 2)
                if beta > 2.0:
                    result["score_penalty"] += 5
                    result["warnings"].append(
                        f"⚡ High Beta ({beta:.1f}) — {beta:.1f}× more volatile than NIFTY. "
                        "Use 50% of normal position size."
                    )
                elif beta > 1.5:
                    result["warnings"].append(
                        f"📊 Beta {beta:.1f} — moderately volatile. "
                        "Consider 75% of normal position size."
                    )

    except Exception as exc:
        logger.debug("Structured event fetch failed for %s: %s", symbol, exc)

    # ── 2. News-based risk scan ───────────────────────────────────────────────
    headlines = _fetch_news_headlines(symbol)
    result["news_headlines"] = headlines

    if headlines:
        combined = " | ".join(headlines)
        news_penalty = 0
        for penalty, label, keywords in _NEWS_RISK_RULES:
            if any(kw in combined for kw in keywords):
                result["news_flags"].append(label)
                # Positive rules use negative penalty (boost), others add penalty
                news_penalty += penalty
                # Build a warning with the matching headline as evidence
                matching = next(
                    (h for h in headlines if any(kw in h for kw in keywords)), None
                )
                if penalty > 0:
                    result["warnings"].append(
                        f"{label} detected in recent news"
                        + (f': "{matching.title()}"' if matching else ".")
                    )

        # Cap news penalty at 30; allow positive news to reduce by up to 5
        news_penalty = max(-5, min(news_penalty, 30))
        result["score_penalty"] += news_penalty

    result["score_penalty"] = max(0, min(result["score_penalty"], 30))
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


# ── India VIX ─────────────────────────────────────────────────────────────────

def get_india_vix() -> Optional[float]:
    """
    Fetch the current India VIX from Yahoo Finance (symbol ^INDIAVIX).
    India VIX measures expected 30-day volatility of Nifty options.

    Interpretation:
      < 13  : Complacency — market very calm, low fear
      13–17 : Normal range — healthy market
      17–20 : Elevated — be cautious, tighten stops
      > 20  : High fear — avoid new positions, use 50% size
      > 25  : Extreme fear — stay in cash or hedge only
    """
    try:
        url = "https://query2.finance.yahoo.com/v8/finance/chart/%5EINDIAVIX"
        r = _SESSION.get(url, params={"interval": "1d", "range": "5d"}, timeout=10)
        if r.status_code == 200:
            res = r.json().get("chart", {}).get("result")
            if res:
                vix = res[0].get("meta", {}).get("regularMarketPrice")
                if vix:
                    return round(float(vix), 2)
    except Exception as exc:
        logger.debug("India VIX fetch failed: %s", exc)
    return None


def get_vix_signal(vix: float) -> tuple[str, str]:
    """
    Returns (label, guidance) for a given VIX level.
    """
    if vix < 13:
        return "😴 Very Low", "Market complacent. Signals are reliable but watch for sudden reversals."
    if vix < 17:
        return "✅ Normal", "Healthy volatility. All strategies can be used with full position size."
    if vix < 20:
        return "⚠️ Elevated", "Use 75% position size. Widen stops by 0.5× ATR."
    if vix < 25:
        return "🔴 High", "Use 50% position size. Only enter strongest signals (score > 75). Tight stops."
    return "🚨 Extreme", "Stay in cash or hedge only. Do NOT open new positions."


# ── F&O Expiry Awareness ──────────────────────────────────────────────────────

def get_fo_expiry_info() -> dict:
    """
    Calculate days to next weekly and monthly NSE F&O expiry.

    NSE F&O expiry schedule:
      Weekly  — every Thursday (Nifty, BankNifty, FinNifty, MidcapNifty)
      Monthly — last Thursday of each month (all stock futures)

    Near expiry, F&O stocks exhibit:
      - Max pain pinning: price is dragged toward the strike with maximum open interest
      - Short gamma squeezes: sharp spikes and reversals in the last hour
      - Intraday strategies become unreliable in the last 60 minutes of expiry day
    """
    import datetime as _dt
    today = _dt.date.today()

    # Next weekly expiry (Thursday = weekday 3)
    days_to_thu = (3 - today.weekday()) % 7
    next_weekly = today + _dt.timedelta(days=days_to_thu)

    # Last Thursday of current month (monthly expiry)
    if today.month == 12:
        first_next = _dt.date(today.year + 1, 1, 1)
    else:
        first_next = _dt.date(today.year, today.month + 1, 1)
    last_day = first_next - _dt.timedelta(days=1)
    while last_day.weekday() != 3:
        last_day -= _dt.timedelta(days=1)
    monthly_expiry = last_day

    weekly_days  = (next_weekly    - today).days
    monthly_days = (monthly_expiry - today).days
    is_monthly   = (next_weekly == monthly_expiry)

    return {
        "next_weekly_expiry":   str(next_weekly),
        "weekly_days_away":     weekly_days,
        "next_monthly_expiry":  str(monthly_expiry),
        "monthly_days_away":    monthly_days,
        "is_expiry_day":        weekly_days == 0,
        "is_expiry_week":       weekly_days <= 2,
        "is_monthly_expiry":    is_monthly,
        "is_monthly_week":      is_monthly and weekly_days <= 2,
    }


# ── NSE session + Delivery % ──────────────────────────────────────────────────

# ── NSE Bhavcopy delivery cache ───────────────────────────────────────────────
# NSE publishes a full-market CSV (bhavcopy) every trading day at ~6 PM.
# It contains DELIV_PER (delivery %) for every equity. We download it once
# per day and cache it in memory — no session/cookie required.

_BHAV_CACHE: dict = {"data": None, "date": None}   # {symbol -> delivery_pct}


def _load_bhavcopy(trade_date=None) -> dict:
    """
    Download NSE equity bhavcopy for trade_date (defaults to today / last trading day).
    Returns dict {SYMBOL: delivery_pct_float}. Returns {} on failure.

    URL format:
      https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv
    Columns include: SYMBOL, SERIES, DELIV_QTY, DELIV_PER
    Only EQ series rows are used.
    """
    import datetime as _dt
    import io

    if trade_date is None:
        trade_date = _dt.date.today()

    # Walk back up to 5 days to find the last trading day with a published file
    for delta in range(6):
        d = trade_date - _dt.timedelta(days=delta)
        if d.weekday() >= 5:        # skip Saturday / Sunday
            continue
        date_str = d.strftime("%d%m%Y")
        url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
        try:
            r = _SESSION.get(url, timeout=20)
            if r.status_code != 200:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]
            # Keep EQ series only, drop rows with missing delivery
            if "SERIES" in df.columns:
                df = df[df["SERIES"].str.strip() == "EQ"]
            if "DELIV_PER" not in df.columns or "SYMBOL" not in df.columns:
                continue
            df["SYMBOL"] = df["SYMBOL"].str.strip()
            df["DELIV_PER"] = pd.to_numeric(df["DELIV_PER"], errors="coerce")
            result = dict(zip(df["SYMBOL"], df["DELIV_PER"]))
            logger.info("Bhavcopy loaded: %s (%d symbols)", d, len(result))
            return result
        except Exception as exc:
            logger.debug("Bhavcopy fetch failed for %s: %s", date_str, exc)

    return {}


def _get_bhavcopy() -> dict:
    """Return in-memory bhavcopy, refreshing once per calendar day."""
    import datetime as _dt
    today = str(_dt.date.today())
    if _BHAV_CACHE["date"] != today or not _BHAV_CACHE["data"]:
        _BHAV_CACHE["data"] = _load_bhavcopy()
        _BHAV_CACHE["date"] = today
    return _BHAV_CACHE["data"] or {}


def get_delivery_pct(symbol: str) -> Optional[float]:
    """
    Return delivery-to-traded % for a symbol from NSE bhavcopy (published daily).

    High delivery % (> 50%) = institutional/retail conviction — supports the signal.
    Low delivery  % (< 25%) = mostly intraday/speculative — signal less trustworthy.

    Uses bhavcopy CSV (no session/cookie required) instead of the live NSE API.
    symbol can be with or without .NS suffix.
    """
    nse_sym = symbol.replace(".NS", "").replace(".BO", "").upper()
    bhav = _get_bhavcopy()
    val  = bhav.get(nse_sym)
    if val is not None and not pd.isna(val):
        return round(float(val), 1)
    return None


def get_delivery_label(pct: Optional[float]) -> str:
    """Return a short label + interpretation for a delivery percentage."""
    if pct is None:
        return "N/A"
    if pct >= 60:
        return f"{pct:.0f}% 💪"
    if pct >= 45:
        return f"{pct:.0f}% ✅"
    if pct >= 25:
        return f"{pct:.0f}% ⚠️"
    return f"{pct:.0f}% ❌"
