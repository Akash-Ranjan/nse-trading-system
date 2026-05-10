"""
Stock screener — runs multiple strategies across the universe and ranks results.

Strategies:
  1. Golden Cross Trend  — EMA50 > EMA200, price above EMA200, MACD bullish
  2. MACD Momentum       — Fresh MACD bullish crossover with RSI in healthy zone
  3. Volume Breakout     — Price > 20-day high with vol > 1.4x average
  4. Oversold Bounce     — RSI < 35 with MACD turning bullish
  5. BB Squeeze Breakout — Bollinger Band squeeze resolving upward with volume
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

from data_fetcher import fetch_ohlcv, fetch_multiple, get_52_week_stats, get_market_cap_tier
from analyzer import analyze, compute_vwap
from stocks_universe import get_sector, get_display_name

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
                and result["vol_ratio"] > 1.2
            ),
            # BB Squeeze: bands were tight (consolidation), now price breaks above
            # EMA20 with MACD bullish and above-average volume. The squeeze itself
            # is already computed in the analyzer as bb_squeeze.
            "strategy_bb_squeeze": (
                result["bb_squeeze"]
                and result["price_above_ema200"]
                and result["macd_bullish"]
                and result["vol_ratio"] > 1.1
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
    Screen all symbols.
    Phase 1: Batch-fetch all OHLCV data (far fewer Yahoo Finance requests).
    Phase 2: Analyze each stock in parallel threads (CPU-bound, no network).
    progress_callback(done, total) is called as each stock finishes analysis.
    Returns a DataFrame sorted by composite score (descending).
    """
    total = len(symbols)

    # Phase 1 — batch download (1–2 Yahoo Finance requests instead of 120+)
    if progress_callback:
        progress_callback(0, total)
    price_data = fetch_multiple(symbols, period="1y", interval="1d")
    if progress_callback:
        progress_callback(min(total // 3, total), total)

    # Phase 2 — parallel analysis (pure computation, no network)
    results = []
    done = 0

    def _analyse(sym: str):
        df = price_data.get(sym)
        if df is None:
            return None
        try:
            result = analyze(df)
            stats = get_52_week_stats(df)
            return {
                "symbol": sym,
                "name": get_display_name(sym),
                "sector": get_sector(sym),
                "cap_tier": get_market_cap_tier(sym),
                **result,
                **stats,
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
                    and result["vol_ratio"] > 1.2
                ),
                "strategy_bb_squeeze": (
                    result["bb_squeeze"]
                    and result["price_above_ema200"]
                    and result["macd_bullish"]
                    and result["vol_ratio"] > 1.1
                ),
            }
        except Exception as exc:
            logger.error("Analysis failed for %s: %s", sym, exc)
            return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(_analyse, sym): sym for sym in symbols}
        for future in as_completed(future_map):
            done += 1
            if progress_callback:
                progress_callback(min(total // 3 + int(done * 2 / 3), total), total)
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
        "BB Squeeze Breakout": "strategy_bb_squeeze",
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
        "bb_squeeze_count": int(df.get("strategy_bb_squeeze", pd.Series(False)).sum()),
    }


# ── Intraday Screener ──────────────────────────────────────────────────────────

def run_intraday_screener(
    symbols: list,
    timeframe: str = "1h",
    progress_callback=None,
) -> pd.DataFrame:
    """
    Screen stocks for intraday trading setups on 1h / 30m / 15m timeframes.

    A stock qualifies as an intraday BUY when ALL of:
      1. Price > VWAP            — bullish intraday bias (most important filter)
      2. RSI 45–68               — healthy momentum, not overbought
      3. MACD bullish            — trend direction confirmed
      4. Vol ratio  > 1.2        — above-average volume (institutions active)
      5. ADX > 20                — stock is trending, not ranging sideways
      6. Price > EMA 20          — short-term trend intact

    Upgrades to STRONG BUY when also:
      - Fresh MACD histogram crossover (momentum just turned)
      - Vol ratio > 1.5          — strong volume surge
      - RSI 50–65                — ideal momentum zone

    Data limits (Yahoo Finance):
      1h  → last 60 days of 1-hour candles
      30m → last 60 days of 30-min candles
      15m → last 10 days of 15-min candles  ← use for same-day only
    """
    _PERIOD_MAP = {"1h": "60d", "30m": "60d", "15m": "10d"}
    period = _PERIOD_MAP.get(timeframe, "60d")
    total  = len(symbols)

    # Phase 1: batch-fetch intraday OHLCV for all symbols
    if progress_callback:
        progress_callback(0, total)
    price_data = fetch_multiple(symbols, period=period, interval=timeframe)
    if progress_callback:
        progress_callback(total // 3, total)

    # Phase 2: parallel analysis
    results = []
    done    = 0

    def _analyse(sym: str):
        df = price_data.get(sym)
        if df is None or len(df) < 30:
            return None
        try:
            result = analyze(df)
            price  = result["price"]

            # VWAP — daily-anchored (resets each calendar day)
            vwap_s    = compute_vwap(df["High"], df["Low"], df["Close"], df["Volume"])
            vwap_last = vwap_s.dropna()
            vwap_price = float(vwap_last.iloc[-1]) if not vwap_last.empty else None
            above_vwap = bool(vwap_price and price > vwap_price)
            vwap_gap_pct = round((price - vwap_price) / vwap_price * 100, 2) if vwap_price else None

            # ── Intraday filter criteria ───────────────────────────────────────
            qualifies = (
                above_vwap
                and result["macd_bullish"]
                and 45 <= result["rsi"] <= 68
                and result["vol_ratio"] > 1.2
                and result["adx"] > 20
                and price > result["ema20"]
            )
            if not qualifies:
                return None

            # Upgrade to STRONG BUY on confluence
            strong = (
                result["macd_crossover"]
                and result["vol_ratio"] > 1.5
                and 50 <= result["rsi"] <= 65
            )

            # Intraday trade setup (tighter than swing: 1.0× ATR stop)
            atr = result["atr"]
            sl   = round(price - atr * 1.0, 2)         # 1× ATR stop
            risk = price - sl
            t1   = round(price + risk * 1.5, 2)        # 1:1.5 R:R
            t2   = round(price + risk * 2.0, 2)        # 1:2   R:R

            return {
                "symbol":       sym,
                "name":         get_display_name(sym),
                "sector":       get_sector(sym),
                "price":        price,
                "entry":        price,
                "stop_loss":    sl,
                "target_1":     t1,
                "target_2":     t2,
                "vwap":         round(vwap_price, 2) if vwap_price else None,
                "vwap_gap":     vwap_gap_pct,
                "signal":       "STRONG BUY" if strong else "BUY",
                "score":        result["score"],
                "rsi":          result["rsi"],
                "adx":          round(result["adx"], 1),
                "vol_ratio":    result["vol_ratio"],
                "macd_bullish": result["macd_bullish"],
                "macd_cross":   result["macd_crossover"],
                "above_vwap":   above_vwap,
                "ema20":        result["ema20"],
                "atr":          atr,
            }
        except Exception as exc:
            logger.debug("Intraday analysis failed for %s: %s", sym, exc)
            return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(_analyse, sym): sym for sym in symbols}
        for future in as_completed(future_map):
            done += 1
            if progress_callback:
                progress_callback(min(total // 3 + int(done * 2 / 3), total), total)
            r = future.result()
            if r is not None:
                results.append(r)

    if not results:
        return pd.DataFrame()

    df_out = pd.DataFrame(results)
    # Sort: STRONG BUY first, then by score descending
    df_out["_rank"] = df_out["signal"].map({"STRONG BUY": 0, "BUY": 1}).fillna(2)
    df_out = df_out.sort_values(["_rank", "score"], ascending=[True, False]).drop(columns="_rank")
    return df_out.reset_index(drop=True)
