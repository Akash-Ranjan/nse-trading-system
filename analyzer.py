"""
Technical analysis engine.

Computes: RSI, MACD, EMA (20/50/200), Bollinger Bands, ATR, ADX, Stochastic,
volume trends, and support/resistance levels from raw OHLCV DataFrames.
"""

import numpy as np
import pandas as pd


# ── Core Indicators ──────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period).mean()


def compute_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper_band, middle_band, lower_band)."""
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    return mid + num_std * std, mid, mid - num_std * std


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Returns ADX series (trend strength, >25 = trending)."""
    up_move = high.diff()
    down_move = -low.diff()

    pos_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=close.index)
    neg_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=close.index)

    atr = compute_atr(high, low, close, period)
    smooth_pos = pos_dm.ewm(com=period - 1, min_periods=period).mean()
    smooth_neg = neg_dm.ewm(com=period - 1, min_periods=period).mean()

    di_pos = 100 * smooth_pos / atr.replace(0, np.nan)
    di_neg = 100 * smooth_neg / atr.replace(0, np.nan)
    dx = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0, np.nan)
    return dx.ewm(com=period - 1, min_periods=period).mean()


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Returns (%K, %D)."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def compute_volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume.rolling(window=period).mean()


# ── Full Analysis Bundle ──────────────────────────────────────────────────────

def analyze(df: pd.DataFrame) -> dict:
    """
    Run all indicators on an OHLCV DataFrame and return a summary dict
    with latest values, signals, and a composite score (0–100).
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ── Indicators ──
    rsi = compute_rsi(close)
    macd_line, signal_line, macd_hist = compute_macd(close)
    ema20 = compute_ema(close, 20)
    ema50 = compute_ema(close, 50)
    ema200 = compute_ema(close, 200)
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close)
    atr = compute_atr(high, low, close)
    adx = compute_adx(high, low, close)
    stoch_k, stoch_d = compute_stochastic(high, low, close)
    vol_ma = compute_volume_ma(volume)

    # ── Latest values ──
    price = float(close.iloc[-1])
    rsi_val = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
    macd_val = float(macd_line.iloc[-1])
    macd_sig = float(signal_line.iloc[-1])
    macd_hist_val = float(macd_hist.iloc[-1])
    macd_hist_prev = float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else 0.0
    adx_val = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0
    stoch_k_val = float(stoch_k.iloc[-1]) if not np.isnan(stoch_k.iloc[-1]) else 50.0
    atr_val = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else price * 0.02

    ema20_val = float(ema20.iloc[-1])
    ema50_val = float(ema50.iloc[-1])
    ema200_val = float(ema200.iloc[-1])
    bb_upper_val = float(bb_upper.iloc[-1])
    bb_lower_val = float(bb_lower.iloc[-1])
    bb_mid_val = float(bb_mid.iloc[-1])

    vol_current = float(volume.iloc[-1])
    vol_avg = float(vol_ma.iloc[-1]) if not np.isnan(vol_ma.iloc[-1]) else vol_current
    vol_ratio = vol_current / vol_avg if vol_avg > 0 else 1.0

    # ── Momentum returns ──
    ret_1w = _pct_change(close, 5)
    ret_1m = _pct_change(close, 21)
    ret_3m = _pct_change(close, 63)
    ret_6m = _pct_change(close, 126)

    # ── Signals ──
    golden_cross = ema50_val > ema200_val
    price_above_ema20 = price > ema20_val
    price_above_ema50 = price > ema50_val
    price_above_ema200 = price > ema200_val

    macd_bullish = macd_val > macd_sig
    macd_crossover = macd_hist_val > 0 and macd_hist_prev <= 0  # fresh crossover
    macd_momentum = macd_hist_val > macd_hist_prev  # histogram expanding

    rsi_healthy = 40 <= rsi_val <= 65
    rsi_oversold = rsi_val < 35
    rsi_overbought = rsi_val > 75

    strong_trend = adx_val > 25

    # 20-day breakout
    high_20d = float(high.rolling(20).max().iloc[-2]) if len(high) > 20 else float(high.max())
    breakout = price > high_20d and vol_ratio > 1.4

    bb_position = (price - bb_lower_val) / (bb_upper_val - bb_lower_val) if (bb_upper_val - bb_lower_val) > 0 else 0.5
    bb_squeeze = (bb_upper_val - bb_lower_val) / bb_mid_val < 0.06 if bb_mid_val > 0 else False

    # ── Composite Score (0–100) ──
    score = _composite_score(
        rsi_val=rsi_val,
        macd_bullish=macd_bullish,
        macd_crossover=macd_crossover,
        macd_momentum=macd_momentum,
        price_above_ema20=price_above_ema20,
        price_above_ema50=price_above_ema50,
        price_above_ema200=price_above_ema200,
        golden_cross=golden_cross,
        strong_trend=strong_trend,
        breakout=breakout,
        vol_ratio=vol_ratio,
        ret_1m=ret_1m,
        ret_3m=ret_3m,
        bb_position=bb_position,
        stoch_k_val=stoch_k_val,
    )

    signal_label, signal_strength = _classify_signal(score, rsi_val, rsi_overbought)

    return {
        # Prices & indicators
        "price": round(price, 2),
        "rsi": round(rsi_val, 1),
        "macd": round(macd_val, 3),
        "macd_signal": round(macd_sig, 3),
        "macd_hist": round(macd_hist_val, 3),
        "ema20": round(ema20_val, 2),
        "ema50": round(ema50_val, 2),
        "ema200": round(ema200_val, 2),
        "atr": round(atr_val, 2),
        "adx": round(adx_val, 1),
        "stoch_k": round(stoch_k_val, 1),
        "vol_ratio": round(vol_ratio, 2),
        "bb_upper": round(bb_upper_val, 2),
        "bb_lower": round(bb_lower_val, 2),
        "bb_position": round(bb_position, 2),

        # Returns
        "ret_1w": round(ret_1w, 2),
        "ret_1m": round(ret_1m, 2),
        "ret_3m": round(ret_3m, 2),
        "ret_6m": round(ret_6m, 2),

        # Boolean signals
        "golden_cross": golden_cross,
        "price_above_ema200": price_above_ema200,
        "macd_bullish": macd_bullish,
        "macd_crossover": macd_crossover,
        "breakout": breakout,
        "strong_trend": strong_trend,
        "rsi_oversold": rsi_oversold,
        "rsi_overbought": rsi_overbought,
        "bb_squeeze": bb_squeeze,

        # Score & recommendation
        "score": score,
        "signal": signal_label,
        "signal_strength": signal_strength,

        # Raw series (for charts, last 60 candles)
        "close_series": close.tail(60).tolist(),
        "dates_series": [str(d.date()) for d in close.tail(60).index],
        "rsi_series": rsi.tail(60).fillna(50).tolist(),
        "macd_hist_series": macd_hist.tail(60).fillna(0).tolist(),
        "volume_series": volume.tail(60).tolist(),
        "vol_ma_series": vol_ma.tail(60).fillna(0).tolist(),
        "ema20_series": ema20.tail(60).tolist(),
        "ema50_series": ema50.tail(60).tolist(),
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pct_change(close: pd.Series, n: int) -> float:
    if len(close) < n + 1:
        return 0.0
    return round((close.iloc[-1] / close.iloc[-n - 1] - 1) * 100, 2)


def _composite_score(
    rsi_val, macd_bullish, macd_crossover, macd_momentum,
    price_above_ema20, price_above_ema50, price_above_ema200,
    golden_cross, strong_trend, breakout, vol_ratio,
    ret_1m, ret_3m, bb_position, stoch_k_val,
) -> int:
    score = 0.0

    # RSI — reward healthy range (40–65), penalise extremes
    if 40 <= rsi_val <= 65:
        score += 15
    elif 35 <= rsi_val < 40:
        score += 8
    elif 65 < rsi_val <= 70:
        score += 5
    elif rsi_val < 35:
        score += 3  # oversold can bounce but risky

    # MACD signals (max 20)
    if macd_bullish:
        score += 8
    if macd_crossover:
        score += 7
    if macd_momentum:
        score += 5

    # EMA trend alignment (max 20)
    if price_above_ema200:
        score += 7
    if price_above_ema50:
        score += 7
    if price_above_ema20:
        score += 4
    if golden_cross:
        score += 2

    # Trend strength (max 8)
    if strong_trend:
        score += 8

    # Breakout with volume (max 12)
    if breakout:
        score += 12
    elif vol_ratio > 1.3:
        score += 5

    # Momentum returns (max 15)
    if ret_1m > 3:
        score += 5
    elif ret_1m > 0:
        score += 2
    if ret_3m > 8:
        score += 6
    elif ret_3m > 3:
        score += 3
    if ret_3m > 15:
        score += 4

    # Bollinger Band position — ideal: lower-mid (not overbought)
    if 0.3 <= bb_position <= 0.65:
        score += 5
    elif bb_position < 0.3:
        score += 2  # near lower band — possible bounce

    # Stochastic — fresh from oversold
    if 30 <= stoch_k_val <= 70:
        score += 3

    return min(100, int(score))


def _classify_signal(score: int, rsi: float, rsi_overbought: bool) -> tuple[str, str]:
    if rsi_overbought:
        return "AVOID", "Overbought"
    if score >= 75:
        return "STRONG BUY", "High"
    if score >= 60:
        return "BUY", "Medium-High"
    if score >= 45:
        return "WATCH", "Medium"
    if score >= 30:
        return "NEUTRAL", "Low"
    return "AVOID", "Very Low"
