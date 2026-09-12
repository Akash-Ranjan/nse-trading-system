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


def compute_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Compute daily-anchored VWAP.
    Resets at the start of each calendar day so it is meaningful for intraday
    timeframes (1h, 30m, 15m). For daily data it degenerates to a cumulative
    average and is not meaningful — only call on intraday DataFrames.
    """
    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume

    dates = close.index.normalize()
    result = pd.Series(index=close.index, dtype=float)

    for day in dates.unique():
        mask = dates == day
        cum_tp_vol = tp_vol[mask].cumsum()
        cum_vol = volume[mask].cumsum()
        result[mask] = cum_tp_vol / cum_vol.replace(0, float("nan"))

    return result


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

    # ── 5-day sustained volume (for accumulation detection) ──
    # Average of the last 5 days vs the 20-day average — tells us if big money
    # has been consistently active all week, not just a single-day spike.
    _vol5 = volume.iloc[-5:] if len(volume) >= 5 else volume
    vol_avg_5d = float(_vol5.mean())
    vol_ratio_5d = vol_avg_5d / vol_avg if vol_avg > 0 else 1.0

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
    # momentum: histogram must be POSITIVE and expanding — not just "less negative".
    # Giving bonus when hist goes -10 → -8 (downtrend) is wrong.
    macd_momentum = macd_hist_val > 0 and macd_hist_val > macd_hist_prev

    rsi_oversold = rsi_val < 35
    rsi_overbought = rsi_val > 75

    strong_trend = adx_val > 25

    # 20-day breakout
    high_20d = float(high.rolling(20).max().iloc[-2]) if len(high) > 20 else float(high.max())
    breakout = price > high_20d and vol_ratio > 1.4

    bb_position = (price - bb_lower_val) / (bb_upper_val - bb_lower_val) if (bb_upper_val - bb_lower_val) > 0 else 0.5
    bb_squeeze = (bb_upper_val - bb_lower_val) / bb_mid_val < 0.06 if bb_mid_val > 0 else False

    # ── Support level detection ──────────────────────────────────────────────
    # "At support" = price is 0–N% ABOVE a known support zone.
    # Four zones checked (first match wins for the label):
    #   EMA20  — dynamic short-term support (0–2% above)
    #   EMA50  — medium-term support       (0–3% above)
    #   BB Lower Band — statistical lower bound (0–3% above)
    #   20-Day Low — recent floor / demand zone (0–3% above)
    near_ema20_support  = 0.0 <= _pct_above(price, ema20_val)  <= 2.0
    near_ema50_support  = 0.0 <= _pct_above(price, ema50_val)  <= 3.0
    near_bb_lower_sup   = 0.0 <= _pct_above(price, bb_lower_val) <= 3.0 if bb_lower_val > 0 else False
    low_20d = float(low.rolling(20).min().iloc[-1]) if len(low) >= 20 else float(low.min())
    near_20d_low_sup    = 0.0 <= _pct_above(price, low_20d) <= 3.0 if low_20d > 0 else False

    near_support = near_ema20_support or near_ema50_support or near_bb_lower_sup or near_20d_low_sup

    if near_ema20_support:
        support_type  = "EMA20"
        support_level = ema20_val
    elif near_ema50_support:
        support_type  = "EMA50"
        support_level = ema50_val
    elif near_bb_lower_sup:
        support_type  = "BB Lower"
        support_level = bb_lower_val
    elif near_20d_low_sup:
        support_type  = "20D Low"
        support_level = low_20d
    else:
        support_type  = "None"
        support_level = 0.0

    # ── Short-term coil / pre-breakout detection ─────────────────────────────
    # NR7 — today's high–low range is the narrowest of the last 7 sessions.
    # Statistically, NR7 days precede above-average directional moves.
    current_range = float(high.iloc[-1] - low.iloc[-1])
    if len(high) >= 7:
        _ranges7 = [float(high.iloc[-i] - low.iloc[-i]) for i in range(1, 8)]
        nr7 = (current_range <= min(_ranges7)) and (current_range < atr_val * 0.6)
    else:
        nr7 = False

    # Inside Bar — today's entire range is enclosed by yesterday's candle.
    # Signals indecision / accumulation: a low-risk entry day.
    inside_bar = (
        len(high) >= 2
        and float(high.iloc[-1]) <= float(high.iloc[-2])
        and float(low.iloc[-1])  >= float(low.iloc[-2])
    )

    # Price Compression — combined range of the last 3 candles vs ATR.
    # Ratio < 0.75 → stock has been coiling in less than 75% of its usual range.
    if len(high) >= 3 and atr_val > 0:
        _r3h = float(high.iloc[-3:].max())
        _r3l = float(low.iloc[-3:].min())
        price_compression_ratio = (_r3h - _r3l) / atr_val
    else:
        price_compression_ratio = 2.0
    price_compressed = price_compression_ratio < 0.75

    # Volume dry-up on pullback — last 2-day average volume < 70% of 20-day avg
    # while price hasn't fallen more than 2%. Healthy, low-conviction pullback.
    _vol2 = volume.iloc[-2:] if len(volume) >= 2 else volume
    vol_avg_2d  = float(_vol2.mean())
    vol_dryup   = (vol_avg_2d / vol_avg < 0.70) and (ret_1w > -2.0) if vol_avg > 0 else False

    # ── Resistance level detection ────────────────────────────────────────────
    # Rolling highs (excluding today) are natural resistance / target zones.
    res_5d  = float(high.iloc[-6:-1].max())  if len(high) >= 6  else float(high.max())
    res_10d = float(high.iloc[-11:-1].max()) if len(high) >= 11 else float(high.max())
    res_20d = float(high.iloc[-21:-1].max()) if len(high) >= 21 else float(high.max())

    dist_to_res_5d  = _dist_to_res(price, res_5d)
    dist_to_res_10d = _dist_to_res(price, res_10d)
    dist_to_res_20d = _dist_to_res(price, res_20d)

    # Nearest resistance strictly above current price
    _above_res = [(d, r) for d, r in [
        (dist_to_res_5d, res_5d),
        (dist_to_res_10d, res_10d),
        (dist_to_res_20d, res_20d),
    ] if d > 0.1]
    if _above_res:
        dist_to_nearest_res, nearest_resistance = min(_above_res, key=lambda x: x[0])
    else:
        dist_to_nearest_res, nearest_resistance = 0.0, 0.0

    # Clean 1–2% target: nearest resistance is 0.5–2.5% above price.
    # This is the sweet-spot for a short swing trade with a clear exit level.
    has_1to2_target = 0.5 <= dist_to_nearest_res <= 2.5

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
        vol_ratio_5d=vol_ratio_5d,
        near_support=near_support,
        ret_1m=ret_1m,
        ret_3m=ret_3m,
        bb_position=bb_position,
        stoch_k_val=stoch_k_val,
        nr7=nr7,
        inside_bar=inside_bar,
        price_compressed=price_compressed,
        vol_dryup=vol_dryup,
        has_1to2_target=has_1to2_target,
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
        "vol_ratio_5d": round(vol_ratio_5d, 2),
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
        "price_above_ema20":  price_above_ema20,
        "price_above_ema50":  price_above_ema50,
        "price_above_ema200": price_above_ema200,
        "macd_bullish": macd_bullish,
        "macd_crossover": macd_crossover,
        "breakout": breakout,
        "strong_trend": strong_trend,
        "rsi_oversold": rsi_oversold,
        "rsi_overbought": rsi_overbought,
        "bb_squeeze": bb_squeeze,
        "near_support": near_support,
        "support_type": support_type,
        "support_level": round(support_level, 2),

        # Coil / pre-breakout patterns
        "nr7": nr7,
        "inside_bar": inside_bar,
        "price_compressed": price_compressed,
        "price_compression_ratio": round(price_compression_ratio, 2),
        "vol_dryup": vol_dryup,

        # Resistance levels & target proximity
        "res_5d":  round(res_5d,  2),
        "res_10d": round(res_10d, 2),
        "res_20d": round(res_20d, 2),
        "dist_to_res_5d":  dist_to_res_5d,
        "dist_to_res_10d": dist_to_res_10d,
        "dist_to_res_20d": dist_to_res_20d,
        "nearest_resistance":   round(nearest_resistance, 2),
        "dist_to_nearest_res":  dist_to_nearest_res,
        "has_1to2_target":      has_1to2_target,

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


def _pct_above(px: float, level: float) -> float:
    """Signed % distance of price above a level. Returns -999 if level is zero."""
    return (px - level) / level * 100 if level > 0 else -999.0


def _dist_to_res(px: float, res: float) -> float:
    """% gap from price UP to resistance. Returns 0 if price is already above."""
    return round((res - px) / px * 100, 2) if px > 0 and res > px else 0.0


def _composite_score(
    rsi_val, macd_bullish, macd_crossover, macd_momentum,
    price_above_ema20, price_above_ema50, price_above_ema200,
    golden_cross, strong_trend, breakout, vol_ratio,
    ret_1m, ret_3m, bb_position, stoch_k_val,
    nr7=False, inside_bar=False, price_compressed=False,
    vol_dryup=False, has_1to2_target=False,
    vol_ratio_5d=1.0, near_support=False,
) -> int:
    """
    Composite score 0–100.  Higher = stronger buy setup.

    Priority / Max points per group:
      EMA trend alignment    20  — is the stock in an uptrend?     (most important)
      MACD signals           18  — momentum direction & timing
      RSI zone               15  — not overbought, healthy range
      Volume (1D + 5D)       14  — institutional participation
      ADX trend strength      8  — is the trend real?
      Coil patterns (NR7…)    8  — pre-breakout spring
      Returns (1M + 3M)       7  — moderate recent momentum
      BB position             5  — not extended above upper band
      Has 1–2% target         5  — defined risk/reward entry
      At support level        4  — better R:R on entry
      Stochastic              3  — secondary timing confirmation
      Volume dry-up           3  — weak sellers on pullback
      Total theoretical max  111 → capped at 100
    """
    score = 0.0

    # ── 1. EMA trend alignment (max 20) ─────────────────────────────────────
    # The most important filter: is the stock above its key moving averages?
    if price_above_ema200:
        score += 7   # in a long-term uptrend
    if price_above_ema50:
        score += 7   # medium-term uptrend intact
    if price_above_ema20:
        score += 4   # short-term trend intact
    if golden_cross:
        score += 2   # EMA50 crossed above EMA200 — structural bull signal

    # ── 2. MACD signals (max 18) ────────────────────────────────────────────
    # FIX: macd_momentum now only fires when histogram is POSITIVE & expanding,
    # so a stock in a downtrend (-10 → -8 histogram) no longer earns +5.
    if macd_bullish:
        score += 8   # MACD line above signal — trend is up
    if macd_crossover:
        score += 7   # fresh histogram crossover — entry trigger
    if macd_momentum:          # positive histogram AND growing
        score += 3   # reduced from 5 (crossover already captures the event)

    # ── 3. RSI zone (max 15) ────────────────────────────────────────────────
    # Sweet spot 40–65: stock has momentum but isn't overbought.
    if 40 <= rsi_val <= 65:
        score += 15
    elif 35 <= rsi_val < 40:
        score += 8   # slightly under sweet-spot — recovery setups
    elif 65 < rsi_val <= 70:
        score += 5   # extended but not overbought yet
    elif rsi_val < 35:
        score += 3   # oversold — can bounce, but risky alone

    # ── 4. Volume — single day + 5-day sustained (max 14) ───────────────────
    # Single-day spike vs multi-day accumulation are both rewarded.
    # FIX: vol_ratio_5d was computed but never scored before.
    if breakout:
        score += 10  # price > 20D high on vol > 1.4× avg — the strongest signal
    else:
        # Single-day volume (no breakout)
        if vol_ratio > 1.5:
            score += 5
        elif vol_ratio > 1.3:
            score += 3
        elif vol_ratio > 1.1:
            score += 1

    # 5-day SUSTAINED volume (institutional accumulation over multiple days)
    if vol_ratio_5d >= 2.0:
        score += 4   # very strong consistent buying
    elif vol_ratio_5d >= 1.5:
        score += 3
    elif vol_ratio_5d >= 1.3:
        score += 2

    # ── 5. Trend strength — ADX (max 8) ─────────────────────────────────────
    if strong_trend:            # ADX > 25
        score += 8

    # ── 6. Coil / pre-breakout patterns (max 8) ─────────────────────────────
    if nr7:
        score += 8   # narrowest range in 7 days — spring fully coiled
    elif inside_bar:
        score += 5   # today inside yesterday — low-risk entry day
    elif price_compressed:
        score += 3   # 3-day range < 0.75× ATR

    # ── 7. Returns — moderate momentum (max 8) ──────────────────────────────
    # FIX: removed the double-bonus where ret_3m > 15% gave 6+4=10 pts.
    # A stock that already ran 15%+ in 3 months may have less room left.
    # Now: clean if-elif, max 6 pts for 3M + 2 pts for 1M.
    if ret_1m > 3:
        score += 2
    if ret_3m > 15:
        score += 5   # strong multi-month trend
    elif ret_3m > 8:
        score += 5   # healthy medium momentum
    elif ret_3m > 3:
        score += 3   # mild momentum
    elif ret_3m > 0:
        score += 1   # at least positive

    # ── 8. Bollinger Band position (max 5) ──────────────────────────────────
    # Ideal: lower-to-mid band — room to move up, not extended.
    if 0.3 <= bb_position <= 0.65:
        score += 5
    elif bb_position < 0.3:
        score += 2   # near lower band — possible bounce

    # ── 9. Clear 1–2% resistance target (max 5) ─────────────────────────────
    if has_1to2_target:
        score += 5

    # ── 10. At support level (max 4) ─────────────────────────────────────────
    # FIX: near_support was computed but contributed 0 to score.
    # Entering at support = better risk/reward, tighter stop.
    if near_support:
        score += 4

    # ── 11. Stochastic (max 3) ───────────────────────────────────────────────
    if 30 <= stoch_k_val <= 70:
        score += 3

    # ── 12. Volume dry-up on pullback (max 3) ────────────────────────────────
    if vol_dryup:
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
