"""
Simple walk-forward backtester.

For each strategy, it scans historical data day-by-day (rolling window),
generates a signal on day N, and measures the return over the next 10/20 trading days.
Reports win rate, average return, best/worst trade, and profit factor.
"""

import numpy as np
import pandas as pd

from analyzer import (
    compute_rsi, compute_macd, compute_ema, compute_atr,
    compute_adx, compute_volume_ma,
)


# ── Strategy signal functions (operate on slices ending at day i) ────────────

def _signal_golden_cross(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    ema50 = compute_ema(close, 50)
    ema200 = compute_ema(close, 200)
    ema20 = compute_ema(close, 20)
    macd_line, signal_line, _ = compute_macd(close)
    adx = compute_adx(df["High"], df["Low"], close)

    buy = (
        (ema50 > ema200)
        & (close > ema20)
        & (macd_line > signal_line)
        & (adx > 20)
        & (ema50.shift(1) <= ema200.shift(1))  # fresh crossover
    )
    return buy


def _signal_macd_momentum(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    macd_line, signal_line, hist = compute_macd(close)
    rsi = compute_rsi(close)

    buy = (
        (hist > 0) & (hist.shift(1) <= 0)  # fresh crossover
        & (rsi >= 40) & (rsi <= 68)
    )
    return buy


def _signal_volume_breakout(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    high = df["High"]
    volume = df["Volume"]
    vol_ma = compute_volume_ma(volume, 20)
    high_20d = high.rolling(20).max().shift(1)

    buy = (
        (close > high_20d)
        & (volume > vol_ma * 1.4)
    )
    return buy


def _signal_oversold_bounce(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    rsi = compute_rsi(close)
    macd_line, signal_line, hist = compute_macd(close)

    buy = (
        (rsi < 38)
        & (hist > hist.shift(1))  # MACD hist rising
        & (macd_line > macd_line.shift(2))  # MACD line turning up
    )
    return buy


STRATEGIES = {
    "Golden Cross Trend": _signal_golden_cross,
    "MACD Momentum": _signal_macd_momentum,
    "Volume Breakout": _signal_volume_breakout,
    "Oversold Bounce": _signal_oversold_bounce,
}


# ── Core backtester ───────────────────────────────────────────────────────────

def backtest_strategy(
    df: pd.DataFrame,
    strategy_name: str,
    hold_days: int = 15,
    stop_loss_pct: float = 5.0,
) -> dict:
    """
    Walk-forward backtest for a single strategy on one stock's OHLCV DataFrame.

    Returns a dict with:
      trades, win_rate, avg_return, max_gain, max_loss, profit_factor, returns list
    """
    if df is None or len(df) < 250:
        return _empty_result(strategy_name)

    signal_fn = STRATEGIES.get(strategy_name)
    if signal_fn is None:
        return _empty_result(strategy_name)

    try:
        signals = signal_fn(df)
    except Exception:
        return _empty_result(strategy_name)

    close = df["Close"].values
    dates = df.index

    trades = []
    i = 200  # skip warm-up period

    while i < len(close) - hold_days - 1:
        if signals.iloc[i]:
            entry = close[i + 1]  # buy at next open (approx close)
            exit_prices = close[i + 2 : i + 2 + hold_days]
            stop = entry * (1 - stop_loss_pct / 100)

            # Find first stop hit or hold until end
            hit_stop = False
            exit_price = exit_prices[-1]
            for ep in exit_prices:
                if ep <= stop:
                    exit_price = stop
                    hit_stop = True
                    break

            pct_return = (exit_price / entry - 1) * 100
            trades.append({
                "date": str(dates[i].date()),
                "entry": round(entry, 2),
                "exit": round(exit_price, 2),
                "return_pct": round(pct_return, 2),
                "hit_stop": hit_stop,
            })
            i += hold_days  # no overlapping trades
        else:
            i += 1

    if not trades:
        return _empty_result(strategy_name)

    returns = [t["return_pct"] for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    return {
        "strategy": strategy_name,
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_return": round(np.mean(returns), 2),
        "median_return": round(float(np.median(returns)), 2),
        "max_gain": round(max(returns), 2),
        "max_loss": round(min(returns), 2),
        "profit_factor": profit_factor,
        "trade_list": trades,
        "return_series": returns,
        "expectancy": round(np.mean(returns), 2),
    }


def backtest_all_strategies(
    df: pd.DataFrame,
    hold_days: int = 15,
    stop_loss_pct: float = 5.0,
) -> list[dict]:
    """Run all 4 strategies and return a list of result dicts."""
    return [
        backtest_strategy(df, name, hold_days, stop_loss_pct)
        for name in STRATEGIES
    ]


def _empty_result(strategy_name: str) -> dict:
    return {
        "strategy": strategy_name,
        "trades": 0,
        "win_rate": 0,
        "avg_return": 0,
        "median_return": 0,
        "max_gain": 0,
        "max_loss": 0,
        "profit_factor": 0,
        "trade_list": [],
        "return_series": [],
        "expectancy": 0,
    }


# ── Equity curve ──────────────────────────────────────────────────────────────

def compute_equity_curve(returns: list[float], initial: float = 100_000) -> list[float]:
    """Compound a list of % returns starting from initial capital."""
    equity = [initial]
    capital = initial
    for r in returns:
        capital *= (1 + r / 100)
        equity.append(round(capital, 2))
    return equity


def compute_max_drawdown(equity_curve: list[float]) -> float:
    """Maximum peak-to-trough drawdown as a percentage."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 2)
