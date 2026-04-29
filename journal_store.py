"""
Trade journal persistence — logs every trade with full context.

Each journal entry:
  {
    "id": "20260429_RELIANCE_buy",
    "date_entered": "2026-04-29",
    "date_exited": "2026-05-10",
    "symbol": "RELIANCE",
    "direction": "BUY",
    "entry_price": 2845.0,
    "exit_price": 2980.0,
    "stop_loss": 2738.0,
    "target_1": 2952.0,
    "quantity": 35,
    "strategy": "Golden Cross Trend",
    "score_at_entry": 78,
    "rsi_at_entry": 54.3,
    "pnl": 4725.0,
    "pnl_pct": 4.74,
    "outcome": "WIN",   # WIN / LOSS / BREAKEVEN
    "exit_reason": "Hit Target 1",
    "mistakes": "Should have held to Target 2",
    "learnings": "Volume was 2x — strong signal, could have held longer",
  }
"""

import json
import uuid
from datetime import date
from pathlib import Path

import pandas as pd

JOURNAL_FILE = Path(__file__).parent / "trade_journal.json"


def load_journal() -> list[dict]:
    if not JOURNAL_FILE.exists():
        return []
    try:
        with open(JOURNAL_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_journal(entries: list[dict]) -> None:
    try:
        with open(JOURNAL_FILE, "w") as f:
            json.dump(entries, f, indent=2)
    except Exception:
        pass


def add_trade(
    entries: list[dict],
    symbol: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    stop_loss: float,
    target_1: float,
    quantity: int,
    date_entered: str,
    date_exited: str,
    strategy: str = "",
    score_at_entry: int = 0,
    rsi_at_entry: float = 0.0,
    exit_reason: str = "",
    mistakes: str = "",
    learnings: str = "",
) -> list[dict]:
    pnl_per_share = exit_price - entry_price if direction == "BUY" else entry_price - exit_price
    pnl = round(pnl_per_share * quantity, 2)
    pnl_pct = round((pnl_per_share / entry_price) * 100, 2)

    if pnl > 0:
        outcome = "WIN"
    elif pnl < 0:
        outcome = "LOSS"
    else:
        outcome = "BREAKEVEN"

    entry = {
        "id": f"{date_entered}_{symbol}_{str(uuid.uuid4())[:6]}",
        "date_entered": date_entered,
        "date_exited": date_exited,
        "symbol": symbol.upper(),
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "quantity": quantity,
        "strategy": strategy,
        "score_at_entry": score_at_entry,
        "rsi_at_entry": rsi_at_entry,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "outcome": outcome,
        "exit_reason": exit_reason,
        "mistakes": mistakes,
        "learnings": learnings,
    }

    entries.append(entry)
    save_journal(entries)
    return entries


def delete_trade(entries: list[dict], trade_id: str) -> list[dict]:
    updated = [e for e in entries if e["id"] != trade_id]
    save_journal(updated)
    return updated


def get_journal_stats(entries: list[dict]) -> dict:
    """Compute overall trading statistics from journal."""
    if not entries:
        return {}

    wins = [e for e in entries if e["outcome"] == "WIN"]
    losses = [e for e in entries if e["outcome"] == "LOSS"]
    total = len(entries)

    total_pnl = sum(e["pnl"] for e in entries)
    gross_profit = sum(e["pnl"] for e in wins) if wins else 0
    gross_loss = abs(sum(e["pnl"] for e in losses)) if losses else 0
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    avg_win = round(sum(e["pnl_pct"] for e in wins) / len(wins), 2) if wins else 0
    avg_loss = round(sum(e["pnl_pct"] for e in losses) / len(losses), 2) if losses else 0

    # Best strategy
    strategy_pnl: dict[str, float] = {}
    for e in entries:
        s = e.get("strategy", "Unknown")
        strategy_pnl[s] = strategy_pnl.get(s, 0) + e["pnl"]
    best_strategy = max(strategy_pnl, key=strategy_pnl.get) if strategy_pnl else "N/A"

    return {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / total * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": profit_factor,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "best_trade": round(max((e["pnl"] for e in entries), default=0), 2),
        "worst_trade": round(min((e["pnl"] for e in entries), default=0), 2),
        "best_strategy": best_strategy,
        "strategy_pnl": strategy_pnl,
        "expectancy": round(total_pnl / total, 2) if total > 0 else 0,
    }


def to_dataframe(entries: list[dict]) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame()
    df = pd.DataFrame(entries)
    show_cols = [
        "date_entered", "date_exited", "symbol", "direction", "strategy",
        "entry_price", "exit_price", "quantity", "pnl", "pnl_pct", "outcome",
        "exit_reason", "mistakes", "learnings",
    ]
    return df[[c for c in show_cols if c in df.columns]]
