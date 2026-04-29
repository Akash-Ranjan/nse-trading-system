"""
Portfolio persistence — saves and loads open positions to/from a local JSON file
so they survive Streamlit page refreshes.
"""

import json
import os
from pathlib import Path

PORTFOLIO_FILE = Path(__file__).parent / "portfolio.json"


def load_portfolio() -> list[dict]:
    """Load saved positions from disk. Returns empty list if file missing."""
    if not PORTFOLIO_FILE.exists():
        return []
    try:
        with open(PORTFOLIO_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_portfolio(positions: list[dict]) -> None:
    """Persist positions list to disk."""
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(positions, f, indent=2)
    except Exception:
        pass


def add_position(positions: list[dict], new_position: dict) -> list[dict]:
    positions.append(new_position)
    save_portfolio(positions)
    return positions


def remove_position(positions: list[dict], symbol: str) -> list[dict]:
    updated = [p for p in positions if p["symbol"] != symbol]
    save_portfolio(updated)
    return updated


def update_prices(positions: list[dict], price_map: dict[str, float]) -> list[dict]:
    """Refresh current prices in all positions from a symbol->price dict."""
    for pos in positions:
        sym = pos["symbol"] + ".NS"
        if sym in price_map:
            pos["current"] = round(price_map[sym], 2)
    save_portfolio(positions)
    return positions
