"""
Watchlist persistence — saves pinned stocks and their price alert levels to disk.

Each watchlist entry:
  {
    "symbol": "RELIANCE.NS",
    "name": "RELIANCE",
    "added_at": "2026-04-29",
    "alert_above": 3100.0,   # None if not set
    "alert_below": 2600.0,   # None if not set
    "alert_above_triggered": False,
    "alert_below_triggered": False,
    "notes": "Watching for breakout above 3100",
  }
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional

WATCHLIST_FILE = Path(__file__).parent / "watchlist.json"


def load_watchlist() -> list[dict]:
    if not WATCHLIST_FILE.exists():
        return []
    try:
        with open(WATCHLIST_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_watchlist(entries: list[dict]) -> None:
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(entries, f, indent=2)
    except Exception:
        pass


def add_to_watchlist(
    entries: list[dict],
    symbol: str,
    alert_above: Optional[float] = None,
    alert_below: Optional[float] = None,
    notes: str = "",
) -> list[dict]:
    name = symbol.replace(".NS", "")
    full_sym = symbol if symbol.endswith(".NS") else f"{symbol}.NS"

    # Avoid duplicates
    if any(e["symbol"] == full_sym for e in entries):
        return entries

    entries.append({
        "symbol": full_sym,
        "name": name,
        "added_at": str(date.today()),
        "alert_above": alert_above,
        "alert_below": alert_below,
        "alert_above_triggered": False,
        "alert_below_triggered": False,
        "notes": notes,
    })
    save_watchlist(entries)
    return entries


def remove_from_watchlist(entries: list[dict], symbol: str) -> list[dict]:
    full_sym = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    updated = [e for e in entries if e["symbol"] != full_sym]
    save_watchlist(updated)
    return updated


def update_alert_levels(
    entries: list[dict],
    symbol: str,
    alert_above: Optional[float],
    alert_below: Optional[float],
    notes: str = "",
) -> list[dict]:
    full_sym = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    for entry in entries:
        if entry["symbol"] == full_sym:
            entry["alert_above"] = alert_above
            entry["alert_below"] = alert_below
            entry["notes"] = notes
            # Reset trigger state when levels are updated
            entry["alert_above_triggered"] = False
            entry["alert_below_triggered"] = False
    save_watchlist(entries)
    return entries


def check_price_alerts(
    entries: list[dict],
    price_map: dict[str, float],
) -> list[dict]:
    """
    Check each watchlist entry against current prices.
    Returns list of triggered alert dicts (to send notifications).
    Marks triggered entries so they don't fire again until reset.
    """
    triggered = []

    for entry in entries:
        sym = entry["symbol"]
        price = price_map.get(sym)
        if price is None:
            continue

        above = entry.get("alert_above")
        below = entry.get("alert_below")

        if above and price >= above and not entry.get("alert_above_triggered"):
            triggered.append({
                "symbol": sym,
                "name": entry["name"],
                "price": price,
                "alert_type": "ABOVE",
                "level": above,
            })
            entry["alert_above_triggered"] = True

        if below and price <= below and not entry.get("alert_below_triggered"):
            triggered.append({
                "symbol": sym,
                "name": entry["name"],
                "price": price,
                "alert_type": "BELOW",
                "level": below,
            })
            entry["alert_below_triggered"] = True

    if triggered:
        save_watchlist(entries)

    return triggered
