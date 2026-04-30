"""
Risk management — position sizing, stop-loss, targets, and portfolio rules.

Core rules this system enforces:
  • Never risk more than 2% of trading capital on any single trade.
  • Minimum Risk:Reward ratio of 1:2 (target must be 2x the risk amount).
  • ATR-based stop-loss: 1.5× ATR below entry for trending stocks.
  • Maximum 10 open positions at a time (10% capital allocation per position).
  • No single sector should exceed 30% of portfolio.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class TradeSetup:
    symbol: str
    entry_price: float
    stop_loss: float
    target_1: float       # Conservative target (1:2 R:R)
    target_2: float       # Aggressive target (1:3 R:R)
    quantity: int
    capital_at_risk: float
    risk_pct: float
    potential_gain_1: float
    potential_gain_2: float
    atr: float
    signal: str
    score: int
    notes: list[str] = field(default_factory=list)


def calculate_trade_setup(
    symbol: str,
    entry_price: float,
    atr: float,
    capital: float,
    risk_per_trade_pct: float = 2.0,
    signal: str = "BUY",
    score: int = 50,
    atr_stop_multiplier: float = 1.5,
) -> TradeSetup:
    """
    Given an entry price and ATR, calculate a complete trade setup.

    stop_loss  = entry - (ATR × multiplier)
    target_1   = entry + 2 × risk_per_share  (1:2 R:R)
    target_2   = entry + 3 × risk_per_share  (1:3 R:R)
    quantity   = floor(max_risk_in_rupees / risk_per_share)
    """
    risk_per_share = atr * atr_stop_multiplier
    stop_loss = round(entry_price - risk_per_share, 2)

    # Floor: stop-loss can never be more than 20% below entry (prevents absurdly wide stops)
    stop_loss = max(stop_loss, entry_price * 0.80)
    risk_per_share = entry_price - stop_loss

    max_risk_rupees = capital * (risk_per_trade_pct / 100)
    quantity = max(1, math.floor(max_risk_rupees / risk_per_share))

    # Cap position size at 15% of capital
    max_position_value = capital * 0.15
    quantity = min(quantity, math.floor(max_position_value / entry_price))
    quantity = max(quantity, 1)

    capital_at_risk = round(quantity * risk_per_share, 2)
    actual_risk_pct = round((capital_at_risk / capital) * 100, 2)

    reward_per_share_1 = risk_per_share * 2
    reward_per_share_2 = risk_per_share * 3

    target_1 = round(entry_price + reward_per_share_1, 2)
    target_2 = round(entry_price + reward_per_share_2, 2)

    potential_gain_1 = round(quantity * reward_per_share_1, 2)
    potential_gain_2 = round(quantity * reward_per_share_2, 2)

    notes = _generate_notes(entry_price, stop_loss, atr, signal, score, capital)

    return TradeSetup(
        symbol=symbol,
        entry_price=round(entry_price, 2),
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        quantity=quantity,
        capital_at_risk=capital_at_risk,
        risk_pct=actual_risk_pct,
        potential_gain_1=potential_gain_1,
        potential_gain_2=potential_gain_2,
        atr=round(atr, 2),
        signal=signal,
        score=score,
        notes=notes,
    )


def _generate_notes(
    entry: float,
    stop: float,
    atr: float,
    signal: str,
    score: int,
    capital: float,
) -> list[str]:
    notes = []

    stop_pct = abs((entry - stop) / entry) * 100
    if stop_pct > 8:
        notes.append("Wide stop-loss (>8%) — consider scaling position smaller.")

    if score >= 75:
        notes.append("High-confidence signal — full position sizing appropriate.")
    elif score >= 60:
        notes.append("Medium-confidence signal — consider starting with 75% of position.")
    else:
        notes.append("Moderate signal — start with 50% position, add on confirmation.")

    notes.append(f"Trail stop-loss to break-even once price moves 1× ATR (₹{atr:.2f}) in your favour.")
    notes.append("Book 50% profit at Target 1, let rest ride to Target 2 with trailing stop.")

    return notes


# ── Portfolio-level checks ────────────────────────────────────────────────────

def portfolio_health_check(
    positions: list[dict],
    capital: float,
) -> dict:
    """
    Check portfolio-level rules given a list of open positions.
    Each position dict: {symbol, sector, entry, current, quantity, stop_loss}
    """
    issues = []
    warnings = []

    # Rule 1: max 10 positions
    if len(positions) >= 10:
        issues.append("Maximum 10 positions reached. Close a trade before opening new ones.")

    # Rule 2: total deployed capital
    total_deployed = sum(p["entry"] * p["quantity"] for p in positions)
    deployed_pct = (total_deployed / capital) * 100 if capital > 0 else 0
    if deployed_pct > 80:
        warnings.append(f"Deployed {deployed_pct:.0f}% of capital — keep 20% as cash buffer.")

    # Rule 3: sector concentration
    sector_exposure: dict[str, float] = {}
    for p in positions:
        sector = p.get("sector", "Other")
        value = p["entry"] * p["quantity"]
        sector_exposure[sector] = sector_exposure.get(sector, 0) + value

    for sector, value in sector_exposure.items():
        pct = (value / capital) * 100
        if pct > 30:
            issues.append(f"Over-concentrated in {sector} ({pct:.0f}% of capital). Limit sector to 30%.")
        elif pct > 20:
            warnings.append(f"{sector} is {pct:.0f}% of capital. Consider diversifying.")

    # Rule 4: total risk
    total_risk = sum(
        (p["entry"] - p["stop_loss"]) * p["quantity"]
        for p in positions
        if p.get("stop_loss") and p["entry"] > p["stop_loss"]
    )
    total_risk_pct = (total_risk / capital) * 100 if capital > 0 else 0
    if total_risk_pct > 15:
        issues.append(f"Total portfolio risk {total_risk_pct:.1f}% exceeds safe limit of 15%.")

    return {
        "positions": len(positions),
        "deployed_pct": round(deployed_pct, 1),
        "total_risk_pct": round(total_risk_pct, 1),
        "sector_exposure": {k: round((v / capital) * 100, 1) for k, v in sector_exposure.items()},
        "issues": issues,
        "warnings": warnings,
        "status": "Critical" if issues else ("Warning" if warnings else "Healthy"),
    }


# ── Quick sizing helper ───────────────────────────────────────────────────────

def quick_position_size(capital: float, entry: float, stop: float, risk_pct: float = 2.0) -> dict:
    """Simple position sizing when you already know entry and stop price."""
    if entry <= stop:
        return {"error": "Entry must be above stop-loss."}
    risk_per_share = entry - stop
    max_risk = capital * (risk_pct / 100)
    quantity = max(1, math.floor(max_risk / risk_per_share))
    position_value = quantity * entry
    actual_risk = quantity * risk_per_share
    return {
        "quantity": quantity,
        "position_value": round(position_value, 2),
        "capital_at_risk": round(actual_risk, 2),
        "risk_pct": round((actual_risk / capital) * 100, 2),
        "position_pct": round((position_value / capital) * 100, 2),
    }
