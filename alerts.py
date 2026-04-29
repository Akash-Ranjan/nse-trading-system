"""
Telegram alert system — sends buy signal notifications to a Telegram bot.

Setup (one time):
  1. Open Telegram → search @BotFather → /newbot → get your BOT_TOKEN
  2. Open your bot, send /start
  3. Visit https://api.telegram.org/bot<BOT_TOKEN>/getUpdates to get your CHAT_ID
  4. Enter both in the app's Alerts tab settings

No token stored in code — entered at runtime via the Streamlit UI and kept
in st.session_state only.
"""

import requests


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> tuple[bool, str]:
    """
    Send a message via Telegram Bot API.
    Returns (success: bool, message: str).
    """
    if not bot_token or not chat_id:
        return False, "Bot token or chat ID missing."

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }

    try:
        resp = requests.post(url, json=payload, timeout=10, verify=False)
        if resp.status_code == 200:
            return True, "Message sent successfully."
        else:
            return False, f"Telegram error: {resp.json().get('description', resp.text)}"
    except Exception as e:
        return False, f"Failed to send: {e}"


def format_buy_alert(stock: dict) -> str:
    """Format a single stock's signal as a Telegram HTML message."""
    signal_emoji = "🟢" if stock.get("signal") == "STRONG BUY" else "🟩"
    return (
        f"{signal_emoji} <b>NSE Buy Signal — {stock['name']}</b>\n\n"
        f"Signal: <b>{stock['signal']}</b>  |  Score: <b>{stock['score']}/100</b>\n"
        f"Sector: {stock.get('sector', 'N/A')}\n\n"
        f"Price:  ₹{stock['price']:,.2f}\n"
        f"RSI:    {stock['rsi']}\n"
        f"ADX:    {stock['adx']}\n"
        f"Vol Ratio: {stock['vol_ratio']}x\n\n"
        f"1M Return: {stock['ret_1m']:+.1f}%  |  3M: {stock['ret_3m']:+.1f}%\n\n"
        f"Strategies triggered:\n"
        f"{'✅' if stock.get('strategy_golden_cross') else '❌'} Golden Cross\n"
        f"{'✅' if stock.get('strategy_macd_momentum') else '❌'} MACD Momentum\n"
        f"{'✅' if stock.get('strategy_breakout') else '❌'} Volume Breakout\n"
        f"{'✅' if stock.get('strategy_oversold_bounce') else '❌'} Oversold Bounce\n\n"
        f"<i>Entry: ₹{stock['price']:,.2f} | "
        f"Stop: ₹{round(stock['price'] - stock['atr'] * 1.5, 2):,.2f}</i>\n"
        f"<i>Not financial advice. DYOR.</i>"
    )


def send_bulk_alerts(
    bot_token: str,
    chat_id: str,
    top_stocks: list[dict],
    min_score: int = 65,
) -> tuple[int, list[str]]:
    """
    Send alerts for all stocks above min_score.
    Returns (sent_count, list_of_error_messages).
    """
    sent = 0
    errors = []

    eligible = [s for s in top_stocks if s.get("score", 0) >= min_score
                and s.get("signal") in ("STRONG BUY", "BUY")]

    for stock in eligible[:10]:  # cap at 10 alerts per run
        msg = format_buy_alert(stock)
        ok, err = send_telegram_message(bot_token, chat_id, msg)
        if ok:
            sent += 1
        else:
            errors.append(f"{stock['name']}: {err}")

    return sent, errors


def send_test_message(bot_token: str, chat_id: str) -> tuple[bool, str]:
    """Send a test ping to confirm the bot is configured correctly."""
    return send_telegram_message(
        bot_token, chat_id,
        "✅ <b>NSE Trading System</b>\n\nBot connected successfully! "
        "You'll receive buy signal alerts here after each screener run."
    )
