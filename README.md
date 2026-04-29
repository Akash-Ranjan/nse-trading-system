# NSE Trading System

A multi-strategy technical analysis dashboard for NSE stocks, built with Python and Streamlit.

## Features

| Module | What it does |
|---|---|
| **Stock Screener** | Scans 120+ NSE stocks, scores each 0–100, surfaces best buy opportunities |
| **4 Strategies** | Golden Cross, MACD Momentum, Volume Breakout, Oversold Bounce |
| **Stock Analysis** | Full chart (candlestick + EMA + RSI + MACD), signal summary, trade setup |
| **Portfolio Tracker** | Track open positions, P&L, sector exposure, portfolio health check |
| **Risk Calculator** | Position sizing, R:R visualizer, ATR-based stop-loss calculation |

## Setup

### 1. Create a virtual environment (recommended)

```bash
cd /Users/aranj7/Document/trd_psnl
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## How to Use

### Step 1 — Set your capital
In the sidebar, enter your trading capital and how much % you want to risk per trade (default 2%).

### Step 2 — Run the screener
Go to the **Stock Screener** tab. Choose a stock universe (start with NIFTY 50 for safety) and click **Run Screener**. Wait ~60–90 seconds while it fetches and analyses all stocks.

### Step 3 — Review top buys
Stocks with **STRONG BUY** or **BUY** signal and score above 60 are your best candidates. You can filter by strategy or sector.

### Step 4 — Deep-dive individual stocks
Go to **Stock Analysis**, type the NSE symbol (e.g. `RELIANCE`), and click Analyse. You'll see:
- Full candlestick chart with EMA 20/50 overlay
- RSI and MACD histogram panels
- Signal checklist (what's bullish, what's cautionary)
- Suggested entry, stop-loss, and two target prices

### Step 5 — Check risk and position size
The trade setup on the analysis page tells you exactly how many shares to buy. You can also use the **Risk Calculator** tab to manually enter any entry/stop price.

### Step 6 — Track your trades
Add your trades to the **Portfolio Tracker**. It monitors P&L, sector concentration, and alerts you if portfolio risk exceeds safe limits.

---

## Strategies Explained

### 1. Golden Cross Trend
**When:** EMA 50 crosses above EMA 200 (long-term uptrend confirmed).  
**Additional filters:** Price > EMA 20, MACD bullish, ADX > 20 (trend is real, not sideways).  
**Best for:** Riding established uptrends in quality large-caps (NIFTY 50 stocks).  
**Hold time:** 4–12 weeks.

### 2. MACD Momentum
**When:** MACD line freshly crosses above signal line, RSI between 40–68.  
**Why RSI filter:** Excludes overbought stocks where the crossover is too late.  
**Best for:** Catching early momentum moves before they become crowded trades.  
**Hold time:** 2–6 weeks.

### 3. Volume Breakout
**When:** Price breaks above the 20-day high AND volume is 1.4× average or more.  
**Why volume matters:** High volume = institutional buying. Low volume breakouts fail 70% of the time.  
**Best for:** High-momentum mid-caps with a strong business catalyst.  
**Hold time:** 1–4 weeks.

### 4. Oversold Bounce
**When:** RSI drops below 38 (stock is oversold/beaten down) AND MACD starts turning bullish.  
**Best for:** Quality large-cap stocks that fell due to market correction, not business trouble.  
**Risk:** Higher volatility. Use tight stops.  
**Hold time:** 1–3 weeks.

---

## Composite Score (0–100)

| Score | Signal | Meaning |
|---|---|---|
| 75–100 | STRONG BUY | Multiple strategies aligned. High-confidence setup. |
| 60–74 | BUY | Good setup. Most indicators positive. |
| 45–59 | WATCH | Mixed signals. Monitor for confirmation. |
| 30–44 | NEUTRAL | No clear setup. Stay out. |
| 0–29 | AVOID | Bearish signals dominate. Do not buy. |

---

## Risk Rules (Non-Negotiable)

1. **Never risk more than 2% of capital per trade.** If you have ₹1,00,000 capital, max loss per trade = ₹2,000.
2. **Set stop-loss before entering.** No exceptions.
3. **Minimum 1:2 Risk:Reward.** Only enter if your target is at least 2× your risk amount.
4. **Maximum 10 open positions.** Concentration kills accounts.
5. **Book 50% at Target 1.** Let the rest ride with a trailing stop.
6. **No sector should exceed 30% of capital.**
7. **Keep at least 20% cash** for averaging down or new opportunities.

---

## Important Disclaimer

This system is for **educational and research purposes only**. It is **not SEBI-registered investment advice**. Stock markets involve risk of loss. Always do your own research (DYOR) and consult a SEBI-registered advisor before investing real money. Past patterns do not guarantee future results.

---

## Data Source

Stock data is fetched from **Yahoo Finance** via the `yfinance` library. Data refreshes every 15 minutes (cached in memory). Market data may be delayed by 15 minutes.
