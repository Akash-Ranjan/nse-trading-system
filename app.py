"""
NSE Trading Dashboard — main Streamlit application.

Run: streamlit run app.py
"""

import time
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from stocks_universe import ALL_STOCKS, NIFTY_50, NIFTY_NEXT_50, HIGH_MOMENTUM_MIDCAP, get_display_name
from data_fetcher import fetch_ohlcv, get_52_week_stats
from analyzer import analyze
from screener import run_screener, filter_by_strategy, get_top_buys, get_sector_breakdown, get_summary_stats
from risk_manager import calculate_trade_setup, quick_position_size, portfolio_health_check

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NSE Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header  { color: #666; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .metric-card { background: #f8f9fa; border-radius: 8px; padding: 1rem;
                   border-left: 4px solid #1f77b4; }
    .signal-STRONG-BUY { color: #00a651; font-weight: 700; }
    .signal-BUY        { color: #4caf50; font-weight: 600; }
    .signal-WATCH      { color: #ff9800; font-weight: 600; }
    .signal-NEUTRAL    { color: #9e9e9e; }
    .signal-AVOID      { color: #f44336; font-weight: 600; }
    .stDataFrame thead tr th { background-color: #1f2937; color: white; }
    div[data-testid="metric-container"] { background: #f8f9fa;
        border-radius: 8px; padding: 0.5rem 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    capital = st.number_input(
        "Trading Capital (₹)",
        min_value=10_000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
        help="Total capital you want to deploy for trading.",
    )

    risk_per_trade = st.slider(
        "Risk per Trade (%)",
        min_value=0.5,
        max_value=3.0,
        value=2.0,
        step=0.25,
        help="Maximum % of capital you're willing to lose on a single trade.",
    )

    st.divider()

    universe_choice = st.selectbox(
        "Stock Universe",
        options=["NIFTY 50 (Safest)", "NIFTY 50 + Next 50", "Full Universe (Slower)"],
        index=1,
    )

    universe_map = {
        "NIFTY 50 (Safest)": NIFTY_50,
        "NIFTY 50 + Next 50": NIFTY_50 + NIFTY_NEXT_50,
        "Full Universe (Slower)": ALL_STOCKS,
    }
    selected_universe = universe_map[universe_choice]

    st.divider()
    st.markdown("### 📋 Quick Facts")
    st.info(
        f"**Capital:** ₹{capital:,.0f}\n\n"
        f"**Max risk/trade:** ₹{capital * risk_per_trade / 100:,.0f}\n\n"
        f"**Max positions:** 10\n\n"
        f"**Stocks in scan:** {len(selected_universe)}"
    )
    st.divider()
    st.caption("Data from Yahoo Finance. Refresh every 15 min. Not SEBI-registered advice.")


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Dashboard",
    "🔍 Stock Screener",
    "📊 Stock Analysis",
    "💼 Portfolio Tracker",
    "🔢 Risk Calculator",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="main-header">NSE Trading System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-strategy technical screening for NSE stocks · '
        'Data refreshes every 15 minutes</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Capital", f"₹{capital:,.0f}")
    with col2:
        st.metric("Risk/Trade", f"{risk_per_trade}% = ₹{capital * risk_per_trade / 100:,.0f}")
    with col3:
        st.metric("Universe", f"{len(selected_universe)} stocks")
    with col4:
        st.metric("Max Positions", "10 concurrent")

    st.divider()

    st.markdown("### How This System Works")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        **Strategy 1 — Golden Cross Trend**

        EMA 50 crosses above EMA 200. Price stays above EMA 20. MACD bullish.
        ADX > 20 confirms a real trend — not just noise.
        """)
    with c2:
        st.markdown("""
        **Strategy 2 — MACD Momentum**

        MACD line freshly crosses above signal line. RSI between 40–68 (healthy,
        not overbought). Good for catching early momentum moves.
        """)
    with c3:
        st.markdown("""
        **Strategy 3 — Volume Breakout**

        Price breaks above 20-day high with volume 1.4× its average.
        High volume confirms institutional participation — not a fakeout.
        """)
    with c4:
        st.markdown("""
        **Strategy 4 — Oversold Bounce**

        RSI drops below 38 (oversold) and MACD starts turning bullish. Best
        for catching reversals in quality large-cap stocks.
        """)

    st.divider()
    st.markdown("### Risk Management Rules You Must Follow")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.error("**Never risk > 2% per trade**\nSet stop-loss before you enter. No trade without a stop.")
    with r2:
        st.warning("**Minimum 1:2 Risk:Reward**\nOnly take a trade if potential gain is 2× your risk.")
    with r3:
        st.success("**Book 50% at Target 1**\nLet rest ride with trailing stop. Never let a profit turn into a loss.")

    st.divider()
    st.markdown("### NSE Market Timings")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.info("**Pre-Market:** 9:00 AM – 9:15 AM\nBid/ask orders collected. Price discovery happens here.")
    with t2:
        st.info("**Market Hours:** 9:15 AM – 3:30 PM\nMain trading session. Best liquidity 9:15–11 AM and 2:30–3:30 PM.")
    with t3:
        st.info("**Post-Market:** 3:40 PM – 4:00 PM\nClosing session. Use for limit orders at closing price.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCREENER
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## Stock Screener")
    st.caption("Scans all stocks, scores them 0–100, and surfaces the best buy opportunities.")

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        strategy_filter = st.selectbox(
            "Filter by Strategy",
            options=["All Strategies", "Golden Cross Trend", "MACD Momentum",
                     "Volume Breakout", "Oversold Bounce"],
        )
    with col_b:
        sector_options = ["All Sectors", "Banking", "IT", "FMCG", "Pharma", "Auto",
                          "Energy", "Metals", "Infra", "NBFC", "Insurance", "Cement", "Consumer"]
        sector_filter = st.selectbox("Filter by Sector", sector_options)
    with col_c:
        min_score = st.number_input("Min Score", min_value=0, max_value=100, value=50)

    run_col, _ = st.columns([1, 4])
    with run_col:
        run_btn = st.button("🔍 Run Screener", type="primary", use_container_width=True)

    if "screener_df" not in st.session_state:
        st.session_state.screener_df = pd.DataFrame()
        st.session_state.screener_summary = {}

    if run_btn:
        progress_bar = st.progress(0, text="Scanning stocks...")
        status_text = st.empty()

        def update_progress(done, total):
            pct = done / total
            progress_bar.progress(pct, text=f"Scanning {done}/{total} stocks...")

        with st.spinner("Fetching data and running analysis..."):
            df_all = run_screener(selected_universe, progress_callback=update_progress)

        progress_bar.empty()
        status_text.empty()

        st.session_state.screener_df = df_all
        st.session_state.screener_summary = get_summary_stats(df_all)
        st.success(f"Screened {len(df_all)} stocks successfully!")

    if not st.session_state.screener_df.empty:
        df_display = st.session_state.screener_df.copy()
        summary = st.session_state.screener_summary

        # Summary metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Screened", summary.get("total_screened", 0))
        m2.metric("Strong Buy", summary.get("strong_buy", 0), delta_color="normal")
        m3.metric("Buy", summary.get("buy", 0))
        m4.metric("Watch", summary.get("watch", 0))
        m5.metric("Avg Score", summary.get("avg_score", 0))

        st.divider()

        # Strategy signal counts chart
        strat_data = {
            "Golden Cross": summary.get("golden_cross_count", 0),
            "MACD Momentum": summary.get("macd_momentum_count", 0),
            "Breakout": summary.get("breakout_count", 0),
            "Oversold Bounce": summary.get("oversold_bounce_count", 0),
        }
        fig_strat = px.bar(
            x=list(strat_data.keys()),
            y=list(strat_data.values()),
            title="Stocks Triggering Each Strategy",
            labels={"x": "Strategy", "y": "Count"},
            color=list(strat_data.values()),
            color_continuous_scale="Blues",
        )
        fig_strat.update_layout(height=300, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_strat, use_container_width=True)

        # Apply filters
        if strategy_filter != "All Strategies":
            df_display = filter_by_strategy(df_display, strategy_filter)
        if sector_filter != "All Sectors":
            df_display = df_display[df_display["sector"] == sector_filter]
        df_display = df_display[df_display["score"] >= min_score]

        st.markdown(f"### Results — {len(df_display)} stocks")

        # Colour the signal column
        def signal_color(val):
            colors = {
                "STRONG BUY": "background-color: #c8e6c9; color: #1b5e20; font-weight:bold",
                "BUY": "background-color: #dcedc8; color: #33691e",
                "WATCH": "background-color: #fff9c4; color: #f57f17",
                "NEUTRAL": "background-color: #f5f5f5; color: #616161",
                "AVOID": "background-color: #ffcdd2; color: #b71c1c",
            }
            return colors.get(val, "")

        cols_show = ["name", "sector", "price", "score", "signal", "rsi", "adx",
                     "macd_bullish", "golden_cross", "breakout",
                     "ret_1m", "ret_3m", "vol_ratio"]
        cols_available = [c for c in cols_show if c in df_display.columns]

        renamed = {
            "name": "Stock", "sector": "Sector", "price": "Price (₹)",
            "score": "Score", "signal": "Signal", "rsi": "RSI",
            "adx": "ADX", "macd_bullish": "MACD↑", "golden_cross": "GoldenX",
            "breakout": "Breakout", "ret_1m": "1M %", "ret_3m": "3M %",
            "vol_ratio": "Vol Ratio",
        }

        display_df = df_display[cols_available].rename(columns=renamed)
        styled = display_df.style.map(signal_color, subset=["Signal"])
        st.dataframe(styled, use_container_width=True, height=450)

        # Sector distribution of buy signals
        buy_stocks = df_display[df_display["signal"].isin(["BUY", "STRONG BUY"])]
        if not buy_stocks.empty and "sector" in buy_stocks.columns:
            sector_counts = buy_stocks["sector"].value_counts()
            fig_sector = px.pie(
                values=sector_counts.values,
                names=sector_counts.index,
                title="Sector Distribution of Buy Signals",
                hole=0.4,
            )
            fig_sector.update_layout(height=350)
            st.plotly_chart(fig_sector, use_container_width=True)

    else:
        st.info("Click **Run Screener** to scan the selected universe of stocks.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STOCK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## Individual Stock Analysis")

    symbol_input = st.text_input(
        "Enter NSE Symbol (e.g. RELIANCE, TCS, INFY)",
        value="RELIANCE",
        help="Type the NSE ticker without .NS suffix.",
    ).strip().upper()

    analyze_btn = st.button("📊 Analyse", type="primary")

    if analyze_btn and symbol_input:
        full_symbol = f"{symbol_input}.NS"
        with st.spinner(f"Fetching data for {symbol_input}..."):
            df_stock = fetch_ohlcv(full_symbol, period="1y", force_refresh=True)

        if df_stock is None:
            st.error(f"Could not fetch data for **{symbol_input}**. Check the symbol and try again.")
        else:
            result = analyze(df_stock)
            stats = get_52_week_stats(df_stock)

            # Store in session state for reuse
            st.session_state[f"analysis_{symbol_input}"] = (result, stats, df_stock)

    # Display from session state
    session_key = f"analysis_{symbol_input}"
    if session_key in st.session_state:
        result, stats, df_stock = st.session_state[session_key]

        # ── Header ──
        signal_colors = {
            "STRONG BUY": "🟢", "BUY": "🟩", "WATCH": "🟡",
            "NEUTRAL": "⬜", "AVOID": "🔴",
        }
        sig_icon = signal_colors.get(result["signal"], "⬜")
        st.markdown(f"### {symbol_input}  {sig_icon} {result['signal']}  (Score: {result['score']}/100)")

        # ── Key metrics row ──
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Price", f"₹{result['price']:,.2f}")
        col2.metric("RSI (14)", result["rsi"],
                    delta="Oversold" if result["rsi_oversold"] else ("Overbought" if result["rsi_overbought"] else None))
        col3.metric("ADX", result["adx"], delta="Trending" if result["strong_trend"] else None)
        col4.metric("1M Return", f"{result['ret_1m']:+.1f}%")
        col5.metric("3M Return", f"{result['ret_3m']:+.1f}%")
        col6.metric("Vol Ratio", f"{result['vol_ratio']:.1f}x")

        st.divider()

        # ── 52-week stats ──
        wk1, wk2, wk3, wk4 = st.columns(4)
        wk1.metric("52W High", f"₹{stats['high_52w']:,.2f}")
        wk2.metric("52W Low", f"₹{stats['low_52w']:,.2f}")
        wk3.metric("Current", f"₹{stats['current']:,.2f}")
        wk4.metric("Position in Range", f"{stats['position_pct']}%")

        st.divider()

        # ── Price chart with EMAs ──
        dates = result["dates_series"]
        closes = result["close_series"]
        ema20s = result["ema20_series"]
        ema50s = result["ema50_series"]

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20],
            vertical_spacing=0.03,
            subplot_titles=["Price + EMAs", "RSI (14)", "MACD Histogram"],
        )

        # Candlestick
        df_tail = df_stock.tail(60)
        fig.add_trace(go.Candlestick(
            x=[str(d.date()) for d in df_tail.index],
            open=df_tail["Open"], high=df_tail["High"],
            low=df_tail["Low"], close=df_tail["Close"],
            name="Price", increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=dates, y=ema20s, name="EMA 20",
                                  line=dict(color="#1f77b4", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=ema50s, name="EMA 50",
                                  line=dict(color="#ff7f0e", width=1.5)), row=1, col=1)

        # RSI
        rsi_series = result["rsi_series"]
        rsi_colors = ["#ef5350" if v > 70 else "#26a69a" if v < 30 else "#1f77b4" for v in rsi_series]
        fig.add_trace(go.Scatter(x=dates, y=rsi_series, name="RSI",
                                  line=dict(color="#9c27b0", width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)

        # MACD Histogram
        macd_hist = result["macd_hist_series"]
        macd_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in macd_hist]
        fig.add_trace(go.Bar(x=dates, y=macd_hist, name="MACD Hist",
                              marker_color=macd_colors), row=3, col=1)

        fig.update_layout(
            height=700,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Signal summary ──
        st.markdown("### Signal Summary")
        sig_col1, sig_col2 = st.columns(2)
        with sig_col1:
            st.markdown("**Bullish signals:**")
            signals_bull = [
                ("Golden Cross (EMA50 > EMA200)", result["golden_cross"]),
                ("Price above EMA 200", result["price_above_ema200"]),
                ("MACD Bullish", result["macd_bullish"]),
                ("Fresh MACD Crossover", result["macd_crossover"]),
                ("Volume Breakout", result["breakout"]),
                ("Strong Trend (ADX > 25)", result["strong_trend"]),
                ("BB Squeeze (pending move)", result["bb_squeeze"]),
            ]
            for name, val in signals_bull:
                icon = "✅" if val else "❌"
                st.markdown(f"{icon} {name}")

        with sig_col2:
            st.markdown("**Caution signals:**")
            st.markdown(f"{'⚠️' if result['rsi_overbought'] else '✅'} RSI Overbought (>{75}): "
                        f"{'Yes — Avoid buying' if result['rsi_overbought'] else 'No'}")
            st.markdown(f"{'⚠️' if result['rsi_oversold'] else 'ℹ️'} RSI Oversold (<35): "
                        f"{'Yes — Bounce possible' if result['rsi_oversold'] else 'No'}")
            st.markdown(f"\n**EMA Alignment:**")
            st.markdown(f"EMA20: ₹{result['ema20']:,.2f} | "
                        f"EMA50: ₹{result['ema50']:,.2f} | "
                        f"EMA200: ₹{result['ema200']:,.2f}")

        # ── Trade setup ──
        st.divider()
        st.markdown("### Suggested Trade Setup")
        setup = calculate_trade_setup(
            symbol=symbol_input,
            entry_price=result["price"],
            atr=result["atr"],
            capital=capital,
            risk_per_trade_pct=risk_per_trade,
            signal=result["signal"],
            score=result["score"],
        )

        ts1, ts2, ts3, ts4, ts5 = st.columns(5)
        ts1.metric("Entry Price", f"₹{setup.entry_price:,.2f}")
        ts2.metric("Stop-Loss", f"₹{setup.stop_loss:,.2f}",
                   delta=f"-{((setup.entry_price - setup.stop_loss)/setup.entry_price*100):.1f}%",
                   delta_color="inverse")
        ts3.metric("Target 1 (1:2)", f"₹{setup.target_1:,.2f}",
                   delta=f"+{((setup.target_1 - setup.entry_price)/setup.entry_price*100):.1f}%")
        ts4.metric("Target 2 (1:3)", f"₹{setup.target_2:,.2f}",
                   delta=f"+{((setup.target_2 - setup.entry_price)/setup.entry_price*100):.1f}%")
        ts5.metric("Quantity", f"{setup.quantity} shares")

        ts_a, ts_b, ts_c = st.columns(3)
        ts_a.metric("Capital at Risk", f"₹{setup.capital_at_risk:,.2f}", delta=f"{setup.risk_pct}% of capital")
        ts_b.metric("Potential Gain T1", f"₹{setup.potential_gain_1:,.2f}")
        ts_c.metric("Potential Gain T2", f"₹{setup.potential_gain_2:,.2f}")

        st.markdown("**Trade Notes:**")
        for note in setup.notes:
            st.markdown(f"- {note}")

    else:
        st.info("Enter an NSE symbol above and click **Analyse**.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PORTFOLIO TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("## Portfolio Tracker")
    st.caption("Track your open positions and monitor portfolio health.")

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

    # Add position form
    with st.expander("➕ Add New Position", expanded=len(st.session_state.portfolio) == 0):
        p1, p2, p3, p4, p5 = st.columns(5)
        p_symbol = p1.text_input("Symbol", key="p_sym", placeholder="e.g. TCS")
        p_sector = p2.selectbox("Sector", ["Banking", "IT", "FMCG", "Pharma", "Auto",
                                             "Energy", "Metals", "Infra", "NBFC",
                                             "Insurance", "Cement", "Consumer", "Other"], key="p_sec")
        p_entry = p3.number_input("Entry Price (₹)", min_value=1.0, value=100.0, key="p_entry")
        p_sl = p4.number_input("Stop-Loss (₹)", min_value=1.0, value=90.0, key="p_sl")
        p_qty = p5.number_input("Quantity", min_value=1, value=10, key="p_qty")

        if st.button("Add Position", key="add_pos"):
            if p_symbol and p_entry > p_sl:
                # Try to fetch current price
                current_price = p_entry
                try:
                    df_temp = fetch_ohlcv(f"{p_symbol.upper()}.NS", period="5d")
                    if df_temp is not None:
                        current_price = float(df_temp["Close"].iloc[-1])
                except Exception:
                    pass

                st.session_state.portfolio.append({
                    "symbol": p_symbol.upper(),
                    "sector": p_sector,
                    "entry": p_entry,
                    "current": current_price,
                    "quantity": int(p_qty),
                    "stop_loss": p_sl,
                })
                st.success(f"Added {p_symbol.upper()} to portfolio!")
                st.rerun()
            else:
                st.error("Entry must be above stop-loss.")

    if st.session_state.portfolio:
        # Build display DataFrame
        rows = []
        for pos in st.session_state.portfolio:
            invested = pos["entry"] * pos["quantity"]
            current_val = pos["current"] * pos["quantity"]
            pnl = current_val - invested
            pnl_pct = (pnl / invested) * 100 if invested > 0 else 0
            at_risk = (pos["entry"] - pos["stop_loss"]) * pos["quantity"]
            rows.append({
                "Symbol": pos["symbol"],
                "Sector": pos["sector"],
                "Entry (₹)": pos["entry"],
                "Current (₹)": pos["current"],
                "Qty": pos["quantity"],
                "Stop-Loss (₹)": pos["stop_loss"],
                "Invested (₹)": round(invested, 2),
                "P&L (₹)": round(pnl, 2),
                "P&L (%)": round(pnl_pct, 2),
                "At Risk (₹)": round(at_risk, 2),
            })

        port_df = pd.DataFrame(rows)

        def pnl_color(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return "color: #26a69a; font-weight: bold"
                elif val < 0:
                    return "color: #ef5350; font-weight: bold"
            return ""

        styled_port = port_df.style.map(pnl_color, subset=["P&L (₹)", "P&L (%)"])

        st.dataframe(styled_port, use_container_width=True)

        # Portfolio summary
        total_invested = sum(p["entry"] * p["quantity"] for p in st.session_state.portfolio)
        total_current = sum(p["current"] * p["quantity"] for p in st.session_state.portfolio)
        total_pnl = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("Total Invested", f"₹{total_invested:,.2f}")
        pm2.metric("Current Value", f"₹{total_current:,.2f}")
        pm3.metric("Total P&L", f"₹{total_pnl:+,.2f}", delta=f"{total_pnl_pct:+.1f}%")
        pm4.metric("Cash Remaining", f"₹{max(0, capital - total_invested):,.2f}")

        st.divider()

        # Portfolio health check
        health = portfolio_health_check(st.session_state.portfolio, capital)

        st.markdown("### Portfolio Health")
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("Open Positions", health["positions"])
        hc2.metric("Capital Deployed", f"{health['deployed_pct']}%")
        hc3.metric("Total Risk", f"{health['total_risk_pct']}%")

        if health["issues"]:
            for issue in health["issues"]:
                st.error(f"⛔ {issue}")
        if health["warnings"]:
            for warning in health["warnings"]:
                st.warning(f"⚠️ {warning}")
        if not health["issues"] and not health["warnings"]:
            st.success("✅ Portfolio health looks good!")

        # Sector exposure chart
        if health["sector_exposure"]:
            sector_exp = health["sector_exposure"]
            fig_exp = px.bar(
                x=list(sector_exp.keys()),
                y=list(sector_exp.values()),
                title="Sector Exposure (% of Capital)",
                labels={"x": "Sector", "y": "% of Capital"},
                color=list(sector_exp.values()),
                color_continuous_scale="RdYlGn_r",
            )
            fig_exp.add_hline(y=30, line_dash="dash", line_color="red",
                               annotation_text="30% limit")
            fig_exp.update_layout(height=300, coloraxis_showscale=False)
            st.plotly_chart(fig_exp, use_container_width=True)

        if st.button("Clear All Positions", type="secondary"):
            st.session_state.portfolio = []
            st.rerun()

    else:
        st.info("No positions tracked yet. Add a position using the form above.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RISK CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("## Risk Calculator")
    st.caption("Calculate position size and trade setup for any stock before you enter.")

    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("### Position Sizer")
        rc_entry = st.number_input("Entry Price (₹)", min_value=0.01, value=500.0, key="rc_entry")
        rc_stop = st.number_input("Stop-Loss Price (₹)", min_value=0.01, value=475.0, key="rc_stop")
        rc_risk = st.slider("Risk per Trade (%)", 0.5, 3.0, risk_per_trade, 0.25, key="rc_risk")

        if st.button("Calculate", key="rc_calc"):
            if rc_entry > rc_stop:
                result_rc = quick_position_size(capital, rc_entry, rc_stop, rc_risk)
                st.success("**Trade Setup:**")
                st.metric("Quantity to Buy", f"{result_rc['quantity']} shares")
                st.metric("Position Value", f"₹{result_rc['position_value']:,.2f}")
                st.metric("Capital at Risk", f"₹{result_rc['capital_at_risk']:,.2f} ({result_rc['risk_pct']}%)")
                st.metric("% of Capital", f"{result_rc['position_pct']}%")

                # Targets
                risk_per_share = rc_entry - rc_stop
                t1 = round(rc_entry + 2 * risk_per_share, 2)
                t2 = round(rc_entry + 3 * risk_per_share, 2)
                st.metric("Target 1 (1:2 R:R)", f"₹{t1:,.2f}", delta=f"+{((t1-rc_entry)/rc_entry*100):.1f}%")
                st.metric("Target 2 (1:3 R:R)", f"₹{t2:,.2f}", delta=f"+{((t2-rc_entry)/rc_entry*100):.1f}%")
            else:
                st.error("Entry price must be above stop-loss price.")

    with rc2:
        st.markdown("### Risk:Reward Visualizer")

        # Interactive scenario builder
        viz_entry = st.number_input("Entry", value=500.0, key="viz_entry")
        viz_stop = st.number_input("Stop", value=475.0, key="viz_stop")
        viz_target = st.number_input("Target (your goal)", value=550.0, key="viz_target")

        if viz_entry > viz_stop and viz_target > viz_entry:
            risk = viz_entry - viz_stop
            reward = viz_target - viz_entry
            rr_ratio = reward / risk if risk > 0 else 0
            qty_rr = quick_position_size(capital, viz_entry, viz_stop, risk_per_trade)

            fig_rr = go.Figure()
            fig_rr.add_hline(y=viz_stop, line_color="red", line_dash="dash",
                              annotation_text=f"Stop ₹{viz_stop}", annotation_position="right")
            fig_rr.add_hline(y=viz_entry, line_color="gray",
                              annotation_text=f"Entry ₹{viz_entry}", annotation_position="right")
            fig_rr.add_hline(y=viz_target, line_color="green", line_dash="dash",
                              annotation_text=f"Target ₹{viz_target}", annotation_position="right")

            fig_rr.add_vrect(x0=0, x1=1, y0=viz_stop, y1=viz_entry,
                             fillcolor="red", opacity=0.1, layer="below")
            fig_rr.add_vrect(x0=0, x1=1, y0=viz_entry, y1=viz_target,
                             fillcolor="green", opacity=0.1, layer="below")

            fig_rr.update_layout(
                title=f"Risk:Reward = 1:{rr_ratio:.1f}",
                yaxis_title="Price (₹)",
                height=350,
                showlegend=False,
                xaxis=dict(showticklabels=False),
            )
            st.plotly_chart(fig_rr, use_container_width=True)

            rr_color = "success" if rr_ratio >= 2 else "warning" if rr_ratio >= 1.5 else "error"
            if rr_color == "success":
                st.success(f"R:R = 1:{rr_ratio:.1f} ✅ Good trade — proceed with confidence.")
            elif rr_color == "warning":
                st.warning(f"R:R = 1:{rr_ratio:.1f} ⚠️ Marginal — try to widen target or tighten stop.")
            else:
                st.error(f"R:R = 1:{rr_ratio:.1f} ❌ Poor trade setup — skip this trade.")

    st.divider()
    st.markdown("### Quick Reference — Golden Rules")
    gr1, gr2, gr3, gr4 = st.columns(4)
    with gr1:
        st.info("**2% Rule**\nNever risk more than 2% of your capital on a single trade.")
    with gr2:
        st.info("**1:2 Minimum R:R**\nOnly enter if potential profit is at least 2× your risk.")
    with gr3:
        st.info("**Trail Your Stop**\nOnce up 1× ATR, move stop to break-even. Protect your capital.")
    with gr4:
        st.info("**Book 50% at T1**\nTake half the position off at Target 1. Let the rest run to T2.")
