"""
NSE Trading Dashboard — Enhanced v2
Run: streamlit run app.py
"""

import time
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from stocks_universe import ALL_STOCKS, NIFTY_50, NIFTY_NEXT_50, HIGH_MOMENTUM_MIDCAP, get_display_name
from data_fetcher import fetch_ohlcv, get_52_week_stats, get_ticker_info, clear_cache
from analyzer import analyze
from screener import run_screener, filter_by_strategy, get_top_buys, get_sector_breakdown, get_summary_stats
from risk_manager import calculate_trade_setup, quick_position_size, portfolio_health_check
from backtest import backtest_all_strategies, compute_equity_curve, compute_max_drawdown
from portfolio_store import load_portfolio, save_portfolio, add_position, remove_position
from alerts import send_bulk_alerts, send_test_message, send_price_alerts
from watchlist_store import (load_watchlist, save_watchlist, add_to_watchlist,
                              remove_from_watchlist, update_alert_levels, check_price_alerts)
from journal_store import (load_journal, add_trade, delete_trade,
                            get_journal_stats, to_dataframe)

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NSE Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header  { color: #666; font-size: 0.95rem; margin-bottom: 1.5rem; }
    div[data-testid="metric-container"] { background: #f8f9fa;
        border-radius: 8px; padding: 0.5rem 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    capital = st.number_input("Trading Capital (₹)", min_value=10_000,
                               max_value=10_000_000, value=100_000, step=10_000)
    risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 3.0, 2.0, 0.25)

    st.divider()

    universe_choice = st.selectbox("Stock Universe", options=[
        "NIFTY 50 (Safest)", "NIFTY 50 + Next 50", "Full Universe (Slower)"])
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
        f"**Stocks in scan:** {len(selected_universe)}"
    )
    st.caption("Data: Yahoo Finance · Refreshed every 15 min · Not SEBI-registered advice.")


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🏠 Dashboard",
    "🔍 Screener",
    "📊 Stock Analysis",
    "🕯️ Intraday",
    "🔬 Backtest",
    "💼 Portfolio",
    "🔔 Alerts",
    "⭐ Watchlist",
    "📓 Trade Journal",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="main-header">NSE Trading System v2</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">4 strategies · Backtest · Intraday charts · '
        'Telegram alerts · Persistent portfolio</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Capital", f"₹{capital:,.0f}")
    col2.metric("Risk/Trade", f"{risk_per_trade}% = ₹{capital * risk_per_trade / 100:,.0f}")
    col3.metric("Universe", f"{len(selected_universe)} stocks")
    col4.metric("Max Positions", "10 concurrent")

    st.divider()

    # NIFTY 50 benchmark
    st.markdown("### NIFTY 50 Benchmark")
    with st.spinner("Fetching NIFTY data..."):
        nifty_df = fetch_ohlcv("^NSEI", period="6mo")

    if nifty_df is not None and len(nifty_df) > 0:
        n = len(nifty_df)
        nifty_price = round(nifty_df["Close"].iloc[-1], 2)
        nifty_ret_1m = round((nifty_df["Close"].iloc[-1] / nifty_df["Close"].iloc[max(-22, -n)] - 1) * 100, 2)
        nifty_ret_3m = round((nifty_df["Close"].iloc[-1] / nifty_df["Close"].iloc[max(-63, -n)] - 1) * 100, 2)
        nb1, nb2, nb3 = st.columns(3)
        nb1.metric("NIFTY 50", f"{nifty_price:,.2f}")
        nb2.metric("1M Return", f"{nifty_ret_1m:+.1f}%")
        nb3.metric("3M Return", f"{nifty_ret_3m:+.1f}%")

        fig_nifty = go.Figure()
        fig_nifty.add_trace(go.Scatter(
            x=[str(d.date()) for d in nifty_df.tail(120).index],
            y=nifty_df["Close"].tail(120).tolist(),
            line=dict(color="#1f77b4", width=2),
            fill="tozeroy", fillcolor="rgba(31,119,180,0.08)",
            name="NIFTY 50"
        ))
        fig_nifty.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0),
                                  showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig_nifty, use_container_width=True)
    else:
        st.info("NIFTY benchmark unavailable — check internet connection.")

    st.divider()
    st.markdown("### Strategies at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Golden Cross**\nEMA50 > EMA200 + price > EMA20 + MACD bullish.\n*Hold: 4–12 weeks*")
    with c2:
        st.markdown("**MACD Momentum**\nFresh MACD crossover + RSI 40–68.\n*Hold: 2–6 weeks*")
    with c3:
        st.markdown("**Volume Breakout**\nPrice > 20D high + volume > 1.4× avg.\n*Hold: 1–4 weeks*")
    with c4:
        st.markdown("**Oversold Bounce**\nRSI < 38 + MACD turning up.\n*Hold: 1–3 weeks*")

    st.divider()
    r1, r2, r3 = st.columns(3)
    with r1:
        st.error("**Never risk > 2% per trade**\nSet stop-loss before entering.")
    with r2:
        st.warning("**Minimum 1:2 Risk:Reward**\nTarget must be ≥ 2× your risk.")
    with r3:
        st.success("**Book 50% at Target 1**\nLet rest ride with trailing stop.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCREENER
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## Stock Screener")
    st.caption("Batch-fetches all data then scores 0–100. Typical scan: 60–90 seconds.")

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        strategy_filter = st.selectbox("Filter by Strategy", [
            "All Strategies", "Golden Cross Trend", "MACD Momentum",
            "Volume Breakout", "Oversold Bounce"])
    with col_b:
        sector_options = ["All Sectors", "Banking", "IT", "FMCG", "Pharma", "Auto",
                          "Energy", "Metals", "Infra", "NBFC", "Insurance", "Cement", "Consumer"]
        sector_filter = st.selectbox("Filter by Sector", sector_options)
    with col_c:
        min_score = st.number_input("Min Score", 0, 100, 50)

    btn_col, clear_col, info_col = st.columns([1, 1, 4])
    with btn_col:
        run_btn = st.button("🔍 Run Screener", type="primary", use_container_width=True)
    with clear_col:
        if st.button("🗑️ Clear Cache", use_container_width=True,
                     help="If screener returns 0 stocks, click this then run again"):
            clear_cache()
            st.session_state.screener_df = pd.DataFrame()
            st.session_state.screener_summary = {}
            st.success("Cache cleared!")
    with info_col:
        st.info("If scan returns 0 stocks: Yahoo Finance may be rate-limiting. "
                "Wait 5 minutes, click **Clear Cache**, then run again.")

    if "screener_df" not in st.session_state:
        st.session_state.screener_df = pd.DataFrame()
        st.session_state.screener_summary = {}

    if run_btn:
        progress_bar = st.progress(0, text="Starting scan...")

        def update_progress(done, total):
            progress_bar.progress(done / total, text=f"Scanning {done}/{total}...")

        with st.spinner("Fetching and analysing..."):
            df_all = run_screener(selected_universe, progress_callback=update_progress)

        progress_bar.empty()
        st.session_state.screener_df = df_all
        st.session_state.screener_summary = get_summary_stats(df_all)

        # Auto-send Telegram alerts if configured
        if (st.session_state.get("tg_token") and st.session_state.get("tg_chat_id")
                and not df_all.empty):
            top = df_all.head(20).to_dict("records")
            sent, errs = send_bulk_alerts(
                st.session_state["tg_token"],
                st.session_state["tg_chat_id"],
                top,
                min_score=65,
            )
            if sent:
                st.success(f"📲 Sent {sent} Telegram alert(s)!")

        st.success(f"Screened {len(df_all)} stocks!")

    if not st.session_state.screener_df.empty:
        df_display = st.session_state.screener_df.copy()
        summary = st.session_state.screener_summary

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Screened", summary.get("total_screened", 0))
        m2.metric("Strong Buy", summary.get("strong_buy", 0))
        m3.metric("Buy", summary.get("buy", 0))
        m4.metric("Watch", summary.get("watch", 0))
        m5.metric("Avg Score", summary.get("avg_score", 0))

        st.divider()

        # Strategy distribution
        strat_data = {
            "Golden Cross": summary.get("golden_cross_count", 0),
            "MACD Momentum": summary.get("macd_momentum_count", 0),
            "Breakout": summary.get("breakout_count", 0),
            "Oversold Bounce": summary.get("oversold_bounce_count", 0),
        }
        fig_strat = px.bar(x=list(strat_data.keys()), y=list(strat_data.values()),
                           title="Stocks Triggering Each Strategy",
                           color=list(strat_data.values()), color_continuous_scale="Blues")
        fig_strat.update_layout(height=280, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_strat, use_container_width=True)

        # Filters
        if strategy_filter != "All Strategies":
            df_display = filter_by_strategy(df_display, strategy_filter)
        if sector_filter != "All Sectors":
            df_display = df_display[df_display["sector"] == sector_filter]
        df_display = df_display[df_display["score"] >= min_score]

        cols_show = ["name", "sector", "price", "score", "signal", "rsi", "adx",
                     "macd_bullish", "golden_cross", "breakout",
                     "ret_1m", "ret_3m", "vol_ratio"]
        cols_available = [c for c in cols_show if c in df_display.columns]
        renamed = {
            "name": "Stock", "sector": "Sector", "price": "Price (₹)",
            "score": "Score", "signal": "Signal", "rsi": "RSI",
            "adx": "ADX", "macd_bullish": "MACD↑", "golden_cross": "GoldenX",
            "breakout": "Breakout", "ret_1m": "1M %", "ret_3m": "3M %", "vol_ratio": "Vol Ratio",
        }

        res_col, csv_col = st.columns([4, 1])
        res_col.markdown(f"### Results — {len(df_display)} stocks")
        with csv_col:
            csv_bytes = df_display[cols_available].rename(columns=renamed).to_csv(index=False).encode()
            st.download_button(
                "⬇️ Export CSV", data=csv_bytes,
                file_name="nse_screener_results.csv", mime="text/csv",
                use_container_width=True,
            )

        def signal_color(val):
            colors = {
                "STRONG BUY": "background-color:#c8e6c9;color:#1b5e20;font-weight:bold",
                "BUY": "background-color:#dcedc8;color:#33691e",
                "WATCH": "background-color:#fff9c4;color:#f57f17",
                "NEUTRAL": "background-color:#f5f5f5;color:#616161",
                "AVOID": "background-color:#ffcdd2;color:#b71c1c",
            }
            return colors.get(val, "")

        display_df = df_display[cols_available].rename(columns=renamed)
        st.dataframe(display_df.style.map(signal_color, subset=["Signal"]),
                     use_container_width=True, height=420)

        # Sector pie
        buy_stocks = df_display[df_display["signal"].isin(["BUY", "STRONG BUY"])]
        if not buy_stocks.empty and "sector" in buy_stocks.columns:
            sector_counts = buy_stocks["sector"].value_counts()
            fig_sector = px.pie(values=sector_counts.values, names=sector_counts.index,
                                title="Sector Distribution of Buy Signals", hole=0.4)
            fig_sector.update_layout(height=320)
            st.plotly_chart(fig_sector, use_container_width=True)

    else:
        st.info("Click **Run Screener** to start.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STOCK ANALYSIS (Daily)
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## Stock Analysis — Daily")

    symbol_input = st.text_input("NSE Symbol (e.g. RELIANCE, TCS)", value="RELIANCE",
                                  key="daily_sym").strip().upper()
    show_fundamentals = st.checkbox("Show Fundamental Data (P/E, Market Cap)", value=True)
    analyse_btn = st.button("📊 Analyse", type="primary", key="analyse_daily")

    if analyse_btn and symbol_input:
        full_symbol = f"{symbol_input}.NS"
        with st.spinner(f"Fetching {symbol_input}..."):
            df_stock = fetch_ohlcv(full_symbol, period="1y", force_refresh=True)
            fund_info = get_ticker_info(full_symbol) if show_fundamentals else {}

        if df_stock is None:
            st.error(f"No data for **{symbol_input}**.")
        else:
            result = analyze(df_stock)
            stats = get_52_week_stats(df_stock)
            st.session_state[f"analysis_{symbol_input}"] = (result, stats, df_stock, fund_info)

    session_key = f"analysis_{symbol_input}"
    if session_key in st.session_state:
        result, stats, df_stock, fund_info = st.session_state[session_key]
        price = result["price"]

        sig_icons = {"STRONG BUY": "🟢", "BUY": "🟩", "WATCH": "🟡", "NEUTRAL": "⬜", "AVOID": "🔴"}
        st.markdown(f"### {symbol_input}  {sig_icons.get(result['signal'], '')} "
                    f"{result['signal']}  (Score: {result['score']}/100)")

        # Fundamentals row
        if fund_info:
            f1, f2, f3, f4 = st.columns(4)
            pe = fund_info.get("pe_ratio")
            mcap = fund_info.get("market_cap")
            beta = fund_info.get("beta")
            div = fund_info.get("dividend_yield")
            f1.metric("P/E Ratio", f"{pe:.1f}" if pe else "N/A")
            f2.metric("Market Cap", f"₹{mcap/1e12:.2f}T" if mcap and mcap > 1e12
                       else (f"₹{mcap/1e9:.0f}B" if mcap else "N/A"))
            f3.metric("Beta", f"{beta:.2f}" if beta else "N/A",
                      help="Beta > 1 = more volatile than NIFTY")
            f4.metric("Dividend Yield", f"{div*100:.1f}%" if div else "N/A")
            st.divider()

        # Technicals
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Price", f"₹{price:,.2f}")
        col2.metric("RSI (14)", result["rsi"],
                    delta="Overbought" if result["rsi_overbought"] else
                    ("Oversold" if result["rsi_oversold"] else None))
        col3.metric("ADX", result["adx"],
                    delta="Strong trend" if result["strong_trend"] else None)
        col4.metric("1M Return", f"{result['ret_1m']:+.1f}%")
        col5.metric("3M Return", f"{result['ret_3m']:+.1f}%")
        col6.metric("Vol Ratio", f"{result['vol_ratio']:.1f}x")

        # Relative strength vs NIFTY
        nifty_df = fetch_ohlcv("^NSEI", period="3mo")
        if nifty_df is not None and len(nifty_df) >= 63 and len(df_stock) >= 63:
            nifty_ret = (nifty_df["Close"].iloc[-1] / nifty_df["Close"].iloc[-63] - 1) * 100
            rs = result["ret_3m"] - nifty_ret
            col4.metric("RS vs NIFTY (3M)", f"{rs:+.1f}% vs NIFTY")

        st.divider()

        # 52-week
        wk1, wk2, wk3, wk4 = st.columns(4)
        wk1.metric("52W High", f"₹{stats['high_52w']:,.2f}")
        wk2.metric("52W Low", f"₹{stats['low_52w']:,.2f}")
        wk3.metric("Current", f"₹{stats['current']:,.2f}")
        wk4.metric("In 52W Range", f"{stats['position_pct']}%")

        st.divider()

        # Chart
        dates = result["dates_series"]
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.55, 0.25, 0.20],
                            vertical_spacing=0.03,
                            subplot_titles=["Price + EMA 20/50", "RSI (14)", "MACD Histogram"])

        df_tail = df_stock.tail(60)
        fig.add_trace(go.Candlestick(
            x=[str(d.date()) for d in df_tail.index],
            open=df_tail["Open"], high=df_tail["High"],
            low=df_tail["Low"], close=df_tail["Close"],
            name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=result["ema20_series"], name="EMA 20",
                                  line=dict(color="#1f77b4", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=result["ema50_series"], name="EMA 50",
                                  line=dict(color="#ff7f0e", width=1.5)), row=1, col=1)

        rsi_series = result["rsi_series"]
        fig.add_trace(go.Scatter(x=dates, y=rsi_series, name="RSI",
                                  line=dict(color="#9c27b0", width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

        macd_hist = result["macd_hist_series"]
        macd_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in macd_hist]
        fig.add_trace(go.Bar(x=dates, y=macd_hist, name="MACD Hist",
                              marker_color=macd_colors), row=3, col=1)

        fig.update_layout(height=680, xaxis_rangeslider_visible=False,
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        # Signals
        st.markdown("### Signal Summary")
        sig1, sig2 = st.columns(2)
        with sig1:
            st.markdown("**Bullish signals:**")
            for name, val in [
                ("Golden Cross (EMA50 > EMA200)", result["golden_cross"]),
                ("Price above EMA 200", result["price_above_ema200"]),
                ("MACD Bullish", result["macd_bullish"]),
                ("Fresh MACD Crossover", result["macd_crossover"]),
                ("Volume Breakout", result["breakout"]),
                ("Strong Trend (ADX > 25)", result["strong_trend"]),
                ("Bollinger Band Squeeze", result["bb_squeeze"]),
            ]:
                st.markdown(f"{'✅' if val else '❌'} {name}")
        with sig2:
            st.markdown("**EMA levels:**")
            st.markdown(f"EMA20: ₹{result['ema20']:,.2f}")
            st.markdown(f"EMA50: ₹{result['ema50']:,.2f}")
            st.markdown(f"EMA200: ₹{result['ema200']:,.2f}")
            st.markdown(f"\n**Caution:**")
            st.markdown(f"{'⚠️' if result['rsi_overbought'] else '✅'} "
                        f"RSI Overbought: {'Yes — skip entry' if result['rsi_overbought'] else 'No'}")

        # Trade setup
        st.divider()
        st.markdown("### Trade Setup")
        setup = calculate_trade_setup(symbol_input, price, result["atr"], capital,
                                       risk_per_trade, result["signal"], result["score"])
        ts1, ts2, ts3, ts4, ts5 = st.columns(5)
        ts1.metric("Entry", f"₹{setup.entry_price:,.2f}")
        ts2.metric("Stop-Loss", f"₹{setup.stop_loss:,.2f}",
                   delta=f"-{((setup.entry_price-setup.stop_loss)/setup.entry_price*100):.1f}%",
                   delta_color="inverse")
        ts3.metric("Target 1 (1:2)", f"₹{setup.target_1:,.2f}",
                   delta=f"+{((setup.target_1-setup.entry_price)/setup.entry_price*100):.1f}%")
        ts4.metric("Target 2 (1:3)", f"₹{setup.target_2:,.2f}",
                   delta=f"+{((setup.target_2-setup.entry_price)/setup.entry_price*100):.1f}%")
        ts5.metric("Qty", f"{setup.quantity} shares")

        ra, rb, rc = st.columns(3)
        ra.metric("Capital at Risk", f"₹{setup.capital_at_risk:,.2f}",
                  delta=f"{setup.risk_pct}% of capital")
        rb.metric("Gain at T1", f"₹{setup.potential_gain_1:,.2f}")
        rc.metric("Gain at T2", f"₹{setup.potential_gain_2:,.2f}")

        for note in setup.notes:
            st.markdown(f"- {note}")

    else:
        st.info("Enter a symbol and click **Analyse**.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — INTRADAY
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("## Intraday Chart")
    st.caption("1-hour candles for last 60 days. Use for same-day entry timing.")

    i_col1, i_col2 = st.columns([3, 1])
    with i_col1:
        intraday_sym = st.text_input("NSE Symbol", value="RELIANCE", key="intra_sym").strip().upper()
    with i_col2:
        intraday_tf = st.selectbox("Timeframe", ["1h", "30m", "15m"], index=0)

    intraday_btn = st.button("📈 Load Chart", type="primary", key="intra_btn")

    if intraday_btn and intraday_sym:
        full_sym = f"{intraday_sym}.NS"
        period_map = {"1h": "60d", "30m": "30d", "15m": "10d"}
        period = period_map[intraday_tf]

        with st.spinner(f"Fetching {intraday_tf} data for {intraday_sym}..."):
            df_intra = fetch_ohlcv(full_sym, period=period, interval=intraday_tf, force_refresh=True)

        if df_intra is None or df_intra.empty:
            st.error(f"No intraday data for {intraday_sym}. Note: 15m data only for last 10 days.")
        else:
            st.session_state[f"intra_{intraday_sym}_{intraday_tf}"] = df_intra

    intra_key = f"intra_{intraday_sym}_{intraday_tf}"
    if intra_key in st.session_state:
        df_intra = st.session_state[intra_key]
        result_intra = analyze(df_intra)

        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Current Price", f"₹{result_intra['price']:,.2f}")
        i2.metric("RSI", result_intra["rsi"],
                  delta="Overbought" if result_intra["rsi_overbought"] else
                  ("Oversold" if result_intra["rsi_oversold"] else None))
        i3.metric("Signal", result_intra["signal"])
        i4.metric("Score", f"{result_intra['score']}/100")

        st.divider()

        # Intraday candlestick + volume
        fig_i = make_subplots(rows=3, cols=1, shared_xaxes=True,
                              row_heights=[0.55, 0.25, 0.20],
                              vertical_spacing=0.03,
                              subplot_titles=["Price + EMA 20/50", "RSI", "Volume"])

        dates_i = [str(d) for d in df_intra.index]
        tail_df = df_intra.tail(120)
        tail_dates = [str(d) for d in tail_df.index]

        fig_i.add_trace(go.Candlestick(
            x=tail_dates,
            open=tail_df["Open"], high=tail_df["High"],
            low=tail_df["Low"], close=tail_df["Close"],
            name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ), row=1, col=1)

        ema20_i = result_intra["ema20_series"]
        ema50_i = result_intra["ema50_series"]
        dates_tail_60 = result_intra["dates_series"]
        fig_i.add_trace(go.Scatter(x=dates_tail_60, y=ema20_i, name="EMA 20",
                                    line=dict(color="#1f77b4", width=1.2)), row=1, col=1)
        fig_i.add_trace(go.Scatter(x=dates_tail_60, y=ema50_i, name="EMA 50",
                                    line=dict(color="#ff7f0e", width=1.2)), row=1, col=1)

        fig_i.add_trace(go.Scatter(x=dates_tail_60, y=result_intra["rsi_series"],
                                    line=dict(color="#9c27b0", width=1.5), name="RSI"), row=2, col=1)
        fig_i.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.4, row=2, col=1)
        fig_i.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.4, row=2, col=1)

        vol_s = result_intra["volume_series"]
        vol_ma_s = result_intra["vol_ma_series"]
        fig_i.add_trace(go.Bar(x=dates_tail_60, y=vol_s, name="Volume",
                                marker_color="#bbdefb"), row=3, col=1)
        fig_i.add_trace(go.Scatter(x=dates_tail_60, y=vol_ma_s, name="Vol MA",
                                    line=dict(color="#1565c0", width=1.5, dash="dash")), row=3, col=1)

        fig_i.update_layout(height=680, xaxis_rangeslider_visible=False,
                            plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_i, use_container_width=True)

        st.info(
            "**Intraday Trading Tips:**\n"
            "- Best entry times: 9:30–10:30 AM and 1:30–3:00 PM\n"
            "- Avoid entering in the first 15 minutes (9:15–9:30 AM) — too much volatility\n"
            "- Volume bar above MA = institutional activity. Below MA = thin, avoid\n"
            "- RSI crossing 50 upward on hourly chart = intraday bullish momentum"
        )

    else:
        st.info("Enter a symbol and click **Load Chart**.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("## Strategy Backtester")
    st.caption(
        "Walk-forward backtest on 2 years of historical data. "
        "Generates a signal on day N, measures return over next N hold days with stop-loss."
    )

    bt1, bt2, bt3 = st.columns(3)
    with bt1:
        bt_sym = st.text_input("NSE Symbol", value="TCS", key="bt_sym").strip().upper()
    with bt2:
        hold_days = st.slider("Hold Days (exit after N days)", 5, 30, 15)
    with bt3:
        sl_pct = st.slider("Stop-Loss %", 2.0, 10.0, 5.0, 0.5)

    bt_btn = st.button("🔬 Run Backtest", type="primary", key="bt_btn")

    if bt_btn and bt_sym:
        with st.spinner(f"Backtesting {bt_sym} — fetching 2 years of data..."):
            df_bt = fetch_ohlcv(f"{bt_sym}.NS", period="2y", force_refresh=True)

        if df_bt is None or len(df_bt) < 250:
            st.error(f"Insufficient data for {bt_sym}. Need at least 250 trading days.")
        else:
            results_bt = backtest_all_strategies(df_bt, hold_days=hold_days, stop_loss_pct=sl_pct)
            st.session_state[f"bt_{bt_sym}"] = (results_bt, df_bt, hold_days, sl_pct)

    bt_key = f"bt_{bt_sym}"
    if bt_key in st.session_state:
        results_bt, df_bt, bt_hold_days, bt_sl_pct = st.session_state[bt_key]

        st.markdown(f"### {bt_sym} — Backtest Results (hold {bt_hold_days}d, SL {bt_sl_pct}%)")

        # Summary table
        summary_rows = []
        for r in results_bt:
            summary_rows.append({
                "Strategy": r["strategy"],
                "Trades": r["trades"],
                "Win Rate": f"{r['win_rate']}%",
                "Avg Return": f"{r['avg_return']:+.2f}%",
                "Max Gain": f"{r['max_gain']:+.2f}%",
                "Max Loss": f"{r['max_loss']:+.2f}%",
                "Profit Factor": r["profit_factor"],
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        st.divider()

        # Equity curves
        st.markdown("### Equity Curves (₹1,00,000 starting capital)")
        fig_eq = go.Figure()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, r in enumerate(results_bt):
            if r["trades"] > 0:
                equity = compute_equity_curve(r["return_series"], 100_000)
                max_dd = compute_max_drawdown(equity)
                label = f"{r['strategy']} (WR:{r['win_rate']}% DD:{max_dd}%)"
                fig_eq.add_trace(go.Scatter(
                    y=equity, name=label,
                    line=dict(color=colors[i], width=2)
                ))
        fig_eq.add_hline(y=100_000, line_dash="dash", line_color="gray", opacity=0.4)
        fig_eq.update_layout(height=380, plot_bgcolor="white",
                              yaxis_title="Portfolio Value (₹)", xaxis_title="Trade #")
        st.plotly_chart(fig_eq, use_container_width=True)

        # Detailed trade list for best strategy
        best = max(results_bt, key=lambda r: r.get("win_rate", 0))
        if best["trades"] > 0:
            st.markdown(f"### Trade Log — {best['strategy']}")
            trade_df = pd.DataFrame(best["trade_list"])

            def ret_color(val):
                if isinstance(val, (int, float)):
                    return "color:#26a69a;font-weight:bold" if val > 0 else "color:#ef5350;font-weight:bold"
                return ""

            st.dataframe(
                trade_df.style.map(ret_color, subset=["return_pct"]),
                use_container_width=True,
                height=300,
            )

        st.info(
            "**How to read this:**\n"
            "- Win Rate > 50% with Profit Factor > 1.5 = strategy is working on this stock\n"
            "- Check Max Loss — if > 2× Avg Return, the strategy is risky\n"
            "- Equity curve going up-right = strategy profitable over time\n"
            "- Max Drawdown % = worst peak-to-trough loss — keep this < 20%"
        )

    else:
        st.info("Enter a symbol and click **Run Backtest**.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown("## Portfolio Tracker")
    st.caption("Positions are saved to disk — survive page refreshes.")

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = load_portfolio()

    with st.expander("➕ Add Position", expanded=len(st.session_state.portfolio) == 0):
        p1, p2, p3, p4, p5 = st.columns(5)
        p_symbol = p1.text_input("Symbol", key="p_sym", placeholder="e.g. TCS")
        p_sector = p2.selectbox("Sector", ["Banking", "IT", "FMCG", "Pharma", "Auto",
                                             "Energy", "Metals", "Infra", "NBFC",
                                             "Insurance", "Cement", "Consumer", "Other"], key="p_sec")
        p_entry = p3.number_input("Entry (₹)", min_value=1.0, value=100.0, key="p_entry")
        p_sl = p4.number_input("Stop-Loss (₹)", min_value=1.0, value=90.0, key="p_sl")
        p_qty = p5.number_input("Qty", min_value=1, value=10, key="p_qty")

        if st.button("Add Position", key="add_pos"):
            if p_symbol and p_entry > p_sl:
                current_price = p_entry
                try:
                    df_tmp = fetch_ohlcv(f"{p_symbol.upper()}.NS", period="5d")
                    if df_tmp is not None:
                        current_price = float(df_tmp["Close"].iloc[-1])
                except Exception:
                    pass

                new_pos = {
                    "symbol": p_symbol.upper(), "sector": p_sector,
                    "entry": p_entry, "current": current_price,
                    "quantity": int(p_qty), "stop_loss": p_sl,
                }
                st.session_state.portfolio = add_position(st.session_state.portfolio, new_pos)
                st.success(f"Added {p_symbol.upper()}!")
                st.rerun()
            else:
                st.error("Entry must be above stop-loss.")

    if st.session_state.portfolio:
        # Refresh live prices
        refresh_port = st.button("🔄 Refresh Live Prices", type="primary", key="port_refresh")
        if refresh_port:
            with st.spinner("Fetching latest prices..."):
                for pos in st.session_state.portfolio:
                    sym = pos["symbol"] if pos["symbol"].endswith(".NS") else f"{pos['symbol']}.NS"
                    try:
                        df_tmp = fetch_ohlcv(sym, period="5d", force_refresh=True)
                        if df_tmp is not None and not df_tmp.empty:
                            pos["current"] = round(float(df_tmp["Close"].iloc[-1]), 2)
                    except Exception:
                        pass
            from portfolio_store import save_portfolio
            save_portfolio(st.session_state.portfolio)
            st.success("Prices updated!")

        # Build rows
        today = __import__("datetime").date.today()
        rows = []
        for pos in st.session_state.portfolio:
            invested = pos["entry"] * pos["quantity"]
            current_val = pos["current"] * pos["quantity"]
            pnl = current_val - invested
            pnl_pct = (pnl / invested * 100) if invested > 0 else 0
            at_risk = (pos["entry"] - pos["stop_loss"]) * pos["quantity"]
            entry_date = pos.get("entry_date", "—")
            try:
                hold_days_val = (today - __import__("datetime").date.fromisoformat(entry_date)).days
            except Exception:
                hold_days_val = "—"
            rows.append({
                "Symbol": pos["symbol"], "Sector": pos["sector"],
                "Entry Date": entry_date, "Hold Days": hold_days_val,
                "Entry (₹)": pos["entry"], "Current (₹)": pos["current"],
                "Qty": pos["quantity"], "Stop-Loss (₹)": pos["stop_loss"],
                "Invested (₹)": round(invested, 2),
                "P&L (₹)": round(pnl, 2), "P&L (%)": round(pnl_pct, 2),
                "At Risk (₹)": round(at_risk, 2),
            })

        port_df = pd.DataFrame(rows)

        def pnl_color(val):
            if isinstance(val, (int, float)):
                if val > 0: return "color:#26a69a;font-weight:bold"
                elif val < 0: return "color:#ef5350;font-weight:bold"
            return ""

        st.dataframe(
            port_df.style.map(pnl_color, subset=["P&L (₹)", "P&L (%)"]),
            use_container_width=True,
        )

        # Summary
        total_invested = sum(p["entry"] * p["quantity"] for p in st.session_state.portfolio)
        total_current = sum(p["current"] * p["quantity"] for p in st.session_state.portfolio)
        total_pnl = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("Invested", f"₹{total_invested:,.2f}")
        pm2.metric("Current Value", f"₹{total_current:,.2f}")
        pm3.metric("Total P&L", f"₹{total_pnl:+,.2f}", delta=f"{total_pnl_pct:+.1f}%")
        pm4.metric("Cash Remaining", f"₹{max(0, capital - total_invested):,.2f}")

        st.divider()

        # Health check
        health = portfolio_health_check(st.session_state.portfolio, capital)
        st.markdown("### Portfolio Health")
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("Open Positions", health["positions"])
        hc2.metric("Capital Deployed", f"{health['deployed_pct']}%")
        hc3.metric("Total Risk", f"{health['total_risk_pct']}%")
        for issue in health["issues"]:
            st.error(f"⛔ {issue}")
        for warning in health["warnings"]:
            st.warning(f"⚠️ {warning}")
        if not health["issues"] and not health["warnings"]:
            st.success("✅ Portfolio health is good!")

        # Remove position
        st.divider()
        remove_sym = st.selectbox("Remove Position",
                                   ["— select —"] + [p["symbol"] for p in st.session_state.portfolio])
        if st.button("Remove", type="secondary") and remove_sym != "— select —":
            st.session_state.portfolio = remove_position(st.session_state.portfolio, remove_sym)
            st.rerun()

    else:
        st.info("No positions yet. Add one above.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ALERTS (Telegram)
# ═══════════════════════════════════════════════════════════════════════════════

with tab7:
    st.markdown("## Telegram Alerts")
    st.caption("Get buy signal notifications on your phone after each screener run.")

    st.markdown("### Setup (one time)")
    st.markdown("""
1. Open Telegram → search **@BotFather** → send `/newbot` → follow steps → copy **Bot Token**
2. Open your new bot, send `/start`
3. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` to find your **Chat ID**
    """)

    tg1, tg2 = st.columns(2)
    with tg1:
        tg_token = st.text_input("Bot Token", type="password",
                                  value=st.session_state.get("tg_token", ""),
                                  placeholder="123456789:ABCdef...")
    with tg2:
        tg_chat_id = st.text_input("Chat ID",
                                    value=st.session_state.get("tg_chat_id", ""),
                                    placeholder="e.g. 987654321")

    col_save, col_test = st.columns(2)
    with col_save:
        if st.button("💾 Save", type="primary"):
            st.session_state["tg_token"] = tg_token
            st.session_state["tg_chat_id"] = tg_chat_id
            st.success("Saved! Alerts will fire automatically after each screener run.")

    with col_test:
        if st.button("📲 Send Test Message"):
            if tg_token and tg_chat_id:
                ok, msg = send_test_message(tg_token, tg_chat_id)
                if ok:
                    st.success(f"✅ {msg} Check your Telegram!")
                else:
                    st.error(f"❌ {msg}")
            else:
                st.error("Enter both Bot Token and Chat ID first.")

    st.divider()
    st.markdown("### Manual Alert — Send current screener results")

    alert_min_score = st.slider("Minimum score to alert", 50, 90, 65, key="alert_score")

    if st.button("📢 Send Alerts Now"):
        if not tg_token or not tg_chat_id:
            st.error("Configure Telegram credentials above first.")
        elif st.session_state.screener_df.empty:
            st.warning("No screener results. Run the Screener tab first.")
        else:
            top = st.session_state.screener_df.head(20).to_dict("records")
            sent, errs = send_bulk_alerts(tg_token, tg_chat_id, top, min_score=alert_min_score)
            if sent:
                st.success(f"✅ Sent {sent} alert(s) to Telegram!")
            if errs:
                for e in errs:
                    st.warning(f"⚠️ {e}")
            if sent == 0 and not errs:
                st.info(f"No stocks scored ≥ {alert_min_score}. Lower the minimum or run screener first.")

    st.divider()
    st.markdown("### When alerts are sent automatically")
    st.info(
        "Alerts are sent automatically every time you click **Run Screener** "
        "in the Screener tab, for all stocks with score ≥ 65 and signal = BUY/STRONG BUY."
    )

    st.markdown("### Alert message preview")
    st.code("""
🟢 NSE Buy Signal — RELIANCE

Signal: STRONG BUY  |  Score: 82/100
Sector: Energy

Price:  ₹2,845.00
RSI:    54.3
ADX:    31.2
Vol Ratio: 1.8x

1M Return: +4.2%  |  3M: +11.8%

Strategies triggered:
✅ Golden Cross
✅ MACD Momentum
❌ Volume Breakout
❌ Oversold Bounce

Entry: ₹2,845 | Stop: ₹2,738
Not financial advice. DYOR.
    """, language="text")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — WATCHLIST
# ═══════════════════════════════════════════════════════════════════════════════

with tab8:
    st.markdown("## Watchlist")
    st.caption(
        "Pin your favourite stocks. Set price alerts — get a Telegram notification "
        "the moment price crosses your level."
    )

    # Init session state
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = load_watchlist()

    # ── Add stock ──
    with st.expander("➕ Add Stock to Watchlist", expanded=len(st.session_state.watchlist) == 0):
        wa1, wa2, wa3, wa4 = st.columns(4)
        wl_sym = wa1.text_input("Symbol", placeholder="e.g. INFY", key="wl_add_sym").strip().upper()
        wl_above = wa2.number_input("Alert if price goes ABOVE (₹)", min_value=0.0,
                                     value=0.0, key="wl_above",
                                     help="Leave 0 to skip this alert")
        wl_below = wa3.number_input("Alert if price goes BELOW (₹)", min_value=0.0,
                                     value=0.0, key="wl_below",
                                     help="Leave 0 to skip this alert")
        wl_notes = wa4.text_input("Notes", placeholder="Why you're watching this", key="wl_notes")

        if st.button("Add to Watchlist", key="wl_add_btn"):
            if wl_sym:
                st.session_state.watchlist = add_to_watchlist(
                    st.session_state.watchlist,
                    wl_sym,
                    alert_above=wl_above if wl_above > 0 else None,
                    alert_below=wl_below if wl_below > 0 else None,
                    notes=wl_notes,
                )
                st.success(f"Added {wl_sym} to watchlist!")
                st.rerun()
            else:
                st.error("Enter a symbol.")

    if st.session_state.watchlist:
        # ── Refresh signals ──
        refresh_btn = st.button("🔄 Refresh All Signals", type="primary", key="wl_refresh")

        if refresh_btn or "wl_data" not in st.session_state:
            symbols = [e["symbol"] for e in st.session_state.watchlist]
            with st.spinner("Fetching live data for watchlist..."):
                wl_results = []
                price_map = {}
                for sym in symbols:
                    df_wl = fetch_ohlcv(sym, period="3mo", force_refresh=refresh_btn)
                    if df_wl is not None:
                        res = analyze(df_wl)
                        price_map[sym] = res["price"]
                        entry = next((e for e in st.session_state.watchlist if e["symbol"] == sym), {})
                        wl_results.append({
                            "Symbol": entry.get("name", sym.replace(".NS", "")),
                            "Price (₹)": res["price"],
                            "Signal": res["signal"],
                            "Score": res["score"],
                            "RSI": res["rsi"],
                            "ADX": res["adx"],
                            "1M %": f"{res['ret_1m']:+.1f}%",
                            "3M %": f"{res['ret_3m']:+.1f}%",
                            "Vol Ratio": res["vol_ratio"],
                            "Alert Above": entry.get("alert_above", "—") or "—",
                            "Alert Below": entry.get("alert_below", "—") or "—",
                            "Notes": entry.get("notes", ""),
                        })

            st.session_state.wl_data = wl_results

            # ── Check price alerts ──
            triggered = check_price_alerts(st.session_state.watchlist, price_map)
            if triggered:
                tg_tok = st.session_state.get("tg_token")
                tg_cid = st.session_state.get("tg_chat_id")
                for alert in triggered:
                    direction = "above" if alert["alert_type"] == "ABOVE" else "below"
                    st.warning(
                        f"🔔 **Price Alert:** {alert['name']} is now ₹{alert['price']:,.2f} "
                        f"({direction} your ₹{alert['level']:,.2f} level)"
                    )
                if tg_tok and tg_cid:
                    sent, _ = send_price_alerts(tg_tok, tg_cid, triggered)
                    if sent:
                        st.success(f"📲 Sent {sent} price alert(s) to Telegram!")

        # ── Display table ──
        if st.session_state.get("wl_data"):
            wl_df = pd.DataFrame(st.session_state.wl_data)

            def wl_signal_color(val):
                colors = {
                    "STRONG BUY": "background-color:#c8e6c9;color:#1b5e20;font-weight:bold",
                    "BUY": "background-color:#dcedc8;color:#33691e",
                    "WATCH": "background-color:#fff9c4;color:#f57f17",
                    "NEUTRAL": "background-color:#f5f5f5;color:#616161",
                    "AVOID": "background-color:#ffcdd2;color:#b71c1c",
                }
                return colors.get(val, "")

            st.dataframe(
                wl_df.style.map(wl_signal_color, subset=["Signal"]),
                use_container_width=True,
                height=max(200, len(wl_df) * 38 + 40),
            )

        st.divider()

        # ── Edit alert levels ──
        st.markdown("### Edit Alert Levels")
        edit_sym = st.selectbox(
            "Select stock to edit",
            ["— select —"] + [e["name"] for e in st.session_state.watchlist],
            key="wl_edit_select"
        )

        if edit_sym != "— select —":
            entry = next((e for e in st.session_state.watchlist if e["name"] == edit_sym), None)
            if entry:
                ea1, ea2, ea3 = st.columns(3)
                new_above = ea1.number_input(
                    "Alert Above (₹)", value=entry.get("alert_above") or 0.0, key="wl_edit_above")
                new_below = ea2.number_input(
                    "Alert Below (₹)", value=entry.get("alert_below") or 0.0, key="wl_edit_below")
                new_notes = ea3.text_input("Notes", value=entry.get("notes", ""), key="wl_edit_notes")

                ec1, ec2 = st.columns(2)
                if ec1.button("Update Alerts", key="wl_update_btn"):
                    st.session_state.watchlist = update_alert_levels(
                        st.session_state.watchlist,
                        edit_sym,
                        new_above if new_above > 0 else None,
                        new_below if new_below > 0 else None,
                        new_notes,
                    )
                    st.session_state.pop("wl_data", None)
                    st.success(f"Updated alerts for {edit_sym}!")
                    st.rerun()

                if ec2.button("Remove from Watchlist", type="secondary", key="wl_remove_btn"):
                    st.session_state.watchlist = remove_from_watchlist(
                        st.session_state.watchlist, edit_sym)
                    st.session_state.pop("wl_data", None)
                    st.rerun()

    else:
        st.info("Your watchlist is empty. Add stocks above to start tracking them.")

    st.divider()
    st.info(
        "**How price alerts work:**\n"
        "- Set an ABOVE level (e.g. ₹3,100) to be alerted when the stock breaks out\n"
        "- Set a BELOW level (e.g. ₹2,500) to be alerted on a major dip / stop hit\n"
        "- Alerts fire once per session — click Refresh to re-arm them after they trigger\n"
        "- Telegram must be configured in the Alerts tab to receive notifications"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 — TRADE JOURNAL
# ═══════════════════════════════════════════════════════════════════════════════

with tab9:
    st.markdown("## Trade Journal")
    st.caption("Log every trade. Track your win rate, mistakes, and learnings over time.")

    if "journal" not in st.session_state:
        st.session_state.journal = load_journal()

    # ── Log new trade ──
    with st.expander("📝 Log a New Trade", expanded=len(st.session_state.journal) == 0):
        jc1, jc2, jc3 = st.columns(3)
        j_sym = jc1.text_input("Symbol", placeholder="e.g. TCS", key="j_sym").strip().upper()
        j_dir = jc2.selectbox("Direction", ["BUY", "SHORT"], key="j_dir")
        j_strat = jc3.selectbox("Strategy Used", [
            "Golden Cross Trend", "MACD Momentum", "Volume Breakout",
            "Oversold Bounce", "Manual / Other"], key="j_strat")

        jd1, jd2, jd3, jd4, jd5 = st.columns(5)
        j_entry = jd1.number_input("Entry (₹)", min_value=0.01, value=100.0, key="j_entry")
        j_exit = jd2.number_input("Exit (₹)", min_value=0.01, value=110.0, key="j_exit")
        j_sl = jd3.number_input("Stop-Loss (₹)", min_value=0.01, value=92.0, key="j_sl")
        j_t1 = jd4.number_input("Target 1 (₹)", min_value=0.01, value=116.0, key="j_t1")
        j_qty = jd5.number_input("Quantity", min_value=1, value=100, key="j_qty")

        import datetime as _dt
        jdate1, jdate2 = st.columns(2)
        j_date_in = str(jdate1.date_input("Date Entered", value=_dt.date.today(), key="j_date_in"))
        j_date_out = str(jdate2.date_input("Date Exited", value=_dt.date.today(), key="j_date_out"))

        j_exit_reason = st.selectbox("Exit Reason", [
            "Hit Target 1", "Hit Target 2", "Hit Stop-Loss", "Trailing Stop",
            "Manual Exit — Market Weakness", "Manual Exit — Better Opportunity", "Other"
        ], key="j_exit_reason")

        j_mistakes = st.text_area("Mistakes Made (be honest)", placeholder="e.g. Entered too early, didn't wait for volume confirmation", key="j_mistakes", height=80)
        j_learnings = st.text_area("Learnings / What to do next time", placeholder="e.g. Wait for RSI to confirm above 50 before entering", key="j_learnings", height=80)

        if st.button("Log Trade", type="primary", key="j_log_btn"):
            if j_sym and j_entry > 0 and j_exit > 0:
                pnl_preview = (j_exit - j_entry) * j_qty if j_dir == "BUY" else (j_entry - j_exit) * j_qty
                outcome = "WIN" if pnl_preview > 0 else ("LOSS" if pnl_preview < 0 else "BREAKEVEN")
                st.session_state.journal = add_trade(
                    st.session_state.journal,
                    symbol=j_sym, direction=j_dir,
                    entry_price=j_entry, exit_price=j_exit,
                    stop_loss=j_sl, target_1=j_t1, quantity=int(j_qty),
                    date_entered=j_date_in, date_exited=j_date_out,
                    strategy=j_strat, exit_reason=j_exit_reason,
                    mistakes=j_mistakes, learnings=j_learnings,
                )
                icon = "✅" if outcome == "WIN" else "❌"
                st.success(f"{icon} Trade logged! P&L: ₹{pnl_preview:+,.2f}")
                st.rerun()
            else:
                st.error("Fill in all required fields.")

    # ── Statistics ──
    if st.session_state.journal:
        stats = get_journal_stats(st.session_state.journal)

        st.markdown("### Your Trading Statistics")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Trades", stats["total_trades"])
        s2.metric("Win Rate", f"{stats['win_rate']}%",
                  delta="Good" if stats["win_rate"] > 50 else "Needs work")
        s3.metric("Total P&L", f"₹{stats['total_pnl']:+,.2f}",
                  delta_color="normal" if stats["total_pnl"] > 0 else "inverse")
        s4.metric("Profit Factor", stats["profit_factor"],
                  delta="Profitable" if stats["profit_factor"] > 1 else "Losing")
        s5.metric("Expectancy / Trade", f"₹{stats['expectancy']:+,.2f}")

        st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Win", f"{stats['avg_win_pct']:+.2f}%")
        m2.metric("Avg Loss", f"{stats['avg_loss_pct']:+.2f}%")
        m3.metric("Best Trade", f"₹{stats['best_trade']:+,.2f}")
        m4.metric("Worst Trade", f"₹{stats['worst_trade']:+,.2f}")

        # Win/Loss pie
        pie1, pie2 = st.columns(2)
        with pie1:
            fig_wl = px.pie(
                values=[stats["wins"], stats["losses"],
                        stats["total_trades"] - stats["wins"] - stats["losses"]],
                names=["Wins", "Losses", "Breakeven"],
                title="Win / Loss Ratio",
                color_discrete_map={"Wins": "#26a69a", "Losses": "#ef5350", "Breakeven": "#bdbdbd"},
                hole=0.4,
            )
            fig_wl.update_layout(height=280)
            st.plotly_chart(fig_wl, use_container_width=True)

        # Strategy P&L bar chart
        with pie2:
            if stats.get("strategy_pnl"):
                sp = stats["strategy_pnl"]
                fig_sp = px.bar(
                    x=list(sp.keys()), y=list(sp.values()),
                    title="P&L by Strategy (₹)",
                    color=list(sp.values()),
                    color_continuous_scale="RdYlGn",
                )
                fig_sp.update_layout(height=280, coloraxis_showscale=False)
                st.plotly_chart(fig_sp, use_container_width=True)

        # Cumulative P&L line
        journal_sorted = sorted(st.session_state.journal, key=lambda x: x["date_exited"])
        cum_pnl = []
        running = 0
        dates_j = []
        for t in journal_sorted:
            running += t["pnl"]
            cum_pnl.append(round(running, 2))
            dates_j.append(t["date_exited"])

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=dates_j, y=cum_pnl,
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.08)",
            name="Cumulative P&L"
        ))
        fig_cum.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_cum.update_layout(
            title="Cumulative P&L Over Time",
            yaxis_title="P&L (₹)", height=280,
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        st.divider()

        # ── Trade log table ──
        st.markdown("### Trade Log")
        journal_df = to_dataframe(st.session_state.journal)

        def pnl_color_j(val):
            if isinstance(val, (int, float)):
                if val > 0: return "color:#26a69a;font-weight:bold"
                elif val < 0: return "color:#ef5350;font-weight:bold"
            return ""

        def outcome_color(val):
            colors = {
                "WIN": "background-color:#c8e6c9;color:#1b5e20;font-weight:bold",
                "LOSS": "background-color:#ffcdd2;color:#b71c1c;font-weight:bold",
                "BREAKEVEN": "background-color:#f5f5f5;color:#616161",
            }
            return colors.get(val, "")

        st.dataframe(
            journal_df.style
                .map(pnl_color_j, subset=["pnl", "pnl_pct"])
                .map(outcome_color, subset=["outcome"]),
            use_container_width=True,
            height=350,
        )

        # ── Learnings feed ──
        st.divider()
        st.markdown("### Mistakes & Learnings Feed")
        st.caption("Your personal trading knowledge base — built from every trade.")

        for trade in reversed(journal_sorted):
            if trade.get("mistakes") or trade.get("learnings"):
                outcome_icon = "✅" if trade["outcome"] == "WIN" else "❌"
                with st.expander(
                    f"{outcome_icon} {trade['symbol']} · {trade['date_exited']} · "
                    f"₹{trade['pnl']:+,.2f} ({trade['pnl_pct']:+.1f}%)"
                ):
                    if trade.get("mistakes"):
                        st.markdown(f"**Mistakes:** {trade['mistakes']}")
                    if trade.get("learnings"):
                        st.markdown(f"**Learnings:** {trade['learnings']}")
                    st.caption(f"Strategy: {trade.get('strategy', 'N/A')} · Exit: {trade.get('exit_reason', 'N/A')}")

        # ── Delete trade ──
        st.divider()
        del_options = [f"{t['symbol']} ({t['date_entered']}) — ₹{t['pnl']:+,.0f}"
                       for t in st.session_state.journal]
        del_select = st.selectbox("Delete a trade", ["— select —"] + del_options, key="j_del")
        if st.button("Delete Trade", type="secondary", key="j_del_btn") and del_select != "— select —":
            idx = del_options.index(del_select)
            trade_id = st.session_state.journal[idx]["id"]
            st.session_state.journal = delete_trade(st.session_state.journal, trade_id)
            st.rerun()

    else:
        st.info("No trades logged yet. Log your first trade above — including the ones that lost money. That's where the real learning is.")
