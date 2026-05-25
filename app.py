import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
from google import genai

st.set_page_config(page_title="Wall Street AI Dashboard", layout="centered")

st.title("📊 Wall Street AI Scanner & Wealth Dashboard")
st.write("Track custom holdings, calculate advanced trend crossovers, and generate AI insights.")

# API Key input
api_key = st.text_input("🔑 Enter Gemini API Key", type="password")

# Initialize portfolio
if "user_portfolio" not in st.session_state:
    st.session_state.user_portfolio = {}

# Constants
TICKER_POOL = ["PLSE", "BRCB", "CELH", "SOFI", "HOOD", "DKNG"]
PORTFOLIO_FILE = "portfolio.json"

INTERVAL_MAPPING = {
    10: {"history": "3mo", "description": "Short-term momentum & micro-trends"},
    20: {"history": "6mo", "description": "Standard short-term trend-following boundary"},
    50: {"history": "1y", "description": "Medium-term structural support / Institutional tracking"},
    100: {"history": "2y", "description": "Macro cycle support and accumulation zone"},
    200: {"history": "2y", "description": "Long-term macro baseline (The ultimate bull/bear line)"}
}

# --- FUNCTIONS ---
def fetch_technical_data(ticker_symbol, period_window, calculation_type):
    try:
        clean_symbol = ticker_symbol.upper().strip()
        COMMON_CRYPTOS = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT", "LTC"]
        if clean_symbol in COMMON_CRYPTOS:
            clean_symbol = f"{clean_symbol}-USD"
            
        ticker = yf.Ticker(clean_symbol)
        lookback = INTERVAL_MAPPING[period_window]["history"]
        hist = ticker.history(period=lookback)
        
        if hist.empty or len(hist) < period_window:
            return False, {}, None, 0.0
        
        if "Simple" in calculation_type:
            hist['MA_Line'] = hist['Close'].rolling(window=period_window).mean()
            ma_acronym = "SMA"
        else:
            linear_weights = np.arange(1, period_window + 1)
            hist['MA_Line'] = hist['Close'].rolling(window=period_window).apply(
                lambda prices: np.dot(prices, linear_weights) / linear_weights.sum(), raw=True
            )
            ma_acronym = "WMA"
            
        hist['Prev_Close'] = hist['Close'].shift(1)
        hist['Prev_MA'] = hist['MA_Line'].shift(1)
        
        buy_condition = (hist['Close'] > hist['MA_Line']) & (hist['Prev_Close'] <= hist['Prev_MA'])
        sell_condition = (hist['Close'] < hist['MA_Line']) & (hist['Prev_Close'] >= hist['Prev_MA'])
        
        hist['Buy_Signal'] = np.where(buy_condition, hist['Close'], np.nan)
        hist['Sell_Signal'] = np.where(sell_condition, hist['Close'], np.nan)
        
        current_price = hist['Close'].iloc[-1]
        prior_price = hist['Close'].iloc[-20] if len(hist) >= 20 else hist['Close'].iloc[0]
        one_month_change = ((current_price - prior_price) / prior_price) * 100
        current_ma_val = hist['MA_Line'].iloc[-1]
        
        latest_signal = "Neutral (Consolidating)"
        recent_buys = hist['Buy_Signal'].dropna()
        recent_sells = hist['Sell_Signal'].dropna()
        
        if not recent_buys.empty and (recent_sells.empty or recent_buys.index[-1] > recent_sells.index[-1]):
            latest_signal = f"BUY Trigger ({recent_buys.index[-1].strftime('%m/%d')})"
        elif not recent_sells.empty and (recent_buys.empty or recent_sells.index[-1] > recent_buys.index[-1]):
            latest_signal = f"SELL Trigger ({recent_sells.index[-1].strftime('%m/%d')})"

        metrics = {
            "Price": f"${current_price:.2f}",
            "1-Mo Momentum": f"{one_month_change:.1f}%",
            f"{period_window}-Day {ma_acronym}": f"${current_ma_val:.2f}",
            "Current State": latest_signal
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='#1f77b4', width=2)))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA_Line'], name=ma_acronym, line=dict(color='#ff7f0e', width=1.5, dash='dot')))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Buy_Signal'], mode='markers', name='BUY', marker=dict(color='#2ca02c', size=10, symbol='triangle-up')))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Sell_Signal'], mode='markers', name='SELL', marker=dict(color='#d62728', size=10, symbol='triangle-down')))
        fig.update_layout(title=f"{clean_symbol} Technical View", xaxis_title="Date", yaxis_title="Price ($)", hovermode="x unified", template="plotly_dark", margin=dict(l=10,r=10,t=40,b=10))
        
        return one_month_change > 0, metrics, fig, current_price
    except:
        return False, {}, None, 0.0

# --- APP LAYOUT ---
col_ma, col_per = st.columns(2)
with col_ma:
    ma_type = st.radio("📐 Average Methodology", ["Simple Moving Average (SMA)", "Weighted Moving Average (WMA)"])
with col_per:
    sma_period = st.selectbox("📈 Lookback Duration", options=list(INTERVAL_MAPPING.keys()), index=1, format_func=lambda x: f"{x}-Day Window")

mode_selection = st.radio("Select Application Mode", ["Portfolio Dashboard", "Analyze Single Ticker", "Run Market Scanner"])

if mode_selection == "Portfolio Dashboard":
    st.header("💼 Personal Holding Monitor")
    
    # LOCAL FILE STORAGE ENGINE
    with st.sidebar.expander("💾 Local File Storage", expanded=True):
        st.write("Save/Load your portfolio to local disk.")
        col_load, col_save = st.columns(2)
        
        with col_save:
            if st.button("💾 Save Local"):
                with open(PORTFOLIO_FILE, "w") as f:
                    json.dump(st.session_state.user_portfolio, f)
                st.success("Saved!")
        
        with col_load:
            if st.button("🔄 Load Local"):
                if os.path.exists(PORTFOLIO_FILE):
                    with open(PORTFOLIO_FILE, "r") as f:
                        st.session_state.user_portfolio = json.load(f)
                    st.success("Loaded!")
                    st.rerun()
                else:
                    st.error("No file found.")

    with st.sidebar.expander("🛠️ Position Editor", expanded=True):
        existing_assets = [""] + list(st.session_state.user_portfolio.keys())
        selected_edit_ticker = st.selectbox("Quick-Select Asset to Edit", options=existing_assets, index=0)
        
        default_shares = 0.0
        default_cost = 0.0
        ticker_input_val = ""
        
        if selected_edit_ticker:
            ticker_input_val = selected_edit_ticker
            default_shares = float(st.session_state.user_portfolio[selected_edit_ticker]["shares"])
            default_cost = float(st.session_state.user_portfolio[selected_edit_ticker]["cost"])

        edit_ticker = st.text_input("Ticker Symbol", value=ticker_input_val).upper().strip()
        edit_shares = st.number_input("Shares", min_value=0.0, step=0.0001, format="%.4f", value=default_shares)
        edit_cost = st.number_input("Cost Basis ($)", min_value=0.0, step=0.01, value=default_cost)
        
        if st.button("Apply Changes"):
            if edit_ticker:
                if edit_shares > 0:
                    st.session_state.user_portfolio[edit_ticker] = {"shares": edit_shares, "cost": edit_cost}
                    st.success(f"Updated {edit_ticker}")
                    st.rerun()
                elif edit_ticker in st.session_state.user_portfolio and edit_shares == 0:
                    del st.session_state.user_portfolio[edit_ticker]
                    st.warning(f"Purged {edit_ticker}")
                    st.rerun()

    if not st.session_state.user_portfolio:
        st.info("Dashboard empty. Add assets in sidebar.")
    else:
        # ... (Portfolio display logic remains the same)
        total_market_value = 0.0
        total_cost_basis = 0.0
        display_data = []
        pie_labels = []
        pie_values = []
        saved_charts = {}
        
        with st.spinner("Crunching..."):
            for ticker, details in list(st.session_state.user_portfolio.items()):
                passed, metrics, fig, current_price = fetch_technical_data(ticker, sma_period, ma_type)
                if current_price > 0:
                    position_cost = details['shares'] * details['cost']
                    position_value = details['shares'] * current_price
                    total_market_value += position_value
                    total_cost_basis += position_cost
                    saved_charts[ticker] = fig
                    pie_labels.append(ticker)
                    pie_values.append(position_value)
                    display_data.append({
                        "Asset": ticker,
                        "Shares": details['shares'],
                        "Avg Cost": details['cost'],
                        "Current Price": current_price,
                        "Market Value": position_value,
                        "Return ($)": position_value - position_cost,
                        "Return (%)": ((position_value - position_cost) / position_cost * 100) if position_cost > 0 else 0,
                        "Trend Signal": metrics.get("Current State", "Calculating")
                    })
        
        st.metric("Total Portfolio Value", f"${total_market_value:,.2f}")
        
        # Display data
        df_display = pd.DataFrame(display_data)
        st.data_editor(df_display, use_container_width=True)
        
        # AI Report
        if st.button("Generate AI Portfolio Analysis"):
            if not api_key: st.error("Key required.")
            else:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(model='gemini-2.5-flash', contents=f"Analyze this portfolio: {display_data}")
                st.markdown(response.text)

elif mode_selection == "Analyze Single Ticker":
    user_ticker = st.text_input("Ticker:").upper()
    if user_ticker and st.button("Analyze"):
        _, _, fig, _ = fetch_technical_data(user_ticker, sma_period, ma_type)
        if fig: st.plotly_chart(fig)

elif mode_selection == "Run Market Scanner":
    if st.button("Run"):
        triggered = []
        for ticker in TICKER_POOL:
            passed, metrics, _, _ = fetch_technical_data(ticker, sma_period, ma_type)
            if passed: triggered.append({"Ticker": ticker, **metrics})
        if triggered: st.dataframe(pd.DataFrame(triggered))
        else: st.warning("No triggers.")
