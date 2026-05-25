import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import requests
from google import genai  

st.set_page_config(page_title="Wall Street AI Dashboard", layout="centered")

st.title("Wall Street AI Scanner and Wealth Dashboard")
st.write("Track custom holdings, calculate advanced trend crossovers, and generate AI insights.")

api_key = st.text_input("Enter Gemini API Key", type="password")

if "user_portfolio" not in st.session_state:
    st.session_state.user_portfolio = {}

TICKER_POOL = ["PLSE", "BRCB", "CELH", "SOFI", "HOOD", "DKNG"]
INTERVAL_MAPPING = {
    10: {"history": "3mo"}, 20: {"history": "6mo"}, 50: {"history": "1y"}, 
    100: {"history": "2y"}, 200: {"history": "2y"}
}

def fetch_technical_data(ticker_symbol, period_window, calculation_type):
    try:
        clean_symbol = ticker_symbol.upper().strip()
        if clean_symbol in ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE"]: clean_symbol = f"{clean_symbol}-USD"
        ticker = yf.Ticker(clean_symbol)
        hist = ticker.history(period=INTERVAL_MAPPING[period_window]["history"])
        if hist.empty or len(hist) < period_window: return False, {}, None, 0.0
        
        if "Simple" in calculation_type:
            hist['MA_Line'] = hist['Close'].rolling(window=period_window).mean()
        else:
            w = np.arange(1, period_window + 1)
            hist['MA_Line'] = hist['Close'].rolling(window=period_window).apply(lambda p: np.dot(p, w) / w.sum(), raw=True)
            
        current_price = hist['Close'].iloc[-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA_Line'], name='MA'))
        fig.update_layout(template="plotly_dark", height=400)
        return True, {"Price": f"${current_price:.2f}"}, fig, current_price
    except: return False, {}, None, 0.0

mode_selection = st.radio("Select Application Mode", ["Portfolio Dashboard", "Analyze Single Ticker", "Run Market Scanner"])

if mode_selection == "Portfolio Dashboard":
    st.header("Personal Holding Monitor")
    
    with st.sidebar.expander("Cloud Vault Sync"):
        vault_user = st.text_input("Vault ID", value="amin")
        user_pin = st.text_input("PIN", type="password")
        SYSTEM_BUCKET = "wallstreet_ai_wealth_dash_v4"
        if st.button("Save to Cloud"):
            try:
                # Proper Bucket Initialization
                requests.post(f"https://kvdb.io/{SYSTEM_BUCKET}", data={'email': 'user@example.com'})
                res = requests.put(f"https://kvdb.io/{SYSTEM_BUCKET}/{vault_user}_{user_pin}", json=st.session_state.user_portfolio)
                if res.status_code in [200, 201]: st.success("Saved!")
            except: st.error("Sync failed.")
    
    with st.sidebar.expander("Position Editor"):
        edit_ticker = st.text_input("Ticker").upper()
        edit_shares = st.number_input("Shares", min_value=0.0)
        edit_cost = st.number_input("Avg Cost", min_value=0.0)
        if st.button("Apply"):
            if edit_shares > 0: st.session_state.user_portfolio[edit_ticker] = {"shares": edit_shares, "cost": edit_cost}
            elif edit_ticker in st.session_state.user_portfolio: del st.session_state.user_portfolio[edit_ticker]
            st.rerun()

    display_data = []
    for ticker, details in st.session_state.user_portfolio.items():
        _, _, _, cp = fetch_technical_data(ticker, 20, "SMA")
        if cp > 0:
            display_data.append({"Asset": ticker, "Shares": float(details['shares']), "Avg Cost": float(details['cost']), "Market Value": details['shares']*cp})
    
    df = pd.DataFrame(display_data)
    edited_df = st.data_editor(df, key="editor")
    
    if st.button("Save Table Modifications"):
        for _, row in edited_df.iterrows():
            st.session_state.user_portfolio[row["Asset"]] = {"shares": row["Shares"], "cost": row["Avg Cost"]}
        st.rerun()

    if not df.empty:
        df = df.sort_values("Market Value", ascending=False)
        top5, next5, other = df.iloc[:5], df.iloc[5:10], df.iloc[10:]
        
        plot_data = top5.copy()
        if not next5.empty:
            plot_data = pd.concat([plot_data, pd.DataFrame([{"Asset": "Next 5 Mid-Tier", "Market Value": next5["Market Value"].sum()}])])
        if not other.empty:
            plot_data = pd.concat([plot_data, pd.DataFrame([{"Asset": "Other Remaining", "Market Value": other["Market Value"].sum()}])])

        fig = go.Figure(data=[go.Pie(labels=plot_data["Asset"], values=plot_data["Market Value"], hole=0.4)])
        fig.update_layout(height=500, template="plotly_dark")
        st.plotly_chart(fig)

elif mode_selection == "Analyze Single Ticker":
    ticker = st.text_input("Enter Ticker").upper()
    if ticker and st.button("Analyze"):
        _, _, fig, _ = fetch_technical_data(ticker, 20, "SMA")
        st.plotly_chart(fig)
