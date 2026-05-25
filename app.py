import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import requests
from google import genai  

# --- PAGE CONFIG ---
st.set_page_config(page_title="Wall Street AI Dashboard", layout="wide")

# --- INITIALIZATION ---
if "user_portfolio" not in st.session_state:
    st.session_state.user_portfolio = {}

# --- CORE FUNCTIONS ---
def fetch_technical_data(ticker_symbol, period_window, calculation_type):
    try:
        ticker = yf.Ticker(ticker_symbol.upper().strip())
        hist = ticker.history(period="1y")
        if hist.empty: return False, {}, None, 0.0
        
        # Calculate Moving Average
        if "Simple" in calculation_type:
            hist['MA_Line'] = hist['Close'].rolling(window=period_window).mean()
        else:
            weights = np.arange(1, period_window + 1)
            hist['MA_Line'] = hist['Close'].rolling(window=period_window).apply(lambda p: np.dot(p, weights) / weights.sum(), raw=True)
            
        current_price = hist['Close'].iloc[-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA_Line'], name='MA'))
        fig.update_layout(template="plotly_dark", height=400)
        
        return True, {"Price": f"${current_price:.2f}"}, fig, current_price
    except Exception as e: return False, {}, None, 0.0

# --- UI STRUCTURE ---
st.title("📊 Wall Street AI Scanner & Wealth Dashboard")
api_key = st.text_input("🔑 Enter Gemini API Key", type="password")

mode = st.radio("Mode", ["Portfolio Dashboard", "Market Scanner", "Ticker Analysis"])

# --- PORTFOLIO LOGIC ---
if mode == "Portfolio Dashboard":
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("🛠️ Editor")
        t = st.text_input("Ticker").upper()
        s = st.number_input("Shares", min_value=0.0)
        c = st.number_input("Cost", min_value=0.0)
        if st.button("Update/Add"):
            if s > 0: st.session_state.user_portfolio[t] = {"shares": s, "cost": c}
            elif t in st.session_state.user_portfolio: del st.session_state.user_portfolio[t]
            st.rerun()

    with col2:
        st.subheader("📋 Assets Summary")
        data = [{"Asset": k, "Shares": float(v['shares']), "Avg Cost": float(v['cost'])} for k, v in st.session_state.user_portfolio.items()]
        df = pd.DataFrame(data)
        
        # Dynamic Market Data Fetch
        for i, row in df.iterrows():
            _, _, _, cp = fetch_technical_data(row["Asset"], 20, "SMA")
            df.at[i, "Market Value"] = row["Shares"] * cp
        
        edited_df = st.data_editor(df, key="main_editor", use_container_width=True)
        if st.button("Save Table Modifications"):
            for _, row in edited_df.iterrows():
                st.session_state.user_portfolio[row["Asset"]] = {"shares": row["Shares"], "cost": row["Avg Cost"]}
            st.rerun()

        # Binning logic for Chart
        if not df.empty:
            df = df.sort_values("Market Value", ascending=False)
            top5 = df.iloc[:5]
            next5 = df.iloc[5:10]
            others = df.iloc[10:]
            
            plot_data = top5.copy()
            if not next5.empty:
                plot_data = pd.concat([plot_data, pd.DataFrame([{"Asset": "Next 5 Mid-Tier", "Market Value": next5["Market Value"].sum()}])])
            if not others.empty:
                plot_data = pd.concat([plot_data, pd.DataFrame([{"Asset": "Other Remaining", "Market Value": others["Market Value"].sum()}])])

            fig = go.Figure(data=[go.Pie(labels=plot_data["Asset"], values=plot_data["Market Value"], hole=0.4)])
            fig.update_layout(title="Allocation Breakdown", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

# --- SCANNER & ANALYSIS LOGIC ---
elif mode == "Market Scanner":
    if st.button("Run Global Scan"):
        st.write("Scanning pool...")
        # Add loop for TICKER_POOL here

elif mode == "Ticker Analysis":
    ticker = st.text_input("Enter Ticker")
    if ticker:
        _, _, fig, _ = fetch_technical_data(ticker, 50, "SMA")
        st.plotly_chart(fig, use_container_width=True)
