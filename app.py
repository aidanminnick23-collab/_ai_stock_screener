import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import requests
from google import genai  

st.set_page_config(page_title="Wall Street AI Dashboard", layout="centered")

st.title("📊 Wall Street AI Scanner & Wealth Dashboard")
st.write("Track custom holdings, calculate advanced trend crossovers, and generate AI insights.")

# API Key input on the main screen
api_key = st.text_input("🔑 Enter Gemini API Key", type="password")

# Initialize isolated user portfolio tracking in session memory
if "user_portfolio" not in st.session_state:
    st.session_state.user_portfolio = {}

# Predefined pool to scan
TICKER_POOL = ["PLSE", "BRCB", "CELH", "SOFI", "HOOD", "DKNG"]

INTERVAL_MAPPING = {
    10: {"history": "3mo", "description": "Short-term momentum & micro-trends"},
    20: {"history": "6mo", "description": "Standard short-term trend-following boundary"},
    50: {"history": "1y", "description": "Medium-term structural support / Institutional tracking"},
    100: {"history": "2y", "description": "Macro cycle support and accumulation zone"},
    200: {"history": "2y", "description": "Long-term macro baseline (The ultimate bull/bear line)"}
}

# Layout Setup: Primary App Selectors
col_ma, col_per = st.columns(2)
with col_ma:
    ma_type = st.radio("📐 Average Methodology", ["Simple Moving Average (SMA)", "Weighted Moving Average (WMA)"])
with col_per:
    sma_period = st.selectbox("📈 Lookback Duration", options=list(INTERVAL_MAPPING.keys()), index=1, format_func=lambda x: f"{x}-Day Window")

# Math Engine: Calculates SMA/WMA, signals, and builds interactive visuals
def fetch_technical_data(ticker_symbol, period_window, calculation_type):
    try:
        clean_symbol = ticker_symbol.upper().strip()
        
        # --- SMART CRYPTO AUTO-CORRECTION PATCH ---
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

mode = st.radio("Select Application Mode", ["💼 My Portfolio Dashboard", "Analyze Single Ticker", "Run Market Scanner"])

# ----------------------------------------------------
# MODE 1: THE USER PORTFOLIO DASHBOARD (INTEGRATED VIEW)
# ----------------------------------------------------
if mode == "💼 My Portfolio Dashboard":
    st.header("Personal Holding Monitor")
    
    # Dual-Parameter Sync Engine Panel
    with st.sidebar.expander("🌐 Cloud Vault Sync (Auto-Save)", expanded=True):
        st.write("Save or retrieve your portfolio automatically from any device.")
        vault_user = st.text_input("Vault Username/ID", value="amin")
        user_pin = st.text_input("Secret Security PIN", max_chars=4, type="password")
        col_load, col_save = st.columns(2)
        SYSTEM_BUCKET = "wallstreet_ai_wealth_dash_v4"
        
        # Unique combined data key routing path
        composite_key = f"{vault_user.strip()}_{user_pin.strip()}" if vault_user and user_pin else ""
        
        with col_save:
            if st.button("💾 Cloud Save"):
                if not vault_user or not user_pin or len(user_pin) != 4 or not user_pin.isdigit():
                    st.error("Please enter a valid username and 4-digit numeric PIN.")
                elif not st.session_state.user_portfolio:
                    st.warning("Portfolio workspace is empty.")
                else:
                    try:
                        url = f"https://kvdb.io/{SYSTEM_BUCKET}/{composite_key}"
                        res = requests.put(url, json=st.session_state.user_portfolio, timeout=7)
                        if res.status_code in [200, 201]: st.success("Portfolio backed up!")
                        else: st.error("Sync registration failed.")
                    except: st.error("Cloud server unavailable.")
                        
        with col_load:
            if st.button("🔄 Load Portfolio"):
                if not vault_user or not user_pin or len(user_pin) != 4 or not user_pin.isdigit():
                    st.error("Please enter a username and 4-digit PIN.")
                else:
                    try:
                        url = f"https://kvdb.io/{SYSTEM_BUCKET}/{composite_key}"
                        res = requests.get(url, timeout=7)
                        if res.status_code == 200:
                            st.session_state.user_portfolio = res.json()
                            st.success("Records loaded!")
                            st.rerun()
                        elif res.status_code == 404: st.error("Vault records not found.")
                        else: st.error("Transmission aborted.")
                    except: st.error("Cloud server unavailable.")

    with st.sidebar.expander("🛠️ Position Editor"):
        new_ticker = st.text_input("Ticker Symbol").upper().strip()
        new_shares = st.number_input("Shares Owned", min_value=0.0, step=0.0001, format="%.4f")
        new_cost = st.number_input("Average Purchase Cost ($)", min_value=0.0, step=0.01)
        
        if st.button("Update Position"):
            if new_ticker and new_shares > 0:
                st.session_state.user_portfolio[new_ticker] = {"shares": new_shares, "cost": new_cost}
                st.success(f"Updated {new_ticker}.")
                st.rerun()
            elif new_ticker in st.session_state.user_portfolio and new_shares == 0:
                del st.session_state.user_portfolio[new_ticker]
                st.warning(f"Removed {new_ticker}.")
                st.rerun()

    if not st.session_state.user_portfolio:
        st.info("Your dashboard workspace is currently empty. Use the tools in the sidebar to add assets or input your sync details.")
    else:
        total_market_value = 0.0
        total_cost_basis = 0.0
        display_data = []
        pie_labels = []
        pie_values = []
        saved_charts = {}
        
        with st.spinner("Streaming live quotes and matching dynamic trendlines..."):
            for ticker, details in list(st.session_state.user_portfolio.items()):
                passed, metrics, fig, current_price = fetch_technical_data(ticker, sma_period, ma_type)
                
                if current_price > 0:
                    position_cost = details['shares'] * details['cost']
                    position_value = details['shares'] * current_price
                    position_gain = position_value - position_cost
                    position_gain_pct = (position_gain / position_cost * 100) if position_cost > 0 else 0.0
                    
                    total_market_value += position_value
                    total_cost_basis += position_cost
                    saved_charts[ticker] = fig
                    
                    display_name = f"{ticker}-USD" if ticker in ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE"] else ticker
                    pie_labels.append(display_name)
                    pie_values.append(position_value)
                    
                    display_data.append({
                        "Asset": display_name,
                        "Shares": f"{details['shares']:.4f}" if details['shares'] % 1 != 0 else f"{details['shares']:.1f}",
                        "Avg Cost": f"${details['cost']:.2f}",
                        "Current Price": f"${current_price:.2f}",
                        "Market Value": f"${position_value:.2f}",
                        "Return ($)": f"${position_gain:.2f}",
                        "Return (%)": f"{position_gain_pct:.1f}%",
                        "Trend Signal": metrics.get("Current State", "Calculating")
                    })

        total_gain = total_market_value - total_cost_basis
        total_gain_pct = (total_gain / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
        
        # Performance KPIs Ribbon Banner
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Asset Value", f"${total_market_value:,.2f}")
        kpi2.metric("Net Cost Basis", f"${total_cost_basis:,.2f}")
        kpi3.metric("Total Performance Return", f"${total_gain:,.2f}", f"{total_gain_pct:.2f}%")
        
        # --- UPGRADED: ULTRA-CLEAN CLUTTER-FREE DONUT CHART ENGINE ---
        if pie_values:
            df_pie = pd.DataFrame({"Asset": pie_labels, "Value": pie_values})
            df_pie = df_pie.sort_values(by="Value", ascending=False).reset_index(drop=True)
            
            # Compute real portfolio weights
            df_pie["Percentage"] = (df_pie["Value"] / total_market_value) * 100
            
            # Isolate core holdings vs tiny positions (< 2.5% allocation)
            MIN_THRESHOLD = 2.5
            core_holdings = df_pie[df_pie["Percentage"] >= MIN_THRESHOLD]
            micro_holdings = df_pie[df_pie["Percentage"] < MIN_THRESHOLD]
            
            if not micro_holdings.empty:
                other_row = pd.DataFrame([{
                    "Asset": f"Other ({len(micro_holdings)} Small Positions)",
                    "Value": micro_holdings["Value"].sum(),
                    "Percentage": micro_holdings["Percentage"].sum()
                }])
                df_final_pie = pd.concat([core_holdings, other_row], ignore_index=True)
            else:
                df_final_pie = df_pie

            # Sophisticated, High-Contrast FinTech Palette
            fintech_colors = [
                "#1f77b4", "#00b4d8", "#0077b6", "#0096c7", "#03045e",
                "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#457b9d",
                "#a8dadc", "#1d3557", "#48cae4", "#b5e2fa", "#4a4e69"
            ]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=df_final_pie["Asset"], 
                values=df_final_pie["Value"], 
                hole=0.45,
                hoverinfo="label+value+percent",
                textinfo="percent+label",
                textposition="outside",  # Pushes text clean of the pie borders
                automargin=True,         # Ensures canvas shrinks to fit labels safely
                marker=dict(
                    colors=fintech_colors[:len(df_final_pie)],
                    line=dict(color='#1e1e1e', width=2)
                )
            )])
            
            fig_pie.update_layout(
                title=dict(
                    text="🎯 Real-Time Strategic Asset Allocation Weighting", 
                    x=0.5,
                    font=dict(size=16, family="Helvetica Neue, Arial, sans-serif", color="#ffffff")
                ),
                template="plotly_dark",
                margin=dict(l=50, r=50, t=60, b=50),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.05,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11, color="#cccccc")
                )
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Your Monitored Assets Summary")
        st.dataframe(pd.DataFrame(display_data), use_container_width=True)
        
        # Aggregate Full Portfolio AI Strategic Analysis Matrix
        st.markdown("---")
        st.subheader("🧠 Holistic Wealth & Diversification Audit")
        st.write("Passes your entire portfolio to Gemini to run cross-asset correlation checks and risk reviews.")
        
        if st.button("Generate Full Portfolio AI Macro Report"):
            if not api_key:
                st.error("⚠️ Gemini API Key required to run aggregate AI evaluations.")
            else:
                with st.spinner("Executing structural asset-correlation matrix analysis..."):
                    portfolio_analysis_prompt = f"""
                    You are an elite Wall Street Managing Director and Chief Wealth Management Strategist. 
                    Perform a high-level strategic review on this client investment portfolio matrix:
                    Total Wealth Under Management: ${total_market_value:,.2f}
                    Consolidated Cost Basis: ${total_cost_basis:,.2f}
                    Net Unreleased Returns: ${total_gain:,.2f} ({total_gain_pct:.2f}%)
                    
                    Held Asset Layout:
                    {json.dumps(display_data, indent=2)}
                    
                    Active Macro Tracking Rule: {sma_period}-Day lookback using {ma_type} parameters.

                    Provide a response with the following exact components:
                    1. A Markdown table named 'Portfolio Diversification Analysis' ranking assets by weight, concentration tier, and risk status.
                    2. Macro Risk & Correlation Assessment auditing asset vulnerabilities.
                    3. Actionable Rebalancing Recommendations detailing what to hold, skim, or accumulate.
                    4. A final bolded 'Chief Investment Officer (CIO) Mandate'.
                    """
                    try:
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(model='gemini-2.5-flash', contents=portfolio_analysis_prompt)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Portfolio AI Engine Error: {e}")
        
        # Single Ticker Deep Dive Section
        st.markdown("---")
        st.subheader("🎯 Single Asset Chart Drill-Down")
        selected_chart_ticker = st.selectbox("Choose a holding to pull up historical indicators", options=list(st.session_state.user_portfolio.keys()))
        
        if selected_chart_ticker in saved_charts and saved_charts[selected_chart_ticker] is not None:
            st.plotly_chart(saved_charts[selected_chart_ticker], use_container_width=True)
            
            if st.button("Generate Single Ticker Analyst Report") and api_key:
                with st.spinner("Compiling individual asset waves..."):
                    holding_metrics = [x for x in display_data if x["Asset"].startswith(selected_chart_ticker)][0]
                    
                    analysis_prompt_template = """
                    You are an institutional Wall Street wealth analyst. Analyze held asset {ticker}:
                    Current Account Tracking Parameters: {metrics}
                    Lookback Calculation Rules: {window}-Day {method} boundaries.

                    Provide a response with the following components:
                    1. 'Quantitative Position Tear Sheet' table.
                    2. An Elliott Wave Technical Framework.
                    3. A clear, actionable bolded final 'Portfolio Strategy Suggestion'.
                    """
                    try:
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=analysis_prompt_template.format(ticker=selected_chart_ticker, metrics=holding_metrics, window=sma_period, method=ma_type)
                        )
                        st.markdown(response.text)
                    except Exception as e: st.error(f"AI Error: {e}")

# ----------------------------------------------------
# RETAINED OPERATIONAL MODES (PROMPT INTEGRITY MAINTAINED)
# ----------------------------------------------------
elif mode == "Analyze Single Ticker":
    user_ticker = st.text_input("Enter Stock Ticker:").upper()
    if user_ticker and st.button("Generate Tear Sheet"):
        if not api_key: st.error("⚠️ Gemini API Key required.")
        else:
            with st.spinner("Crunching data..."):
                passed, metrics, fig, _ = fetch_technical_data(user_ticker, sma_period, ma_type)
                if fig is not None: st.plotly_chart(fig, use_container_width=True)
                try:
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"Institutional Analyst Assessment for {user_ticker}. Financial Data Profile: {metrics}. Apply a {sma_period}-day {ma_type} overlay framework. Generate a Quantitative Screen Tear Sheet table, core real-world metric analogies, an Elliott Wave psychological map, and a final bolded Analyst Verdict."
                    )
                    st.markdown(response.text)
                except Exception as e: st.error(f"Error: {e}")

elif mode == "Run Market Scanner":
    if st.button("Launch Scan Now"):
        if not api_key: st.error("⚠️ Gemini API Key required.")
        else:
            triggered_stocks = []
            saved_figs = {}
            for ticker in TICKER_POOL:
                passed, metrics, fig, _ = fetch_technical_data(ticker, sma_period, ma_type)
                if passed and fig is not None:
                    triggered_stocks.append({"Ticker": ticker, **metrics})
                    saved_figs[ticker] = fig
            if triggered_stocks:
                st.success("Scanning complete!")
                st.dataframe(triggered_stocks)
                top_stock = triggered_stocks[0]["Ticker"]
                st.plotly_chart(saved_figs[top_stock], use_container_width=True)
            else:
                st.warning("No assets currently trigger momentum thresholds.")
