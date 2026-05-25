import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from google import genai  

st.set_page_config(page_title="Wall Street AI Scanner", layout="centered")

st.title("📊 Wall Street AI Scanner & Tear Sheet Tool")
st.write("Calculates core technical overlays, identifies trend signals, and maps psychological market waves.")

# API Key input on the main screen
api_key = st.text_input("🔑 Enter Gemini API Key", type="password")

# Predefined pool to scan
TICKER_POOL = ["PLSE", "BRCB", "CELH", "SOFI", "HOOD", "DKNG"]

# NEW: Interval Selection Mapping
# Pairs user selection with the minimum safe data window required for calculation
INTERVAL_MAPPING = {
    10: {"history": "3mo", "description": "Short-term momentum & micro-trends"},
    20: {"history": "6mo", "description": "Standard short-term trend-following boundary"},
    50: {"history": "1y", "description": "Medium-term structural support / Institutional tracking"},
    100: {"history": "2y", "description": "Macro cycle support and accumulation zone"},
    200: {"history": "2y", "description": "Long-term macro baseline (The ultimate bull/bear line)"}
}

# Dynamic UI Selector for SMA Duration
sma_period = st.selectbox(
    "📈 Select Moving Average (SMA) Duration",
    options=list(INTERVAL_MAPPING.keys()),
    index=1, # Defaults to 20
    format_func=lambda x: f"{x}-Day SMA ({INTERVAL_MAPPING[x]['description']})"
)

# Advanced adaptive data processing engine
def fetch_technical_data(ticker_symbol, period_window):
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Dynamically allocate historical lookback period based on SMA choice
        lookback = INTERVAL_MAPPING[period_window]["history"]
        hist = ticker.history(period=lookback)
        
        if hist.empty or len(hist) < period_window:
            return False, {}, None
        
        # Calculate Dynamic Technical Indicators
        sma_col = f'SMA_{period_window}'
        hist[sma_col] = hist['Close'].rolling(window=period_window).mean()
        hist['Prev_Close'] = hist['Close'].shift(1)
        hist['Prev_SMA'] = hist[sma_col].shift(1)
        
        # Define algorithmic Crossover Signals based on dynamic boundaries
        buy_condition = (hist['Close'] > hist[sma_col]) & (hist['Prev_Close'] <= hist['Prev_SMA'])
        sell_condition = (hist['Close'] < hist[sma_col]) & (hist['Prev_Close'] >= hist['Prev_SMA'])
        
        hist['Buy_Signal'] = np.where(buy_condition, hist['Close'], np.nan)
        hist['Sell_Signal'] = np.where(sell_condition, hist['Close'], np.nan)
        
        # Capture current conditions
        current_price = hist['Close'].iloc[-1]
        prior_price = hist['Close'].iloc[-20] if len(hist) >= 20 else hist['Close'].iloc[0]
        one_month_change = ((current_price - prior_price) / prior_price) * 100
        current_sma = hist[sma_col].iloc[-1]
        
        # Determine latest structural signal state and historical dates
        latest_signal = "Neutral (Consolidating)"
        recent_buys = hist['Buy_Signal'].dropna()
        recent_sells = hist['Sell_Signal'].dropna()
        
        if not recent_buys.empty and (recent_sells.empty or recent_buys.index[-1] > recent_sells.index[-1]):
            latest_signal = f"Active BUY Trigger (Price crossed ABOVE the {period_window}-day SMA on {recent_buys.index[-1].strftime('%m/%d/%Y')})"
        elif not recent_sells.empty and (recent_buys.empty or recent_sells.index[-1] > recent_buys.index[-1]):
            latest_signal = f"Active SELL Trigger (Price crossed BELOW the {period_window}-day SMA on {recent_sells.index[-1].strftime('%m/%d/%Y')})"

        metrics = {
            "Price": f"${current_price:.2f}",
            "Approx. 1-Mo Momentum": f"{one_month_change:.1f}%",
            f"{period_window}-Day SMA Value": f"${current_sma:.2f}",
            "Algorithmic Trend State": latest_signal,
            "Lookback Window Applied": lookback
        }
        
        # Build the dynamic Plotly visualization chart
        fig = go.Figure()
        
        # Underlying Price Line
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['Close'], name='Stock Price',
            line=dict(color='#1f77b4', width=2.5)
        ))
        
        # Dynamic Moving Average Overlay
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist[sma_col], name=f'{period_window}-Day SMA',
            line=dict(color='#ff7f0e', width=1.5, dash='dash')
        ))
        
        # Visual Buy Markers
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['Buy_Signal'], mode='markers', name='BUY Signal',
            marker=dict(color='#2ca02c', size=11, symbol='triangle-up', line=dict(width=1, color='black'))
        ))
        
        # Visual Sell Markers
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['Sell_Signal'], mode='markers', name='SELL Signal',
            marker=dict(color='#d62728', size=11, symbol='triangle-down', line=dict(width=1, color='black'))
        ))
        
        fig.update_layout(
            title=f"Technical Trend Analysis Overview ({ticker_symbol} - {period_window} Day Horizon)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10),
            template="plotly_dark"
        )
        
        passed = one_month_change > 0
        return passed, metrics, fig
        
    except Exception as e:
        return False, {}, None

mode = st.radio("Select Mode", ["Analyze Single Ticker", "Run Market Scanner (Buy Trigger)"])

# Configured the prompt template to ingest the chosen lookback scope context dynamically
analysis_prompt_template = """
You are an institutional-grade Wall Street Equity Analyst specializing in macro technical frameworks. 
Analyze the following data for ticker {ticker}:
Live Calculated Technical Metrics: {metrics}
Selected Time Horizon Scope: {window}-Day Simple Moving Average

Provide a response with the following exact components:
1. A Markdown table named 'Quantitative Screen Tear Sheet' detailing all the calculated metrics provided, what institutional analysts look for when using a {window}-day scope, and their current screen status. Include the specific date of the last active trigger if noted.
2. A beginner-friendly breakdown defining the technical crossover mechanics of a {window}-day average using intuitive real-world analogies.
3. An Elliott Wave Technical Perspective mapping market psychology and current presumed wave structures based on how price interacts with this specific {window}-day macro framework.
4. A bolded final 'Analyst Verdict' summarizing the specific risk profile and signal confirmation.
"""

if mode == "Analyze Single Ticker":
    user_ticker = st.text_input("Enter Stock Ticker (e.g., HOOD):").upper()
    if user_ticker and st.button("Generate Tear Sheet"):
        if not api_key:
            st.error("⚠️ You must enter your Gemini API Key first!")
        else:
            with st.spinner("Processing market charts and data streams..."):
                passed, metrics, fig = fetch_technical_data(user_ticker, sma_period)
                
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not render historical charts. Try a shorter SMA window if this ticker has limited trading history.")
                    metrics = {"Price": "Data pipeline restriction, verifying core parameters via synthesis."}
                
                with st.spinner("Executing structural AI evaluation..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=analysis_prompt_template.format(ticker=user_ticker, metrics=metrics, window=sma_period)
                        )
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI Error: {e}")

elif mode == "Run Market Scanner (Buy Trigger)":
    if st.button("Launch Scan Now"):
        if not api_key:
            st.error("⚠️ You must enter your Gemini API Key first!")
        else:
            st.write(f"Scanning institutional watchlists using a {sma_period}-day operational threshold...")
            triggered_stocks = []
            saved_figs = {}
            
            for ticker in TICKER_POOL:
                passed, metrics, fig = fetch_technical_data(ticker, sma_period)
                if passed and fig is not None:
                    triggered_stocks.append({"Ticker": ticker, **metrics})
                    saved_figs[ticker] = fig
            
            if triggered_stocks:
                st.success(f"Found {len(triggered_stocks)} assets demonstrating technical momentum relative to the {sma_period}-day line!")
                st.dataframe(triggered_stocks)
                
                top_stock = triggered_stocks[0]["Ticker"]
                st.subheader(f"Deep-Dive Analyst Evaluation for Top Asset: {top_stock}")
                
                st.plotly_chart(saved_figs[top_stock], use_container_width=True)
                
                with st.spinner("Executing Wall Street wave synthesis..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=analysis_prompt_template.format(ticker=top_stock, metrics=triggered_stocks[0], window=sma_period)
                        )
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI Error: {e}")
            else:
                st.warning(f"No tracking assets currently meet the dynamic {sma_period}-day momentum thresholds.")
