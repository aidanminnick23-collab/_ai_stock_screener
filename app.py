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

# Interval Selection Mapping
INTERVAL_MAPPING = {
    10: {"history": "3mo", "description": "Short-term momentum & micro-trends"},
    20: {"history": "6mo", "description": "Standard short-term trend-following boundary"},
    50: {"history": "1y", "description": "Medium-term structural support / Institutional tracking"},
    100: {"history": "2y", "description": "Macro cycle support and accumulation zone"},
    200: {"history": "2y", "description": "Long-term macro baseline (The ultimate bull/bear line)"}
}

# Layout Setup: Interactive selectors placed side-by-side
col1, col2 = st.columns(2)

with col1:
    ma_type = st.radio(
        "📐 Select Average Methodology",
        options=["Simple Moving Average (SMA)", "Weighted Moving Average (WMA)"],
        help="SMA treats all days equally. WMA gives heavier statistical weight to recent days."
    )

with col2:
    sma_period = st.selectbox(
        "📈 Select Lookback Duration",
        options=list(INTERVAL_MAPPING.keys()),
        index=1, # Defaults to 20
        format_func=lambda x: f"{x}-Day Window"
    )

st.caption(f"**Current Configuration Strategy:** {INTERVAL_MAPPING[sma_period]['description']}")

# Advanced adaptive data processing engine supporting SMA and WMA calculations
def fetch_technical_data(ticker_symbol, period_window, calculation_type):
    try:
        ticker = yf.Ticker(ticker_symbol)
        lookback = INTERVAL_MAPPING[period_window]["history"]
        hist = ticker.history(period=lookback)
        
        if hist.empty or len(hist) < period_window:
            return False, {}, None
        
        # --- Dynamic Core Math Block ---
        if "Simple" in calculation_type:
            hist['MA_Line'] = hist['Close'].rolling(window=period_window).mean()
            ma_acronym = "SMA"
        else:
            # Generate a linear weighting array (e.g., [1, 2, 3... N] for an N-day window)
            linear_weights = np.arange(1, period_window + 1)
            # Apply rolling custom linear dot-product to prices
            hist['MA_Line'] = hist['Close'].rolling(window=period_window).apply(
                lambda prices: np.dot(prices, linear_weights) / linear_weights.sum(), 
                raw=True
            )
            ma_acronym = "WMA"
            
        hist['Prev_Close'] = hist['Close'].shift(1)
        hist['Prev_MA'] = hist['MA_Line'].shift(1)
        
        # Algorithmic Crossover Signals mapped to the dynamic line output
        buy_condition = (hist['Close'] > hist['MA_Line']) & (hist['Prev_Close'] <= hist['Prev_MA'])
        sell_condition = (hist['Close'] < hist['MA_Line']) & (hist['Prev_Close'] >= hist['Prev_MA'])
        
        hist['Buy_Signal'] = np.where(buy_condition, hist['Close'], np.nan)
        hist['Sell_Signal'] = np.where(sell_condition, hist['Close'], np.nan)
        
        # Capture parameters
        current_price = hist['Close'].iloc[-1]
        prior_price = hist['Close'].iloc[-20] if len(hist) >= 20 else hist['Close'].iloc[0]
        one_month_change = ((current_price - prior_price) / prior_price) * 100
        current_ma_val = hist['MA_Line'].iloc[-1]
        
        # Determine latest trigger timestamps
        latest_signal = "Neutral (Consolidating)"
        recent_buys = hist['Buy_Signal'].dropna()
        recent_sells = hist['Sell_Signal'].dropna()
        
        if not recent_buys.empty and (recent_sells.empty or recent_buys.index[-1] > recent_sells.index[-1]):
            latest_signal = f"Active BUY Trigger (Price crossed ABOVE the {period_window}-day {ma_acronym} on {recent_buys.index[-1].strftime('%m/%d/%Y')})"
        elif not recent_sells.empty and (recent_buys.empty or recent_sells.index[-1] > recent_buys.index[-1]):
            latest_signal = f"Active SELL Trigger (Price crossed BELOW the {period_window}-day {ma_acronym} on {recent_sells.index[-1].strftime('%m/%d/%Y')})"

        metrics = {
            "Price": f"${current_price:.2f}",
            "Approx. 1-Mo Momentum": f"{one_month_change:.1f}%",
            f"{period_window}-Day {ma_acronym} Value": f"${current_ma_val:.2f}",
            "Algorithmic Trend State": latest_signal,
            "Calculation Methodology": calculation_type
        }
        
        # Build the Plotly visualization chart
        fig = go.Figure()
        
        # Underlying Price Line
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['Close'], name='Stock Price',
            line=dict(color='#1f77b4', width=2.5)
        ))
        
        # Dynamic Overlay Line (Swaps dynamically between SMA and WMA styles)
        line_color = '#ff7f0e' if ma_acronym == "SMA" else '#9467bd'
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['MA_Line'], name=f'{period_window}-Day {ma_acronym}',
            line=dict(color=line_color, width=1.5, dash='dash' if ma_acronym == "SMA" else 'solid')
        ))
        
        # Visual Buy/Sell Markers
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['Buy_Signal'], mode='markers', name='BUY Signal',
            marker=dict(color='#2ca02c', size=11, symbol='triangle-up', line=dict(width=1, color='black'))
        ))
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['Sell_Signal'], mode='markers', name='SELL Signal',
            marker=dict(color='#d62728', size=11, symbol='triangle-down', line=dict(width=1, color='black'))
        ))
        
        fig.update_layout(
            title=f"Technical Trend Analysis Overview ({ticker_symbol} - {period_window} Day {ma_acronym})",
            xaxis_title="Date", yaxis_title="Price ($)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark"
        )
        
        passed = one_month_change > 0
        return passed, metrics, fig
        
    except Exception as e:
        return False, {}, None

mode = st.radio("Select Mode", ["Analyze Single Ticker", "Run Market Scanner (Buy Trigger)"])

# Programmed the system instructions to analyze the exact architectural variant chosen
analysis_prompt_template = """
You are an institutional-grade Wall Street Equity Analyst.
Analyze the following data parameters for ticker {ticker}:
Live Calculated Technical Metrics: {metrics}
Selected Strategy Horizon: {window}-Day lookback using {method} rules.

Provide a response with the following exact components:
1. A Markdown table named 'Quantitative Screen Tear Sheet' detailing all the calculated metrics provided, what institutional analysts look for when applying a {window}-day {method} boundary configuration, and their current screen status. 
2. A beginner-friendly breakdown defining the difference between equal-weight calculations and linear-weighting distribution using intuitive real-world analogies based on the selected setting.
3. An Elliott Wave Technical Perspective mapping market psychology and current presumed wave structures. If WMA was selected, focus heavily on near-term sensitivity and momentum lag minimization. If SMA was selected, focus on structural institutional support walls and macro behavioral cycles.
4. A bolded final 'Analyst Verdict' summarizing the clear risk profile and signal validation.
"""

if mode == "Analyze Single Ticker":
    user_ticker = st.text_input("Enter Stock Ticker (e.g., CELH):").upper()
    if user_ticker and st.button("Generate Tear Sheet"):
        if not api_key:
            st.error("⚠️ You must enter your Gemini API Key first!")
        else:
            with st.spinner("Processing market charts and data streams..."):
                passed, metrics, fig = fetch_technical_data(user_ticker, sma_period, ma_type)
                
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not render historical charts. Try a shorter lookback window if history is limited.")
                    metrics = {"Price": "Data pipeline restriction, verifying parameters via synthesis."}
                
                with st.spinner("Executing structural AI evaluation..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=analysis_prompt_template.format(ticker=user_ticker, metrics=metrics, window=sma_period, method=ma_type)
                        )
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI Error: {e}")

elif mode == "Run Market Scanner (Buy Trigger)":
    if st.button("Launch Scan Now"):
        if not api_key:
            st.error("⚠️ You must enter your Gemini API Key first!")
        else:
            st.write(f"Scanning watchlists using a {sma_period}-day {ma_type} trend threshold...")
            triggered_stocks = []
            saved_figs = {}
            
            for ticker in TICKER_POOL:
                passed, metrics, fig = fetch_technical_data(ticker, sma_period, ma_type)
                if passed and fig is not None:
                    triggered_stocks.append({"Ticker": ticker, **metrics})
                    saved_figs[ticker] = fig
            
            if triggered_stocks:
                st.success(f"Found {len(triggered_stocks)} assets demonstrating technical momentum relative to the selected line configuration!")
                st.dataframe(triggered_stocks)
                
                top_stock = triggered_stocks[0]["Ticker"]
                st.subheader(f"Deep-Dive Analyst Evaluation for Top Asset: {top_stock}")
                
                st.plotly_chart(saved_figs[top_stock], use_container_width=True)
                
                with st.spinner("Executing Wall Street wave synthesis..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=analysis_prompt_template.format(ticker=top_stock, metrics=triggered_stocks[0], window=sma_period, method=ma_type)
                        )
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI Error: {e}")
            else:
                st.warning(f"No tracking assets currently meet the dynamic {sma_period}-day momentum thresholds.")
