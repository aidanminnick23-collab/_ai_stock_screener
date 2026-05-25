import streamlit as st
import yfinance as yf
import google.generativeai as genai
import pandas as pd

# 1. Setup Gemini API Configuration
# Securely fetch API key from Streamlit secrets or user input
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if api_key:
    genai.configure(api_key=api_key)

st.title("📊 Wall Street AI Scanner & Tear Sheet Tool")
st.write("Scans tickers, applies a quantitative 'Buy Trigger', and generates AI insights.")

# 2. Define the Ticker Pool to Scan (Example: High-Growth Small Caps)
TICKER_POOL = ["PLSE", "BRCB", "VCIG", "CELH", "SOFI", "HOOD", "DKNG"]

# 3. Quantitative "Buy Trigger" Filter Function
def check_quantitative_trigger(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        hist = ticker.history(period="1y")
        
        # Hard Metrics
        rev_growth = info.get("revenueGrowth", 0) or 0
        p_sales = info.get("priceToSalesTrailing12Months", 999) or 999
        
        # Simple Technical Momentum (Current Price vs 50-day Moving Average)
        current_price = hist['Close'].iloc[-1]
        ma_50 = hist['Close'].tail(50).mean()
        
        # THE BUY TRIGGER LOGIC: 
        # Must have positive revenue growth AND trade at a reasonable P/S ratio
        if rev_growth > 0.05 and p_sales < 4.0:
            return True, {
                "Price": f"${current_price:.2f}",
                "Rev Growth": f"{rev_growth * 100:.1f}%",
                "P/S Ratio": f"{p_sales:.2f}x",
                "Above 50MA": "Yes" if current_price > ma_50 else "No"
            }
        return False, {}
    except Exception as e:
        return False, {}

# --- UI Interface ---
mode = st.radio("Select Mode", ["Analyze Single Ticker", "Run Market Scanner (Buy Trigger)"])

# Protocol instructions for Gemini
analysis_prompt_template = """
You are an institutional-grade Wall Street Equity Analyst. Analyze the following data for ticker {ticker}:
Live Data Metrics: {metrics}

Provide a response with the following exact components:
1. A Markdown table named 'BRCB Quantitative Screen Tear Sheet' (adapted for this ticker) with headers: | Financial Metric | Current Data | What the Analyst Looks For | Screen Status |
2. A beginner-friendly breakdown defining the key metrics used with easy real-world analogies.
3. An Elliott Wave Technical Perspective mapping market psychology and current presumed wave structures (if technical history allows).
4. A bolded final 'Analyst Verdict' summarizing the risk profile.
"""

if mode == "Analyze Single Ticker":
    user_ticker = st.text_input("Enter Stock Ticker (e.g., BRCB):").upper()
    if user_ticker and st.button("Generate Tear Sheet"):
        with st.spinner("Analyzing data and generating report..."):
            _, metrics = check_quantitative_trigger(user_ticker)
            if not metrics:
                # If it didn't pass the trigger, pull basic info anyway for manual analysis
                t = yf.Ticker(user_ticker)
                metrics = {"Price": t.history(period="1d")['Close'].iloc[-1]}
            
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(analysis_prompt_template.format(ticker=user_ticker, metrics=metrics))
            st.markdown(response.text)

elif mode == "Run Market Scanner (Buy Trigger)":
    if st.button("Launch Scan Now"):
        st.write("Scanning predefined watchlists for 'Buy Triggers'...")
        triggered_stocks = []
        
        for ticker in TICKER_POOL:
            passed, metrics = check_quantitative_trigger(ticker)
            if passed:
                triggered_stocks.append({"Ticker": ticker, **metrics})
        
        if triggered_stocks:
            st.success(f"Found {len(triggered_stocks)} stocks meeting the Buy Trigger criteria!")
            df = pd.DataFrame(triggered_stocks)
            st.dataframe(df)
            
            # Run the deep AI analysis on the top triggered stock as an example
            top_stock = triggered_stocks[0]["Ticker"]
            st.subheader(label=f"Deep-Dive Analyst Report for Top Triggered Stock: {top_stock}")
            
            with st.spinner("Executing Wall Street evaluation..."):
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = model.generate_content(analysis_prompt_template.format(ticker=top_stock, metrics=triggered_stocks[0]))
                st.markdown(response.text)
        else:
            st.warning("No stocks currently meet the strict quantitative Buy Trigger criteria.")