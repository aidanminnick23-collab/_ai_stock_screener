import streamlit as st
import yfinance as yf
from google import genai  # Updated to modern 2026 SDK

st.title("📊 Wall Street AI Scanner & Tear Sheet Tool")
st.write("Scans tickers, applies a quantitative 'Buy Trigger', and generates AI insights.")

# API Key input on the main screen
api_key = st.text_input("🔑 Enter Gemini API Key", type="password")

# Predefined pool to scan
TICKER_POOL = ["PLSE", "BRCB", "CELH", "SOFI", "HOOD", "DKNG"]

# Quantitative momentum logic using historical price boundaries
def check_quantitative_trigger(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1mo")
        if hist.empty:
            return False, {}
        
        current_price = hist['Close'].iloc[-1]
        prior_price = hist['Close'].iloc[0]
        one_month_change = ((current_price - prior_price) / prior_price) * 100
        
        if one_month_change > 0:
            return True, {
                "Price": f"${current_price:.2f}",
                "1-Mo Momentum": f"{one_month_change:.1f}%"
            }
        return False, {}
    except Exception as e:
        return False, {}

mode = st.radio("Select Mode", ["Analyze Single Ticker", "Run Market Scanner (Buy Trigger)"])

analysis_prompt_template = """
You are an institutional-grade Wall Street Equity Analyst. Analyze the following data for ticker {ticker}:
Live Price/Momentum Data: {metrics}

Provide a response with the following exact components:
1. A Markdown table named 'Quantitative Screen Tear Sheet' with headers: | Financial Metric | Current Data | What the Analyst Looks For | Screen Status |
2. A beginner-friendly breakdown defining the key metrics used with easy real-world analogies.
3. An Elliott Wave Technical Perspective mapping market psychology and current presumed wave structures (if technical history allows).
4. A bolded final 'Analyst Verdict' summarizing the risk profile.
"""

if mode == "Analyze Single Ticker":
    user_ticker = st.text_input("Enter Stock Ticker (e.g., BRCB):").upper()
    if user_ticker and st.button("Generate Tear Sheet"):
        if not api_key:
            st.error("⚠️ You must enter your Gemini API Key first!")
        else:
            with st.spinner("Analyzing data and generating report..."):
                passed, metrics = check_quantitative_trigger(user_ticker)
                if not metrics:
                    metrics = {"Price": "Data fetch restricted, relying on AI synthesis."}
                
                try:
                    # Updated client construction and modern gemini-2.5-pro model execution
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model='gemini-2.5-pro',
                        contents=analysis_prompt_template.format(ticker=user_ticker, metrics=metrics)
                    )
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI Error: {e}")

elif mode == "Run Market Scanner (Buy Trigger)":
    if st.button("Launch Scan Now"):
        if not api_key:
            st.error("⚠️ You must enter your Gemini API Key first!")
        else:
            st.write("Scanning watchlists...")
            triggered_stocks = []
            
            for ticker in TICKER_POOL:
                passed, metrics = check_quantitative_trigger(ticker)
                if passed:
                    triggered_stocks.append({"Ticker": ticker, **metrics})
            
            if triggered_stocks:
                st.success(f"Found {len(triggered_stocks)} stocks meeting momentum thresholds!")
                st.dataframe(triggered_stocks)
                
                top_stock = triggered_stocks[0]["Ticker"]
                st.subheader(f"Deep-Dive Analyst Report for Top Triggered Stock: {top_stock}")
                
                with st.spinner("Executing Wall Street evaluation..."):
                    try:
                        # Updated client construction and modern gemini-2.5-pro model execution
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model='gemini-2.5-pro',
                            contents=analysis_prompt_template.format(ticker=top_stock, metrics=triggered_stocks[0])
                        )
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI Error: {e}")
            else:
                st.warning("No stocks currently meet the criteria.")
