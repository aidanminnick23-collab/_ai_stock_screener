import streamlit as st
import yfinance as yf
import google.generativeai as genai

st.title("📊 Wall Street AI Scanner & Tear Sheet Tool")
st.write("Scans tickers, applies a quantitative 'Buy Trigger', and generates AI insights.")

# FIXED: Moved API Key input to the main screen so it's impossible to miss on mobile
api_key = st.text_input("🔑 Enter Gemini API Key", type="password")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.info("Please enter your Gemini API Key above to activate the AI generation.")

# Predefined pool to scan
TICKER_POOL = ["PLSE", "BRCB", "CELH", "SOFI", "HOOD", "DKNG"]

# FIXED: Removed ticker.info entirely to prevent Streamlit Cloud from freezing.
# We now calculate momentum metrics using historical price data, which safely bypasses blocks.
def check_quantitative_trigger(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1mo")
        if hist.empty:
            return False, {}
        
        current_price = hist['Close'].iloc[-1]
        prior_price = hist['Close'].iloc[0]
        one_month_change = ((current_price - prior_price) / prior_price) * 100
        
        # Simple Trigger: Flagging positive momentum over the last month
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
                    model = genai.GenerativeModel('gemini-1.5-pro')
                    response = model.generate_content(analysis_prompt_template.format(ticker=user_ticker, metrics=metrics))
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
                    model = genai.GenerativeModel('gemini-1.5-pro')
                    response = model.generate_content(analysis_prompt_template.format(ticker=top_stock, metrics=triggered_stocks[0]))
                    st.markdown(response.text)
            else:
                st.warning("No stocks currently meet the criteria.")
