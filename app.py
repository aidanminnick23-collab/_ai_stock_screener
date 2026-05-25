import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import requests
from google import genai  

# ==========================================
# CONFIGURATION & INITIALIZATION
# ==========================================
st.set_page_config(page_title="Wall Street AI Dashboard", layout="wide")

st.title("📊 Wall Street AI Scanner & Wealth Dashboard")
st.write("Track custom holdings, calculate advanced trend crossovers, and generate AI insights.")

api_key = st.text_input("🔑 Enter Gemini API Key", type="password")

if "user_portfolio" not in st.session_state:
    st.session_state.user_portfolio = {}

TICKER_POOL = ["PLSE", "BRCB", "CELH", "SOFI", "HOOD", "DKNG"]

INTERVAL_MAPPING = {
    10: {"history": "3mo"},
    20: {"history": "6mo"},
    50: {"history": "1y"},
    100: {"history": "2y"},
    200: {"history": "2y"}
}

# ==========================================
# CORE MATHEMATICAL ENGINE
# ==========================================
def fetch_technical_data(ticker_symbol, period_window, calculation_type):
    try:
        clean_symbol = ticker_symbol.upper().strip()
        
        # Smart Crypto Auto-Correction Patch
        COMMON_CRYPTOS = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT", "LTC"]
        if clean_symbol in COMMON_CRYPTOS:
            clean_symbol = f"{clean_symbol}-USD"
            
        ticker = yf.Ticker(clean_symbol)
        lookback = INTERVAL_MAPPING[period_window]["history"]
        hist = ticker.history(period=lookback)
        
        if hist.empty or len(hist) < period_window:
            return False, {}, None, 0.0
        
        # High-Precision Trend Calculations (Upgraded to handle micro-holdings)
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
        
        current_price = float(hist['Close'].iloc[-1])
        prior_price = float(hist['Close'].iloc[-20]) if len(hist) >= 20 else float(hist['Close'].iloc[0])
        one_month_change = ((current_price - prior_price) / prior_price) * 100
        current_ma_val = float(hist['MA_Line'].iloc[-1])
        
        latest_signal = "Neutral (Consolidating)"
        recent_buys = hist['Buy_Signal'].dropna()
        recent_sells = hist['Sell_Signal'].dropna()
        
        if not recent_buys.empty and (recent_sells.empty or recent_buys.index[-1] > recent_sells.index[-1]):
            latest_signal = f"BUY Trigger ({recent_buys.index[-1].strftime('%m/%d')})"
        elif not recent_sells.empty and (recent_buys.empty or recent_sells.index[-1] > recent_buys.index[-1]):
            latest_signal = f"SELL Trigger ({recent_sells.index[-1].strftime('%m/%d')})"

        metrics = {
            "Price": f"${current_price:.5f}",
            "1-Mo Momentum": f"{one_month_change:.1f}%",
            f"{period_window}-Day {ma_acronym}": f"${current_ma_val:.5f}",
            "Current State": latest_signal
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='#1f77b4', width=2)))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA_Line'], name=ma_acronym, line=dict(color='#ff7f0e', width=1.5, dash='dot')))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Buy_Signal'], mode='markers', name='BUY', marker=dict(color='#2ca02c', size=10, symbol='triangle-up')))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Sell_Signal'], mode='markers', name='SELL', marker=dict(color='#d62728', size=10, symbol='triangle-down')))
        fig.update_layout(title=f"{clean_symbol} Technical View", xaxis_title="Date", yaxis_title="Price ($)", hovermode="x unified", template="plotly_dark", margin=dict(l=10,r=10,t=40,b=10))
        
        return one_month_change > 0, metrics, fig, current_price
    except Exception as e:
        return False, {}, None, 0.0

# ==========================================
# APP ROUTING & UI LAYOUT
# ==========================================
col_ma, col_per = st.columns(2)
with col_ma:
    ma_type = st.radio("📐 Average Methodology", ["Simple Moving Average (SMA)", "Weighted Moving Average (WMA)"])
with col_per:
    sma_period = st.selectbox("📈 Lookback Duration", options=list(INTERVAL_MAPPING.keys()), index=1, format_func=lambda x: f"{x}-Day Window")

mode_selection = st.radio("Select Application Mode", ["Portfolio Dashboard", "Analyze Single Ticker", "Run Market Scanner"])

# ==========================================
# MODE 1: PORTFOLIO DASHBOARD (MAIN VIEW)
# ==========================================
if mode_selection == "Portfolio Dashboard":
    st.header("💼 Personal Holding Monitor")
    
    # 1. Robust Cloud Sync Engine
    with st.sidebar.expander("🌐 Cloud Vault Sync", expanded=True):
        st.write("Sync your portfolio instantly using a secure connection.")
        vault_user = st.text_input("Vault Username/ID", value="amin")
        user_pin = st.text_input("Secret Security PIN", max_chars=4, type="password")
        col_load, col_save = st.columns(2)
        SYSTEM_BUCKET = "wallstreet_ai_wealth_dash_v4"
        composite_key = f"{vault_user.strip()}_{user_pin.strip()}" if vault_user and user_pin else ""
        
        with col_save:
            if st.button("💾 Cloud Save"):
                if not vault_user or not user_pin or len(user_pin) != 4 or not user_pin.isdigit():
                    st.error("Enter a valid username and 4-digit numeric PIN.")
                elif not st.session_state.user_portfolio:
                    st.warning("Portfolio workspace is empty.")
                else:
                    try:
                        # Proper Bucket Initialization to prevent 400/404 errors
                        init_url = f"https://kvdb.io/{SYSTEM_BUCKET}"
                        requests.post(init_url, data={'email': 'admin@wallstreetdash.com'}, timeout=5)
                        
                        # Now push payload
                        url = f"https://kvdb.io/{SYSTEM_BUCKET}/{composite_key}"
                        res = requests.put(url, json=st.session_state.user_portfolio, timeout=7)
                        if res.status_code in [200, 201]: 
                            st.success("Portfolio backed up successfully!")
                        else: 
                            st.error(f"Sync failed (Status {res.status_code})")
                    except Exception as e: 
                        st.error(f"Cloud vault connection failed: {e}")
                        
        with col_load:
            if st.button("🔄 Cloud Load"):
                if not vault_user or not user_pin or len(user_pin) != 4 or not user_pin.isdigit():
                    st.error("Enter a username and 4-digit PIN.")
                else:
                    try:
                        url = f"https://kvdb.io/{SYSTEM_BUCKET}/{composite_key}"
                        res = requests.get(url, timeout=7)
                        if res.status_code == 200:
                            st.session_state.user_portfolio = res.json()
                            st.success("Records loaded!")
                            st.rerun()
                        elif res.status_code == 404: 
                            st.error("PIN configuration not found.")
                        else: 
                            st.error(f"Transmission aborted (Status {res.status_code})")
                    except Exception as e: 
                        st.error("Cloud interface unreachable.")

    # 2. Sidebar Quick Position Editor (High Precision Handling)
    with st.sidebar.expander("🛠️ Add/Remove Position", expanded=True):
        existing_assets = [""] + list(st.session_state.user_portfolio.keys())
        selected_edit_ticker = st.selectbox("Quick-Select Active Asset to Edit", options=existing_assets, index=0)
        
        default_shares = 0.0
        default_cost = 0.0
        ticker_input_val = selected_edit_ticker if selected_edit_ticker else ""
        
        if selected_edit_ticker:
            ticker_input_val = selected_edit_ticker
            default_shares = float(st.session_state.user_portfolio[selected_edit_ticker]["shares"])
            default_cost = float(st.session_state.user_portfolio[selected_edit_ticker]["cost"])

        edit_ticker = st.text_input("Ticker Symbol", value=ticker_input_val).upper().strip()
        edit_shares = st.number_input("Shares Owned (Set to 0 to Delete)", min_value=0.0, step=0.00001, format="%.5f", value=default_shares)
        edit_cost = st.number_input("Average Purchase Cost ($)", min_value=0.0, step=0.00001, format="%.5f", value=default_cost)
        
        if st.button("Apply Quick Change"):
            if edit_ticker:
                if edit_shares > 0:
                    st.session_state.user_portfolio[edit_ticker] = {"shares": edit_shares, "cost": edit_cost}
                    st.rerun()
                elif edit_ticker in st.session_state.user_portfolio and edit_shares == 0:
                    del st.session_state.user_portfolio[edit_ticker]
                    st.rerun()

    if not st.session_state.user_portfolio:
        st.info("Your dashboard workspace is currently empty. Use the sidebar tools to add assets or load a Cloud Vault.")
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
                    # High precision float handling essential for micro crypto fractions
                    position_cost = float(details['shares']) * float(details['cost'])
                    position_value = float(details['shares']) * current_price
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
                        "Shares": float(details['shares']),
                        "Avg Cost": float(details['cost']),
                        "Current Price": current_price,
                        "Market Value": position_value,
                        "Return ($)": position_gain,
                        "Return (%)": position_gain_pct,
                        "Trend Signal": metrics.get("Current State", "Calculating")
                    })

        total_gain = total_market_value - total_cost_basis
        total_gain_pct = (total_gain / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
        
        # Performance KPIs Ribbon Banner
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Asset Value", f"${total_market_value:,.2f}")
        kpi2.metric("Net Cost Basis", f"${total_cost_basis:,.2f}")
        kpi3.metric("Total Performance Return", f"${total_gain:,.2f}", f"{total_gain_pct:.2f}%")
        
        # 3. Intelligent Stratified Donut Chart Engine
        if pie_values:
            df_pie = pd.DataFrame({"Asset": pie_labels, "Value": pie_values})
            df_pie = df_pie.sort_values(by="Value", ascending=False).reset_index(drop=True)
            df_pie["Percentage"] = (df_pie["Value"] / total_market_value) * 100
            
            # Use Pandas slicing to segment the positions safely
            top_5_core = df_pie.iloc[0:5]
            next_5_mid = df_pie.iloc[5:10]
            remaining_micro = df_pie.iloc[10:]
            
            final_slices = []
            hover_templates = []
            
            # A. Core Top 5 (Labeled and always visible)
            for _, row in top_5_core.iterrows():
                final_slices.append({
                    "Asset": row["Asset"],
                    "Value": row["Value"],
                    "Percentage": row["Percentage"],
                    "StaticLabel": f"{row['Asset']}<br>{row['Percentage']:.1f}%"
                })
                hover_templates.append(f"<b>Core Position:</b> {row['Asset']}<br><b>Market Value:</b> ${row['Value']:,.2f}<br><b>Allocation Weight:</b> {row['Percentage']:.1f}%<extra></extra>")
            
            # B. Next 5 Mid-Tier (Binned with dynamic hover breakdown)
            if not next_5_mid.empty:
                mid_val_sum = next_5_mid["Value"].sum()
                mid_pct_sum = next_5_mid["Percentage"].sum()
                
                # Compile dynamic hover data listing contents of the Mid-Tier bin
                mid_lines = [f"• {r['Asset']}: ${r['Value']:,.2f} ({r['Percentage']:.1f}%)" for _, r in next_5_mid.iterrows()]
                detailed_mid_hover = "<br><b>Group Holdings:</b><br>" + "<br>".join(mid_lines)
                
                final_slices.append({
                    "Asset": "Next 5 Mid-Tier Holdings",
                    "Value": mid_val_sum,
                    "Percentage": mid_pct_sum,
                    "StaticLabel": f"Mid-Tier Holdings<br>{mid_pct_sum:.1f}%"
                })
                hover_templates.append(f"<b>Next 5 Mid-Tier Holdings</b><br>Total Group Value: ${mid_val_sum:,.2f}<br>Total Group Weight: {mid_pct_sum:.1f}%{detailed_mid_hover}<extra></extra>")
                
            # C. Troubleshooting Bin (Any other remaining assets spill over here)
            if not remaining_micro.empty:
                rem_val_sum = remaining_micro["Value"].sum()
                rem_pct_sum = remaining_micro["Percentage"].sum()
                
                # Compile spillover breakdown hover list
                rem_lines = [f"• {r['Asset']}: ${r['Value']:,.2f} ({r['Percentage']:.1f}%)" for _, r in remaining_micro.iterrows()]
                detailed_rem_hover = "<br><b>Remaining Assets:</b><br>" + "<br>".join(rem_lines)
                
                final_slices.append({
                    "Asset": "Other Remaining Positions",
                    "Value": rem_val_sum,
                    "Percentage": rem_pct_sum,
                    "StaticLabel": "" # Hidden static text prevents chart clumping
                })
                hover_templates.append(f"<b>Other Tail End Assets</b><br>Combined Spillover Value: ${rem_val_sum:,.2f}<br>Combined Spillover Weight: {rem_pct_sum:.1f}%{detailed_rem_hover}<extra></extra>")

            df_final_pie = pd.DataFrame(final_slices)
            fintech_colors = ["#1f77b4", "#00b4d8", "#0077b6", "#0096c7", "#03045e", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#457b9d"]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=df_final_pie["Asset"], 
                values=df_final_pie["Value"], 
                hole=0.45,
                text=df_final_pie["StaticLabel"],
                textinfo="text",
                textposition="inside", 
                hovertext=hover_templates,
                hoverinfo="text",
                automargin=True,
                marker=dict(colors=fintech_colors[:len(df_final_pie)], line=dict(color='#111111', width=2))
            )])
            
            fig_pie.update_layout(
                title=dict(text="🎯 Real-Time Strategic Asset Allocation Weighting", x=0.5, y=0.97, font=dict(size=18, family="Helvetica Neue, Arial, sans-serif", color="#ffffff")),
                height=650, 
                template="plotly_dark",
                margin=dict(l=40, r=40, t=90, b=80), 
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5, font=dict(size=11, color="#cccccc"))
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # 4. Fully Interactive High-Precision Position Editor Table
        st.subheader("📋 Your Monitored Assets Summary")
        st.write("💡 *Tip: Double-click any cell under 'Shares' or 'Avg Cost' to edit your holdings right inside the table row.*")
        
        df_summary = pd.DataFrame(display_data)
        
        edited_df = st.data_editor(
            df_summary,
            column_config={
                "Asset": st.column_config.TextColumn("Asset", disabled=True),
                # shares upgraded to support up to 5 decimal significant figures
                "Shares": st.column_config.NumberColumn("Shares owned", min_value=0.0, format="%.5f", step=0.00001),
                "Avg Cost": st.column_config.NumberColumn("Avg Cost ($)", min_value=0.0, format="$%.5f", step=0.00001),
                "Current Price": st.column_config.NumberColumn("Current Price", disabled=True, format="$%.5f"),
                "Market Value": st.column_config.NumberColumn("Market Value", disabled=True, format="$%.2f"),
                "Return ($)": st.column_config.NumberColumn("Return ($)", disabled=True, format="$%.2f"),
                "Return (%)": st.column_config.NumberColumn("Return (%)", disabled=True, format="%.1f%%"),
                "Trend Signal": st.column_config.TextColumn("Trend Signal", disabled=True)
            },
            disabled=["Asset", "Current Price", "Market Value", "Return ($)", "Return (%)", "Trend Signal"],
            use_container_width=True,
            key="portfolio_inline_editor"
        )
        
        # Commit inline modifications back to session state on save
        if st.button("💾 Save Table Modifications", type="primary"):
            has_changes = False
            for idx, row in edited_df.iterrows():
                clean_ticker = row["Asset"].replace("-USD", "")
                target_shares = float(row["Shares"])
                target_cost = float(row["Avg Cost"])
                baseline = st.session_state.user_portfolio.get(clean_ticker, {"shares": 0.0, "cost": 0.0})
                
                # Check for significant numerical deviations using float tolerance
                if abs(target_shares - float(baseline["shares"])) > 1e-7 or abs(target_cost - float(baseline["cost"])) > 1e-7:
                    has_changes = True
                    if target_shares == 0:
                        if clean_ticker in st.session_state.user_portfolio:
                            del st.session_state.user_portfolio[clean_ticker]
                    else:
                        st.session_state.user_portfolio[clean_ticker] = {"shares": target_shares, "cost": target_cost}
            
            if has_changes:
                st.success("Modifications saved successfully!")
                st.rerun()
            else:
                st.info("No modifications detected.")

        # 5. Holistic AI Reporting Engine
        st.markdown("---")
        st.subheader("🧠 Holistic Wealth & Diversification Audit")
        st.write("Passes your entire portfolio to Gemini to run cross-asset correlation checks and risk reviews.")
        
        if st.button("Generate Full Portfolio AI Macro Report"):
            if not api_key:
                st.error("⚠️ Gemini API Key required.")
            else:
                with st.spinner("Executing structural asset-correlation matrix analysis..."):
                    # Cast summary view back to formatted string representation for the AI context
                    ai_export_df = edited_df.copy()
                    ai_export_df["Shares"] = ai_export_df["Shares"].map(lambda x: f"{x:.5f}")
                    ai_export_df["Market Value"] = ai_export_df["Market Value"].map(lambda x: f"${x:,.2f}")
                    ai_export_df["Return ($)"] = ai_export_df["Return ($)"].map(lambda x: f"${x:,.2f}")
                    ai_export_df["Return (%)"] = ai_export_df["Return (%)"].map(lambda x: f"{x:.1f}%")
                    
                    portfolio_analysis_prompt = f"""
                    You are an elite Wall Street Managing Director and Chief Wealth Management Strategist. 
                    Perform a high-level strategic review on this client investment portfolio matrix:
                    Total Wealth Under Management: ${total_market_value:,.2f}
                    Consolidated Cost Basis: ${total_cost_basis:,.2f}
                    Net Unreleased Returns: ${total_gain:,.2f} ({total_gain_pct:.2f}%)
                    
                    Held Asset Layout:
                    {ai_export_df.to_json(orient="records", indent=2)}
                    
                    Active Macro Tracking Rule: {sma_period}-Day lookback using {ma_type} parameters.

                    Provide a comprehensive executive advisory response with the following exact components:
                    1. A Markdown table named 'Portfolio Diversification Analysis' ranking assets by capital weight, concentration tier, and risk status.
                    2. Macro Risk & Correlation Assessment explicitly auditing underlying vulnerabilities (sector over-exposure, cash balance lags).
                    3. Actionable Rebalancing Recommendations detailing exactly which assets to hold, skim, or accumulate.
                    4. A final bolded 'Chief Investment Officer (CIO) Mandate' outlining immediate steps for wealth preservation and capital growth.
                    """
                    try:
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(model='gemini-2.5-flash', contents=portfolio_analysis_prompt)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI Engine Error: {e}")
        
        # 6. Single Asset Chart Drill-Down
        st.markdown("---")
        st.subheader("🎯 Single Asset Chart Drill-Down")
        selected_chart_ticker = st.selectbox("Choose a holding select for historical indicators", options=list(st.session_state.user_portfolio.keys()))
        
        if selected_chart_ticker in saved_charts and saved_charts[selected_chart_ticker] is not None:
            st.plotly_chart(saved_charts[selected_chart_ticker], use_container_width=True)

# ==========================================
# MODE 2: SINGLE TICKER TEAR SHEET
# ==========================================
elif mode_selection == "Analyze Single Ticker":
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
                        contents=f"Institutional Analyst Assessment for {user_ticker}. Financial Profile: {metrics}. Apply {sma_period}-day {ma_type} overlay. Generate Quantitative Screen Tear Sheet table, Elliott Wave psychological map, and final bolded Analyst Verdict."
                    )
                    st.markdown(response.text)
                except Exception as e: st.error(f"Error: {e}")

# ==========================================
# MODE 3: MARKET SCANNER
# ==========================================
elif mode_selection == "Run Market Scanner":
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
                st.dataframe(pd.DataFrame(triggered_stocks), use_container_width=True)
                st.plotly_chart(saved_figs[triggered_stocks[0]["Ticker"]], use_container_width=True)
            else:
                st.warning("No assets currently trigger momentum thresholds.")
