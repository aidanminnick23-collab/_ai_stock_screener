# ================================================================
# WALL STREET AI DASHBOARD — Production Build v2.7
# Fix:  KeyError crash (empty rows, 100/200-day periods)
# Fix:  Wilder RSI (ewm alpha=1/14), NaN-safe extraction
# Fix:  200-day now fetches 5y of data
# New:  Multi-indicator confluence scoring for scanner
#       Every ticker scored 0-14 across MA, RSI, MACD, Volume,
#       Momentum, and Fibonacci — only top setups surface
# ================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib, json, requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai

st.set_page_config(
    page_title="Wall Street AI Dashboard", page_icon="📊",
    layout="wide", initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# SECRETS
# ================================================================
def _secret(k):
    try: return st.secrets[k]
    except: return ""

GEMINI_API_KEY = _secret("GEMINI_API_KEY")
SUPABASE_URL   = _secret("SUPABASE_URL")
SUPABASE_KEY   = _secret("SUPABASE_KEY")

AI_AVAILABLE  = bool(GEMINI_API_KEY)
gemini_client = None
GEMINI_MODEL_CANDIDATES = [
    "gemini-2.5-pro", "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-exp-03-25", "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-flash", "gemini-1.5-pro",
]
if AI_AVAILABLE:
    try: gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except: AI_AVAILABLE = False

SUPABASE_AVAILABLE = False
db = None
try:
    if SUPABASE_URL and SUPABASE_KEY:
        from supabase import create_client
        db = create_client(SUPABASE_URL, SUPABASE_KEY)
        SUPABASE_AVAILABLE = True
except: pass

# ================================================================
# CONSTANTS
# ================================================================
CRYPTO_TICKERS = {
    "Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD",
    "XRP": "XRP-USD", "Solana (SOL)": "SOL-USD",
}
KNOWN_CRYPTO_SYMBOLS = {
    "BTC","ETH","XRP","SOL","DOGE","ADA","DOT","AVAX","MATIC","LINK",
    "LTC","BCH","UNI","ATOM","FIL","ALGO","VET","THETA","TRX","EOS",
    "XLM","NEO","IOTA","DASH","SHIB","PEPE","WIF","BONK","SUI","APT",
    "ARB","OP","INJ","NEAR","FTM","HBAR","ICP","SAND","MANA","AXS",
    "CRO","EGLD","FLOW","CHZ","ZEC","XTZ","AAVE","MKR","SNX","COMP",
    "YFI","SUSHI","CRV","1INCH","BAT","ENJ","GRT","BAL","ZRX","LRC",
}

# FIX: 200-day now fetches 5 years to ensure enough bars for all stocks
INTERVAL_MAP = {
    10:  {"history": "3mo",  "label": "10-Day  · Short-Term Momentum"},
    20:  {"history": "6mo",  "label": "20-Day  · Short-Term Trend"},
    50:  {"history": "1y",   "label": "50-Day  · Institutional Baseline"},
    100: {"history": "2y",   "label": "100-Day · Macro Cycle Support"},
    200: {"history": "5y",   "label": "200-Day · Ultimate Bull/Bear Line"},
}
FIB_RATIOS = {
    "78.6%": 0.786, "61.8%": 0.618, "50.0%": 0.500,
    "38.2%": 0.382, "23.6%": 0.236,
}
FIB_COLORS = {
    "78.6%": "rgba(255,82,82,0.55)",   "61.8%": "rgba(255,167,38,0.65)",
    "50.0%": "rgba(255,238,88,0.60)",  "38.2%": "rgba(102,187,106,0.65)",
    "23.6%": "rgba(79,195,247,0.55)",
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":     "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_BUY_KEYWORDS   = ["purchase","bought","acqui"]
_SELL_KEYWORDS  = ["sale","sold"]
_NOISE_KEYWORDS = ["gift","award","grant","automatic","plan sale","tax withholding",
                   "exercise","dispose","reclassif","return","forfeiture","conversion"]

# ================================================================
# CONFLUENCE SCORING ENGINE
# Every indicator contributes points — only high-conviction setups
# score high enough to surface in the scanner
# ================================================================
# Maximum score: 14 points
# 🔥 Exceptional: 11-14   (all indicators aligned)
# 🟢 Strong Buy:  8-10    (most indicators aligned)
# 🟡 Moderate Buy:6-7     (good setup, some confirmation missing)
# ⚪ Watch List:  4-5     (early signal, wait for confirmation)
# —  Weak:       0-3     (insufficient evidence)

SCORE_THRESHOLDS = {
    "🔥 Exceptional only  (≥11)":      11,
    "🟢 Strong Buy+       (≥8)":        8,
    "🟡 Moderate Buy+     (≥6)":        6,
    "⚪ Watch List+       (≥4)":        4,
}

def calc_confluence_score(
    is_ma_crossover: bool,
    price_above_ma: bool,
    rsi: float,
    macd: float,
    macd_sig: float,
    price: float,
    bbu: float,
    bbl: float,
    vol_ratio: float,
    momentum: float,
    fib_levels: dict,
) -> tuple:
    """
    Score a ticker on 6 dimensions. Returns (score, label, breakdown_dict).

    Scoring breakdown:
      MA Signal   — 0–3 pts  (fresh crossover most valuable)
      RSI         — 0–2 pts  (ideal 40-65 range, or oversold <35)
      MACD        — 0–2 pts  (bullish cross + positive territory)
      Volume      — 0–2 pts  (confirms conviction)
      Momentum    — 0–2 pts  (1-month price trend)
      Fibonacci   — 0–1 pt   (near key support = higher probability)
      Bollinger   — 0–2 pts  (position within bands)
    """
    score = 0
    breakdown = {}

    # ── 1. MA Signal (0–3) ────────────────────────────────────
    if is_ma_crossover:
        score += 3
        breakdown["MA Crossover"] = ("🟢 Fresh BUY crossover — price just broke above MA", 3)
    elif price_above_ma:
        score += 1
        breakdown["MA Crossover"] = ("🟡 Price above MA — uptrend intact, no fresh signal", 1)
    else:
        breakdown["MA Crossover"] = ("🔴 Price below MA — bearish structure", 0)

    # ── 2. RSI / Wilder (0–2) ─────────────────────────────────
    # Ideal: not overbought but with building momentum (40–65)
    # Also flag oversold bounces as moderate setup
    if 40 <= rsi <= 65:
        score += 2
        breakdown["RSI"] = (f"🟢 RSI {rsi:.0f} — healthy momentum, room to run", 2)
    elif 65 < rsi <= 70:
        score += 1
        breakdown["RSI"] = (f"🟡 RSI {rsi:.0f} — approaching overbought, still ok", 1)
    elif rsi < 35:
        score += 1
        breakdown["RSI"] = (f"🟡 RSI {rsi:.0f} — oversold bounce setup", 1)
    else:
        breakdown["RSI"] = (f"🔴 RSI {rsi:.0f} — overbought or weak momentum", 0)

    # ── 3. MACD (0–2) ─────────────────────────────────────────
    # Best: bullish cross AND in positive territory (both EMAs converging up)
    macd_bullish = macd > macd_sig
    macd_positive = macd > 0
    if macd_bullish and macd_positive:
        score += 2
        breakdown["MACD"] = ("🟢 Bullish cross + positive territory — strong momentum confirmation", 2)
    elif macd_bullish:
        score += 1
        breakdown["MACD"] = ("🟡 Bullish cross, negative territory — early reversal signal", 1)
    else:
        breakdown["MACD"] = ("🔴 Bearish MACD cross — selling pressure dominant", 0)

    # ── 4. Volume (0–2) ───────────────────────────────────────
    if vol_ratio >= 1.5:
        score += 2
        breakdown["Volume"] = (f"🟢 {vol_ratio:.1f}× average — strong conviction behind the move", 2)
    elif vol_ratio >= 1.0:
        score += 1
        breakdown["Volume"] = (f"🟡 {vol_ratio:.1f}× average — moderate, acceptable confirmation", 1)
    else:
        breakdown["Volume"] = (f"🔴 {vol_ratio:.1f}× average — light volume, move lacks conviction", 0)

    # ── 5. 1-Month Momentum (0–2) ─────────────────────────────
    if momentum >= 7.0:
        score += 2
        breakdown["Momentum"] = (f"🟢 +{momentum:.1f}% (1-month) — strong directional trend", 2)
    elif momentum > 0:
        score += 1
        breakdown["Momentum"] = (f"🟡 +{momentum:.1f}% (1-month) — positive, building", 1)
    else:
        breakdown["Momentum"] = (f"🔴 {momentum:.1f}% (1-month) — negative momentum", 0)

    # ── 6. Bollinger Band Position (0–2) ──────────────────────
    bb_range = (bbu - bbl) if (bbu > 0 and bbl > 0 and bbu > bbl) else 0
    if bb_range > 0:
        bb_pos = (price - bbl) / bb_range   # 0=lower band, 1=upper band
        if bb_pos <= 0.30:
            score += 2
            breakdown["Bollinger"] = ("🟢 Near lower band — oversold bounce, high probability setup", 2)
        elif bb_pos <= 0.60:
            score += 1
            breakdown["Bollinger"] = ("🟡 Lower-mid of bands — early upward positioning", 1)
        elif bb_pos >= 0.95:
            breakdown["Bollinger"] = ("🔴 At upper band — overbought, chasing risk", 0)
        else:
            breakdown["Bollinger"] = ("⚪ Mid-upper portion of bands", 0)
    else:
        breakdown["Bollinger"] = ("⚪ Bollinger data unavailable", 0)

    # ── 7. Fibonacci support proximity (0–1 bonus) ────────────
    # Extra point if price is within 2% above a key support level (38.2% or 61.8%)
    key_fibs = {k: v for k, v in fib_levels.items() if k in ("38.2%", "61.8%")}
    for lbl, lvl in key_fibs.items():
        if lvl > 0 and 0 <= (price - lvl) / lvl <= 0.02:
            score += 1
            breakdown["Fibonacci"] = (f"🟢 Within 2% above {lbl} Fib support — high-probability bounce zone (+1 bonus)", 1)
            break
    else:
        if "Fibonacci" not in breakdown:
            breakdown["Fibonacci"] = ("⚪ Not at a key Fibonacci support level", 0)

    # ── Label ──────────────────────────────────────────────────
    if score >= 11:   label = "🔥 Exceptional"
    elif score >= 8:  label = "🟢 Strong Buy"
    elif score >= 6:  label = "🟡 Moderate Buy"
    elif score >= 4:  label = "⚪ Watch List"
    else:             label = "—  Weak Signal"

    return score, label, breakdown

# ================================================================
# TICKER NORMALISATION
# ================================================================
def normalize_ticker(raw: str) -> str:
    t = raw.upper().strip()
    if not t: return t
    if t.endswith("-USD"): return t
    if t in KNOWN_CRYPTO_SYMBOLS: return t + "-USD"
    return t

def ticker_label(ticker: str) -> str:
    if ticker.endswith("-USD"):
        base = ticker[:-4]
        if base in KNOWN_CRYPTO_SYMBOLS: return base
    return ticker

# ================================================================
# INDICATOR GUIDE
# ================================================================
def render_indicator_guide():
    with st.expander("📚  Indicator Guide — What does everything mean?  (click to expand)", expanded=False):
        st.markdown("#### Learn the indicators used in every chart and analysis report")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("---\n### 📈 Moving Averages: EMA, SMA & WMA")
            st.info("**EMA (Exponential) ⭐:** Exponential decay weighting — reacts fast, stays smooth. Industry standard (Bloomberg, TradingView, MACD).\n\n**SMA (Simple):** Equal weight to every day — slower but excellent for 200-day macro trend.\n\n**WMA (Weighted):** Linear weighting — fastest, most false signals.\n\n🔺 Green ▲ = price crossed above → Buy signal\n🔻 Red ▼ = price crossed below → Sell signal")
            st.markdown("---\n### 📉 RSI (Wilder's Smoothing)")
            st.info("**What it is:** 0–100 momentum gauge using Wilder's exponential smoothing — the same method as TradingView, Bloomberg, and Reuters.\n\n🔴 >70 = Overbought — pullback risk\n🟢 <30 = Oversold — bounce candidate\n🟢 40–65 = Ideal range — room to run\n⚪ 30–70 = Neutral")
            st.markdown("---\n### 📊 Volume")
            st.info("**What it is:** Total shares traded. Validates whether a price move has real conviction.\n\n🟢 >1.5× avg + rising price = Strong confirmed move\n🟡 1.0–1.5× avg = Moderate confirmation\n🔴 <1.0× avg = Light volume — suspect move")
        with c2:
            st.markdown("---\n### ⚡ MACD (12 / 26 / 9)")
            st.info("**What it is:** Fast EMA(12) minus slow EMA(26). Always EMA-based regardless of MA selection.\n\n🟢 MACD above Signal + positive territory = Strongest setup\n🟡 MACD above Signal, negative territory = Early reversal\n🔴 MACD below Signal = Bearish momentum")
            st.markdown("---\n### 🎯 Bollinger Bands")
            st.info("**What it is:** 20-day MA ± 2 std deviations.\n\n🟢 Near lower band = Oversold bounce setup\n🟡 Lower-mid = Early upward positioning\n🔴 Near upper band = Overbought, chasing risk\n⚠️ Squeeze = Big move loading")
            st.markdown("---\n### 👔 Insider / CEO Buying")
            st.info("**What it is:** Open-market purchases by executives (SEC Form 4). Personal cash = genuine conviction.\n\n🟢 CEO buying = Strongest bullish signal\n📌 3+ insiders = Cluster buy (highest conviction)\n🔴 Heavy selling = Monitor carefully")
        with c3:
            st.markdown("---\n### 🔢 Scanner Signal Strength Score")
            st.info(
                "Every scanned ticker is scored across 6 dimensions (max 14 pts):\n\n"
                "| Dimension | Max Pts |\n|---|---|\n"
                "| MA Crossover | 3 |\n| RSI Range | 2 |\n"
                "| MACD Position | 2 |\n| Volume Conviction | 2 |\n"
                "| 1-Month Momentum | 2 |\n| Bollinger Position | 2 |\n"
                "| Fibonacci Bonus | 1 |\n\n"
                "🔥 **11+** = Exceptional — all indicators aligned\n"
                "🟢 **8–10** = Strong Buy — most indicators confirmed\n"
                "🟡 **6–7** = Moderate Buy — solid setup\n"
                "⚪ **4–5** = Watch List — early signal\n"
                "— **<4** = Weak — insufficient evidence"
            )
            st.markdown("---\n### 🌊 Elliott Wave & Fibonacci")
            st.info("**Elliott Wave:** Markets move in 5-wave impulse (up) then 3-wave corrective (A-B-C) patterns.\n\n**Fibonacci Levels:**\n🔵 23.6% — Shallow pullback\n🟢 38.2% — Common healthy dip\n🟡 50.0% — Psychological midpoint\n🟠 61.8% — Golden Ratio — strongest support\n🔴 78.6% — Deep retracement\n\n**Scanner bonus:** +1 pt for price within 2% above 38.2% or 61.8% support")

# ================================================================
# PORTFOLIO AUTH
# ================================================================
def hash_pin(pin): return hashlib.sha256(pin.encode()).hexdigest()

def load_portfolio_from_db(user_id, pin):
    try:
        result = db.table("portfolios").select("*").eq("user_id", user_id).execute()
        rows = result.data
        if not rows: return {}
        if rows[0]["pin_hash"] != hash_pin(pin): return None
        return {row["ticker"]: {"shares": row["shares"], "cost": row["cost"]} for row in rows}
    except Exception as e:
        st.error(f"Database error: {e}"); return None

def save_position_to_db(user_id, pin, ticker, shares, cost):
    try:
        if shares == 0:
            db.table("portfolios").delete().eq("user_id", user_id).eq("ticker", ticker).execute()
        else:
            db.table("portfolios").upsert({
                "user_id": user_id, "pin_hash": hash_pin(pin),
                "ticker": ticker, "shares": shares, "cost": cost,
            }, on_conflict="user_id,ticker").execute()
        return True
    except Exception as e:
        st.error(f"Save error: {e}"); return False

# ================================================================
# TICKER LOADERS
# ================================================================
def _read_html_safe(url, **kwargs):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text), **kwargs)

@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers():
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv")
        t = sorted(df["Symbol"].str.replace(".", "-", regex=False).tolist())
        if len(t) > 400: return t
    except: pass
    try:
        tables = _read_html_safe("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", flavor="lxml")
        return sorted(tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception as e:
        st.warning(f"S&P 500 fetch failed: {e}"); return []

@st.cache_data(ttl=86400, show_spinner=False)
def get_sp1500_tickers():
    tickers = set(get_sp500_tickers())
    for url, col in [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Ticker"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Ticker"),
    ]:
        try:
            tables = _read_html_safe(url, flavor="lxml")
            tickers.update(tables[0][col].str.replace(".", "-", regex=False).tolist())
        except: pass
    return sorted(list(tickers))

@st.cache_data(ttl=86400, show_spinner=False)
def get_russell2000_tickers():
    try:
        r = requests.get(
            "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund",
            headers=HEADERS, timeout=20
        )
        df = pd.read_csv(StringIO(r.text), skiprows=9)
        df = df[df.get("Asset Class", df.columns[0]) == "Equity"]
        valid = sorted([t for t in df["Ticker"].dropna().str.strip().tolist() if t and t != "-" and len(t) <= 6])
        if len(valid) > 500: return valid
    except: pass
    try:
        tables = _read_html_safe("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", flavor="lxml")
        st.caption("ℹ️ Russell 2000 unavailable — using S&P SmallCap 600 as proxy.")
        return sorted(tables[0]["Ticker"].str.replace(".", "-", regex=False).tolist())
    except:
        st.warning("Russell 2000 unavailable. Try S&P 500."); return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_us_equities():
    try:
        r = requests.get(
            "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&download=true",
            headers=HEADERS, timeout=25
        )
        df = pd.DataFrame(r.json()["data"]["rows"])
        df["volume"] = pd.to_numeric(df["volume"].astype(str).str.replace(",","",regex=False), errors="coerce")
        t = sorted(df[df["volume"] > 500_000]["symbol"].str.strip().tolist())
        if len(t) > 100: return t
    except: pass
    st.caption("ℹ️ Full equity list unavailable — using S&P 1500.")
    return get_sp1500_tickers()

# ================================================================
# INSIDER DATA
# ================================================================
def _categorize_tx(text):
    t = str(text).lower()
    if any(x in t for x in _NOISE_KEYWORDS): return "⚪ Non-Market"
    if any(x in t for x in _BUY_KEYWORDS):   return "🟢 Open Market Buy"
    if any(x in t for x in _SELL_KEYWORDS):  return "🔴 Open Market Sale"
    return "⚪ Other"

@st.cache_data(ttl=600, show_spinner=False)
def get_insider_transactions(symbol):
    try:
        raw = yf.Ticker(symbol).insider_transactions
        if raw is None or raw.empty: return None
        df = raw.copy()
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)
            df = df[df[date_col] >= cutoff]
            df["Date"] = df[date_col].dt.strftime("%m/%d/%Y")
        text_col = next((c for c in df.columns if c.lower() in ["text","description","transaction"]), None)
        df["Transaction Type"] = df[text_col].apply(_categorize_tx) if text_col else "⚪ Unknown"
        def _find(patterns):
            for p in patterns:
                col = next((c for c in df.columns if p in c.lower()), None)
                if col: return col
            return None
        name_col = _find(["insider","name"]); pos_col = _find(["position","title","role"])
        share_col = _find(["share"]); val_col = _find(["value"])
        clean = pd.DataFrame()
        if "Date" in df.columns:  clean["Date"]     = df["Date"]
        if name_col:              clean["Insider"]   = df[name_col]
        if pos_col:               clean["Role"]      = df[pos_col]
        clean["Transaction Type"] = df["Transaction Type"]
        if share_col:
            clean["Shares"] = pd.to_numeric(df[share_col], errors="coerce").apply(
                lambda x: f"{int(x):,}" if pd.notna(x) else "—"
            )
        if val_col:
            vals = pd.to_numeric(df[val_col], errors="coerce")
            clean["Est. Value ($)"] = vals.apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "—")
            clean["_raw_value"] = vals
        if text_col: clean["Description"] = df[text_col]
        return clean if not clean.empty else None
    except: return None

def insider_summary(df):
    if df is None: return "No insider transaction data available."
    if "Transaction Type" not in df.columns: return "Insider data available but format unrecognised."
    buys  = df[df["Transaction Type"] == "🟢 Open Market Buy"]
    sells = df[df["Transaction Type"] == "🔴 Open Market Sale"]
    buy_val  = df.loc[df["Transaction Type"] == "🟢 Open Market Buy",  "_raw_value"].sum() if "_raw_value" in df.columns else 0
    sell_val = df.loc[df["Transaction Type"] == "🔴 Open Market Sale", "_raw_value"].sum() if "_raw_value" in df.columns else 0
    s = f"Insider open-market activity (last 12 months): {len(buys)} purchase(s) ~${buy_val:,.0f}, {len(sells)} sale(s) ~${sell_val:,.0f}. "
    if len(buys) > 0 and len(sells) == 0: s += "✅ Buying with no selling — very bullish."
    elif len(buys) >= 3:                  s += "✅ Cluster insider buying — high conviction."
    elif len(buys) > 0:                   s += "Mild insider buying — modestly bullish."
    elif sell_val > buy_val * 3 > 0:      s += "⚠️ Heavy net insider selling — warrants caution."
    else:                                 s += "No open-market purchases detected."
    return s

def generate_insider_ai_analysis(symbol, df):
    if not AI_AVAILABLE: return "⚠️ AI unavailable."
    if df is None: return "No insider data available."
    display_cols = [c for c in ["Date","Insider","Role","Transaction Type","Shares","Est. Value ($)","Description"] if c in df.columns]
    table_text = df[display_cols].to_string(index=False) if display_cols else "Data unavailable"
    prompt = f"""SEC Form 4 insider analysis for {symbol}.\n{table_text}\n
"Open Market Buy" = personal cash = TRUE signal. "Non-Market" = compensation, IGNORE.

Four sections:
## 👔 Insider Transaction Summary Table (Open Market only, bold purchases)
## 🔍 Signal Interpretation (role context, cluster signal, plan vs discretionary sales)
## 📊 Historical Context (typical level, academic evidence, red/green flags)
## 🎯 Insider Signal Verdict (**Bold:** Bullish/Neutral/Bearish, weight vs technicals, 1 sentence for beginners)"""
    errors = []
    for model in GEMINI_MODEL_CANDIDATES:
        try:
            resp = gemini_client.models.generate_content(model=model, contents=prompt)
            return f"*Model: `{model}`*\n\n" + resp.text
        except Exception as e: errors.append(f"**{model}:** {str(e)[:100]}")
    return "⚠️ Failed.\n" + "\n".join(f"- {e}" for e in errors)

def render_insider_section(symbol):
    with st.expander("👔  Insider & Executive Transactions (last 12 months)", expanded=False):
        with st.spinner("Loading SEC Form 4 data…"):
            df = get_insider_transactions(symbol)
        if df is None:
            st.info("No recent insider transaction data found."); return
        st.caption("Source: SEC Form 4 via Yahoo Finance. Open Market = personal cash = genuine conviction.")
        show_noise = st.toggle("Show Non-Market entries (awards, grants, gifts)", value=False, key=f"noise_{symbol}")
        display_df = df if show_noise else df[df["Transaction Type"] != "⚪ Non-Market"]
        display_df = display_df[[c for c in display_df.columns if c != "_raw_value"]]
        if display_df.empty:
            st.info("No open-market transactions after filtering noise.")
        else:
            def highlight_row(row):
                if "Buy"  in str(row.get("Transaction Type","")): return ["background-color: rgba(76,175,80,0.12)"] * len(row)
                if "Sale" in str(row.get("Transaction Type","")): return ["background-color: rgba(244,67,54,0.08)"] * len(row)
                return [""] * len(row)
            st.dataframe(display_df.style.apply(highlight_row, axis=1), use_container_width=True, hide_index=True)
        buys  = df[df["Transaction Type"] == "🟢 Open Market Buy"]
        sells = df[df["Transaction Type"] == "🔴 Open Market Sale"]
        other = df[df["Transaction Type"] == "⚪ Non-Market"]
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Open Market Purchases", len(buys), delta="Bullish" if len(buys) > 0 else None)
        k2.metric("Open Market Sales", len(sells), delta="Monitor" if len(sells) > len(buys)*2 else None,
                  delta_color="inverse" if len(sells) > len(buys)*2 else "normal")
        k3.metric("Non-Market (Noise)", len(other))
        net = len(buys)-len(sells)
        k4.metric("Net Signal", f"{'🟢 Bullish' if net>0 else '🔴 Caution' if net<-2 else '⚪ Neutral'}",
                  delta=f"{abs(net)} tx net {'buying' if net>0 else 'selling'}")
        ai_key = f"insider_ai_{symbol}"
        if st.button("🤖  AI Insider Signal Analysis", key=f"btn_{ai_key}"):
            with st.spinner("Analysing insider patterns…"):
                st.session_state[ai_key] = generate_insider_ai_analysis(symbol, df)
        if ai_key in st.session_state:
            st.markdown("---"); st.markdown(st.session_state[ai_key])
            if st.button("🗑️  Clear Insider Report", key=f"clr_{ai_key}"):
                del st.session_state[ai_key]; st.rerun()

# ================================================================
# PERFORMANCE CACHE
# ================================================================
@st.cache_data(ttl=300, show_spinner=False)
def _cached_history(symbol: str, period: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=period)
        return df.ffill() if not df.empty else df
    except: return pd.DataFrame()

# ================================================================
# CORE TECHNICAL ANALYSIS ENGINE — bulletproof v2.7
# Returns 7 values: (is_bullish, metrics, fig, price, fib, score, error)
# ================================================================
def _safe_last(series, fallback=0.0):
    """Return last non-NaN value or fallback."""
    valid = series.dropna()
    return float(valid.iloc[-1]) if not valid.empty else fallback

def fetch_technical_data(symbol: str, period_window: int, calc_type: str):
    err_pfx = f"[{symbol}/{period_window}d/{calc_type[:3]}]"
    try:
        lookback = INTERVAL_MAP[period_window]["history"]
        hist     = _cached_history(symbol, lookback)

        # Adaptive minimum — require the MA window + enough for MACD/RSI/BB
        # but allow 80% of the MA window if data is limited (e.g. small-caps)
        hard_need = max(period_window, 35, 20) + 2
        soft_min  = max(int(period_window * 0.80), 35)

        if hist.empty:
            return False, {}, None, 0.0, {}, 0, f"{err_pfx} No data returned from yfinance."
        if len(hist) < soft_min:
            return False, {}, None, 0.0, {}, 0, (
                f"{err_pfx} Only {len(hist)} bars available — need ≥{soft_min} "
                f"for {period_window}-day analysis. Try a shorter lookback."
            )

        # Trim to working window
        hist  = hist.tail(min(len(hist), hard_need + 100)).copy()
        close = hist["Close"].copy()

        # ── Moving Average ─────────────────────────────────────
        if "Exponential" in calc_type:
            hist["MA"] = close.ewm(span=period_window, adjust=False).mean()
            ma_label   = f"EMA-{period_window}"
        elif "Simple" in calc_type:
            hist["MA"] = close.rolling(period_window).mean()
            ma_label   = f"SMA-{period_window}"
        else:
            w = np.arange(1, period_window + 1, dtype=float)
            hist["MA"] = close.rolling(period_window).apply(
                lambda p: float(np.dot(p, w) / w.sum()), raw=True
            )
            ma_label = f"WMA-{period_window}"

        # ── Bollinger Bands ────────────────────────────────────
        bb_mid           = close.rolling(20).mean()
        bb_std           = close.rolling(20).std(ddof=0)
        hist["BB_Upper"] = bb_mid + 2 * bb_std
        hist["BB_Mid"]   = bb_mid
        hist["BB_Lower"] = bb_mid - 2 * bb_std

        # ── RSI — Wilder's smoothing (ewm alpha=1/14) ─────────
        delta    = close.diff()
        avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        hist["RSI"] = (100 - (100 / (1 + rs))).clip(0, 100)

        # ── MACD (always EMA 12/26/9) ─────────────────────────
        ema12            = close.ewm(span=12, adjust=False).mean()
        ema26            = close.ewm(span=26, adjust=False).mean()
        hist["MACD"]     = ema12 - ema26
        hist["MACD_Sig"] = hist["MACD"].ewm(span=9, adjust=False).mean()
        hist["MACD_H"]   = hist["MACD"] - hist["MACD_Sig"]

        # ── Fibonacci ──────────────────────────────────────────
        sh = float(close.max()); sl = float(close.min())
        fib_levels = {lbl: sl + r * (sh - sl) for lbl, r in FIB_RATIOS.items()}

        # ── MA Crossover signals ───────────────────────────────
        ma_valid   = hist["MA"].notna()
        prev_close = close.shift(1); prev_ma = hist["MA"].shift(1)
        hist["Buy"]  = np.where(ma_valid & (close > hist["MA"]) & (prev_close <= prev_ma), close, np.nan)
        hist["Sell"] = np.where(ma_valid & (close < hist["MA"]) & (prev_close >= prev_ma), close, np.nan)

        # ── Snapshot values (NaN-safe) ─────────────────────────
        cur_price = _safe_last(close)
        cur_ma    = _safe_last(hist["MA"])
        cur_rsi   = _safe_last(hist["RSI"], 50.0)
        cur_macd  = _safe_last(hist["MACD"])
        cur_sig   = _safe_last(hist["MACD_Sig"])
        cur_bbu   = _safe_last(hist["BB_Upper"])
        cur_bbl   = _safe_last(hist["BB_Lower"])

        prior_idx = max(0, len(hist) - 21)
        prior     = float(close.iloc[prior_idx]) if prior_idx < len(hist) else cur_price
        momentum  = ((cur_price - prior) / prior * 100) if prior > 0 else 0.0

        avg_vol   = float(hist["Volume"].rolling(20, min_periods=1).mean().iloc[-1])
        last_vol  = float(hist["Volume"].iloc[-1])
        vol_ratio = (last_vol / avg_vol) if avg_vol > 0 else 1.0

        # ── Signal logic ───────────────────────────────────────
        buys  = hist["Buy"].dropna()
        sells = hist["Sell"].dropna()
        is_crossover = not buys.empty and (sells.empty or buys.index[-1] > sells.index[-1])

        if is_crossover:
            ma_signal  = f"🟢 BUY  ({buys.index[-1].strftime('%m/%d')})"
            is_bullish = True
        elif not sells.empty and (buys.empty or sells.index[-1] > buys.index[-1]):
            ma_signal  = f"🔴 SELL ({sells.index[-1].strftime('%m/%d')})"
            is_bullish = False
        else:
            ma_signal  = "⚪ Neutral"
            is_bullish = cur_price > cur_ma

        # Nearest Fibonacci level
        nearest_fib  = min(fib_levels.items(), key=lambda x: abs(x[1] - cur_price))
        fib_distance = ((cur_price - nearest_fib[1]) / nearest_fib[1] * 100) if nearest_fib[1] > 0 else 0.0

        # ── Confluence Score ───────────────────────────────────
        conf_score, conf_label, conf_breakdown = calc_confluence_score(
            is_ma_crossover=is_crossover,
            price_above_ma=(cur_price > cur_ma and cur_ma > 0),
            rsi=cur_rsi,
            macd=cur_macd, macd_sig=cur_sig,
            price=cur_price, bbu=cur_bbu, bbl=cur_bbl,
            vol_ratio=vol_ratio, momentum=momentum,
            fib_levels=fib_levels,
        )

        metrics = {
            "Price":              f"${cur_price:.2f}",
            "1-Mo Momentum":      f"{momentum:+.1f}%",
            f"{ma_label}":        f"${cur_ma:.2f}" if cur_ma > 0 else "N/A",
            "MA Signal":          ma_signal,
            "RSI (Wilder)":       f"{cur_rsi:.1f} — " + (
                "🔴 Overbought" if cur_rsi > 70 else
                "🟢 Oversold"   if cur_rsi < 30 else
                "🟢 Ideal range" if 40 <= cur_rsi <= 65 else "⚪ Neutral"
            ),
            "MACD":               "🟢 Bullish +" if cur_macd > cur_sig and cur_macd > 0 else
                                  "🟡 Bullish ±" if cur_macd > cur_sig else "🔴 Bearish",
            "Bollinger":          "🔴 Upper" if cur_bbu > 0 and cur_price >= cur_bbu * 0.99 else
                                  "🟢 Lower"  if cur_bbl > 0 and cur_price <= cur_bbl * 1.01 else "⚪ Mid",
            "Volume vs 20-avg":   f"{vol_ratio:.2f}x — " + (
                "🟢 High" if vol_ratio > 1.5 else
                "🔴 Light" if vol_ratio < 0.5 else "⚪ Average"
            ),
            "Fibonacci Zone":     f"📐 Near {nearest_fib[0]} (${nearest_fib[1]:.2f}) — " +
                                  f"{'above' if fib_distance > 0 else 'below'} by {abs(fib_distance):.1f}%",
            "Signal Strength":    f"{conf_label}  ({conf_score}/14)",
            "_score":             conf_score,          # private — used for sorting
            "_breakdown":         conf_breakdown,      # private — for detail view
        }

        # ── Build chart ────────────────────────────────────────
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.50, 0.17, 0.17, 0.16], vertical_spacing=0.025,
            subplot_titles=[
                f"{symbol} — Price, {ma_label}, Bollinger Bands & Fibonacci",
                "Volume", "MACD  (12 / 26 / 9)",
                "RSI  (14 Wilder)  |  70 = Overbought · 30 = Oversold",
            ],
        )
        fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_Upper"], showlegend=False, line=dict(color="rgba(120,120,255,0.35)",width=1)), row=1,col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_Lower"], fill="tonexty", fillcolor="rgba(120,120,255,0.07)", showlegend=False, line=dict(color="rgba(120,120,255,0.35)",width=1)), row=1,col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_Mid"], showlegend=False, line=dict(color="rgba(180,180,255,0.4)",width=1,dash="dot")), row=1,col=1)
        for lbl, lvl in fib_levels.items():
            fig.add_hline(y=lvl, line_dash="dot", line_color=FIB_COLORS[lbl], line_width=1.2,
                          annotation_text=f" Fib {lbl}  ${lvl:.2f}", annotation_position="right",
                          annotation_font_size=9, annotation_font_color=FIB_COLORS[lbl], row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=close, name="Price", line=dict(color="#4fc3f7",width=2)), row=1,col=1)
        if cur_ma > 0:
            fig.add_trace(go.Scatter(x=hist.index, y=hist["MA"], name=ma_label, line=dict(color="#ffa726",width=1.8,dash="dot")), row=1,col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Buy"], mode="markers", name="BUY", marker=dict(color="#4caf50",size=11,symbol="triangle-up",line=dict(color="white",width=1))), row=1,col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Sell"], mode="markers", name="SELL", marker=dict(color="#f44336",size=11,symbol="triangle-down",line=dict(color="white",width=1))), row=1,col=1)
        vol_colors = ["#4caf50" if float(c)>=float(o) else "#f44336" for c,o in zip(hist["Close"],hist["Open"])]
        fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], marker_color=vol_colors, showlegend=False), row=2,col=1)
        hist_colors = ["#4caf50" if float(v)>=0 else "#f44336" for v in hist["MACD_H"].fillna(0)]
        fig.add_trace(go.Bar(x=hist.index, y=hist["MACD_H"], marker_color=hist_colors, showlegend=False), row=3,col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD"], name="MACD", line=dict(color="#4fc3f7",width=1.5)), row=3,col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD_Sig"], name="Signal", line=dict(color="#ffa726",width=1.5,dash="dot")), row=3,col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["RSI"], name="RSI (Wilder)", line=dict(color="#ce93d8",width=1.8)), row=4,col=1)
        for y_val,color in [(70,"rgba(244,67,54,0.55)"),(30,"rgba(76,175,80,0.55)")]:
            fig.add_hline(y=y_val, line_dash="dash", line_color=color, line_width=1.2, row=4,col=1)
        fig.add_hrect(y0=70,y1=100,fillcolor="rgba(244,67,54,0.04)",line_width=0,row=4,col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(76,175,80,0.04)", line_width=0,row=4,col=1)
        fig.add_hrect(y0=40,y1=65, fillcolor="rgba(79,195,247,0.03)",line_width=0,row=4,col=1)  # ideal zone

        fig.update_layout(
            template="plotly_dark", hovermode="x unified", height=820,
            margin=dict(l=10,r=80,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,25,1)",
            legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1,font=dict(size=11)),
        )
        fig.update_yaxes(title_text="Price ($)", row=1,col=1,gridcolor="#1e1e2e")
        fig.update_yaxes(title_text="Volume",   row=2,col=1,gridcolor="#1e1e2e")
        fig.update_yaxes(title_text="MACD",     row=3,col=1,gridcolor="#1e1e2e")
        fig.update_yaxes(title_text="RSI",      row=4,col=1,range=[0,100],gridcolor="#1e1e2e")

        return is_bullish, metrics, fig, cur_price, fib_levels, conf_score, None

    except Exception as exc:
        return False, {}, None, 0.0, {}, 0, f"{err_pfx} {type(exc).__name__}: {exc}"

# ================================================================
# AI ANALYSIS
# ================================================================
def generate_ai_analysis(symbol, metrics, period, method, fib_levels=None, extra_context=""):
    if not AI_AVAILABLE: return "⚠️ AI unavailable — GEMINI_API_KEY not configured in Secrets."
    # Strip private keys before sending to AI
    public_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
    fib_text = ("\nFibonacci Levels:\n" + "\n".join(f"  {lbl}: ${lvl:.2f}" for lbl, lvl in fib_levels.items())) if fib_levels else ""
    prompt = f"""
You are an elite institutional analyst. Analyse {symbol} — client may be a beginner. Define jargon on first use.
Note: RSI uses Wilder's smoothing (same as TradingView/Bloomberg). Signal Strength score is a 0-14 multi-indicator confluence score.

Live Data:
{json.dumps(public_metrics, indent=2)}
{fib_text}
Framework: {period}-day {method}
{f"Context: {extra_context}" if extra_context else ""}

EXACTLY five sections:
## 📋 Quantitative Tear Sheet
Table: | Metric | Value | Plain-English Meaning |
## 🌊 Elliott Wave & Trend Structure
Wave position, Fibonacci support/resistance, price targets AND invalidation level.
## 🔀 Multi-Indicator Confluence
Where indicators agree/conflict, signal strength justification, one actionable sentence for beginners.
## ⚠️ Risk Assessment
Bull risk, bear risk, invalidation price, stop-loss zone (specific range, not a concept).
## 🎯 Portfolio Strategy Suggestion
**Bold:** action, entry zone, target, stop-loss, position size tier (aggressive/moderate/conservative). Risk/reward summary.
"""
    errors = []
    for model in GEMINI_MODEL_CANDIDATES:
        try:
            resp = gemini_client.models.generate_content(model=model, contents=prompt)
            return f"*Model: `{model}`*\n\n" + resp.text
        except Exception as e: errors.append(f"**{model}:** {str(e)[:120]}")
    return ("### ⚠️ All Gemini models failed\n\n"
            "1. [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) → Create fresh key\n"
            "2. Streamlit → Settings → Secrets → `GEMINI_API_KEY = \"your-key-here\"`\n\n"
            "**Errors:**\n" + "\n".join(f"- {e}" for e in errors))

# ================================================================
# BATCH SCANNER — confluence-scored
# ================================================================
def scan_tickers(ticker_list, period, calc_type, min_score: int = 6, max_workers: int = 15):
    """
    Scans tickers and returns only those meeting min_score threshold,
    sorted highest score first.
    """
    results, figs = [], {}
    progress = st.progress(0.0, text="Preparing scan…")
    total, done = len(ticker_list), 0

    def _scan_one(sym):
        return sym, *fetch_technical_data(sym, period, calc_type)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_scan_one, t): t for t in ticker_list}
        for future in as_completed(futures):
            done += 1
            progress.progress(done / total, text=f"Scanning … {done}/{total}")
            sym, bullish, metrics, fig, price, fib, score, err = future.result()
            if not err and score >= min_score and metrics:
                public = {k: v for k, v in metrics.items() if not k.startswith("_")}
                results.append({"Ticker": sym, "_score": score, **public})
                if fig is not None: figs[sym] = (fig, fib, metrics.get("_breakdown", {}))

    progress.empty()
    # Sort by score descending — highest conviction setups first
    results.sort(key=lambda x: x.pop("_score", 0), reverse=True)
    return results, figs

# ================================================================
# DISPLAY HELPERS
# ================================================================
def display_metrics_grid(metrics):
    public = {k: v for k, v in metrics.items() if not k.startswith("_")}
    items  = list(public.items())
    for i in range(0, len(items), 4):
        chunk = items[i:i+4]; cols = st.columns(len(chunk))
        for col, (k, v) in zip(cols, chunk): col.metric(k, v)

def show_ai_report(report_key, symbol, metrics, period, method, fib_levels,
                   extra_context="", button_label="🤖  Generate AI Analyst Report"):
    if st.button(button_label, key=f"btn_{report_key}"):
        with st.spinner("Gemini is analysing — may take 20–40 seconds…"):
            st.session_state[f"rpt_{report_key}"] = generate_ai_analysis(
                symbol, metrics, period, method, fib_levels, extra_context
            )
    stored = st.session_state.get(f"rpt_{report_key}")
    if stored:
        st.markdown("---"); st.markdown(stored)
        if st.button("🗑️  Clear Report", key=f"clr_{report_key}"):
            del st.session_state[f"rpt_{report_key}"]; st.rerun()

def render_signal_breakdown(breakdown: dict):
    """Show the confluence score breakdown for a scanned stock."""
    if not breakdown: return
    with st.expander("📊  Signal Score Breakdown — why this ticker was selected", expanded=False):
        st.caption("Each indicator contributes points toward the 14-pt confluence score.")
        for dimension, (explanation, pts) in breakdown.items():
            bar = "█" * pts + "░" * (3 - min(pts, 3))
            st.markdown(f"**{dimension}** `{pts}pt` {bar}  \n{explanation}")

def render_diversity_chart(rows, total_value):
    if not rows or total_value == 0: return
    labels, values = [], []
    for row in rows:
        val   = float(row.get("Mkt Value","$0").replace("$","").replace(",",""))
        label = row.get("Display", row.get("Asset","?"))
        if val > 0: labels.append(label); values.append(val)
    if not values: return
    st.subheader("📊  Portfolio Diversity")
    n = len(values)
    if n <= 8:
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.45,
                                     textinfo="label+percent" if n <= 4 else "percent",
                                     hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
                                     marker=dict(line=dict(color="#0f0f19",width=2)))])
        fig.update_layout(template="plotly_dark", showlegend=(n>4), height=380,
                          margin=dict(l=20,r=20,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)",
                          annotations=[dict(text=f"<b>${total_value:,.0f}</b>",x=0.5,y=0.5,
                                            font_size=15,showarrow=False,font=dict(color="#4fc3f7"))])
    else:
        df_c = pd.DataFrame({"Asset":labels,"Value":values,"Pct":[v/total_value*100 for v in values]})
        fig = go.Figure(go.Treemap(labels=df_c["Asset"],parents=[""]*len(df_c),values=df_c["Value"],
                                   texttemplate="<b>%{label}</b><br>%{customdata:.1f}%",
                                   customdata=df_c["Pct"],
                                   hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{customdata:.1f}%<extra></extra>",
                                   marker=dict(colorscale="Blues",line=dict(width=2,color="#0f0f19"))))
        fig.update_layout(template="plotly_dark",height=380,
                          margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=3600, show_spinner=False)
def get_openinsider_cluster_buys():
    try:
        r = requests.get("https://openinsider.com/latest-cluster-buys", headers=HEADERS, timeout=20)
        dfs = pd.read_html(StringIO(r.text), flavor="lxml")
        if dfs:
            df = dfs[0]
            col = next((c for c in df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()),
                       df.columns[2] if len(df.columns)>2 else None)
            if col: return set(df[col].dropna().astype(str).str.upper().str.strip().tolist())
    except: pass
    return set()

# ================================================================
# PORTFOLIO EDIT TABLE
# ================================================================
def render_portfolio_editor(portfolio: dict, uid: str, pin: str):
    with st.expander("✏️  Edit / Delete / Add Positions", expanded=False):
        st.caption("Edit cells · Delete rows (checkbox) · Add new rows · Crypto auto-detected · Save when done.")
        edit_rows = [
            {"Ticker": ticker_label(sym), "Full Ticker": sym,
             "Shares / Units": float(pos["shares"]), "Avg Cost ($)": float(pos["cost"])}
            for sym, pos in portfolio.items()
        ]
        edit_df = (pd.DataFrame(edit_rows) if edit_rows
                   else pd.DataFrame(columns=["Ticker","Full Ticker","Shares / Units","Avg Cost ($)"]))
        edited_df = st.data_editor(
            edit_df, use_container_width=True, hide_index=True, num_rows="dynamic",
            column_config={
                "Ticker":         st.column_config.TextColumn("Ticker",
                    help="Stock: AAPL · Crypto: ETH, BTC, XRP (auto-normalised)", width="small"),
                "Full Ticker":    st.column_config.TextColumn("Full Ticker", disabled=True, width="small"),
                "Shares / Units": st.column_config.NumberColumn("Shares / Units",
                    min_value=0.0, format="%.8f", width="medium"),
                "Avg Cost ($)":   st.column_config.NumberColumn("Avg Cost ($)",
                    min_value=0.0, format="$%.2f", width="medium"),
            },
            key="portfolio_editor_table",
        )
        col_save, col_hint = st.columns([1,3])
        with col_save:
            save_edits = st.button("💾  Save All Changes", use_container_width=True, key="save_all_edits")
        with col_hint:
            st.caption("ℹ️ Deleted rows permanently removed from cloud storage.")
        if save_edits:
            new_portfolio = {}
            for _, row in edited_df.iterrows():
                raw    = str(row.get("Ticker") or row.get("Full Ticker") or "").strip()
                ticker = normalize_ticker(raw)
                shares = float(row.get("Shares / Units") or 0)
                cost   = float(row.get("Avg Cost ($)") or 0)
                if not ticker or shares == 0: continue
                new_portfolio[ticker] = {"shares": shares, "cost": cost}
            deleted = set(portfolio.keys()) - set(new_portfolio.keys())
            if SUPABASE_AVAILABLE:
                for t, pos in new_portfolio.items(): save_position_to_db(uid, pin, t, pos["shares"], pos["cost"])
                for t in deleted: save_position_to_db(uid, pin, t, 0, 0)
            st.session_state["user_portfolio"] = new_portfolio
            if deleted: st.info(f"🗑️  Removed: {', '.join(deleted)}")
            st.success(f"✅ Portfolio saved — {len(new_portfolio)} position(s)")
            st.rerun()

# ================================================================
# MAIN APP
# ================================================================
st.title("📊  Wall Street AI Dashboard")
st.caption("Institutional-grade analysis · Gemini 2.5 Pro · EMA / SMA / WMA · Wilder RSI · Multi-Indicator Confluence Scoring · Elliott Wave · Insider Activity")
render_indicator_guide()
st.divider()

with st.sidebar:
    st.header("⚙️  Analysis Settings")
    ma_type = st.radio(
        "Moving Average Type",
        ["Exponential Moving Average (EMA)","Simple Moving Average (SMA)","Weighted Moving Average (WMA)"],
        help="EMA: industry standard. SMA: best for 200-day macro lines. WMA: fastest, most noise."
    )
    sma_period = st.selectbox(
        "Lookback Period", options=list(INTERVAL_MAP.keys()), index=1,
        format_func=lambda x: INTERVAL_MAP[x]["label"],
        help="200-day fetches 5 years. Small-caps may still have limited history."
    )
    st.divider()
    if not AI_AVAILABLE:
        st.error("⚠️ GEMINI_API_KEY missing — AI disabled.")
    elif len(GEMINI_API_KEY) < 20:
        st.error("⚠️ GEMINI_API_KEY too short — check Secrets.")
    else:
        masked = GEMINI_API_KEY[:6] + "•" * 8 + GEMINI_API_KEY[-4:]
        st.success(f"✅ Gemini key: `{masked}`")
    if SUPABASE_AVAILABLE:
        st.success("✅ Cloud portfolio connected")
    else:
        st.warning("⚠️ Supabase not set — portfolios won't persist")
    st.divider()
    st.caption("🔒 Shared institutional API — no personal key needed.")

mode = st.radio("Mode", ["💼  My Portfolio","🔍  Analyze Single Asset","🌐  Market Scanner"],
                horizontal=True, label_visibility="collapsed")
st.markdown("<br>", unsafe_allow_html=True)

# ================================================================
# MODE 1 — PORTFOLIO
# ================================================================
if mode == "💼  My Portfolio":
    st.header("💼  Portfolio Dashboard")

    if not SUPABASE_AVAILABLE:
        for k,v in [("user_portfolio",{}),("auth_user","local"),("auth_pin","")]:
            if k not in st.session_state: st.session_state[k] = v
        st.info("☁️ Cloud storage not configured — portfolio resets on page refresh.")

    if SUPABASE_AVAILABLE and "auth_user" not in st.session_state:
        st.subheader("🔐  Access Your Portfolio")
        st.caption("First time? Choose any username and 4-digit PIN to create your account.")
        with st.form("login_form"):
            uid = st.text_input("Username / Investor ID", placeholder="e.g. john_trader")
            pin = st.text_input("4-Digit PIN", type="password", max_chars=4)
            sub = st.form_submit_button("Access Portfolio →", use_container_width=True)
        if sub:
            uid = uid.strip().lower()
            if not uid or not pin: st.error("Enter both.")
            elif len(pin)!=4 or not pin.isdigit(): st.error("PIN must be 4 digits.")
            else:
                with st.spinner("Authenticating…"):
                    portfolio = load_portfolio_from_db(uid, pin)
                if portfolio is None: st.error("❌ Incorrect PIN.")
                else:
                    st.session_state.update({"auth_user":uid,"auth_pin":pin,"user_portfolio":portfolio})
                    st.rerun()
        st.stop()

    with st.sidebar:
        st.subheader("🛠️  Quick Add Position")
        if SUPABASE_AVAILABLE: st.caption(f"Logged in as: **{st.session_state.get('auth_user','—')}**")
        with st.form("position_form"):
            raw_ticker = st.text_input("Ticker Symbol",
                placeholder="AAPL · NVDA · ETH · BTC · XRP · SOL",
                help="Crypto auto-detected — no suffix needed.").strip()
            new_shares = st.number_input("Shares / Units", min_value=0.0, step=0.00000001, format="%.8f")
            new_cost   = st.number_input("Avg Purchase Price ($)", min_value=0.0, step=0.01)
            save_btn   = st.form_submit_button("➕  Add Position", use_container_width=True)
        if save_btn and raw_ticker:
            new_ticker = normalize_ticker(raw_ticker)
            uid = st.session_state.get("auth_user","local"); pin = st.session_state.get("auth_pin","")
            if new_shares == 0:
                st.session_state["user_portfolio"].pop(new_ticker, None)
                if SUPABASE_AVAILABLE: save_position_to_db(uid, pin, new_ticker, 0, 0)
                st.warning(f"Removed {new_ticker}")
            else:
                st.session_state["user_portfolio"][new_ticker] = {"shares":new_shares,"cost":new_cost}
                if SUPABASE_AVAILABLE: save_position_to_db(uid, pin, new_ticker, new_shares, new_cost)
                st.success(f"✅ {new_ticker} added")
        st.divider()
        if st.session_state.get("user_portfolio"):
            st.download_button("📥  Download Backup",
                               data=json.dumps(st.session_state["user_portfolio"]),
                               file_name="portfolio_backup.json", mime="application/json",
                               use_container_width=True)
        uploaded = st.file_uploader("📤  Restore from Backup", type="json")
        if uploaded:
            try: st.session_state["user_portfolio"] = json.load(uploaded); st.success("Restored!")
            except: st.error("Invalid backup file.")
        if SUPABASE_AVAILABLE:
            st.divider()
            if st.button("🚪  Log Out", use_container_width=True):
                for k in ["auth_user","auth_pin","user_portfolio"]: st.session_state.pop(k,None)
                st.rerun()

    portfolio = st.session_state.get("user_portfolio", {})
    uid = st.session_state.get("auth_user","local")
    pin = st.session_state.get("auth_pin","")

    if not portfolio:
        st.info("Portfolio is empty. Use **Quick Add** in the sidebar or the edit table below.")
        render_portfolio_editor({}, uid, pin)
    else:
        total_value = total_cost = 0.0
        rows, charts, load_errors = [], {}, []

        with st.spinner("Fetching live data (5-min cache active)…"):
            for sym, pos in list(portfolio.items()):
                _, metrics, fig, price, fib, score, err = fetch_technical_data(sym, sma_period, ma_type)
                if err:
                    load_errors.append(f"**{ticker_label(sym)}:** {err}"); continue
                if price > 0:
                    pos_cost  = pos["shares"] * pos["cost"]
                    pos_value = pos["shares"] * price
                    pos_gain  = pos_value - pos_cost
                    pos_pct   = (pos_gain / pos_cost * 100) if pos_cost > 0 else 0.0
                    total_value += pos_value; total_cost += pos_cost
                    if fig: charts[sym] = (fig, fib, metrics)
                    rows.append({
                        "Display":        ticker_label(sym),
                        "Asset":          sym,
                        "Shares":         f"{pos['shares']:.8f}".rstrip("0").rstrip("."),
                        "Avg Cost":       f"${pos['cost']:.2f}",
                        "Current Price":  f"${price:.2f}",
                        "Mkt Value":      f"${pos_value:,.2f}",
                        "Return ($)":     f"${pos_gain:+,.2f}",
                        "Return (%)":     f"{pos_pct:+.1f}%",
                        "MA Signal":      metrics.get("MA Signal","—"),
                        "RSI":            metrics.get("RSI (Wilder)","—"),
                        "Signal Strength":metrics.get("Signal Strength","—"),
                    })

        if load_errors:
            with st.expander(f"⚠️  {len(load_errors)} position(s) could not load — click to see why", expanded=True):
                for e in load_errors: st.warning(e)
                st.caption("Common causes: insufficient history for the selected period, or invalid ticker. Use the Edit table below to fix.")

        total_gain = total_value - total_cost
        total_pct  = (total_gain / total_cost * 100) if total_cost > 0 else 0.0
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total Portfolio Value", f"${total_value:,.2f}")
        k2.metric("Total Cost Basis",      f"${total_cost:,.2f}")
        k3.metric("Total Return",          f"${total_gain:+,.2f}", f"{total_pct:+.2f}%")
        k4.metric("Open Positions",        str(len(rows)))

        st.subheader("Holdings Summary")
        if rows:
            df_display = pd.DataFrame(rows)
            display_cols = [c for c in ["Display","Shares","Avg Cost","Current Price",
                                         "Mkt Value","Return ($)","Return (%)","MA Signal",
                                         "RSI","Signal Strength"] if c in df_display.columns]
            st.dataframe(df_display[display_cols].rename(columns={"Display":"Asset"}),
                         use_container_width=True, hide_index=True)
        elif not load_errors:
            st.info("No positions loaded. Check tickers or try a shorter lookback period.")

        render_portfolio_editor(portfolio, uid, pin)

        if rows:
            render_diversity_chart(rows, total_value)
            st.subheader("📈  Deep-Dive Chart & AI Analysis")
            sym_options = {ticker_label(s): s for s in portfolio.keys() if s in charts}
            if sym_options:
                chosen_label  = st.selectbox("Select a holding", list(sym_options.keys()))
                chosen        = sym_options[chosen_label]
                fig, fib, chosen_metrics = charts[chosen]
                st.plotly_chart(fig, use_container_width=True)
                if not chosen.endswith("-USD"): render_insider_section(chosen)
                pos_detail = portfolio.get(chosen, {})
                ins_df     = get_insider_transactions(chosen) if not chosen.endswith("-USD") else None
                pnl_row    = next((r for r in rows if r["Asset"]==chosen), {})
                context    = (f"Held: {pos_detail.get('shares',0):.8f} units at "
                              f"${pos_detail.get('cost',0):.2f} avg. "
                              f"P&L: {pnl_row.get('Return (%)', 'unknown')}. "
                              f"{insider_summary(ins_df)}")
                show_ai_report(f"portfolio_{chosen}", chosen, chosen_metrics, sma_period, ma_type, fib, extra_context=context)
            else:
                st.info("No charts available. Try a shorter lookback period.")

# ================================================================
# MODE 2 — SINGLE ASSET
# ================================================================
elif mode == "🔍  Analyze Single Asset":
    st.header("🔍  Single Asset Analysis")
    st.caption("Any US stock or crypto — crypto auto-detected.")
    c1,c2 = st.columns([4,1])
    with c1:
        raw_input = st.text_input("Ticker", label_visibility="collapsed",
                                  placeholder="NVDA · AAPL · ETH · BTC · XRP · SOL").strip()
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        go_btn = st.button("Analyse →", use_container_width=True)

    if go_btn and raw_input:
        symbol_input = normalize_ticker(raw_input)
        with st.spinner(f"Fetching {symbol_input}…"):
            result = fetch_technical_data(symbol_input, sma_period, ma_type)
        _, metrics, fig, price, fib, score, err = result
        if err or price == 0.0:
            st.error(err or f"No data returned for **{symbol_input}**.")
        else:
            st.session_state["single_result"] = (symbol_input, result, sma_period, ma_type)

    if "single_result" in st.session_state:
        sym, (_, metrics, fig, price, fib, score, _err), period, method = st.session_state["single_result"]
        if st.button("🔄  Analyse a different ticker", key="clear_single"):
            del st.session_state["single_result"]; st.rerun()
        st.markdown(f"**Showing: `{sym}`** — {INTERVAL_MAP.get(period,{}).get('label','')} · Signal Score: **{score}/14**")
        display_metrics_grid(metrics)
        render_signal_breakdown(metrics.get("_breakdown", {}))
        st.plotly_chart(fig, use_container_width=True)
        if not sym.endswith("-USD"): render_insider_section(sym)
        ins_df = get_insider_transactions(sym) if not sym.endswith("-USD") else None
        show_ai_report(f"single_{sym}", sym, metrics, period, method, fib,
                       extra_context=insider_summary(ins_df),
                       button_label="🤖  Generate Full AI Report")

# ================================================================
# MODE 3 — MARKET SCANNER (Multi-indicator confluence)
# ================================================================
elif mode == "🌐  Market Scanner":
    st.header("🌐  Market Scanner")
    st.caption(
        "Every ticker is scored 0–14 across MA, RSI, MACD, Volume, Momentum, Bollinger, and Fibonacci. "
        "Only tickers above your chosen threshold are shown, sorted highest score first."
    )

    tab_stocks, tab_crypto = st.tabs(["📈  Stocks","₿  Crypto"])

    with tab_stocks:
        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
        with c1:
            universe = st.selectbox("Market Universe", [
                "S&P 500  (~500 stocks, ~2–4 min)",
                "Russell 2000  (~2,000 stocks, ~10–20 min)",
                "All US Equities  (~1,500–3,000 stocks, ~15–30 min)",
            ])
        with c2:
            threshold_label = st.selectbox(
                "Minimum Signal Strength",
                options=list(SCORE_THRESHOLDS.keys()),
                index=1,   # default: Strong Buy+
                help="Higher threshold = fewer but higher-quality signals"
            )
            min_score = SCORE_THRESHOLDS[threshold_label]
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            insider_filter = st.checkbox(
                "🔍 Insider cluster buy filter",
                help="Pre-filter to stocks with 3+ insiders buying recently (OpenInsider). Combines with score threshold."
            )
        with c4:
            st.markdown("<br>", unsafe_allow_html=True)
            scan_btn = st.button("🚀  Launch Scan", use_container_width=True)

        if "Russell" in universe: st.warning("⚠️ Large scan — expect 10–20 min.")
        elif "All US" in universe: st.warning("⚠️ Very large scan — expect 15–30 min.")
        else: st.info(f"ℹ️ ~500 stocks · 2–4 min · minimum score: **{min_score}/14**")

        if scan_btn:
            with st.spinner("Loading ticker universe…"):
                if "S&P 500" in universe: tickers = get_sp500_tickers()
                elif "Russell" in universe: tickers = get_russell2000_tickers()
                else: tickers = get_all_us_equities()
            if not tickers:
                st.error("Failed to load ticker list.")
            else:
                if insider_filter:
                    with st.spinner("Loading OpenInsider cluster buy data…"):
                        insider_tickers = get_openinsider_cluster_buys()
                    if insider_tickers:
                        before = len(tickers); tickers = [t for t in tickers if t in insider_tickers]
                        st.info(f"Insider filter: {before} → **{len(tickers)}** tickers with cluster buying")
                    else:
                        st.warning("OpenInsider unavailable — running without insider filter.")
                if not tickers:
                    st.warning("No tickers after filter. Try without it.")
                else:
                    st.info(f"Scanning **{len(tickers)}** tickers for confluence score ≥ **{min_score}/14**…")
                    results, figs = scan_tickers(tickers, sma_period, ma_type, min_score=min_score)

                    if results:
                        st.success(f"✅ **{len(results)}** high-conviction setups found (score ≥ {min_score}/14, sorted by score)")

                        # Summary bar
                        exceptional = sum(1 for r in results if "Exceptional" in r.get("Signal Strength",""))
                        strong      = sum(1 for r in results if "Strong Buy" in r.get("Signal Strength","") and "Exceptional" not in r.get("Signal Strength",""))
                        moderate    = sum(1 for r in results if "Moderate" in r.get("Signal Strength",""))
                        s1,s2,s3 = st.columns(3)
                        s1.metric("🔥 Exceptional", exceptional)
                        s2.metric("🟢 Strong Buy",  strong)
                        s3.metric("🟡 Moderate Buy", moderate)

                        # Results table — key columns front and centre
                        display_cols_scanner = ["Ticker","Signal Strength","Price","1-Mo Momentum",
                                                 "MA Signal","RSI (Wilder)","MACD","Volume vs 20-avg",
                                                 "Fibonacci Zone"]
                        df_results = pd.DataFrame(results)
                        safe_cols  = [c for c in display_cols_scanner if c in df_results.columns]
                        st.dataframe(df_results[safe_cols], use_container_width=True, hide_index=True)

                        st.subheader("📊  Deep-Dive Chart")
                        triggered = [r["Ticker"] for r in results]
                        view_sym  = st.selectbox("Select a stock", triggered,
                                                  help="Results sorted by confluence score — highest conviction first")
                        if view_sym in figs:
                            fig, fib, breakdown = figs[view_sym]
                            st.plotly_chart(fig, use_container_width=True)
                            render_signal_breakdown(breakdown)
                            render_insider_section(view_sym)
                            stock_metrics = next((r for r in results if r["Ticker"]==view_sym), {})
                            ins_df        = get_insider_transactions(view_sym)
                            show_ai_report(f"scanner_{view_sym}", view_sym, stock_metrics,
                                           sma_period, ma_type, fib,
                                           extra_context=insider_summary(ins_df),
                                           button_label="🤖  AI Analysis for this stock")
                    else:
                        st.warning(
                            f"No stocks scored ≥ {min_score}/14 with the current settings. "
                            f"Try lowering the threshold, switching to EMA, or using a shorter lookback period."
                        )

    with tab_crypto:
        st.subheader("₿  Major Crypto Dashboard")
        st.caption("Bitcoin · Ethereum · XRP · Solana — all shown with confluence scores")
        crypto_btn = st.button("📡  Refresh Crypto Data")
        if crypto_btn or "crypto_data" not in st.session_state:
            crypto_rows, crypto_fig_map = [], {}
            with st.spinner("Fetching crypto data…"):
                for name, sym in CRYPTO_TICKERS.items():
                    _, metrics, fig, _, fib, score, err = fetch_technical_data(sym, sma_period, ma_type)
                    if not err and metrics:
                        public = {k:v for k,v in metrics.items() if not k.startswith("_")}
                        crypto_rows.append({"Asset": f"{name} ({sym})", **public})
                    if not err and fig:
                        crypto_fig_map[sym] = (fig, fib, metrics.get("_breakdown",{}))
            st.session_state["crypto_data"] = (crypto_rows, crypto_fig_map)
        crypto_rows, crypto_fig_map = st.session_state.get("crypto_data", ([], {}))
        if crypto_rows:
            st.dataframe(pd.DataFrame(crypto_rows), use_container_width=True, hide_index=True)
            chosen_crypto = st.selectbox("Select crypto for chart", list(CRYPTO_TICKERS.values()),
                                         format_func=lambda s: next((k for k,v in CRYPTO_TICKERS.items() if v==s),s))
            if chosen_crypto in crypto_fig_map:
                fig, fib, breakdown = crypto_fig_map[chosen_crypto]
                st.plotly_chart(fig, use_container_width=True)
                render_signal_breakdown(breakdown)
                crypto_metrics = next((r for r in crypto_rows if chosen_crypto in r.get("Asset","")), {})
                show_ai_report(f"crypto_{chosen_crypto}", chosen_crypto, crypto_metrics,
                               sma_period, ma_type, fib,
                               extra_context="Cryptocurrency: 24/7 trading, higher volatility, no traditional fundamentals, no insider data.",
                               button_label="🤖  Generate Crypto AI Analysis")
        else:
            st.info("Click 'Refresh Crypto Data' to load.")
