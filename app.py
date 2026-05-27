# ================================================================
# WALL STREET AI DASHBOARD — Production Build v2.9
# NEW:   Fresh-vs-stale crossover scoring (3/2/1/0 pts by bars since signal)
# NEW:   Portfolio notifications filtered to last 5 trading days, compact UI
# NEW:   Batch yfinance fetching for scanner (5–10x speedup)
# FIX:   LaTeX/$-sign rendering in AI panels (escape before st.markdown)
# IMPR:  Retry-with-backoff for transient yfinance failures
# IMPR:  Cache TTL extended from 5min → 15min for stability under load
# IMPR:  Parallel workers reduced from 12 → 8 (multi-user friendliness)
# IMPR:  Disk-persisted ticker lists survive app restarts
# IMPR:  Scanner universe time estimates updated to reflect new fetch speed
# ================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib, json, requests, re, time
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
    "gemini-2.5-pro","gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-exp-03-25","gemini-2.5-flash-preview-04-17",
    "gemini-2.5-flash","gemini-1.5-pro",
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
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
_BUY_KEYWORDS   = ["purchase","bought","acqui"]
_SELL_KEYWORDS  = ["sale","sold"]
_NOISE_KEYWORDS = ["gift","award","grant","automatic","plan sale","tax withholding",
                   "exercise","dispose","reclassif","return","forfeiture","conversion"]

SCORE_THRESHOLDS = {
    "🔥 Exceptional only  (≥11)": 11,
    "🟢 Strong Buy+       (≥8)":   8,
    "🟡 Moderate Buy+     (≥6)":   6,
    "⚪ Watch List+       (≥4)":   4,
}

# Freshness window for portfolio alerts (trading days)
FRESH_SIGNAL_WINDOW = 5

# ── International ADR / Dual-Listed Universe ──────────────────────
INTERNATIONAL_ADRS = sorted(list(set([
    # UK
    "SHEL","BP","GSK","AZN","RIO","HSBC","BATS","PRU",
    # Switzerland
    "NVS","RHHBY","UBS","NESNY",
    # Germany
    "SAP","SIEGY","BAYRY","MBGYY","BMWYY","VWAGY",
    # Netherlands
    "ASML","PHG","ING","HEIA","AKZOY",
    # France
    "LVMUY","TTE","SNY","AXAHY","LRLCY","BNPQY","ORAN",
    # Nordics
    "NVO","EQNR","NHYDY","VOLVY",
    # Spain / Italy
    "SAN","TEF","RACE","ENIAY",
    # Other Europe
    "ABB","CRH",
    # Japan
    "SONY","TM","HMC","NTDOY","SFTBY","CAJ","FUJIY","MSBHY","TOELY",
    # China (US-listed ADRs)
    "BABA","JD","PDD","BIDU","NIO","LI","XPEV","NTES","TME","VIPS","YUMC","WB",
    # Taiwan
    "TSM",
    # India
    "INFY","WIT","HDB","IBN",
    # Canada
    "SHOP","RY","TD","BMO","BNS","SU","CNQ","CP","CNI","MFC","SLF",
    # Australia
    "BHP","ANZBY",
    # Brazil / LatAm
    "PBR","VALE","ITUB","BBD","ABEV",
    # Southeast Asia
    "SE","GRAB",
    # Global / Other
    "SLB","ACN",
])))

# ================================================================
# MARKDOWN SANITISER — FIX for LaTeX/$-sign rendering bug
# ================================================================
# Streamlit's st.markdown uses KaTeX for math rendering. When AI output
# contains "$105.00 to $107.50", KaTeX reads it as LaTeX math mode,
# strips whitespace, and italicises everything between the $ signs.
# The fix: escape every unescaped $ as \$ before st.markdown renders it.
# This is applied to ALL AI-generated text panels.
def _sanitize_ai_markdown(text: str) -> str:
    """Escape $ signs to prevent Streamlit/KaTeX from rendering as LaTeX."""
    if not text:
        return text
    return re.sub(r'(?<!\\)\$', r'\\$', text)

# ================================================================
# CONFLUENCE SCORING — Fresh-vs-stale crossover refinement
# ================================================================
def calc_confluence_score(
    is_ma_buy_crossover,       # True if most recent crossover signal is BUY
    bars_since_buy,            # trading days since last BUY signal (999 if none)
    price_above_ma,            # price > MA today
    rsi, macd, macd_sig,
    price, bbu, bbl, vol_ratio, momentum,
    fib_levels,
):
    """
    Multi-indicator confluence score (0-14).
    The MA dimension now distinguishes fresh (≤5d), recent (≤20d),
    and established (>20d) crossovers — preventing stale signals from
    masquerading as fresh entries.
    """
    score = 0; breakdown = {}

    # ── MA Crossover (0-3) — now AGE-aware ──────────────────────
    if is_ma_buy_crossover and bars_since_buy <= 5:
        score += 3
        breakdown["MA Crossover"] = (
            f"🟢 Fresh BUY crossover ({bars_since_buy}d ago) — actionable now", 3
        )
    elif is_ma_buy_crossover and bars_since_buy <= 20:
        score += 2
        breakdown["MA Crossover"] = (
            f"🟢 Recent BUY crossover ({bars_since_buy}d ago) — trend confirmed", 2
        )
    elif price_above_ma:
        age_note = f" (last crossover {bars_since_buy}d ago)" if bars_since_buy < 999 else ""
        score += 1
        breakdown["MA Crossover"] = (
            f"🟡 Price above MA — established uptrend, no fresh signal{age_note}", 1
        )
    else:
        breakdown["MA Crossover"] = ("🔴 Price below MA — bearish structure", 0)

    # ── RSI (0-2) ───────────────────────────────────────────────
    if 40 <= rsi <= 65:
        score += 2; breakdown["RSI"] = (f"🟢 RSI {rsi:.0f} — ideal range, room to run", 2)
    elif 65 < rsi <= 70:
        score += 1; breakdown["RSI"] = (f"🟡 RSI {rsi:.0f} — approaching overbought", 1)
    elif rsi < 35:
        score += 1; breakdown["RSI"] = (f"🟡 RSI {rsi:.0f} — oversold bounce setup", 1)
    else:
        breakdown["RSI"] = (f"🔴 RSI {rsi:.0f} — overbought or very weak", 0)

    # ── MACD (0-2) ──────────────────────────────────────────────
    if macd > macd_sig and macd > 0:
        score += 2; breakdown["MACD"] = ("🟢 Bullish cross + positive territory", 2)
    elif macd > macd_sig:
        score += 1; breakdown["MACD"] = ("🟡 Bullish cross, negative territory — early reversal", 1)
    else:
        breakdown["MACD"] = ("🔴 Bearish MACD — selling pressure dominant", 0)

    # ── Volume (0-2) ────────────────────────────────────────────
    if vol_ratio >= 1.5:
        score += 2; breakdown["Volume"] = (f"🟢 {vol_ratio:.1f}x average — strong conviction", 2)
    elif vol_ratio >= 1.0:
        score += 1; breakdown["Volume"] = (f"🟡 {vol_ratio:.1f}x average — moderate confirmation", 1)
    else:
        breakdown["Volume"] = (f"🔴 {vol_ratio:.1f}x average — light volume, suspect", 0)

    # ── Momentum (0-2) ──────────────────────────────────────────
    if momentum >= 7.0:
        score += 2; breakdown["Momentum"] = (f"🟢 +{momentum:.1f}% (1-month) — strong trend", 2)
    elif momentum > 0:
        score += 1; breakdown["Momentum"] = (f"🟡 +{momentum:.1f}% (1-month) — positive", 1)
    else:
        breakdown["Momentum"] = (f"🔴 {momentum:.1f}% (1-month) — negative", 0)

    # ── Bollinger Position (0-2) ────────────────────────────────
    bb_range = (bbu - bbl) if (bbu > 0 and bbl > 0 and bbu > bbl) else 0
    if bb_range > 0:
        bb_pos = (price - bbl) / bb_range
        if bb_pos <= 0.30:
            score += 2; breakdown["Bollinger"] = ("🟢 Near lower band — oversold bounce setup", 2)
        elif bb_pos <= 0.60:
            score += 1; breakdown["Bollinger"] = ("🟡 Lower-mid of bands — early upward positioning", 1)
        elif bb_pos >= 0.95:
            breakdown["Bollinger"] = ("🔴 At upper band — overbought, chasing risk", 0)
        else:
            breakdown["Bollinger"] = ("⚪ Mid-upper portion of bands", 0)
    else:
        breakdown["Bollinger"] = ("⚪ Bollinger unavailable", 0)

    # ── Fibonacci bonus (0-1) ───────────────────────────────────
    for lbl, lvl in {k: v for k, v in fib_levels.items() if k in ("38.2%","61.8%")}.items():
        if lvl > 0 and 0 <= (price - lvl) / lvl <= 0.02:
            score += 1
            breakdown["Fibonacci"] = (f"🟢 Within 2% above {lbl} support — bounce zone (+1 bonus)", 1)
            break
    else:
        breakdown["Fibonacci"] = ("⚪ Not at a key Fibonacci support", 0)

    if score >= 11:   label = "🔥 Exceptional"
    elif score >= 8:  label = "🟢 Strong Buy"
    elif score >= 6:  label = "🟡 Moderate Buy"
    elif score >= 4:  label = "⚪ Watch List"
    else:             label = "—  Weak"

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
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("---\n### 📈 Moving Averages")
            st.info("**EMA ⭐ (Exponential):** Industry standard — fast reaction, smooth line. Used by Bloomberg, TradingView, MACD.\n\n**SMA (Simple):** Slow and stable, best for the 200-day macro trend.\n\n**WMA (Weighted):** Fastest, most false signals.\n\n🔺 Green triangle = price crossed above → Buy signal\n🔻 Red triangle = price crossed below → Sell signal")
            st.markdown("---\n### 📉 RSI (Wilder's Smoothing)")
            st.info("0–100 momentum gauge using Wilder's EWM — same method as TradingView/Bloomberg.\n\n🔴 >70 = Overbought\n🟢 <30 = Oversold\n🟢 40–65 = Ideal — room to run\n⚪ 30–70 = Neutral")
            st.markdown("---\n### 📊 Volume")
            st.info("Total shares traded. Validates conviction behind a price move.\n\n🟢 >1.5x avg + rising price = Confirmed\n🔴 <1.0x avg = Suspect")
        with c2:
            st.markdown("---\n### ⚡ MACD")
            st.info("Fast EMA(12) minus slow EMA(26). Always EMA-based.\n\n🟢 Bullish cross + positive territory = Strongest setup\n🟡 Bullish cross, negative territory = Early reversal\n🔴 Below signal line = Bearish")
            st.markdown("---\n### 🎯 Bollinger Bands")
            st.info("20-day MA ± 2 standard deviations.\n\n🟢 Near lower band = Oversold bounce setup\n🔴 Near upper band = Overbought\n⚠️ Squeeze = Big move loading")
            st.markdown("---\n### 👔 Insider / CEO Buying")
            st.info("Open-market purchases by executives (SEC Form 4). Personal cash = genuine conviction.\n\n🟢 CEO buying = Strongest signal\n📌 3+ insiders = Cluster buy (highest conviction)\n🔴 Heavy selling = Monitor")
        with c3:
            st.markdown("---\n### 🔢 Scanner Confluence Score")
            st.info("Every ticker scored 0–14 pts across:\n\n| Dimension | Max |\n|---|---|\n| MA Crossover (age-aware) | 3 |\n| RSI Range | 2 |\n| MACD Position | 2 |\n| Volume | 2 |\n| Momentum | 2 |\n| Bollinger | 2 |\n| Fibonacci Bonus | 1 |\n\n**MA scoring is age-aware:**\n• Fresh (≤5d): 3 pts\n• Recent (6–20d): 2 pts\n• Established: 1 pt\n• Below MA: 0 pts\n\n🔥 11+ = Exceptional\n🟢 8–10 = Strong Buy\n🟡 6–7 = Moderate Buy\n⚪ 4–5 = Watch List")
            st.markdown("---\n### 📐 Fibonacci & Elliott Wave")
            st.info("🔵 23.6% — Shallow pullback\n🟢 38.2% — Common dip\n🟡 50.0% — Midpoint\n🟠 61.8% — Golden Ratio (strongest)\n🔴 78.6% — Deep retracement\n\nElliott Waves: 5-wave impulse (up) then A-B-C corrective. AI identifies your likely current position.")

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
# TICKER LOADERS  (disk-persisted for multi-user efficiency)
# ================================================================
def _read_html_safe(url, **kwargs):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text), **kwargs)

@st.cache_data(ttl=86400, persist="disk", show_spinner=False)
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

@st.cache_data(ttl=86400, persist="disk", show_spinner=False)
def get_sp1500_tickers():
    tickers = set(get_sp500_tickers())
    for url, col in [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies","Ticker"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies","Ticker"),
    ]:
        try:
            tables = _read_html_safe(url, flavor="lxml")
            tickers.update(tables[0][col].str.replace(".", "-", regex=False).tolist())
        except: pass
    return sorted(list(tickers))

@st.cache_data(ttl=86400, persist="disk", show_spinner=False)
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
        st.caption("ℹ️ Russell 2000 live feed unavailable — using S&P SmallCap 600 as proxy.")
        return sorted(tables[0]["Ticker"].str.replace(".", "-", regex=False).tolist())
    except:
        st.warning("Russell 2000 unavailable."); return []

@st.cache_data(ttl=3600, persist="disk", show_spinner=False)
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
    return get_sp1500_tickers()

@st.cache_data(ttl=3600, persist="disk", show_spinner=False)
def get_international_tickers():
    """International stocks: ADRs and dual-listed companies on US exchanges."""
    extras = []
    try:
        r = requests.get(
            "https://www.ishares.com/us/products/239726/ishares-msci-eafe-etf/1467271812596.ajax?fileType=csv&fileName=EFA_holdings&dataType=fund",
            headers=HEADERS, timeout=20
        )
        df = pd.read_csv(StringIO(r.text), skiprows=9)
        df = df[df.get("Asset Class", df.columns[0]) == "Equity"]
        candidates = df["Ticker"].dropna().str.strip().tolist()
        extras = [t for t in candidates if t and "." not in t and len(t) <= 5 and t != "-"]
    except: pass
    combined = sorted(list(set(INTERNATIONAL_ADRS + extras)))
    return combined if combined else INTERNATIONAL_ADRS

@st.cache_data(ttl=3600, persist="disk", show_spinner=False)
def get_smallcap_growth_tickers():
    """Small-cap growth: market cap ~$50M-$3B, volume > 100K."""
    def _parse_mcap(s):
        s = str(s).replace("$","").replace(",","").strip()
        try:
            if s.upper().endswith("B"): return float(s[:-1]) * 1e9
            if s.upper().endswith("M"): return float(s[:-1]) * 1e6
            return float(s)
        except: return 0.0
    try:
        r = requests.get(
            "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&download=true",
            headers=HEADERS, timeout=25
        )
        df = pd.DataFrame(r.json()["data"]["rows"])
        df["volume"] = pd.to_numeric(df["volume"].astype(str).str.replace(",","",regex=False), errors="coerce")
        df = df[df["volume"] > 100_000]
        if "marketCap" in df.columns:
            df["mcap_num"] = df["marketCap"].apply(_parse_mcap)
            df = df[(df["mcap_num"] >= 50_000_000) & (df["mcap_num"] <= 3_000_000_000)]
        tickers = sorted(df["symbol"].dropna().str.strip().tolist())
        if len(tickers) > 100: return tickers
    except: pass
    try:
        tables = _read_html_safe("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", flavor="lxml")
        st.caption("ℹ️ NASDAQ small-cap screener unavailable — using S&P SmallCap 600.")
        return sorted(tables[0]["Ticker"].str.replace(".", "-", regex=False).tolist())
    except:
        st.warning("Small-cap list unavailable."); return []

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
        if "Date" in df.columns:  clean["Date"]    = df["Date"]
        if name_col:              clean["Insider"]  = df[name_col]
        if pos_col:               clean["Role"]     = df[pos_col]
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
    if "Transaction Type" not in df.columns: return "Insider data format unrecognised."
    buys  = df[df["Transaction Type"] == "🟢 Open Market Buy"]
    sells = df[df["Transaction Type"] == "🔴 Open Market Sale"]
    bv = df.loc[df["Transaction Type"]=="🟢 Open Market Buy","_raw_value"].sum() if "_raw_value" in df.columns else 0
    sv = df.loc[df["Transaction Type"]=="🔴 Open Market Sale","_raw_value"].sum() if "_raw_value" in df.columns else 0
    s = f"Insider open-market (last 12mo): {len(buys)} buy(s) ~${bv:,.0f}, {len(sells)} sale(s) ~${sv:,.0f}. "
    if len(buys) > 0 and len(sells) == 0: s += "✅ Buying with no selling — very bullish."
    elif len(buys) >= 3:                  s += "✅ Cluster buying — high conviction."
    elif len(buys) > 0:                   s += "Mild buying — modestly bullish."
    elif sv > bv * 3 > 0:                 s += "⚠️ Heavy net selling — warrants caution."
    else:                                 s += "No open-market purchases."
    return s

def generate_insider_ai_analysis(symbol, df):
    if not AI_AVAILABLE: return "⚠️ AI unavailable."
    if df is None: return "No insider data."
    dcols = [c for c in ["Date","Insider","Role","Transaction Type","Shares","Est. Value ($)","Description"] if c in df.columns]
    table = df[dcols].to_string(index=False) if dcols else "Data unavailable"
    prompt = f"""SEC Form 4 insider analysis for {symbol}.
{table}
RULES: "Open Market Buy" = personal cash = real signal. "Non-Market" entries = compensation = IGNORE.
Respond in plain markdown. Do not use nested asterisks or malformed bold markers.

## 👔 Insider Transaction Summary (Open Market only, bold buy rows)
## 🔍 Signal Interpretation (role, cluster, pre-scheduled vs discretionary)
## 📊 Historical Context (typical level, research evidence, red/green flags)
## 🎯 Insider Signal Verdict (**Bold:** Bullish/Neutral/Bearish. Weight vs technicals. One beginner sentence.)"""
    errors = []
    for m in GEMINI_MODEL_CANDIDATES:
        try:
            r = gemini_client.models.generate_content(model=m, contents=prompt)
            return f"*Model: `{m}`*\n\n" + r.text
        except Exception as e: errors.append(f"**{m}:** {str(e)[:100]}")
    return "⚠️ Failed.\n" + "\n".join(f"- {e}" for e in errors)

def render_insider_section(symbol):
    with st.expander("👔  Insider & Executive Transactions (last 12 months)", expanded=False):
        with st.spinner("Loading SEC Form 4 data…"):
            df = get_insider_transactions(symbol)
        if df is None:
            st.info("No recent insider data found."); return
        st.caption("Source: SEC Form 4 via Yahoo Finance. Open Market = personal cash = genuine conviction.")
        show_noise = st.toggle("Show Non-Market entries", value=False, key=f"noise_{symbol}")
        display_df = df if show_noise else df[df["Transaction Type"] != "⚪ Non-Market"]
        display_df = display_df[[c for c in display_df.columns if c != "_raw_value"]]
        if display_df.empty:
            st.info("No open-market transactions after filtering noise.")
        else:
            def hr(row):
                if "Buy"  in str(row.get("Transaction Type","")): return ["background-color:rgba(76,175,80,0.12)"]*len(row)
                if "Sale" in str(row.get("Transaction Type","")): return ["background-color:rgba(244,67,54,0.08)"]*len(row)
                return [""]*len(row)
            st.dataframe(display_df.style.apply(hr, axis=1), use_container_width=True, hide_index=True)
        buys  = df[df["Transaction Type"] == "🟢 Open Market Buy"]
        sells = df[df["Transaction Type"] == "🔴 Open Market Sale"]
        other = df[df["Transaction Type"] == "⚪ Non-Market"]
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Open Market Purchases", len(buys), delta="Bullish" if len(buys)>0 else None)
        k2.metric("Open Market Sales", len(sells),
                  delta="Monitor" if len(sells)>len(buys)*2 else None,
                  delta_color="inverse" if len(sells)>len(buys)*2 else "normal")
        k3.metric("Non-Market (Noise)", len(other))
        net = len(buys)-len(sells)
        k4.metric("Net Signal", f"{'🟢 Bullish' if net>0 else '🔴 Caution' if net<-2 else '⚪ Neutral'}",
                  delta=f"{abs(net)} tx net {'buying' if net>0 else 'selling'}")
        ai_key = f"insider_ai_{symbol}"
        if st.button("🤖  AI Insider Analysis", key=f"btn_{ai_key}"):
            with st.spinner("Analysing patterns…"):
                st.session_state[ai_key] = generate_insider_ai_analysis(symbol, df)
        if ai_key in st.session_state:
            st.markdown("---")
            # FIX: sanitize $ signs to prevent LaTeX rendering
            st.markdown(_sanitize_ai_markdown(st.session_state[ai_key]))
            if st.button("🗑️  Clear", key=f"clr_{ai_key}"):
                del st.session_state[ai_key]; st.rerun()

# ================================================================
# OPENINSIDER — multi-URL fallback
# ================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_openinsider_cluster_buys():
    candidate_urls = [
        ("https://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd=5&fdr=&td=0&tdr="
         "&xp=1&sortcol=0&cnt=100&page=1"),
        "https://openinsider.com/latest-cluster-buys",
        "https://openinsider.com/clustered-buys",
    ]
    for url in candidate_urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200: continue
            dfs = pd.read_html(StringIO(r.text), flavor="lxml")
            for df in dfs:
                col = next(
                    (c for c in df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()),
                    None
                )
                if col is None and len(df.columns) > 2:
                    col = df.columns[2]
                if col is not None:
                    tickers = set(df[col].dropna().astype(str).str.upper().str.strip().tolist())
                    tickers = {t for t in tickers if t.isalpha() and 1 <= len(t) <= 5}
                    if len(tickers) > 2:
                        return tickers
        except: continue
    return set()

# ================================================================
# PERFORMANCE CACHE — retry-with-backoff, longer TTL
# ================================================================
@st.cache_data(ttl=900, show_spinner=False)  # 15 min cache
def _cached_history(symbol: str, period: str) -> pd.DataFrame:
    """Single-ticker history with one retry on transient failure."""
    for attempt in range(2):
        try:
            df = yf.Ticker(symbol).history(period=period)
            if not df.empty:
                return df.ffill()
        except Exception:
            if attempt == 0:
                time.sleep(0.5)  # brief backoff before retry
    return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def _batch_history(symbols_tuple: tuple, period: str) -> dict:
    """
    Batch-fetch via yf.download — single biggest performance win.
    Fetches 100 tickers per HTTP request instead of one at a time.
    For 500 tickers: ~5 requests vs 500 → 10-20x throughput.
    """
    symbols = sorted(set(symbols_tuple))
    if not symbols:
        return {}

    result = {}
    chunk_size = 100
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i+chunk_size]
        attempt_succeeded = False
        for attempt in range(2):
            try:
                data = yf.download(
                    chunk, period=period, group_by='ticker',
                    auto_adjust=True, progress=False, threads=True
                )
                for sym in chunk:
                    try:
                        if isinstance(data.columns, pd.MultiIndex) and sym in data.columns.get_level_values(0):
                            df = data[sym]
                        elif len(chunk) == 1:
                            df = data
                        else:
                            df = pd.DataFrame()
                        result[sym] = df.ffill() if df is not None and not df.empty else pd.DataFrame()
                    except Exception:
                        result[sym] = pd.DataFrame()
                attempt_succeeded = True
                break
            except Exception:
                if attempt == 0:
                    time.sleep(0.5)
        if not attempt_succeeded:
            for sym in chunk:
                result[sym] = pd.DataFrame()
    return result

# ================================================================
# CORE TECHNICAL ANALYSIS ENGINE
# Returns 7 values: (is_bullish, metrics, fig, price, fib, score, error)
# ================================================================
def _safe_last(series, fallback=0.0):
    valid = series.dropna()
    return float(valid.iloc[-1]) if not valid.empty else fallback

def fetch_technical_data(symbol: str, period_window: int, calc_type: str,
                          prefetched_hist: pd.DataFrame = None):
    """
    Compute all indicators and confluence score for a single ticker.
    If prefetched_hist is supplied (from scanner batch fetch), uses it
    directly — otherwise falls back to single-ticker cached fetch.
    """
    pfx = f"[{symbol}/{period_window}d/{calc_type[:3]}]"
    try:
        lookback = INTERVAL_MAP[period_window]["history"]
        if prefetched_hist is not None and not prefetched_hist.empty:
            hist = prefetched_hist
        else:
            hist = _cached_history(symbol, lookback)

        hard_need = max(period_window, 35, 20) + 2
        soft_min  = max(int(period_window * 0.80), 35)

        if hist.empty:
            return False,{},None,0.0,{},0,f"{pfx} No data from yfinance."
        if len(hist) < 15:
            return False,{},None,0.0,{},0,f"{pfx} Only {len(hist)} bars — too few for any analysis."

        # Auto-fallback to shorter MA window if history is limited
        effective_window = period_window
        fallback_note    = ""
        if len(hist) < soft_min:
            for fp in [p for p in [50,20,10] if p < period_window]:
                if len(hist) >= max(int(fp * 0.80), 15):
                    effective_window = fp
                    fallback_note    = f" (auto-fallback: {len(hist)} bars, used {fp}d)"
                    break
            else:
                return False,{},None,0.0,{},0, f"{pfx} Only {len(hist)} bars — insufficient even for 10-day analysis."

        hist  = hist.tail(min(len(hist), hard_need + 100)).copy()
        close = hist["Close"].copy()

        # Moving Average
        if "Exponential" in calc_type:
            hist["MA"] = close.ewm(span=effective_window, adjust=False).mean()
            ma_label   = f"EMA-{effective_window}"
        elif "Simple" in calc_type:
            hist["MA"] = close.rolling(effective_window).mean()
            ma_label   = f"SMA-{effective_window}"
        else:
            w = np.arange(1, effective_window + 1, dtype=float)
            hist["MA"] = close.rolling(effective_window).apply(
                lambda p: float(np.dot(p, w) / w.sum()), raw=True
            )
            ma_label = f"WMA-{effective_window}"

        # Bollinger Bands
        bb_mid = close.rolling(20).mean(); bb_std = close.rolling(20).std(ddof=0)
        hist["BB_Upper"] = bb_mid + 2*bb_std; hist["BB_Mid"] = bb_mid; hist["BB_Lower"] = bb_mid - 2*bb_std

        # RSI — Wilder's smoothing
        delta    = close.diff()
        avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        hist["RSI"] = (100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))).clip(0, 100)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean(); ema26 = close.ewm(span=26, adjust=False).mean()
        hist["MACD"] = ema12 - ema26
        hist["MACD_Sig"] = hist["MACD"].ewm(span=9, adjust=False).mean()
        hist["MACD_H"]   = hist["MACD"] - hist["MACD_Sig"]

        # Fibonacci
        sh = float(close.max()); sl = float(close.min())
        fib_levels = {lbl: sl + r*(sh-sl) for lbl,r in FIB_RATIOS.items()}

        # Crossover signals
        ma_valid = hist["MA"].notna()
        prev_close = close.shift(1); prev_ma = hist["MA"].shift(1)
        hist["Buy"]  = np.where(ma_valid & (close>hist["MA"]) & (prev_close<=prev_ma), close, np.nan)
        hist["Sell"] = np.where(ma_valid & (close<hist["MA"]) & (prev_close>=prev_ma), close, np.nan)

        # Snapshots
        cur_price = _safe_last(close); cur_ma  = _safe_last(hist["MA"])
        cur_rsi   = _safe_last(hist["RSI"], 50.0)
        cur_macd  = _safe_last(hist["MACD"]); cur_sig = _safe_last(hist["MACD_Sig"])
        cur_bbu   = _safe_last(hist["BB_Upper"]); cur_bbl = _safe_last(hist["BB_Lower"])
        prior     = float(close.iloc[max(0, len(hist)-21)])
        momentum  = ((cur_price - prior) / prior * 100) if prior > 0 else 0.0
        avg_vol   = float(hist["Volume"].rolling(20, min_periods=1).mean().iloc[-1])
        vol_ratio = float(hist["Volume"].iloc[-1]) / avg_vol if avg_vol > 0 else 1.0
        nearest_fib  = min(fib_levels.items(), key=lambda x: abs(x[1]-cur_price))
        fib_distance = ((cur_price-nearest_fib[1])/nearest_fib[1]*100) if nearest_fib[1]>0 else 0.0

        buys = hist["Buy"].dropna(); sells = hist["Sell"].dropna()
        is_buy_crossover = not buys.empty and (sells.empty or buys.index[-1] > sells.index[-1])

        # ── Bars-since-signal (for fresh/stale scoring + alerts) ───
        bars_since_buy = 999
        if not buys.empty:
            try:
                last_buy_loc = hist.index.get_loc(buys.index[-1])
                bars_since_buy = len(hist) - 1 - last_buy_loc
            except Exception: bars_since_buy = 999

        bars_since_signal = 999
        latest_signal_idx = None
        if not buys.empty and not sells.empty:
            latest_signal_idx = max(buys.index[-1], sells.index[-1])
        elif not buys.empty:
            latest_signal_idx = buys.index[-1]
        elif not sells.empty:
            latest_signal_idx = sells.index[-1]
        if latest_signal_idx is not None:
            try:
                bars_since_signal = len(hist) - 1 - hist.index.get_loc(latest_signal_idx)
            except Exception: bars_since_signal = 999

        if is_buy_crossover:
            ma_signal = f"🟢 BUY  ({buys.index[-1].strftime('%m/%d')})"; is_bullish = True
        elif not sells.empty and (buys.empty or sells.index[-1] > buys.index[-1]):
            ma_signal = f"🔴 SELL ({sells.index[-1].strftime('%m/%d')})"; is_bullish = False
        else:
            ma_signal = "⚪ Neutral"; is_bullish = cur_price > cur_ma

        # Confluence scoring with age-aware MA dimension
        conf_score, conf_label, conf_breakdown = calc_confluence_score(
            is_ma_buy_crossover=is_buy_crossover,
            bars_since_buy=bars_since_buy,
            price_above_ma=(cur_price>cur_ma and cur_ma>0),
            rsi=cur_rsi, macd=cur_macd, macd_sig=cur_sig,
            price=cur_price, bbu=cur_bbu, bbl=cur_bbl,
            vol_ratio=vol_ratio, momentum=momentum, fib_levels=fib_levels,
        )

        # Human-readable age string
        if bars_since_signal == 0:    age_str = "today"
        elif bars_since_signal == 1:  age_str = "1d ago"
        elif bars_since_signal < 999: age_str = f"{bars_since_signal}d ago"
        else:                          age_str = "—"

        metrics = {
            "Price":              f"${cur_price:.2f}",
            "1-Mo Momentum":      f"{momentum:+.1f}%",
            f"{ma_label}":        f"${cur_ma:.2f}" if cur_ma > 0 else "N/A",
            "MA Fallback Note":   fallback_note.strip() if fallback_note else None,
            "MA Signal":          ma_signal,
            "Signal Age":         age_str,
            "RSI (Wilder)":       f"{cur_rsi:.1f} — " + ("🔴 Overbought" if cur_rsi>70 else "🟢 Oversold" if cur_rsi<30 else "🟢 Ideal" if 40<=cur_rsi<=65 else "⚪ Neutral"),
            "MACD":               "🟢 Bullish+" if cur_macd>cur_sig and cur_macd>0 else "🟡 Bullish±" if cur_macd>cur_sig else "🔴 Bearish",
            "Bollinger":          "🔴 Upper" if cur_bbu>0 and cur_price>=cur_bbu*0.99 else "🟢 Lower" if cur_bbl>0 and cur_price<=cur_bbl*1.01 else "⚪ Mid",
            "Volume vs 20-avg":   f"{vol_ratio:.2f}x — " + ("🟢 High" if vol_ratio>1.5 else "🔴 Light" if vol_ratio<0.5 else "⚪ Average"),
            "Fibonacci Zone":     f"Near {nearest_fib[0]} (${nearest_fib[1]:.2f}) — {'above' if fib_distance>0 else 'below'} by {abs(fib_distance):.1f}%",
            "Signal Strength":    f"{conf_label}  ({conf_score}/14)",
            "_score":             conf_score,
            "_breakdown":         conf_breakdown,
            "_bars_since_signal": bars_since_signal,
            "_bars_since_buy":    bars_since_buy,
            "_is_buy_crossover":  is_buy_crossover,
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}

        # ── Chart ──────────────────────────────────────────────
        chart_ma_label = f"{ma_label}{' [fallback]' if fallback_note else ''}"
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            row_heights=[0.50,0.17,0.17,0.16], vertical_spacing=0.025,
                            subplot_titles=[
                                f"{symbol} — Price, {chart_ma_label}, Bollinger & Fibonacci",
                                "Volume","MACD (12/26/9)","RSI (14 Wilder) | 70=Overbought · 30=Oversold"
                            ])
        fig.add_trace(go.Scatter(x=hist.index,y=hist["BB_Upper"],showlegend=False,line=dict(color="rgba(120,120,255,0.35)",width=1)),row=1,col=1)
        fig.add_trace(go.Scatter(x=hist.index,y=hist["BB_Lower"],fill="tonexty",fillcolor="rgba(120,120,255,0.07)",showlegend=False,line=dict(color="rgba(120,120,255,0.35)",width=1)),row=1,col=1)
        fig.add_trace(go.Scatter(x=hist.index,y=hist["BB_Mid"],showlegend=False,line=dict(color="rgba(180,180,255,0.4)",width=1,dash="dot")),row=1,col=1)
        for lbl,lvl in fib_levels.items():
            fig.add_hline(y=lvl,line_dash="dot",line_color=FIB_COLORS[lbl],line_width=1.2,
                          annotation_text=f" Fib {lbl}  ${lvl:.2f}",annotation_position="right",
                          annotation_font_size=9,annotation_font_color=FIB_COLORS[lbl],row=1,col=1)
        fig.add_trace(go.Scatter(x=hist.index,y=close,name="Price",line=dict(color="#4fc3f7",width=2)),row=1,col=1)
        if cur_ma > 0:
            fig.add_trace(go.Scatter(x=hist.index,y=hist["MA"],name=chart_ma_label,line=dict(color="#ffa726",width=1.8,dash="dot")),row=1,col=1)
        fig.add_trace(go.Scatter(x=hist.index,y=hist["Buy"],mode="markers",name="BUY",marker=dict(color="#4caf50",size=11,symbol="triangle-up",line=dict(color="white",width=1))),row=1,col=1)
        fig.add_trace(go.Scatter(x=hist.index,y=hist["Sell"],mode="markers",name="SELL",marker=dict(color="#f44336",size=11,symbol="triangle-down",line=dict(color="white",width=1))),row=1,col=1)
        vol_colors = ["#4caf50" if float(c)>=float(o) else "#f44336" for c,o in zip(hist["Close"],hist["Open"])]
        fig.add_trace(go.Bar(x=hist.index,y=hist["Volume"],marker_color=vol_colors,showlegend=False),row=2,col=1)
        hc = ["#4caf50" if float(v)>=0 else "#f44336" for v in hist["MACD_H"].fillna(0)]
        fig.add_trace(go.Bar(x=hist.index,y=hist["MACD_H"],marker_color=hc,showlegend=False),row=3,col=1)
        fig.add_trace(go.Scatter(x=hist.index,y=hist["MACD"],name="MACD",line=dict(color="#4fc3f7",width=1.5)),row=3,col=1)
        fig.add_trace(go.Scatter(x=hist.index,y=hist["MACD_Sig"],name="Signal",line=dict(color="#ffa726",width=1.5,dash="dot")),row=3,col=1)
        fig.add_trace(go.Scatter(x=hist.index,y=hist["RSI"],name="RSI",line=dict(color="#ce93d8",width=1.8)),row=4,col=1)
        for y_val,color in [(70,"rgba(244,67,54,0.55)"),(30,"rgba(76,175,80,0.55)")]:
            fig.add_hline(y=y_val,line_dash="dash",line_color=color,line_width=1.2,row=4,col=1)
        fig.add_hrect(y0=70,y1=100,fillcolor="rgba(244,67,54,0.04)",line_width=0,row=4,col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(76,175,80,0.04)", line_width=0,row=4,col=1)
        fig.add_hrect(y0=40,y1=65, fillcolor="rgba(79,195,247,0.03)",line_width=0,row=4,col=1)
        fig.update_layout(template="plotly_dark",hovermode="x unified",height=820,
                          margin=dict(l=10,r=80,t=40,b=10),paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(15,15,25,1)",
                          legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1,font=dict(size=11)))
        fig.update_yaxes(title_text="Price ($)",row=1,col=1,gridcolor="#1e1e2e")
        fig.update_yaxes(title_text="Volume",  row=2,col=1,gridcolor="#1e1e2e")
        fig.update_yaxes(title_text="MACD",    row=3,col=1,gridcolor="#1e1e2e")
        fig.update_yaxes(title_text="RSI",     row=4,col=1,range=[0,100],gridcolor="#1e1e2e")

        return is_bullish, metrics, fig, cur_price, fib_levels, conf_score, None

    except Exception as exc:
        return False,{},None,0.0,{},0, f"{pfx} {type(exc).__name__}: {exc}"

# ================================================================
# AI ANALYSIS
# ================================================================
def generate_ai_analysis(symbol, metrics, period, method, fib_levels=None, extra_context=""):
    if not AI_AVAILABLE: return "⚠️ AI unavailable — GEMINI_API_KEY not in Secrets."
    public = {k: v for k, v in metrics.items() if not k.startswith("_")}
    fib_text = ("\nFibonacci Levels:\n" + "\n".join(f"  {lbl}: ${lvl:.2f}" for lbl,lvl in fib_levels.items())) if fib_levels else ""
    prompt = f"""
You are an elite institutional analyst. Analyse {symbol} clearly for a client who may be a beginner.
Define jargon on first use. RSI uses Wilder's smoothing (same as TradingView/Bloomberg).
Signal Strength is a 0-14 multi-indicator confluence score with age-aware MA scoring.

Live Data:
{json.dumps(public, indent=2)}
{fib_text}
Framework: {period}-day {method}
{f"Context: {extra_context}" if extra_context else ""}

EXACTLY five sections:
## Quantitative Tear Sheet
Table: | Metric | Value | Plain-English Meaning |
## Elliott Wave & Trend Structure
Wave position, Fibonacci support/resistance, price targets AND invalidation level.
## Multi-Indicator Confluence
Where indicators agree/conflict, signal strength justification, one actionable sentence for beginners.
## Risk Assessment
Bull risk, bear risk, invalidation price, stop-loss zone (specific price range).
## Portfolio Strategy Suggestion
**Action:** (buy/hold/sell/avoid). Entry zone, target, stop-loss, position size tier. Risk/reward summary.
"""
    errors = []
    for m in GEMINI_MODEL_CANDIDATES:
        try:
            resp = gemini_client.models.generate_content(model=m, contents=prompt)
            return f"*Model: `{m}`*\n\n" + resp.text
        except Exception as e: errors.append(f"**{m}:** {str(e)[:120]}")
    return "### All Gemini models failed\n\n1. [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) → Create fresh key\n2. Streamlit → Settings → Secrets → `GEMINI_API_KEY = \"key\"`\n\n**Errors:**\n" + "\n".join(f"- {e}" for e in errors)

# ================================================================
# BATCH SCANNER — with pre-warm fetch
# ================================================================
def scan_tickers(ticker_list, period, calc_type, min_score=6, max_workers=8):
    """
    Two-phase scan:
      Phase 1: Batch-fetch all histories via yf.download (5-10x faster)
      Phase 2: Parallel indicator computation using pre-fetched data
    """
    results, figs = [], {}
    total = len(ticker_list)
    if total == 0: return results, figs

    # Phase 1: batch fetch
    progress = st.progress(0.0, text=f"📡 Pre-fetching {total} tickers via batch API…")
    lookback = INTERVAL_MAP[period]["history"]
    histories = _batch_history(tuple(sorted(set(ticker_list))), lookback)
    progress.progress(0.30, text=f"✓ Loaded {len(histories)} histories — now scoring…")

    # Phase 2: parallel indicator scoring (no network calls — uses prefetched data)
    done = 0
    def _scan_one(sym):
        return sym, *fetch_technical_data(sym, period, calc_type,
                                          prefetched_hist=histories.get(sym))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_scan_one, t): t for t in ticker_list}
        for future in as_completed(futures):
            done += 1
            progress.progress(0.30 + 0.70 * done/total,
                              text=f"Scoring … {done}/{total}")
            sym, bullish, metrics, fig, price, fib, score, err = future.result()
            if not err and score >= min_score and metrics:
                public = {k:v for k,v in metrics.items() if not k.startswith("_")}
                results.append({"Ticker": sym, "_score": score, **public})
                if fig: figs[sym] = (fig, fib, metrics.get("_breakdown",{}))
    progress.empty()
    results.sort(key=lambda x: x.pop("_score", 0), reverse=True)
    return results, figs

# ================================================================
# PORTFOLIO SIGNAL NOTIFICATIONS — fresh-only, compact
# ================================================================
def render_portfolio_alerts(charts: dict):
    """
    Show only signals where crossover happened in the last FRESH_SIGNAL_WINDOW
    trading days. Compact one-line format. Hides entirely if nothing fresh.
    """
    fresh_buys, fresh_sells = [], []
    for sym, entry in charts.items():
        fig, fib, metrics = _unpack_chart_entry(entry)
        if not isinstance(metrics, dict): continue

        bars_since = metrics.get("_bars_since_signal", 999)
        if not isinstance(bars_since, (int, float)) or bars_since > FRESH_SIGNAL_WINDOW:
            continue

        signal = metrics.get("MA Signal", "")
        label  = ticker_label(sym)
        price  = metrics.get("Price", "—")
        strength = metrics.get("Signal Strength", "—")
        age    = metrics.get("Signal Age", "—")

        if "BUY" in signal:
            fresh_buys.append((label, signal, price, strength, age, int(bars_since)))
        elif "SELL" in signal:
            fresh_sells.append((label, signal, price, strength, age, int(bars_since)))

    if not fresh_buys and not fresh_sells:
        return  # hide entirely — no notification clutter

    # Sort each group by freshness (newest first)
    fresh_buys.sort(key=lambda x: x[5])
    fresh_sells.sort(key=lambda x: x[5])

    total = len(fresh_buys) + len(fresh_sells)
    with st.container(border=True):
        st.markdown(f"### 🔔  Fresh Signals — Last {FRESH_SIGNAL_WINDOW} Trading Days")
        st.caption(f"{total} active signal(s) from your portfolio. Older signals are filtered out.")
        for label, signal, price, strength, age, _bars in fresh_buys:
            st.markdown(
                f"🟢 **{label}** — {signal} · {age} · {price} · {strength}"
            )
        for label, signal, price, strength, age, _bars in fresh_sells:
            st.markdown(
                f"🔴 **{label}** — {signal} · {age} · {price} · {strength}"
            )

# ================================================================
# TRADE LOG — Buy/Sell with cost averaging
# ================================================================
def render_trade_form(portfolio: dict, uid: str, pin: str):
    with st.expander("💹  Log a Trade (Buy / Sell)", expanded=False):
        st.caption(
            "**Buy:** averages your cost basis if position exists. "
            "**Sell:** reduces shares and shows realized P&L. "
            "Changes sync to cloud automatically."
        )
        with st.form("trade_form", clear_on_submit=True):
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                trade_ticker = st.text_input(
                    "Ticker", placeholder="AAPL · ETH · BTC…",
                    help="Crypto auto-detected (ETH → ETH-USD)"
                ).strip()
                trade_shares = st.number_input(
                    "Shares / Units", min_value=0.000001,
                    step=0.000001, format="%.8f"
                )
            with t_col2:
                trade_type  = st.radio("Trade Type", ["🟢 Buy","🔴 Sell"], horizontal=True)
                trade_price = st.number_input(
                    "Price per Share ($)", min_value=0.000001, step=0.01
                )
            submitted = st.form_submit_button("📝  Log Trade", use_container_width=True)

        if submitted and trade_ticker and trade_shares > 0 and trade_price > 0:
            ticker = normalize_ticker(trade_ticker)
            label  = ticker_label(ticker)
            is_buy = "Buy" in trade_type
            new_portfolio = dict(portfolio)

            if is_buy:
                if ticker in new_portfolio:
                    old      = new_portfolio[ticker]
                    total_sh = old["shares"] + trade_shares
                    avg_cost = (old["shares"]*old["cost"] + trade_shares*trade_price) / total_sh
                    new_portfolio[ticker] = {"shares": round(total_sh,8), "cost": round(avg_cost,8)}
                    st.success(
                        f"✅ Added {trade_shares:.6f} {label}  ·  "
                        f"New avg cost: ${avg_cost:.4f}  ·  Total: {total_sh:.6f} units"
                    )
                else:
                    new_portfolio[ticker] = {"shares": round(trade_shares,8), "cost": round(trade_price,8)}
                    st.success(f"✅ New position: {trade_shares:.6f} {label} at ${trade_price:.4f}")
            else:  # Sell
                if ticker not in new_portfolio:
                    st.error(f"❌ {label} is not in your portfolio.")
                else:
                    old       = new_portfolio[ticker]
                    remaining = round(old["shares"] - trade_shares, 8)
                    pnl       = (trade_price - old["cost"]) * trade_shares
                    if remaining <= 0:
                        del new_portfolio[ticker]
                        st.success(
                            f"✅ Closed {label} position · Sold {old['shares']:.6f} at ${trade_price:.4f} · "
                            f"Realized P&L: **${pnl:+,.2f}**"
                        )
                    else:
                        new_portfolio[ticker]["shares"] = remaining
                        st.success(
                            f"✅ Sold {trade_shares:.6f} {label} at ${trade_price:.4f} · "
                            f"Realized P&L: **${pnl:+,.2f}** · Remaining: {remaining:.6f} units"
                        )

            st.session_state["user_portfolio"] = new_portfolio
            if SUPABASE_AVAILABLE:
                for t, pos in new_portfolio.items():
                    save_position_to_db(uid, pin, t, pos["shares"], pos["cost"])
                for t in set(portfolio.keys()) - set(new_portfolio.keys()):
                    save_position_to_db(uid, pin, t, 0, 0)
            st.rerun()

# ================================================================
# DISPLAY HELPERS
# ================================================================
def display_metrics_grid(metrics):
    public = {k:v for k,v in metrics.items() if not k.startswith("_")}
    items  = list(public.items())
    for i in range(0, len(items), 4):
        chunk = items[i:i+4]; cols = st.columns(len(chunk))
        for col,(k,v) in zip(cols,chunk): col.metric(k, str(v))

def show_ai_report(report_key, symbol, metrics, period, method, fib_levels,
                   extra_context="", button_label="🤖  Generate AI Analyst Report"):
    if st.button(button_label, key=f"btn_{report_key}"):
        with st.spinner("Gemini is analysing — may take 20–40 seconds…"):
            st.session_state[f"rpt_{report_key}"] = generate_ai_analysis(
                symbol, metrics, period, method, fib_levels, extra_context
            )
    stored = st.session_state.get(f"rpt_{report_key}")
    if stored:
        st.markdown("---")
        # FIX: sanitize $ signs to prevent LaTeX rendering
        st.markdown(_sanitize_ai_markdown(stored))
        if st.button("🗑️  Clear Report", key=f"clr_{report_key}"):
            del st.session_state[f"rpt_{report_key}"]; st.rerun()

def render_signal_breakdown(breakdown: dict):
    if not breakdown: return
    with st.expander("📊  Signal Score Breakdown — why this ticker was selected", expanded=False):
        st.caption("Each dimension contributes points to the 14-pt confluence score. MA scoring is age-aware.")
        for dim, (explanation, pts) in breakdown.items():
            bar = "█"*pts + "░"*(3-min(pts,3))
            st.markdown(f"**{dim}** `{pts}pt` {bar}  \n{explanation}")

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
                                     textinfo="label+percent" if n<=4 else "percent",
                                     hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
                                     marker=dict(line=dict(color="#0f0f19",width=2)))])
        fig.update_layout(template="plotly_dark",showlegend=(n>4),height=380,
                          margin=dict(l=20,r=20,t=10,b=10),paper_bgcolor="rgba(0,0,0,0)",
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

def render_portfolio_editor(portfolio: dict, uid: str, pin: str):
    with st.expander("✏️  Edit / Delete / Add Positions", expanded=False):
        st.caption("Edit cells · Delete rows (checkbox) · Add rows · Crypto auto-detected · Save when done.")
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
                    help="Stock: AAPL · Crypto: ETH, BTC (auto-normalised)", width="small"),
                "Full Ticker":    st.column_config.TextColumn("Full Ticker", disabled=True, width="small"),
                "Shares / Units": st.column_config.NumberColumn("Shares / Units",
                    min_value=0.0, format="%.8f", width="medium"),
                "Avg Cost ($)":   st.column_config.NumberColumn("Avg Cost ($)",
                    min_value=0.0, format="$%.2f", width="medium"),
            },
            key="portfolio_editor_table",
        )
        col_s, col_h = st.columns([1,3])
        with col_s:
            save_edits = st.button("💾  Save All Changes", use_container_width=True, key="save_all_edits")
        with col_h:
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
                for t,pos in new_portfolio.items(): save_position_to_db(uid,pin,t,pos["shares"],pos["cost"])
                for t in deleted: save_position_to_db(uid,pin,t,0,0)
            st.session_state["user_portfolio"] = new_portfolio
            if deleted: st.info(f"🗑️  Removed: {', '.join(deleted)}")
            st.success(f"✅ Portfolio saved — {len(new_portfolio)} position(s)")
            st.rerun()

# ================================================================
# HELPER: safe chart map unpack
# ================================================================
def _unpack_chart_entry(entry):
    """Safely unpack (fig, fib, metrics_or_breakdown) regardless of tuple length."""
    if isinstance(entry, (list, tuple)):
        if len(entry) >= 3: return entry[0], entry[1], entry[2]
        if len(entry) == 2: return entry[0], entry[1], {}
        if len(entry) == 1: return entry[0], {}, {}
    return None, {}, {}

# ================================================================
# MAIN APP
# ================================================================
st.title("📊  Wall Street AI Dashboard")
st.caption("Institutional-grade analysis · Gemini 2.5 Pro · EMA/SMA/WMA · Wilder RSI · Age-Aware Confluence Scoring · International · Small-Cap · Insider Activity")
render_indicator_guide()
st.divider()

with st.sidebar:
    st.header("⚙️  Analysis Settings")
    ma_type = st.radio("Moving Average Type",
                       ["Exponential Moving Average (EMA)","Simple Moving Average (SMA)","Weighted Moving Average (WMA)"],
                       help="EMA: industry standard. SMA: best for 200-day. WMA: fastest, most noise.")
    sma_period = st.selectbox("Lookback Period", options=list(INTERVAL_MAP.keys()), index=1,
                              format_func=lambda x: INTERVAL_MAP[x]["label"],
                              help="200-day fetches 5 years. Small-caps auto-fallback to shorter window if needed.")
    st.divider()
    if not AI_AVAILABLE:
        st.error("⚠️ GEMINI_API_KEY missing — AI disabled.")
    elif len(GEMINI_API_KEY) < 20:
        st.error("⚠️ GEMINI_API_KEY too short — check Secrets.")
    else:
        masked = GEMINI_API_KEY[:6] + "•"*8 + GEMINI_API_KEY[-4:]
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
                    portfolio_db = load_portfolio_from_db(uid, pin)
                if portfolio_db is None: st.error("❌ Incorrect PIN.")
                else:
                    st.session_state.update({"auth_user":uid,"auth_pin":pin,"user_portfolio":portfolio_db})
                    st.rerun()
        st.stop()

    with st.sidebar:
        st.subheader("🛠️  Quick Add Position")
        if SUPABASE_AVAILABLE:
            st.caption(f"Logged in as: **{st.session_state.get('auth_user','—')}**")

        with st.form("position_form"):
            raw_ticker = st.text_input("Ticker Symbol",
                placeholder="AAPL · NVDA · ETH · BTC",
                help="Crypto auto-detected — no suffix needed.").strip()
            new_shares = st.number_input("Shares / Units", min_value=0.0, step=0.00000001, format="%.8f")
            new_cost   = st.number_input("Avg Purchase Price ($)", min_value=0.0, step=0.01)
            save_btn   = st.form_submit_button("➕  Add Position", use_container_width=True)
        if save_btn and raw_ticker:
            new_ticker = normalize_ticker(raw_ticker)
            uid_ = st.session_state.get("auth_user","local"); pin_ = st.session_state.get("auth_pin","")
            if new_shares == 0:
                st.session_state["user_portfolio"].pop(new_ticker, None)
                if SUPABASE_AVAILABLE: save_position_to_db(uid_, pin_, new_ticker, 0, 0)
                st.warning(f"Removed {new_ticker}")
            else:
                st.session_state["user_portfolio"][new_ticker] = {"shares":new_shares,"cost":new_cost}
                if SUPABASE_AVAILABLE: save_position_to_db(uid_, pin_, new_ticker, new_shares, new_cost)
                st.success(f"✅ {new_ticker} added")

        st.divider()
        render_trade_form(
            st.session_state.get("user_portfolio",{}),
            st.session_state.get("auth_user","local"),
            st.session_state.get("auth_pin","")
        )

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
        st.info("Portfolio is empty. Use Quick Add or the edit table below.")
        render_portfolio_editor({}, uid, pin)
    else:
        total_value = total_cost = 0.0
        rows, charts, load_errors = [], {}, []

        with st.spinner("Fetching live data (15-min cache active)…"):
            for sym, pos in list(portfolio.items()):
                _, metrics, fig, price, fib, score, err = fetch_technical_data(sym, sma_period, ma_type)
                if err:
                    load_errors.append(f"**{ticker_label(sym)}:** {err}"); continue
                if price > 0:
                    pos_cost = pos["shares"]*pos["cost"]; pos_value = pos["shares"]*price
                    pos_gain = pos_value-pos_cost; pos_pct = (pos_gain/pos_cost*100) if pos_cost>0 else 0.0
                    total_value += pos_value; total_cost += pos_cost
                    if fig: charts[sym] = (fig, fib, metrics)
                    rows.append({
                        "Display":        ticker_label(sym), "Asset": sym,
                        "Shares":         f"{pos['shares']:.8f}".rstrip("0").rstrip("."),
                        "Avg Cost":       f"${pos['cost']:.2f}",
                        "Current Price":  f"${price:.2f}",
                        "Mkt Value":      f"${pos_value:,.2f}",
                        "Return ($)":     f"${pos_gain:+,.2f}",
                        "Return (%)":     f"{pos_pct:+.1f}%",
                        "MA Signal":      metrics.get("MA Signal","—"),
                        "Signal Age":     metrics.get("Signal Age","—"),
                        "RSI":            metrics.get("RSI (Wilder)","—"),
                        "Signal Strength":metrics.get("Signal Strength","—"),
                    })

        if load_errors:
            with st.expander(f"⚠️  {len(load_errors)} position(s) could not load — click to see why", expanded=True):
                for e in load_errors: st.warning(e)
                st.caption("Common causes: insufficient history for the selected period. Try a shorter lookback or use the Edit table to fix tickers.")

        # Fresh portfolio notifications (last 5 trading days only)
        if charts:
            render_portfolio_alerts(charts)

        total_gain = total_value-total_cost
        total_pct  = (total_gain/total_cost*100) if total_cost>0 else 0.0
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total Portfolio Value", f"${total_value:,.2f}")
        k2.metric("Total Cost Basis",      f"${total_cost:,.2f}")
        k3.metric("Total Return",          f"${total_gain:+,.2f}", f"{total_pct:+.2f}%")
        k4.metric("Open Positions",        str(len(rows)))

        st.subheader("Holdings Summary")
        if rows:
            df_d = pd.DataFrame(rows)
            dcols = [c for c in ["Display","Shares","Avg Cost","Current Price","Mkt Value",
                                  "Return ($)","Return (%)","MA Signal","Signal Age","RSI","Signal Strength"] if c in df_d.columns]
            st.dataframe(df_d[dcols].rename(columns={"Display":"Asset"}), use_container_width=True, hide_index=True)
        elif not load_errors:
            st.info("No positions loaded. Try a shorter lookback period.")

        render_portfolio_editor(portfolio, uid, pin)

        if rows:
            render_diversity_chart(rows, total_value)
            st.subheader("📈  Deep-Dive Chart & AI Analysis")
            sym_options = {ticker_label(s): s for s in portfolio.keys() if s in charts}
            if sym_options:
                chosen_label = st.selectbox("Select a holding", list(sym_options.keys()))
                chosen = sym_options[chosen_label]
                fig, fib, chosen_metrics = _unpack_chart_entry(charts[chosen])
                if fig: st.plotly_chart(fig, use_container_width=True)
                render_signal_breakdown(chosen_metrics.get("_breakdown", {}) if isinstance(chosen_metrics, dict) else {})
                if not chosen.endswith("-USD"): render_insider_section(chosen)
                pos_detail = portfolio.get(chosen,{})
                ins_df = get_insider_transactions(chosen) if not chosen.endswith("-USD") else None
                pnl_row = next((r for r in rows if r["Asset"]==chosen),{})
                context = (f"Held: {pos_detail.get('shares',0):.8f} units at "
                           f"${pos_detail.get('cost',0):.2f} avg. "
                           f"P&L: {pnl_row.get('Return (%)','unknown')}. {insider_summary(ins_df)}")
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
                                  placeholder="NVDA · AAPL · ETH · BTC · ASML · TSM").strip()
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        go_btn = st.button("Analyse →", use_container_width=True)

    if go_btn and raw_input:
        symbol_input = normalize_ticker(raw_input)
        with st.spinner(f"Fetching {symbol_input}…"):
            result = fetch_technical_data(symbol_input, sma_period, ma_type)
        _, metrics, fig, price, fib, score, err = result
        if err or price == 0.0:
            st.error(err or f"No data for **{symbol_input}**.")
        else:
            st.session_state["single_result"] = (symbol_input, result, sma_period, ma_type)

    if "single_result" in st.session_state:
        sym, (_, metrics, fig, price, fib, score, _err), period, method = st.session_state["single_result"]
        if st.button("🔄  Analyse a different ticker", key="clear_single"):
            del st.session_state["single_result"]; st.rerun()
        st.markdown(f"**Showing: `{sym}`** — {INTERVAL_MAP.get(period,{}).get('label','')} · Confluence Score: **{score}/14**")
        display_metrics_grid(metrics)
        render_signal_breakdown(metrics.get("_breakdown",{}))
        st.plotly_chart(fig, use_container_width=True)
        if not sym.endswith("-USD"): render_insider_section(sym)
        ins_df = get_insider_transactions(sym) if not sym.endswith("-USD") else None
        show_ai_report(f"single_{sym}", sym, metrics, period, method, fib,
                       extra_context=insider_summary(ins_df),
                       button_label="🤖  Generate Full AI Report")

# ================================================================
# MODE 3 — MARKET SCANNER
# ================================================================
elif mode == "🌐  Market Scanner":
    st.header("🌐  Market Scanner")
    st.caption(
        "Every ticker scored 0–14 across 6 indicators (MA is age-aware). "
        "Batch-fetch is 5–10x faster than v2.8. Only tickers above your threshold are shown, sorted highest first."
    )

    tab_stocks, tab_crypto = st.tabs(["📈  Stocks & International","₿  Crypto"])

    with tab_stocks:
        c1,c2,c3,c4 = st.columns([3,2,2,1])
        with c1:
            universe = st.selectbox("Market Universe", [
                "S&P 500  (~500 large-caps, ~1–2 min)",
                "Russell 2000  (~2,000 small-caps, ~3–6 min)",
                "All US Equities  (~1,500–3,000 stocks, ~5–10 min)",
                "International Stocks  (~120 ADRs & dual-listed, ~1–2 min)",
                "Small-Cap Growth  (~200–800 stocks, ~2–4 min)",
            ])
        with c2:
            threshold_label = st.selectbox(
                "Min Signal Strength",
                options=list(SCORE_THRESHOLDS.keys()), index=1,
                help="Higher = fewer but higher-quality signals"
            )
            min_score = SCORE_THRESHOLDS[threshold_label]
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            insider_filter = st.checkbox("🔍 Insider cluster buy filter",
                                         help="Pre-filter to stocks with 3+ insiders buying (OpenInsider). Stacks with score threshold.")
        with c4:
            st.markdown("<br>", unsafe_allow_html=True)
            scan_btn = st.button("🚀  Launch Scan", use_container_width=True)

        if "International" in universe:
            st.info("ℹ️ ~120 international ADRs & dual-listed stocks · Expected: 1–2 minutes")
        elif "Small-Cap" in universe:
            st.info("ℹ️ Small-cap growth ($50M–$3B market cap) · Expected: 2–4 minutes")
        elif "Russell" in universe:
            st.warning("⚠️ Large scan — expect 3–6 min with batch-fetch.")
        elif "All US" in universe:
            st.warning("⚠️ Very large scan — expect 5–10 min with batch-fetch.")
        else:
            st.info(f"ℹ️ ~500 large-cap stocks · 1–2 min · min score: {min_score}/14")

        if scan_btn:
            with st.spinner("Loading ticker universe…"):
                if "S&P 500" in universe:          tickers = get_sp500_tickers()
                elif "Russell" in universe:         tickers = get_russell2000_tickers()
                elif "International" in universe:   tickers = get_international_tickers()
                elif "Small-Cap" in universe:       tickers = get_smallcap_growth_tickers()
                else:                               tickers = get_all_us_equities()

            if not tickers:
                st.error("Failed to load ticker list.")
            else:
                if insider_filter:
                    with st.spinner("Loading OpenInsider cluster buy data…"):
                        insider_tickers = get_openinsider_cluster_buys()
                    if insider_tickers:
                        before = len(tickers)
                        tickers = [t for t in tickers if t in insider_tickers]
                        st.info(f"Insider filter: {before} → **{len(tickers)}** tickers with cluster buying")
                    else:
                        st.warning("OpenInsider data unavailable — running without insider filter.")

                if not tickers:
                    st.warning("No tickers after filter. Try without insider filter.")
                else:
                    st.info(f"Scanning **{len(tickers)}** tickers for confluence score ≥ **{min_score}/14**…")
                    results, figs = scan_tickers(tickers, sma_period, ma_type, min_score=min_score)

                    if results:
                        exc_ct  = sum(1 for r in results if "Exceptional" in r.get("Signal Strength",""))
                        str_ct  = sum(1 for r in results if "Strong Buy"  in r.get("Signal Strength","") and "Exceptional" not in r.get("Signal Strength",""))
                        mod_ct  = sum(1 for r in results if "Moderate"    in r.get("Signal Strength",""))
                        fresh_count = sum(
                            1 for r in results
                            if r.get("Signal Age","") == "today" or
                               any(r.get("Signal Age","").startswith(f"{n}d") for n in range(0,6))
                        )
                        st.success(f"✅ **{len(results)}** setups found · **{fresh_count}** with fresh crossover signals (≤5d)")
                        s1,s2,s3,s4 = st.columns(4)
                        s1.metric("🔥 Exceptional", exc_ct)
                        s2.metric("🟢 Strong Buy",  str_ct)
                        s3.metric("🟡 Moderate Buy", mod_ct)
                        s4.metric("⚡ Fresh (≤5d)", fresh_count)

                        df_r = pd.DataFrame(results)
                        scan_display = [c for c in ["Ticker","Signal Strength","Signal Age","Price","1-Mo Momentum",
                                                     "MA Signal","RSI (Wilder)","MACD","Fibonacci Zone"] if c in df_r.columns]
                        st.dataframe(df_r[scan_display], use_container_width=True, hide_index=True)

                        st.subheader("📊  Deep-Dive Chart")
                        triggered = [r["Ticker"] for r in results]
                        view_sym  = st.selectbox("Select a ticker (sorted by score)", triggered)

                        if view_sym in figs:
                            fig, fib, breakdown = _unpack_chart_entry(figs[view_sym])
                            if fig: st.plotly_chart(fig, use_container_width=True)
                            render_signal_breakdown(breakdown)
                            if not view_sym.endswith("-USD"): render_insider_section(view_sym)
                            stock_metrics = next((r for r in results if r["Ticker"]==view_sym),{})
                            ins_df = get_insider_transactions(view_sym) if not view_sym.endswith("-USD") else None
                            show_ai_report(f"scanner_{view_sym}", view_sym, stock_metrics,
                                           sma_period, ma_type, fib,
                                           extra_context=insider_summary(ins_df),
                                           button_label="🤖  AI Analysis for this stock")
                    else:
                        st.warning(f"No stocks scored ≥{min_score}/14 with current settings. Try lowering the threshold, switching to EMA, or a shorter lookback period.")

    with tab_crypto:
        st.subheader("₿  Major Crypto Dashboard")
        st.caption("Bitcoin · Ethereum · XRP · Solana — all shown with confluence scores")
        crypto_btn = st.button("📡  Refresh Crypto Data")

        # Clear stale session state from old tuple format
        if "crypto_data" in st.session_state:
            cr, cfm = st.session_state["crypto_data"]
            if cfm:
                sample = next(iter(cfm.values()), None)
                if sample is not None and isinstance(sample, tuple) and len(sample) < 3:
                    del st.session_state["crypto_data"]

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
            chosen_crypto = st.selectbox(
                "Select crypto for chart", list(CRYPTO_TICKERS.values()),
                format_func=lambda s: next((k for k,v in CRYPTO_TICKERS.items() if v==s),s)
            )
            if chosen_crypto in crypto_fig_map:
                fig, fib, breakdown = _unpack_chart_entry(crypto_fig_map[chosen_crypto])
                if fig: st.plotly_chart(fig, use_container_width=True)
                render_signal_breakdown(breakdown)
                crypto_metrics = next((r for r in crypto_rows if chosen_crypto in r.get("Asset","")),{})
                show_ai_report(f"crypto_{chosen_crypto}", chosen_crypto, crypto_metrics,
                               sma_period, ma_type, fib,
                               extra_context="Cryptocurrency: 24/7 trading, higher volatility, no traditional fundamentals, no insider data.",
                               button_label="🤖  Generate Crypto AI Analysis")
            else:
                st.info("Click 'Refresh Crypto Data' to reload chart data.")
        else:
            st.info("Click 'Refresh Crypto Data' to load.")
