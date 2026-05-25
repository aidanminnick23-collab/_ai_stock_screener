# ================================================================
# WALL STREET AI DASHBOARD — Production Build v2.2
# Fixes: 403 ticker fetch, API key diagnostics, TypeError in crypto
# New: Insider buying, portfolio diversity chart, performance cache,
#      8-decimal share input, OpenInsider scanner filter
# ================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import json
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Wall Street AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 1.05rem; }
    .insider-buy  { color: #4caf50; font-weight: 600; }
    .insider-sell { color: #f44336; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# SECRETS
# ================================================================
def _secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return ""

GEMINI_API_KEY = _secret("GEMINI_API_KEY")
SUPABASE_URL   = _secret("SUPABASE_URL")
SUPABASE_KEY   = _secret("SUPABASE_KEY")

# ── Gemini client ────────────────────────────────────────────────
AI_AVAILABLE  = bool(GEMINI_API_KEY)
gemini_client = None
ACTIVE_MODEL  = None

# FIX: Full model candidate list — app tries each until one works
GEMINI_MODEL_CANDIDATES = [
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-flash",
    "gemini-1.5-pro",
]

if AI_AVAILABLE:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        AI_AVAILABLE = False

# ── Supabase client ──────────────────────────────────────────────
SUPABASE_AVAILABLE = False
db = None
try:
    if SUPABASE_URL and SUPABASE_KEY:
        from supabase import create_client
        db = create_client(SUPABASE_URL, SUPABASE_KEY)
        SUPABASE_AVAILABLE = True
except Exception:
    pass

# ================================================================
# CONSTANTS
# ================================================================
CRYPTO_TICKERS = {
    "Bitcoin (BTC)":  "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "XRP":            "XRP-USD",
    "Solana (SOL)":   "SOL-USD",
}

INTERVAL_MAP = {
    10:  {"history": "3mo", "label": "10-Day  · Short-Term Momentum"},
    20:  {"history": "6mo", "label": "20-Day  · Short-Term Trend"},
    50:  {"history": "1y",  "label": "50-Day  · Institutional Baseline"},
    100: {"history": "2y",  "label": "100-Day · Macro Cycle Support"},
    200: {"history": "2y",  "label": "200-Day · Ultimate Bull/Bear Line"},
}

FIB_RATIOS = {
    "78.6%": 0.786, "61.8%": 0.618,
    "50.0%": 0.500, "38.2%": 0.382, "23.6%": 0.236,
}
FIB_COLORS = {
    "78.6%": "rgba(255,82,82,0.55)",   "61.8%": "rgba(255,167,38,0.65)",
    "50.0%": "rgba(255,238,88,0.60)",  "38.2%": "rgba(102,187,106,0.65)",
    "23.6%": "rgba(79,195,247,0.55)",
}

# Browser-like headers to avoid 403 blocks from Wikipedia/financial sites
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ================================================================
# INDICATOR + ELLIOTT WAVE EDUCATION PANEL
# ================================================================
def render_indicator_guide():
    with st.expander(
        "📚  Indicator Guide — What does everything mean?  (click to expand)",
        expanded=False
    ):
        st.markdown("#### Learn the indicators used in every chart and analysis report")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("---\n### 📈 Moving Average (SMA / WMA)")
            st.info(
                "**What it is:** Average closing price over N days. SMA weights every day equally; "
                "WMA gives more weight to recent days.\n\n"
                "**Think of it as:** The stock's center of gravity — prices drift back toward it.\n\n"
                "🔺 Green triangle = crossed above → Buy signal\n"
                "🔻 Red triangle   = crossed below → Sell signal"
            )
            st.markdown("---\n### 📉 RSI — Relative Strength Index")
            st.info(
                "**What it is:** A 0–100 speed gauge for recent price movement (14-day).\n\n"
                "**Think of it as:** A runner's fatigue meter. Sprinting too long (>70) leads to "
                "a slowdown. Crawling exhausted (<30) precedes a burst.\n\n"
                "🔴 Above 70 = Overbought\n🟢 Below 30 = Oversold\n⚪ 30–70 = Neutral"
            )
            st.markdown("---\n### 📊 Volume")
            st.info(
                "**What it is:** Total shares traded. Validates whether a move has real conviction.\n\n"
                "**Think of it as:** Votes. Rising price + heavy volume = many investors agree.\n\n"
                "🟢 Rising price + High volume = Confirmed move\n"
                "🔴 Rising price + Low volume  = Suspect move"
            )

        with c2:
            st.markdown("---\n### ⚡ MACD")
            st.info(
                "**What it is:** Compares fast (12-day) vs slow (26-day) EMA. Gap = MACD line, "
                "plotted vs a 9-day signal line.\n\n"
                "**Think of it as:** Two pace cars — when the fast car leads, buyers are accelerating.\n\n"
                "🟢 MACD above Signal = Bullish\n🔴 MACD below Signal = Bearish\n"
                "📊 Histogram growing = Momentum strengthening"
            )
            st.markdown("---\n### 🎯 Bollinger Bands")
            st.info(
                "**What it is:** 20-day MA ± 2 standard deviations. Bands widen on volatility, "
                "narrow on calm markets.\n\n"
                "**Think of it as:** Highway lanes — price drifts to edge then snaps back.\n\n"
                "🔴 At upper band = Overbought\n🟢 At lower band = Oversold\n"
                "⚠️  Squeeze = Big move loading"
            )
            st.markdown("---\n### 👔 Insider / CEO Buying")
            st.info(
                "**What it is:** Purchases of company stock by executives, directors, or major "
                "shareholders — reported publicly via SEC Form 4 filings.\n\n"
                "**Think of it as:** The strongest vote of confidence possible. When a CEO buys "
                "$1M of their own stock with personal money, they're telling you something the "
                "news isn't.\n\n"
                "🟢 CEO / multiple officers buying = Very bullish signal\n"
                "🔴 Heavy insider selling = Caution warranted\n"
                "📌 Cluster buying (3+ insiders at once) = Highest conviction signal"
            )

        with c3:
            st.markdown("---\n### 🌊 Elliott Wave Theory")
            st.info(
                "**What it is:** Markets move in predictable wave patterns driven by crowd psychology.\n\n"
                "**The pattern:**\n"
                "- **Wave 1** — First move up; few notice\n"
                "- **Wave 2** — Pullback; 'Was that it?'\n"
                "- **Wave 3** — Strongest surge; euphoria builds\n"
                "- **Wave 4** — Mild dip; bulls buy the dip\n"
                "- **Wave 5** — Final push; peak optimism\n"
                "- **Wave A–C** — Corrective phase follows\n\n"
                "The AI identifies the likely current wave position."
            )
            st.markdown("---\n### 📐 Fibonacci Retracement Levels")
            st.info(
                "**What it is:** Key price zones from the Fibonacci sequence — the same ratio "
                "found throughout nature. Markets tend to reverse at these exact levels.\n\n"
                "🔵 **23.6%** — Shallow pullback, strong trend\n"
                "🟢 **38.2%** — Common healthy dip\n"
                "🟡 **50.0%** — Psychological midpoint\n"
                "🟠 **61.8%** — The Golden Ratio — strongest support\n"
                "🔴 **78.6%** — Deep retracement, trend may be weakening\n\n"
                "A bounce off the 61.8% level in an uptrend is one of the highest-probability "
                "setups in all of technical analysis."
            )

# ================================================================
# PORTFOLIO AUTH — Supabase
# ================================================================
def hash_pin(pin: str) -> str:
    return hashlib.sha256(pin.encode()).hexdigest()

def load_portfolio_from_db(user_id: str, pin: str):
    try:
        result = db.table("portfolios").select("*").eq("user_id", user_id).execute()
        rows   = result.data
        if not rows:
            return {}
        if rows[0]["pin_hash"] != hash_pin(pin):
            return None
        return {
            row["ticker"]: {"shares": row["shares"], "cost": row["cost"]}
            for row in rows
        }
    except Exception as e:
        st.error(f"Database error: {e}")
        return None

def save_position_to_db(user_id, pin, ticker, shares, cost) -> bool:
    try:
        if shares == 0:
            db.table("portfolios").delete() \
              .eq("user_id", user_id).eq("ticker", ticker).execute()
        else:
            db.table("portfolios").upsert({
                "user_id": user_id, "pin_hash": hash_pin(pin),
                "ticker": ticker, "shares": shares, "cost": cost,
            }, on_conflict="user_id,ticker").execute()
        return True
    except Exception as e:
        st.error(f"Save error: {e}")
        return False

# ================================================================
# TICKER UNIVERSE LOADERS
# FIX: Use browser headers + GitHub CSV primary source to avoid 403
# ================================================================
def _read_html_safe(url: str, **kwargs) -> list:
    """pd.read_html via requests with browser headers to avoid 403."""
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text), **kwargs)

@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers() -> list:
    # Primary: GitHub-hosted CSV (very reliable, no 403 risk)
    try:
        url = (
            "https://raw.githubusercontent.com/datasets/"
            "s-and-p-500-companies/main/data/constituents.csv"
        )
        df = pd.read_csv(url)
        tickers = sorted(df["Symbol"].str.replace(".", "-", regex=False).tolist())
        if len(tickers) > 400:
            return tickers
    except Exception:
        pass
    # Fallback: Wikipedia via requests with browser headers
    try:
        tables = _read_html_safe(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            flavor="lxml"
        )
        return sorted(tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception as e:
        st.warning(f"S&P 500 list fetch failed: {e}")
        return []

@st.cache_data(ttl=86400, show_spinner=False)
def get_sp1500_tickers() -> list:
    tickers = set(get_sp500_tickers())
    for url, col in [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Ticker"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Ticker"),
    ]:
        try:
            tables = _read_html_safe(url, flavor="lxml")
            tickers.update(tables[0][col].str.replace(".", "-", regex=False).tolist())
        except Exception:
            pass
    return sorted(list(tickers))

@st.cache_data(ttl=86400, show_spinner=False)
def get_russell2000_tickers() -> list:
    try:
        url = (
            "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"
            "/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
        )
        r  = requests.get(url, headers=HEADERS, timeout=20)
        df = pd.read_csv(StringIO(r.text), skiprows=9)
        df = df[df.get("Asset Class", df.columns[0]) == "Equity"]
        valid = sorted([
            t for t in df["Ticker"].dropna().str.strip().tolist()
            if t and t != "-" and len(t) <= 6
        ])
        if len(valid) > 500:
            return valid
    except Exception:
        pass
    try:
        tables = _read_html_safe(
            "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", flavor="lxml"
        )
        st.caption("ℹ️ Russell 2000 live feed unavailable — using S&P SmallCap 600 as proxy.")
        return sorted(tables[0]["Ticker"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        st.warning("Russell 2000 unavailable. Try S&P 500.")
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_us_equities() -> list:
    try:
        r    = requests.get(
            "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&download=true",
            headers=HEADERS, timeout=25
        )
        df   = pd.DataFrame(r.json()["data"]["rows"])
        df["volume"] = pd.to_numeric(
            df["volume"].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
        tickers = sorted(df[df["volume"] > 500_000]["symbol"].str.strip().tolist())
        if len(tickers) > 100:
            return tickers
    except Exception:
        pass
    st.caption("ℹ️ Full equity list unavailable — falling back to S&P 1500.")
    return get_sp1500_tickers()

# ================================================================
# INSIDER TRANSACTION DATA
# ================================================================
@st.cache_data(ttl=600, show_spinner=False)
def get_insider_transactions(symbol: str) -> pd.DataFrame | None:
    """
    Fetch insider buy/sell transactions via yfinance.
    Returns cleaned DataFrame or None if unavailable.
    """
    try:
        df = yf.Ticker(symbol).insider_transactions
        if df is None or df.empty:
            return None
        df = df.copy()
        # Normalise date column
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)
            df = df[df[date_col] >= cutoff]
        return df if not df.empty else None
    except Exception:
        return None

def insider_summary(df: pd.DataFrame | None) -> str:
    """One-line summary of insider activity for AI prompt context."""
    if df is None:
        return "No insider transaction data available."
    tx_col = next((c for c in df.columns if "transaction" in c.lower() or "text" in c.lower()), None)
    if tx_col:
        buys  = df[df[tx_col].astype(str).str.contains("Buy|Purchase|Acquired", case=False, na=False)]
        sells = df[df[tx_col].astype(str).str.contains("Sale|Sell|Disposed", case=False, na=False)]
        return (
            f"Insider activity (last 12 months): {len(buys)} purchase(s), "
            f"{len(sells)} sale(s). "
            + ("⚠️ Net insider selling dominates." if len(sells) > len(buys) * 2
               else "✅ Net insider buying — positive signal." if len(buys) > 0
               else "")
        )
    return "Insider data format unrecognised."

@st.cache_data(ttl=3600, show_spinner=False)
def get_openinsider_cluster_buys() -> set:
    """
    Fetch tickers with recent cluster insider buying from OpenInsider.
    Returns a set of uppercase ticker strings, or empty set on failure.
    """
    try:
        r  = requests.get(
            "https://openinsider.com/latest-cluster-buys",
            headers=HEADERS, timeout=20
        )
        dfs = pd.read_html(StringIO(r.text), flavor="lxml")
        if dfs:
            df = dfs[0]
            # Find the ticker column
            col = next(
                (c for c in df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()),
                df.columns[2] if len(df.columns) > 2 else None
            )
            if col:
                return set(df[col].dropna().astype(str).str.upper().str.strip().tolist())
    except Exception:
        pass
    return set()

# ================================================================
# PERFORMANCE CACHE — raw yfinance history
# FIX: Cache prevents redundant network calls on portfolio reloads
# ================================================================
@st.cache_data(ttl=300, show_spinner=False)
def _cached_history(symbol: str, period: str) -> pd.DataFrame:
    try:
        return yf.Ticker(symbol).history(period=period)
    except Exception:
        return pd.DataFrame()

# ================================================================
# CORE TECHNICAL ANALYSIS ENGINE
# ================================================================
def fetch_technical_data(symbol: str, period_window: int, calc_type: str):
    """
    Returns: (is_bullish, metrics, fig, current_price, fib_levels)
    Uses cached history for performance.
    """
    try:
        lookback = INTERVAL_MAP[period_window]["history"]
        hist     = _cached_history(symbol, lookback)   # ← cached

        min_bars = max(period_window, 26, 20) + 5
        if hist.empty or len(hist) < min_bars:
            return False, {}, None, 0.0, {}

        close = hist["Close"]

        # Moving Average
        if "Simple" in calc_type:
            hist["MA"] = close.rolling(period_window).mean()
            ma_label   = f"SMA-{period_window}"
        else:
            weights    = np.arange(1, period_window + 1)
            hist["MA"] = close.rolling(period_window).apply(
                lambda p: np.dot(p, weights) / weights.sum(), raw=True
            )
            ma_label   = f"WMA-{period_window}"

        # Bollinger Bands
        bb_mid           = close.rolling(20).mean()
        bb_std           = close.rolling(20).std()
        hist["BB_Upper"] = bb_mid + 2 * bb_std
        hist["BB_Mid"]   = bb_mid
        hist["BB_Lower"] = bb_mid - 2 * bb_std

        # RSI
        delta       = close.diff()
        gain        = delta.clip(lower=0).rolling(14).mean()
        loss        = (-delta.clip(upper=0)).rolling(14).mean()
        hist["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        # MACD
        ema12            = close.ewm(span=12, adjust=False).mean()
        ema26            = close.ewm(span=26, adjust=False).mean()
        hist["MACD"]     = ema12 - ema26
        hist["MACD_Sig"] = hist["MACD"].ewm(span=9, adjust=False).mean()
        hist["MACD_H"]   = hist["MACD"] - hist["MACD_Sig"]

        # Fibonacci retracement
        swing_high = float(close.max())
        swing_low  = float(close.min())
        diff       = swing_high - swing_low
        fib_levels = {
            label: swing_low + ratio * diff
            for label, ratio in FIB_RATIOS.items()
        }

        # MA crossover signals
        prev_close   = close.shift(1)
        prev_ma      = hist["MA"].shift(1)
        hist["Buy"]  = np.where(
            (close > hist["MA"]) & (prev_close <= prev_ma), close, np.nan
        )
        hist["Sell"] = np.where(
            (close < hist["MA"]) & (prev_close >= prev_ma), close, np.nan
        )

        # Snapshot metrics
        cur_price = float(close.iloc[-1])
        cur_ma    = float(hist["MA"].iloc[-1])
        cur_rsi   = float(hist["RSI"].iloc[-1])
        cur_macd  = float(hist["MACD"].iloc[-1])
        cur_sig   = float(hist["MACD_Sig"].iloc[-1])
        cur_bbu   = float(hist["BB_Upper"].iloc[-1])
        cur_bbl   = float(hist["BB_Lower"].iloc[-1])
        prior     = close.iloc[-20] if len(hist) >= 20 else close.iloc[0]
        momentum  = ((cur_price - float(prior)) / float(prior)) * 100
        avg_vol   = float(hist["Volume"].rolling(20).mean().iloc[-1])
        last_vol  = float(hist["Volume"].iloc[-1])
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

        nearest_fib  = min(fib_levels.items(), key=lambda x: abs(x[1] - cur_price))
        fib_distance = ((cur_price - nearest_fib[1]) / nearest_fib[1]) * 100

        # Signal labels
        buys  = hist["Buy"].dropna()
        sells = hist["Sell"].dropna()
        if not buys.empty and (sells.empty or buys.index[-1] > sells.index[-1]):
            ma_signal  = f"🟢 BUY  ({buys.index[-1].strftime('%m/%d')})"
            is_bullish = True
        elif not sells.empty and (buys.empty or sells.index[-1] > buys.index[-1]):
            ma_signal  = f"🔴 SELL ({sells.index[-1].strftime('%m/%d')})"
            is_bullish = False
        else:
            ma_signal  = "⚪ Neutral"
            is_bullish = momentum > 0

        metrics = {
            "Price":            f"${cur_price:.2f}",
            "1-Mo Momentum":    f"{momentum:+.1f}%",
            f"{ma_label}":      f"${cur_ma:.2f}",
            "MA Signal":        ma_signal,
            "RSI (14)":         f"{cur_rsi:.1f} — " + (
                "🔴 Overbought" if cur_rsi > 70 else
                "🟢 Oversold"   if cur_rsi < 30 else "⚪ Neutral"
            ),
            "MACD":             "🟢 Bullish cross" if cur_macd > cur_sig else "🔴 Bearish cross",
            "Bollinger":        (
                "🔴 At upper band" if cur_price >= cur_bbu * 0.99 else
                "🟢 At lower band" if cur_price <= cur_bbl * 1.01 else "⚪ Within bands"
            ),
            "Volume vs 20-avg": f"{vol_ratio:.2f}x — " + (
                "🟢 High" if vol_ratio > 1.5 else
                "🔴 Light" if vol_ratio < 0.5 else "⚪ Average"
            ),
            "Fibonacci Zone":   (
                f"📐 Near {nearest_fib[0]} (${nearest_fib[1]:.2f}) — "
                f"{'above' if fib_distance > 0 else 'below'} by {abs(fib_distance):.1f}%"
            ),
        }

        # ── Build chart ────────────────────────────────────────
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            row_heights=[0.50, 0.17, 0.17, 0.16],
            vertical_spacing=0.025,
            subplot_titles=[
                f"{symbol} — Price, {ma_label}, Bollinger Bands & Fibonacci",
                "Volume",
                "MACD  (12 / 26 / 9)",
                "RSI  (14-period)  |  70 = Overbought · 30 = Oversold",
            ],
        )
        # Bollinger fill
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["BB_Upper"], showlegend=False,
            line=dict(color="rgba(120,120,255,0.35)", width=1)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["BB_Lower"], fill="tonexty",
            fillcolor="rgba(120,120,255,0.07)", showlegend=False,
            line=dict(color="rgba(120,120,255,0.35)", width=1)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["BB_Mid"], showlegend=False,
            line=dict(color="rgba(180,180,255,0.4)", width=1, dash="dot")
        ), row=1, col=1)
        # Fibonacci levels
        for label, level in fib_levels.items():
            fig.add_hline(
                y=level, line_dash="dot", line_color=FIB_COLORS[label], line_width=1.2,
                annotation_text=f" Fib {label}  ${level:.2f}",
                annotation_position="right",
                annotation_font_size=9, annotation_font_color=FIB_COLORS[label],
                row=1, col=1
            )
        # Price + MA + signals
        fig.add_trace(go.Scatter(
            x=hist.index, y=close, name="Price",
            line=dict(color="#4fc3f7", width=2)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["MA"], name=ma_label,
            line=dict(color="#ffa726", width=1.8, dash="dot")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["Buy"], mode="markers", name="BUY",
            marker=dict(color="#4caf50", size=11, symbol="triangle-up",
                        line=dict(color="white", width=1))
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["Sell"], mode="markers", name="SELL",
            marker=dict(color="#f44336", size=11, symbol="triangle-down",
                        line=dict(color="white", width=1))
        ), row=1, col=1)
        # Volume
        vol_colors = [
            "#4caf50" if float(c) >= float(o) else "#f44336"
            for c, o in zip(hist["Close"], hist["Open"])
        ]
        fig.add_trace(go.Bar(
            x=hist.index, y=hist["Volume"],
            marker_color=vol_colors, showlegend=False
        ), row=2, col=1)
        # MACD
        hist_colors = ["#4caf50" if float(v) >= 0 else "#f44336" for v in hist["MACD_H"]]
        fig.add_trace(go.Bar(
            x=hist.index, y=hist["MACD_H"], marker_color=hist_colors, showlegend=False
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["MACD"], name="MACD",
            line=dict(color="#4fc3f7", width=1.5)
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["MACD_Sig"], name="Signal",
            line=dict(color="#ffa726", width=1.5, dash="dot")
        ), row=3, col=1)
        # RSI
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["RSI"], name="RSI",
            line=dict(color="#ce93d8", width=1.8)
        ), row=4, col=1)
        for y_val, color in [(70, "rgba(244,67,54,0.55)"), (30, "rgba(76,175,80,0.55)")]:
            fig.add_hline(
                y=y_val, line_dash="dash", line_color=color,
                line_width=1.2, row=4, col=1
            )
        fig.update_layout(
            template="plotly_dark", hovermode="x unified", height=820,
            margin=dict(l=10, r=80, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                        font=dict(size=11)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,25,1)",
        )
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, gridcolor="#1e1e2e")
        fig.update_yaxes(title_text="Volume",   row=2, col=1, gridcolor="#1e1e2e")
        fig.update_yaxes(title_text="MACD",     row=3, col=1, gridcolor="#1e1e2e")
        fig.update_yaxes(title_text="RSI",      row=4, col=1, range=[0, 100], gridcolor="#1e1e2e")

        return is_bullish, metrics, fig, cur_price, fib_levels

    except Exception:
        return False, {}, None, 0.0, {}

# ================================================================
# AI ANALYSIS — Gemini with full model fallback chain
# ================================================================
def generate_ai_analysis(
    symbol: str, metrics: dict, period: int, method: str,
    fib_levels: dict = None, extra_context: str = ""
) -> str:
    if not AI_AVAILABLE:
        return "⚠️ AI unavailable — GEMINI_API_KEY not found in secrets."

    fib_text = ""
    if fib_levels:
        fib_text = "\nFibonacci Retracement Levels:\n" + "\n".join(
            f"  {lbl}: ${lvl:.2f}" for lbl, lvl in fib_levels.items()
        )

    prompt = f"""
You are an elite institutional equity and crypto analyst. Analyse {symbol} for a client 
who may be a beginner — be clear and educational while maintaining professional depth.
Define jargon the first time you use it.

Live Market Data:
{json.dumps(metrics, indent=2)}
{fib_text}
Framework: {period}-day {method}
{f"Context: {extra_context}" if extra_context else ""}

Produce a structured report with EXACTLY these five sections:

## 📋 Quantitative Tear Sheet
Markdown table: | Metric | Value | Plain-English Meaning |
Include all metrics and Fibonacci levels above.

## 🌊 Elliott Wave & Trend Structure
- Identify most likely current wave position (explain each wave term)
- How price relates to each Fibonacci level shown
- Which Fib level is acting as current support or resistance, and why
- Specific price targets for next wave up AND the level that invalidates the bullish case

## 🔀 Multi-Indicator Confluence
- Where RSI, MACD, Bollinger Bands, Volume, and any insider data agree
- Where they conflict — which signal takes priority and why
- Signal strength rating: Strong / Moderate / Weak with justification
- One sentence a complete beginner can immediately act on

## ⚠️ Risk Assessment
- Primary bull thesis risk
- Primary bear thesis risk
- Price level that invalidates the setup
- Suggested stop-loss zone (as a specific price range, not a concept)

## 🎯 Portfolio Strategy Suggestion
**Bold and specific:** action, entry zone, target zone, stop-loss, position size tier 
(aggressive/moderate/conservative). End with one sentence summarising the risk/reward.
"""

    errors = []
    for model in GEMINI_MODEL_CANDIDATES:
        try:
            response = gemini_client.models.generate_content(model=model, contents=prompt)
            return f"*Model: `{model}`*\n\n" + response.text
        except Exception as e:
            errors.append(f"**{model}:** {str(e)[:120]}")
            continue

    return (
        "### ⚠️ All Gemini models failed\n\n"
        "**Most likely cause: your API key is invalid or not being read from Secrets.**\n\n"
        "**How to verify your key:**\n"
        "1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)\n"
        "2. Copy the key exactly — no quotes, no extra spaces\n"
        "3. In Streamlit Community Cloud → your app → **Settings → Secrets**\n"
        "4. The entry must be exactly: `GEMINI_API_KEY = \"your-key-here\"`\n"
        "5. Save and wait ~30 seconds for the app to reload\n\n"
        "**Errors tried:**\n" + "\n".join(f"- {e}" for e in errors)
    )

# ================================================================
# BATCH SCANNER — threaded
# ================================================================
def scan_tickers(ticker_list: list, period: int, calc_type: str, max_workers: int = 15):
    results, figs = [], {}
    progress      = st.progress(0.0, text="Preparing scan…")
    total, done   = len(ticker_list), 0

    def _scan_one(sym):
        return sym, *fetch_technical_data(sym, period, calc_type)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_scan_one, t): t for t in ticker_list}
        for future in as_completed(futures):
            done += 1
            progress.progress(done / total, text=f"Scanning … {done}/{total}")
            sym, bullish, metrics, fig, price, fib = future.result()
            if bullish and metrics:
                results.append({"Ticker": sym, **metrics})
                if fig is not None:
                    figs[sym] = (fig, fib)

    progress.empty()
    return results, figs

# ================================================================
# DISPLAY HELPERS
# ================================================================
def display_metrics_grid(metrics: dict):
    items = list(metrics.items())
    for i in range(0, len(items), 4):
        chunk = items[i:i+4]
        cols  = st.columns(len(chunk))
        for col, (k, v) in zip(cols, chunk):
            col.metric(k, v)

def show_ai_report(
    report_key: str, symbol: str, metrics: dict,
    period: int, method: str, fib_levels: dict,
    extra_context: str = "",            # FIX: consistent param name throughout
    button_label: str = "🤖  Generate AI Analyst Report"
):
    """
    FIX: Store AI output in session_state so it survives Streamlit reruns.
    Report persists until user explicitly clears it.
    """
    if st.button(button_label, key=f"btn_{report_key}"):
        with st.spinner("Gemini is analysing — may take 20–40 seconds for Pro…"):
            report = generate_ai_analysis(
                symbol, metrics, period, method, fib_levels, extra_context
            )
            st.session_state[f"rpt_{report_key}"] = report

    stored = st.session_state.get(f"rpt_{report_key}")
    if stored:
        st.markdown("---")
        st.markdown(stored)
        if st.button("🗑️  Clear Report", key=f"clr_{report_key}"):
            del st.session_state[f"rpt_{report_key}"]
            st.rerun()

def render_insider_section(symbol: str):
    """Display insider transaction table and summary for a given ticker."""
    with st.expander("👔  Insider & Executive Transactions (last 12 months)", expanded=False):
        with st.spinner("Loading insider data…"):
            df = get_insider_transactions(symbol)
        if df is None:
            st.info(
                "No recent insider transaction data found for this ticker. "
                "This is common for ETFs, indices, and very small companies."
            )
            return
        st.caption(
            "Data sourced from SEC Form 4 filings via Yahoo Finance. "
            "Purchases made with personal funds by insiders are the strongest vote of confidence."
        )
        # Colour-code the transaction column if present
        tx_col = next(
            (c for c in df.columns if "transaction" in c.lower() or "text" in c.lower()),
            None
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        if tx_col:
            buys  = df[df[tx_col].astype(str).str.contains("Buy|Purchase|Acqui", case=False, na=False)]
            sells = df[df[tx_col].astype(str).str.contains("Sale|Sell|Dispos", case=False, na=False)]
            c1, c2 = st.columns(2)
            c1.metric("Insider Purchases (12mo)", len(buys),
                      delta="Bullish" if len(buys) > 0 else None)
            c2.metric("Insider Sales (12mo)", len(sells),
                      delta="Caution" if len(sells) > len(buys) else None,
                      delta_color="inverse" if len(sells) > 0 else "normal")

# ================================================================
# PORTFOLIO DIVERSITY CHART
# Adaptive: donut for ≤8 positions, treemap for 9+
# ================================================================
def render_diversity_chart(rows: list, total_value: float):
    if not rows or total_value == 0:
        return

    labels, values = [], []
    for row in rows:
        val = float(
            row.get("Mkt Value", "$0").replace("$", "").replace(",", "")
        )
        if val > 0:
            labels.append(row["Asset"])
            values.append(val)

    if not values:
        return

    n = len(values)
    st.subheader("📊  Portfolio Diversity")

    if n <= 8:
        # Donut chart — clean for small portfolios
        pct_labels = [f"{l}<br>{v/total_value*100:.1f}%" for l, v in zip(labels, values)]
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            text=pct_labels,
            textinfo="label+percent" if n <= 4 else "percent",
            hovertemplate="<b>%{label}</b><br>Value: $%{value:,.2f}<br>Share: %{percent}<extra></extra>",
            marker=dict(line=dict(color="#0f0f19", width=2)),
        )])
        fig.update_layout(
            template="plotly_dark",
            showlegend=(n > 4),
            height=380,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                text=f"<b>${total_value:,.0f}</b>",
                x=0.5, y=0.5, font_size=15, showarrow=False,
                font=dict(color="#4fc3f7")
            )],
        )
    else:
        # Treemap — handles many positions cleanly
        df_chart = pd.DataFrame({"Asset": labels, "Value": values})
        df_chart["Pct"] = df_chart["Value"] / total_value * 100
        fig = go.Figure(go.Treemap(
            labels=df_chart["Asset"],
            parents=[""] * len(df_chart),
            values=df_chart["Value"],
            texttemplate="<b>%{label}</b><br>%{customdata:.1f}%",
            customdata=df_chart["Pct"],
            hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{customdata:.1f}%<extra></extra>",
            marker=dict(
                colorscale="Blues",
                line=dict(width=2, color="#0f0f19"),
            ),
        ))
        fig.update_layout(
            template="plotly_dark",
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )

    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# MAIN APP
# ================================================================
st.title("📊  Wall Street AI Dashboard")
st.caption(
    "Institutional-grade analysis · Gemini 2.5 Pro · "
    "Elliott Wave · Fibonacci · Insider Activity · Real-time data"
)

render_indicator_guide()
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Analysis Settings")
    ma_type = st.radio(
        "Moving Average Type",
        ["Simple Moving Average (SMA)", "Weighted Moving Average (WMA)"],
        help="WMA reacts faster. SMA is smoother."
    )
    sma_period = st.selectbox(
        "Lookback Period",
        options=list(INTERVAL_MAP.keys()), index=1,
        format_func=lambda x: INTERVAL_MAP[x]["label"],
    )
    st.divider()

    # API key status
    if not AI_AVAILABLE:
        st.error("⚠️ GEMINI_API_KEY missing — AI disabled.")
    elif len(GEMINI_API_KEY) < 20:
        st.error("⚠️ GEMINI_API_KEY looks too short — check Secrets.")
    else:
        masked = GEMINI_API_KEY[:6] + "•" * 8 + GEMINI_API_KEY[-4:]
        st.success(f"✅ Gemini key loaded: `{masked}`")

    if SUPABASE_AVAILABLE:
        st.success("✅ Cloud portfolio connected")
    else:
        st.warning("⚠️ Supabase not set — portfolios won't persist")
    st.divider()
    st.caption("🔒 Shared institutional API — no personal key needed.")

# ── Mode selector ────────────────────────────────────────────────
mode = st.radio(
    "Mode", ["💼  My Portfolio", "🔍  Analyze Single Asset", "🌐  Market Scanner"],
    horizontal=True, label_visibility="collapsed"
)
st.markdown("<br>", unsafe_allow_html=True)

# ================================================================
# MODE 1 — PORTFOLIO DASHBOARD
# ================================================================
if mode == "💼  My Portfolio":
    st.header("💼  Portfolio Dashboard")

    if not SUPABASE_AVAILABLE:
        for k, v in [("user_portfolio", {}), ("auth_user", "local"), ("auth_pin", "")]:
            if k not in st.session_state:
                st.session_state[k] = v
        st.info("☁️ Cloud storage not configured. Portfolio resets on page refresh.")

    # Login
    if SUPABASE_AVAILABLE and "auth_user" not in st.session_state:
        st.subheader("🔐  Access Your Portfolio")
        st.caption("First time? Choose any username and 4-digit PIN to create your account.")
        with st.form("login_form"):
            uid       = st.text_input("Username / Investor ID", placeholder="e.g. john_trader")
            pin       = st.text_input("4-Digit PIN", type="password", max_chars=4)
            submitted = st.form_submit_button("Access Portfolio →", use_container_width=True)
        if submitted:
            uid = uid.strip().lower()
            if not uid or not pin:
                st.error("Enter both a username and PIN.")
            elif len(pin) != 4 or not pin.isdigit():
                st.error("PIN must be exactly 4 digits.")
            else:
                with st.spinner("Authenticating…"):
                    portfolio = load_portfolio_from_db(uid, pin)
                if portfolio is None:
                    st.error("❌ Incorrect PIN for this username.")
                else:
                    st.session_state.update({
                        "auth_user": uid, "auth_pin": pin, "user_portfolio": portfolio
                    })
                    st.rerun()
        st.stop()

    # ── Sidebar Position Manager ──────────────────────────────────
    with st.sidebar:
        st.subheader("🛠️  Position Manager")
        if SUPABASE_AVAILABLE:
            st.caption(f"Logged in as: **{st.session_state.get('auth_user','—')}**")

        with st.form("position_form"):
            # FIX: Stock/Crypto toggle prevents ETH vs ETH-USD confusion
            asset_type = st.radio(
                "Asset Type", ["📈 Stock", "₿ Crypto"], horizontal=True,
                help="Crypto auto-appends -USD (e.g. ETH → ETH-USD)"
            )
            raw_ticker = st.text_input(
                "Ticker Symbol",
                placeholder="Stock: AAPL  |  Crypto: ETH or ETH-USD"
            ).upper().strip()
            # FIX: 8 decimal places for small crypto positions (e.g. 0.002 BTC)
            new_shares = st.number_input(
                "Shares / Units Owned", min_value=0.0,
                step=0.00000001, format="%.8f",
                help="Supports up to 8 decimal places for crypto (e.g. 0.00250000 BTC)"
            )
            new_cost = st.number_input(
                "Average Purchase Price ($)", min_value=0.0, step=0.01
            )
            save_btn = st.form_submit_button("💾  Save Position", use_container_width=True)

        if save_btn and raw_ticker:
            new_ticker = (
                raw_ticker if raw_ticker.endswith("-USD") else raw_ticker + "-USD"
            ) if "Crypto" in asset_type else raw_ticker

            uid = st.session_state.get("auth_user", "local")
            pin = st.session_state.get("auth_pin", "")

            if new_shares == 0:
                st.session_state["user_portfolio"].pop(new_ticker, None)
                if SUPABASE_AVAILABLE:
                    save_position_to_db(uid, pin, new_ticker, 0, 0)
                st.warning(f"Removed {new_ticker}")
            else:
                st.session_state["user_portfolio"][new_ticker] = {
                    "shares": new_shares, "cost": new_cost
                }
                if SUPABASE_AVAILABLE:
                    save_position_to_db(uid, pin, new_ticker, new_shares, new_cost)
                st.success(f"✅ {new_ticker} saved")

        st.divider()
        if st.session_state.get("user_portfolio"):
            st.download_button(
                "📥  Download Backup",
                data=json.dumps(st.session_state["user_portfolio"]),
                file_name="portfolio_backup.json", mime="application/json",
                use_container_width=True
            )
        uploaded = st.file_uploader("📤  Restore from Backup", type="json")
        if uploaded:
            try:
                st.session_state["user_portfolio"] = json.load(uploaded)
                st.success("Portfolio restored!")
            except Exception:
                st.error("Invalid backup file.")

        if SUPABASE_AVAILABLE:
            st.divider()
            if st.button("🚪  Log Out", use_container_width=True):
                for k in ["auth_user", "auth_pin", "user_portfolio"]:
                    st.session_state.pop(k, None)
                st.rerun()

    # ── Portfolio Display ─────────────────────────────────────────
    portfolio = st.session_state.get("user_portfolio", {})

    if not portfolio:
        st.info(
            "Portfolio is empty. Use the **Position Manager** in the sidebar. "
            "Select ₿ Crypto before entering ETH, BTC, XRP etc."
        )
    else:
        total_value = total_cost = 0.0
        rows, charts = [], {}

        with st.spinner("Fetching live data — using 5-min cache for speed…"):
            for sym, pos in list(portfolio.items()):
                _, metrics, fig, price, fib = fetch_technical_data(sym, sma_period, ma_type)
                if price > 0:
                    pos_cost  = pos["shares"] * pos["cost"]
                    pos_value = pos["shares"] * price
                    pos_gain  = pos_value - pos_cost
                    pos_pct   = (pos_gain / pos_cost * 100) if pos_cost > 0 else 0.0
                    total_value += pos_value
                    total_cost  += pos_cost
                    if fig:
                        charts[sym] = (fig, fib, metrics)
                    rows.append({
                        "Asset":         sym,
                        "Shares":        f"{pos['shares']:.8f}".rstrip("0").rstrip("."),
                        "Avg Cost":      f"${pos['cost']:.2f}",
                        "Current Price": f"${price:.2f}",
                        "Mkt Value":     f"${pos_value:,.2f}",
                        "Return ($)":    f"${pos_gain:+,.2f}",
                        "Return (%)":    f"{pos_pct:+.1f}%",
                        "MA Signal":     metrics.get("MA Signal", "—"),
                        "RSI":           metrics.get("RSI (14)", "—"),
                        "Fibonacci":     metrics.get("Fibonacci Zone", "—"),
                    })

        total_gain = total_value - total_cost
        total_pct  = (total_gain / total_cost * 100) if total_cost > 0 else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Portfolio Value", f"${total_value:,.2f}")
        k2.metric("Total Cost Basis",      f"${total_cost:,.2f}")
        k3.metric("Total Return",          f"${total_gain:+,.2f}", f"{total_pct:+.2f}%")
        k4.metric("Open Positions",        str(len(rows)))

        st.subheader("Holdings Summary")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Adaptive diversity chart
        render_diversity_chart(rows, total_value)

        st.subheader("📈  Deep-Dive Chart & AI Analysis")
        chosen = st.selectbox("Select a holding", list(portfolio.keys()))

        if chosen in charts:
            fig, fib, chosen_metrics = charts[chosen]
            st.plotly_chart(fig, use_container_width=True)

            # Insider activity for chosen stock
            if not chosen.endswith("-USD"):   # skip for crypto
                render_insider_section(chosen)

            pos_detail = portfolio.get(chosen, {})
            ins_df     = get_insider_transactions(chosen) if not chosen.endswith("-USD") else None
            context    = (
                f"Held position: {pos_detail.get('shares',0):.8f} units at "
                f"${pos_detail.get('cost',0):.2f} avg cost. "
                f"Current P&L: {next((r['Return (%)'] for r in rows if r['Asset']==chosen),'unknown')}. "
                f"{insider_summary(ins_df)}"
            )
            show_ai_report(
                f"portfolio_{chosen}", chosen, chosen_metrics,
                sma_period, ma_type, fib, extra_context=context
            )
        else:
            st.warning("Chart unavailable for this asset.")

# ================================================================
# MODE 2 — SINGLE ASSET ANALYSIS
# ================================================================
elif mode == "🔍  Analyze Single Asset":
    st.header("🔍  Single Asset Analysis")
    st.caption(
        "Any US stock or crypto. **Crypto format:** BTC-USD, ETH-USD, XRP-USD, SOL-USD"
    )

    c1, c2 = st.columns([4, 1])
    with c1:
        symbol_input = st.text_input(
            "Ticker", label_visibility="collapsed",
            placeholder="Stock: NVDA, AAPL, TSLA   |   Crypto: BTC-USD, ETH-USD, XRP-USD"
        ).upper().strip()
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        go_btn = st.button("Analyse →", use_container_width=True)

    if go_btn and symbol_input:
        with st.spinner(f"Fetching {symbol_input}…"):
            _, metrics, fig, price, fib = fetch_technical_data(
                symbol_input, sma_period, ma_type
            )

        if fig is None or price == 0.0:
            st.error(
                f"No data for **{symbol_input}**. Check the ticker. "
                "Crypto must use the format `BTC-USD`, `ETH-USD`, etc."
            )
        else:
            display_metrics_grid(metrics)
            st.plotly_chart(fig, use_container_width=True)

            # Insider transactions (stocks only)
            if not symbol_input.endswith("-USD"):
                render_insider_section(symbol_input)

            # AI report with insider context
            ins_df  = get_insider_transactions(symbol_input) if not symbol_input.endswith("-USD") else None
            context = insider_summary(ins_df)

            show_ai_report(
                f"single_{symbol_input}", symbol_input, metrics,
                sma_period, ma_type, fib,
                extra_context=context,
                button_label="🤖  Generate Full AI Report"
            )

# ================================================================
# MODE 3 — MARKET SCANNER
# ================================================================
elif mode == "🌐  Market Scanner":
    st.header("🌐  Market Scanner")
    st.caption(
        "Scans the selected universe for bullish MA crossover + positive 1-month momentum. "
        "Fibonacci zones and insider filter available."
    )

    tab_stocks, tab_crypto = st.tabs(["📈  Stocks", "₿  Crypto"])

    with tab_stocks:
        col_uni, col_ins, col_btn = st.columns([3, 2, 1])
        with col_uni:
            universe = st.selectbox(
                "Market Universe",
                [
                    "S&P 500  (~500 stocks, ~2–4 min)",
                    "Russell 2000  (~2,000 stocks, ~10–20 min)",
                    "All US Equities  (~1,500–3,000 stocks, ~15–30 min)",
                ]
            )
        with col_ins:
            st.markdown("<br>", unsafe_allow_html=True)
            insider_filter = st.checkbox(
                "🔍 Filter: Insider cluster buys only",
                help=(
                    "Pre-filters to stocks where 3+ insiders bought recently "
                    "(sourced from OpenInsider.com). Dramatically reduces scan size "
                    "and surfaces the highest-conviction setups."
                )
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            scan_btn = st.button("🚀  Launch Scan", use_container_width=True)

        if "Russell" in universe:
            st.warning("⚠️ Large scan — expect 10–20 min.")
        elif "All US" in universe:
            st.warning("⚠️ Very large scan — expect 15–30 min.")
        else:
            st.info("ℹ️ ~500 stocks · Expected: 2–4 minutes.")

        if scan_btn:
            with st.spinner("Loading ticker universe…"):
                if "S&P 500" in universe:
                    tickers = get_sp500_tickers()
                elif "Russell" in universe:
                    tickers = get_russell2000_tickers()
                else:
                    tickers = get_all_us_equities()

            if not tickers:
                st.error("Failed to load ticker list. Check internet connection.")
            else:
                # Apply insider filter if requested
                if insider_filter:
                    with st.spinner("Loading OpenInsider cluster buy data…"):
                        insider_tickers = get_openinsider_cluster_buys()
                    if insider_tickers:
                        before = len(tickers)
                        tickers = [t for t in tickers if t in insider_tickers]
                        st.info(
                            f"Insider filter: reduced from {before} → "
                            f"**{len(tickers)}** tickers with cluster buying activity"
                        )
                    else:
                        st.warning(
                            "OpenInsider data unavailable — running full scan without filter."
                        )

                if not tickers:
                    st.warning("No tickers remain after insider filter. Try without the filter.")
                else:
                    st.info(f"Scanning **{len(tickers)}** tickers…")
                    results, figs = scan_tickers(tickers, sma_period, ma_type)

                    if results:
                        st.success(f"✅ **{len(results)}** buy signals found")
                        st.dataframe(
                            pd.DataFrame(results), use_container_width=True, hide_index=True
                        )
                        st.subheader("📊  Chart Deep-Dive")
                        triggered = [r["Ticker"] for r in results]
                        view_sym  = st.selectbox("Select a stock", triggered)

                        if view_sym in figs:
                            fig, fib = figs[view_sym]
                            st.plotly_chart(fig, use_container_width=True)
                            render_insider_section(view_sym)
                            stock_metrics = next(
                                (r for r in results if r["Ticker"] == view_sym), {}
                            )
                            ins_df  = get_insider_transactions(view_sym)
                            context = insider_summary(ins_df)
                            show_ai_report(
                                f"scanner_{view_sym}", view_sym, stock_metrics,
                                sma_period, ma_type, fib,
                                extra_context=context,
                                button_label="🤖  AI Analysis for this stock"
                            )
                    else:
                        st.warning(
                            "No stocks meet the criteria. Try a different MA period in the sidebar."
                        )

    with tab_crypto:
        st.subheader("₿  Major Crypto Dashboard")
        st.caption("Bitcoin · Ethereum · XRP · Solana — all shown regardless of signal")
        crypto_btn = st.button("📡  Refresh Crypto Data")

        if crypto_btn or "crypto_data" not in st.session_state:
            crypto_rows, crypto_figs = [], {}
            with st.spinner("Fetching crypto data…"):
                for name, sym in CRYPTO_TICKERS.items():
                    _, metrics, fig, _, fib = fetch_technical_data(sym, sma_period, ma_type)
                    if metrics:
                        crypto_rows.append({"Asset": f"{name} ({sym})", **metrics})
                    if fig:
                        crypto_figs[sym] = (fig, fib)
            st.session_state["crypto_data"] = (crypto_rows, crypto_figs)

        crypto_rows, crypto_figs = st.session_state.get("crypto_data", ([], {}))

        if crypto_rows:
            st.dataframe(pd.DataFrame(crypto_rows), use_container_width=True, hide_index=True)
            chosen_crypto = st.selectbox(
                "Select crypto for chart",
                list(CRYPTO_TICKERS.values()),
                format_func=lambda s: next(
                    (k for k, v in CRYPTO_TICKERS.items() if v == s), s
                )
            )
            if chosen_crypto in crypto_figs:
                fig, fib = crypto_figs[chosen_crypto]
                st.plotly_chart(fig, use_container_width=True)
                crypto_metrics = next(
                    (r for r in crypto_rows if chosen_crypto in r.get("Asset", "")), {}
                )
                # FIX: use extra_context= keyword consistently
                show_ai_report(
                    f"crypto_{chosen_crypto}", chosen_crypto, crypto_metrics,
                    sma_period, ma_type, fib,
                    extra_context=(
                        "This is a cryptocurrency. Factor in 24/7 trading, "
                        "higher volatility norms, and absence of traditional fundamentals "
                        "like P/E ratios. No insider transaction data applies to crypto."
                    ),
                    button_label="🤖  Generate Crypto AI Analysis"
                )
        else:
            st.info("Click 'Refresh Crypto Data' to load.")
