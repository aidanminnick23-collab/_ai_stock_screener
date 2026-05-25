# ================================================================
# WALL STREET AI DASHBOARD — Production Build v2.1
# Fixes: lxml, Gemini model fallback, crypto ticker, EW/Fibonacci
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
</style>
""", unsafe_allow_html=True)

# ================================================================
# SECRETS
# ================================================================
def _get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return ""

GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")
SUPABASE_URL   = _get_secret("SUPABASE_URL")
SUPABASE_KEY   = _get_secret("SUPABASE_KEY")

# ── Gemini client ────────────────────────────────────────────────
# FIX: Try multiple model strings in order until one succeeds.
# Google periodically changes preview model IDs — this makes the
# app resilient to those changes automatically.
GEMINI_MODEL_CANDIDATES = [
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.5-flash",          # reliable fallback
]

AI_AVAILABLE   = bool(GEMINI_API_KEY)
gemini_client  = None
ACTIVE_MODEL   = None            # resolved at first AI call

if AI_AVAILABLE:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        AI_AVAILABLE = False

def _resolve_model() -> str:
    """Try each candidate model and return the first that responds."""
    global ACTIVE_MODEL
    if ACTIVE_MODEL:
        return ACTIVE_MODEL
    for m in GEMINI_MODEL_CANDIDATES:
        try:
            gemini_client.models.generate_content(
                model=m, contents="ping"
            )
            ACTIVE_MODEL = m
            return m
        except Exception:
            continue
    return GEMINI_MODEL_CANDIDATES[-1]   # last-resort flash

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
    20:  {"history": "6mo", "label": "20-Day  · Short-Term Trend Boundary"},
    50:  {"history": "1y",  "label": "50-Day  · Institutional Baseline"},
    100: {"history": "2y",  "label": "100-Day · Macro Cycle Support"},
    200: {"history": "2y",  "label": "200-Day · Ultimate Bull/Bear Line"},
}

# Fibonacci retracement ratios used in Elliott Wave analysis
FIB_RATIOS = {
    "78.6%": 0.786,
    "61.8%": 0.618,
    "50.0%": 0.500,
    "38.2%": 0.382,
    "23.6%": 0.236,
}
FIB_COLORS = {
    "78.6%": "rgba(255,82,82,0.55)",
    "61.8%": "rgba(255,167,38,0.65)",
    "50.0%": "rgba(255,238,88,0.60)",
    "38.2%": "rgba(102,187,106,0.65)",
    "23.6%": "rgba(79,195,247,0.55)",
}

# ================================================================
# INDICATOR & ELLIOTT WAVE EDUCATION PANEL
# ================================================================
def render_indicator_guide():
    with st.expander(
        "📚  Indicator Guide — What does everything mean?  (click to expand)",
        expanded=False
    ):
        st.markdown("#### Learn the indicators used in every chart and analysis report")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("---")
            st.markdown("### 📈  Moving Average (SMA / WMA)")
            st.info(
                "**What it is:** The average closing price over N days. "
                "SMA weights every day equally; WMA gives more weight to recent days.\n\n"
                "**Think of it as:** The stock's center of gravity. Prices drift back toward "
                "it over time. When price punches above, buyers are in control.\n\n"
                "🔺 Green triangle = price just crossed above → Buy signal\n"
                "🔻 Red triangle   = price just crossed below → Sell signal"
            )

            st.markdown("---")
            st.markdown("### 📉  RSI — Relative Strength Index")
            st.info(
                "**What it is:** A 0–100 speed gauge measuring how aggressively price "
                "has moved over the last 14 days.\n\n"
                "**Think of it as:** A runner's fatigue meter. Sprinting too long (>70) "
                "leads to a slowdown. Crawling exhausted (<30) often precedes a burst.\n\n"
                "🔴 Above 70 = Overbought — pullback risk\n"
                "🟢 Below 30 = Oversold   — bounce candidate\n"
                "⚪ 30–70    = Neutral territory"
            )

            st.markdown("---")
            st.markdown("### 📊  Volume")
            st.info(
                "**What it is:** Total shares traded in a session. Validates whether "
                "a price move has real conviction.\n\n"
                "**Think of it as:** Votes. A rise on heavy volume = thousands agree. "
                "A rise on thin volume = easy to reverse.\n\n"
                "🟢 Rising price + High volume = Strong confirmed move\n"
                "🔴 Rising price + Low volume  = Weak, suspect move"
            )

        with c2:
            st.markdown("---")
            st.markdown("### ⚡  MACD")
            st.info(
                "**What it is:** Compares a fast (12-day) EMA to a slow (26-day) EMA. "
                "The gap between them is the MACD line, plotted against a 9-day signal line.\n\n"
                "**Think of it as:** Two pace cars. When the fast car pulls ahead of the "
                "slow one (MACD crosses above Signal), buying acceleration is building.\n\n"
                "🟢 MACD above Signal  = Bullish momentum\n"
                "🔴 MACD below Signal  = Bearish momentum\n"
                "📊 Histogram growing  = Momentum strengthening"
            )

            st.markdown("---")
            st.markdown("### 🎯  Bollinger Bands")
            st.info(
                "**What it is:** Three lines — a 20-day MA centre flanked by upper/lower "
                "bands 2 standard deviations away. Bands widen on high volatility, "
                "narrow on calm markets.\n\n"
                "**Think of it as:** Highway lane markers. Price drifts to the edge then "
                "snaps back. Narrow bands = big move loading.\n\n"
                "🔴 At upper band = Potentially overbought\n"
                "🟢 At lower band = Potentially oversold\n"
                "⚠️  Squeeze        = Breakout incoming"
            )

            st.markdown("---")
            st.markdown("### 🔀  Reading Indicators Together")
            st.info(
                "**Strongest buy setup (confluence):**\n"
                "- Price crosses above MA 🔺\n"
                "- RSI between 40–60 (room to run)\n"
                "- MACD just crossed above Signal\n"
                "- High volume on the breakout day\n"
                "- Price bouncing off lower BB or key Fibonacci level\n\n"
                "The more signals agree, the higher the conviction. "
                "The AI report synthesises all of this automatically."
            )

        with c3:
            st.markdown("---")
            st.markdown("### 🌊  Elliott Wave Theory")
            st.info(
                "**What it is:** A framework that says markets move in predictable "
                "wave patterns driven by crowd psychology — repeating cycles of "
                "optimism and pessimism.\n\n"
                "**The basic pattern:**\n"
                "- **Wave 1** — First move up; few notice\n"
                "- **Wave 2** — Pullback; 'Was that it?'\n"
                "- **Wave 3** — Strongest surge; news turns bullish\n"
                "- **Wave 4** — Mild dip; bulls buy the dip\n"
                "- **Wave 5** — Final push; euphoria peaks\n"
                "- **Wave A–C** — Corrective phase follows\n\n"
                "The AI analysis identifies the likely current wave position."
            )

            st.markdown("---")
            st.markdown("### 📐  Fibonacci Retracement Levels")
            st.info(
                "**What it is:** Key price zones derived from the Fibonacci sequence "
                "(the same ratio found in nature, art, and architecture). Markets tend "
                "to reverse or pause at these exact levels — traders actively watch them.\n\n"
                "**The key levels on every chart:**\n"
                "🔵 **23.6%** — Shallow pullback, strong trend\n"
                "🟢 **38.2%** — Common retracement, healthy dip\n"
                "🟡 **50.0%** — Psychological midpoint, watch closely\n"
                "🟠 **61.8%** — The 'Golden Ratio', strongest support\n"
                "🔴 **78.6%** — Deep retracement, trend may be weakening\n\n"
                "**How to use:** After a big move up, price often pulls back to one "
                "of these levels before continuing. A bounce off 61.8% in an uptrend "
                "is one of the highest-probability trade setups in all of technical analysis."
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
                "user_id":  user_id,
                "pin_hash": hash_pin(pin),
                "ticker":   ticker,
                "shares":   shares,
                "cost":     cost,
            }, on_conflict="user_id,ticker").execute()
        return True
    except Exception as e:
        st.error(f"Save error: {e}")
        return False

# ================================================================
# TICKER UNIVERSE LOADERS  (lxml required — now in requirements.txt)
# ================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers() -> list:
    try:
        table = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            flavor="lxml"
        )[0]
        return sorted(table["Symbol"].str.replace(".", "-", regex=False).tolist())
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
            df = pd.read_html(url, flavor="lxml")[0]
            tickers.update(df[col].str.replace(".", "-", regex=False).tolist())
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
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r  = requests.get(url, headers=headers, timeout=20)
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), skiprows=9)
        df = df[df.get("Asset Class", df.columns[0]) == "Equity"]
        tickers = df["Ticker"].dropna().str.strip().tolist()
        valid   = sorted([t for t in tickers if t and t != "-" and len(t) <= 6])
        if len(valid) > 500:
            return valid
    except Exception:
        pass
    try:
        df = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", flavor="lxml"
        )[0]
        st.caption("ℹ️ Russell 2000 live feed unavailable — using S&P SmallCap 600 as proxy.")
        return sorted(df["Ticker"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        st.warning("Russell 2000 data unavailable. Try S&P 500.")
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_us_equities() -> list:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept":     "application/json, text/plain, */*",
        }
        url = (
            "https://api.nasdaq.com/api/screener/stocks"
            "?tableonly=true&limit=5000&offset=0&download=true"
        )
        r    = requests.get(url, headers=headers, timeout=25)
        data = r.json()
        df   = pd.DataFrame(data["data"]["rows"])
        df["volume"] = pd.to_numeric(
            df["volume"].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
        df = df[df["volume"] > 500_000]
        tickers = sorted(df["symbol"].str.strip().tolist())
        if len(tickers) > 100:
            return tickers
    except Exception:
        pass
    st.caption("ℹ️ Full equity list unavailable — falling back to S&P 1500.")
    return get_sp1500_tickers()

# ================================================================
# CORE TECHNICAL ANALYSIS ENGINE  (+ Fibonacci retracement)
# ================================================================
def fetch_technical_data(symbol: str, period_window: int, calc_type: str):
    """
    Returns (is_bullish, metrics, plotly_fig, current_price, fib_levels)
    """
    try:
        ticker   = yf.Ticker(symbol)
        lookback = INTERVAL_MAP[period_window]["history"]
        hist     = ticker.history(period=lookback)

        min_bars = max(period_window, 26, 20) + 5
        if hist.empty or len(hist) < min_bars:
            return False, {}, None, 0.0, {}

        close = hist["Close"]

        # ── Moving Average ─────────────────────────────────────
        if "Simple" in calc_type:
            hist["MA"] = close.rolling(period_window).mean()
            ma_label   = f"SMA-{period_window}"
        else:
            weights    = np.arange(1, period_window + 1)
            hist["MA"] = close.rolling(period_window).apply(
                lambda p: np.dot(p, weights) / weights.sum(), raw=True
            )
            ma_label   = f"WMA-{period_window}"

        # ── Bollinger Bands (20-day ±2σ) ──────────────────────
        bb_mid           = close.rolling(20).mean()
        bb_std           = close.rolling(20).std()
        hist["BB_Upper"] = bb_mid + 2 * bb_std
        hist["BB_Mid"]   = bb_mid
        hist["BB_Lower"] = bb_mid - 2 * bb_std

        # ── RSI (14-period) ────────────────────────────────────
        delta       = close.diff()
        gain        = delta.clip(lower=0).rolling(14).mean()
        loss        = (-delta.clip(upper=0)).rolling(14).mean()
        rs          = gain / loss.replace(0, np.nan)
        hist["RSI"] = 100 - (100 / (1 + rs))

        # ── MACD (12, 26, 9) ───────────────────────────────────
        ema12            = close.ewm(span=12, adjust=False).mean()
        ema26            = close.ewm(span=26, adjust=False).mean()
        hist["MACD"]     = ema12 - ema26
        hist["MACD_Sig"] = hist["MACD"].ewm(span=9, adjust=False).mean()
        hist["MACD_H"]   = hist["MACD"] - hist["MACD_Sig"]

        # ── Fibonacci Retracement (Elliott Wave anchor) ────────
        swing_high = float(close.max())
        swing_low  = float(close.min())
        diff       = swing_high - swing_low
        fib_levels = {
            label: swing_low + ratio * diff
            for label, ratio in FIB_RATIOS.items()
        }

        # ── MA Crossover Signals ───────────────────────────────
        prev_close   = close.shift(1)
        prev_ma      = hist["MA"].shift(1)
        hist["Buy"]  = np.where(
            (close > hist["MA"]) & (prev_close <= prev_ma), close, np.nan
        )
        hist["Sell"] = np.where(
            (close < hist["MA"]) & (prev_close >= prev_ma), close, np.nan
        )

        # ── Snapshot Values ────────────────────────────────────
        cur_price  = float(close.iloc[-1])
        cur_ma     = float(hist["MA"].iloc[-1])
        cur_rsi    = float(hist["RSI"].iloc[-1])
        cur_macd   = float(hist["MACD"].iloc[-1])
        cur_sig    = float(hist["MACD_Sig"].iloc[-1])
        cur_bbu    = float(hist["BB_Upper"].iloc[-1])
        cur_bbl    = float(hist["BB_Lower"].iloc[-1])
        prior      = close.iloc[-20] if len(hist) >= 20 else close.iloc[0]
        momentum   = ((cur_price - float(prior)) / float(prior)) * 100
        avg_vol    = float(hist["Volume"].rolling(20).mean().iloc[-1])
        last_vol   = float(hist["Volume"].iloc[-1])
        vol_ratio  = last_vol / avg_vol if avg_vol > 0 else 1.0

        # ── Nearest Fib level to current price ────────────────
        nearest_fib = min(fib_levels.items(), key=lambda x: abs(x[1] - cur_price))
        fib_distance = ((cur_price - nearest_fib[1]) / nearest_fib[1]) * 100

        # ── Signal Labels ──────────────────────────────────────
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

        rsi_label  = (
            "🔴 Overbought (>70)" if cur_rsi > 70 else
            "🟢 Oversold  (<30)" if cur_rsi < 30 else "⚪ Neutral"
        )
        macd_label = "🟢 Bullish cross" if cur_macd > cur_sig else "🔴 Bearish cross"
        bb_label   = (
            "🔴 At upper band" if cur_price >= cur_bbu * 0.99 else
            "🟢 At lower band" if cur_price <= cur_bbl * 1.01 else "⚪ Within bands"
        )
        vol_label  = (
            "🟢 High volume"  if vol_ratio > 1.5 else
            "🔴 Light volume" if vol_ratio < 0.5 else "⚪ Average volume"
        )
        fib_label  = (
            f"📐 Near {nearest_fib[0]} (${nearest_fib[1]:.2f}) — "
            f"{'above' if fib_distance > 0 else 'below'} by {abs(fib_distance):.1f}%"
        )

        metrics = {
            "Price":             f"${cur_price:.2f}",
            "1-Mo Momentum":     f"{momentum:+.1f}%",
            f"{ma_label}":       f"${cur_ma:.2f}",
            "MA Signal":         ma_signal,
            "RSI (14)":          f"{cur_rsi:.1f} — {rsi_label}",
            "MACD":              macd_label,
            "Bollinger":         bb_label,
            "Volume vs 20-avg":  f"{vol_ratio:.2f}x — {vol_label}",
            "Fibonacci Zone":    fib_label,
        }

        # ── Build Multi-Panel Chart ────────────────────────────
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

        # Bollinger Bands fill
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["BB_Upper"], name="BB Upper",
            line=dict(color="rgba(120,120,255,0.35)", width=1), showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["BB_Lower"], name="BB Lower",
            fill="tonexty", fillcolor="rgba(120,120,255,0.07)",
            line=dict(color="rgba(120,120,255,0.35)", width=1), showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["BB_Mid"], name="BB Mid",
            line=dict(color="rgba(180,180,255,0.4)", width=1, dash="dot"),
            showlegend=False
        ), row=1, col=1)

        # Fibonacci horizontal lines (drawn before price so price sits on top)
        for label, level in fib_levels.items():
            fig.add_hline(
                y=level,
                line_dash="dot",
                line_color=FIB_COLORS[label],
                line_width=1.2,
                annotation_text=f" Fib {label}  ${level:.2f}",
                annotation_position="right",
                annotation_font_size=9,
                annotation_font_color=FIB_COLORS[label],
                row=1, col=1
            )

        # Price line + MA
        fig.add_trace(go.Scatter(
            x=hist.index, y=close, name="Price",
            line=dict(color="#4fc3f7", width=2)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["MA"], name=ma_label,
            line=dict(color="#ffa726", width=1.8, dash="dot")
        ), row=1, col=1)

        # Buy / Sell signal markers
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["Buy"], mode="markers", name="BUY Signal",
            marker=dict(color="#4caf50", size=11, symbol="triangle-up",
                        line=dict(color="white", width=1))
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["Sell"], mode="markers", name="SELL Signal",
            marker=dict(color="#f44336", size=11, symbol="triangle-down",
                        line=dict(color="white", width=1))
        ), row=1, col=1)

        # Volume bars
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
            x=hist.index, y=hist["MACD_H"],
            marker_color=hist_colors, showlegend=False
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
                y=y_val, line_dash="dash",
                line_color=color, line_width=1.2,
                row=4, col=1
            )

        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            height=820,
            margin=dict(l=10, r=80, t=40, b=10),
            legend=dict(
                orientation="h", yanchor="bottom",
                y=1.01, xanchor="right", x=1,
                font=dict(size=11)
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,25,1)",
        )
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, gridcolor="#1e1e2e")
        fig.update_yaxes(title_text="Volume",   row=2, col=1, gridcolor="#1e1e2e")
        fig.update_yaxes(title_text="MACD",     row=3, col=1, gridcolor="#1e1e2e")
        fig.update_yaxes(
            title_text="RSI", range=[0, 100],
            row=4, col=1, gridcolor="#1e1e2e"
        )

        return is_bullish, metrics, fig, cur_price, fib_levels

    except Exception:
        return False, {}, None, 0.0, {}

# ================================================================
# AI ANALYSIS — Gemini with automatic model fallback
# ================================================================
def generate_ai_analysis(
    symbol: str,
    metrics: dict,
    period: int,
    method: str,
    fib_levels: dict = None,
    extra_context: str = ""
) -> str:
    if not AI_AVAILABLE:
        return "⚠️ AI unavailable — GEMINI_API_KEY not found in secrets."

    fib_text = ""
    if fib_levels:
        fib_text = "\nFibonacci Retracement Levels (full range of visible history):\n"
        for label, price in fib_levels.items():
            fib_text += f"  {label}: ${price:.2f}\n"

    prompt = f"""
You are an elite institutional equity and crypto analyst with 20 years of Wall Street experience.
A client — who may be a beginner — needs a professional analysis of {symbol}.

Live market data:
{json.dumps(metrics, indent=2)}
{fib_text}
Analysis framework: {period}-day {method}
{f"Additional context: {extra_context}" if extra_context else ""}

Write a structured institutional report with EXACTLY these five sections.
Be clear enough for a novice but deep enough for a professional.
Define any jargon the first time you use it.

---

## 📋 Quantitative Tear Sheet
A markdown table with columns: | Metric | Value | What It Means (plain English) |
Include every metric provided above including Fibonacci levels.

---

## 🌊 Elliott Wave & Trend Structure
- Identify the most likely current Elliott Wave position (label each wave concept the first time)
- Explain where price sits relative to the Fibonacci retracement levels above
- State which Fibonacci level is acting as current support or resistance and why
- Give specific price targets for the next wave up AND the key level that would invalidate the bullish case

---

## 🔀 Multi-Indicator Confluence
- Where RSI, MACD, Bollinger Bands, and Volume all agree: state the consensus
- Where they conflict: explain the tension and which signal takes priority and why
- Rate overall signal strength: Strong / Moderate / Weak with justification
- One sentence summary a complete beginner can act on

---

## ⚠️ Risk Assessment
- Primary bull thesis risk
- Primary bear thesis risk
- Specific price level that invalidates the current setup
- Suggested stop-loss zone (as a price range)

---

## 🎯 Portfolio Strategy Suggestion
**Bold and specific.** Include: action (buy/hold/sell/avoid), entry zone, target zone,
stop-loss level, and position size tier (aggressive/moderate/conservative).
End with one sentence summarising the risk/reward ratio.
"""

    # FIX: Try model candidates in order until one works
    errors = []
    for model in GEMINI_MODEL_CANDIDATES:
        try:
            response = gemini_client.models.generate_content(
                model=model, contents=prompt
            )
            return f"*Analysis generated using model: `{model}`*\n\n" + response.text
        except Exception as e:
            errors.append(f"{model}: {e}")
            continue

    return (
        "⚠️ All Gemini models failed. Check that:\n"
        "1. Your `GEMINI_API_KEY` is correct in Streamlit Secrets\n"
        "2. The key has billing enabled at https://aistudio.google.com\n"
        "3. You have access to at least one Gemini model\n\n"
        f"**Errors tried:**\n" + "\n".join(f"- {e}" for e in errors)
    )

# ================================================================
# BATCH SCANNER — threaded
# ================================================================
def scan_tickers(ticker_list: list, period: int, calc_type: str, max_workers: int = 15):
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

def show_ai_report(report_key: str, symbol: str, metrics: dict,
                   period: int, method: str, fib_levels: dict,
                   context: str = "", button_label: str = "🤖  Generate AI Analyst Report"):
    """
    FIX: Store AI output in session_state so it survives Streamlit reruns.
    The report won't vanish when the user interacts with other widgets.
    """
    if st.button(button_label, key=f"btn_{report_key}"):
        with st.spinner("Gemini is analysing — this may take 20–40 seconds for Pro…"):
            report = generate_ai_analysis(symbol, metrics, period, method, fib_levels, context)
            st.session_state[f"report_{report_key}"] = report

    if f"report_{report_key}" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state[f"report_{report_key}"])
        if st.button("🗑️  Clear Report", key=f"clear_{report_key}"):
            del st.session_state[f"report_{report_key}"]
            st.rerun()

# ================================================================
# MAIN APP
# ================================================================
st.title("📊  Wall Street AI Dashboard")
st.caption("Institutional-grade analysis · Gemini 2.5 Pro · Elliott Wave · Fibonacci · Real-time data")

render_indicator_guide()
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Analysis Settings")
    ma_type = st.radio(
        "Moving Average Type",
        ["Simple Moving Average (SMA)", "Weighted Moving Average (WMA)"],
        help="WMA reacts faster to recent price. SMA is smoother."
    )
    sma_period = st.selectbox(
        "Lookback Period",
        options=list(INTERVAL_MAP.keys()),
        index=1,
        format_func=lambda x: INTERVAL_MAP[x]["label"],
        help="Longer = slower signals, fewer false alarms. Shorter = faster but noisier."
    )
    st.divider()
    if not AI_AVAILABLE:
        st.error("⚠️ GEMINI_API_KEY not in secrets — AI disabled.")
    else:
        st.success("✅ Gemini AI connected")
    if SUPABASE_AVAILABLE:
        st.success("✅ Cloud portfolio connected")
    else:
        st.warning("⚠️ Supabase not configured — portfolios won't persist")
    st.divider()
    st.caption("🔒 Shared institutional API — no personal key needed.")

# ── Mode Selector ────────────────────────────────────────────────
mode = st.radio(
    "Mode",
    ["💼  My Portfolio", "🔍  Analyze Single Asset", "🌐  Market Scanner"],
    horizontal=True,
    label_visibility="collapsed"
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
        st.caption("New users: choose any username and PIN to create your account automatically.")
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
                    st.session_state["auth_user"]      = uid
                    st.session_state["auth_pin"]       = pin
                    st.session_state["user_portfolio"] = portfolio
                    st.rerun()
        st.stop()

    # ── Sidebar Position Manager ──────────────────────────────────
    with st.sidebar:
        st.subheader("🛠️  Position Manager")
        if SUPABASE_AVAILABLE:
            st.caption(f"Logged in as: **{st.session_state.get('auth_user','—')}**")

        with st.form("position_form"):
            # FIX: Asset type toggle prevents ETH (stock) vs ETH-USD (crypto) confusion
            asset_type = st.radio(
                "Asset Type",
                ["📈 Stock", "₿ Crypto"],
                horizontal=True,
                help="Select Crypto to auto-format the ticker (e.g. ETH → ETH-USD)"
            )
            raw_ticker = st.text_input(
                "Ticker Symbol",
                placeholder="Stock: AAPL  |  Crypto: ETH or ETH-USD"
            ).upper().strip()
            new_shares = st.number_input(
                "Shares / Units Owned", min_value=0.0, step=0.001,
                help="Set to 0 to remove this position."
            )
            new_cost   = st.number_input(
                "Average Purchase Price ($)", min_value=0.0, step=0.01
            )
            save_btn   = st.form_submit_button("💾  Save Position", use_container_width=True)

        if save_btn and raw_ticker:
            # Auto-append -USD for crypto if not already present
            if "Crypto" in asset_type:
                new_ticker = raw_ticker if raw_ticker.endswith("-USD") else raw_ticker + "-USD"
            else:
                new_ticker = raw_ticker

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
                "📥  Download Portfolio Backup",
                data=json.dumps(st.session_state["user_portfolio"]),
                file_name="portfolio_backup.json",
                mime="application/json",
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
            "Your portfolio is empty. Use the **Position Manager** in the sidebar. "
            "For crypto, select the ₿ Crypto toggle before entering the symbol."
        )
    else:
        total_value = total_cost = 0.0
        rows, charts = [], {}

        with st.spinner("Fetching live quotes and calculating all indicators…"):
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
                        charts[sym] = (fig, fib)
                    rows.append({
                        "Asset":         sym,
                        "Shares":        f"{pos['shares']:.4f}",
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

        st.subheader("📈  Deep-Dive Chart & AI Analysis")
        chosen = st.selectbox("Select a holding", list(portfolio.keys()))

        if chosen in charts:
            fig, fib = charts[chosen]
            st.plotly_chart(fig, use_container_width=True)
            chosen_metrics = next((r for r in rows if r["Asset"] == chosen), {})
            pos_detail     = portfolio.get(chosen, {})
            context        = (
                f"Held position: {pos_detail.get('shares',0):.4f} units "
                f"at ${pos_detail.get('cost',0):.2f} avg cost. "
                f"Current P&L: {chosen_metrics.get('Return (%)', 'unknown')}."
            )
            show_ai_report(
                f"portfolio_{chosen}", chosen, chosen_metrics,
                sma_period, ma_type, fib, context
            )
        else:
            st.warning("Chart unavailable — data may be insufficient for this asset.")

# ================================================================
# MODE 2 — SINGLE ASSET ANALYSIS
# ================================================================
elif mode == "🔍  Analyze Single Asset":
    st.header("🔍  Single Asset Analysis")

    st.caption(
        "Enter any US stock ticker or crypto symbol. "
        "**Crypto format:** BTC-USD, ETH-USD, XRP-USD, SOL-USD"
    )
    c1, c2 = st.columns([4, 1])
    with c1:
        symbol_input = st.text_input(
            "Ticker", label_visibility="collapsed",
            placeholder="Stock: NVDA, AAPL   |   Crypto: BTC-USD, ETH-USD, XRP-USD"
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
                f"Could not retrieve data for **{symbol_input}**. "
                "Check the ticker. Crypto must use the format `BTC-USD`, `ETH-USD`, etc."
            )
        else:
            display_metrics_grid(metrics)
            st.plotly_chart(fig, use_container_width=True)
            show_ai_report(
                f"single_{symbol_input}", symbol_input, metrics,
                sma_period, ma_type, fib,
                button_label="🤖  Generate Full AI Report"
            )

# ================================================================
# MODE 3 — MARKET SCANNER
# ================================================================
elif mode == "🌐  Market Scanner":
    st.header("🌐  Market Scanner")
    st.caption(
        "Scans the selected universe and returns every ticker with a bullish MA crossover "
        "signal combined with positive 1-month momentum. Fibonacci zones shown in results."
    )

    tab_stocks, tab_crypto = st.tabs(["📈  Stocks", "₿  Crypto"])

    # ── Stocks ────────────────────────────────────────────────────
    with tab_stocks:
        col_uni, col_btn = st.columns([3, 1])
        with col_uni:
            universe = st.selectbox(
                "Select Market Universe",
                [
                    "S&P 500  (~500 stocks, ~2–4 min)",
                    "Russell 2000  (~2,000 stocks, ~10–20 min)",
                    "All US Equities  (~1,500–3,000 stocks, ~15–30 min)",
                ],
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            scan_btn = st.button("🚀  Launch Scan", use_container_width=True)

        if "Russell" in universe:
            st.warning("⚠️ Large scan — expect 10–20 minutes. Best run during off-hours.")
        elif "All US" in universe:
            st.warning("⚠️ Very large scan — expect 15–30 minutes.")
        else:
            st.info("ℹ️ ~500 stocks. Expected time: 2–4 minutes.")

        if scan_btn:
            with st.spinner("Loading ticker universe…"):
                if "S&P 500" in universe:
                    tickers = get_sp500_tickers()
                elif "Russell" in universe:
                    tickers = get_russell2000_tickers()
                else:
                    tickers = get_all_us_equities()

            if not tickers:
                st.error("Failed to load tickers. Check your internet connection.")
            else:
                st.info(f"Loaded **{len(tickers)}** tickers. Starting scan…")
                results, figs = scan_tickers(tickers, sma_period, ma_type)

                if results:
                    st.success(f"✅ Scan complete — **{len(results)}** buy signals found")
                    st.dataframe(
                        pd.DataFrame(results), use_container_width=True, hide_index=True
                    )
                    st.subheader("📊  Deep-Dive Chart")
                    triggered = [r["Ticker"] for r in results]
                    view_sym  = st.selectbox("Select a triggered stock", triggered)

                    if view_sym in figs:
                        fig, fib = figs[view_sym]
                        st.plotly_chart(fig, use_container_width=True)
                        stock_metrics = next(
                            (r for r in results if r["Ticker"] == view_sym), {}
                        )
                        show_ai_report(
                            f"scanner_{view_sym}", view_sym, stock_metrics,
                            sma_period, ma_type, fib,
                            button_label="🤖  AI Analysis for this stock"
                        )
                else:
                    st.warning(
                        "No stocks meet the buy signal criteria for the chosen MA period. "
                        "Try a different lookback window in the sidebar."
                    )

    # ── Crypto ────────────────────────────────────────────────────
    with tab_crypto:
        st.subheader("₿  Major Crypto Dashboard")
        st.caption("Bitcoin · Ethereum · XRP · Solana — all shown regardless of signal direction")
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
            st.dataframe(
                pd.DataFrame(crypto_rows), use_container_width=True, hide_index=True
            )
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
                    (r for r in crypto_rows if chosen_crypto in r["Asset"]), {}
                )
                show_ai_report(
                    f"crypto_{chosen_crypto}", chosen_crypto, crypto_metrics,
                    sma_period, ma_type, fib,
                    extra_context=(
                        "This is a cryptocurrency. Factor in 24/7 trading, higher volatility "
                        "norms, and absence of traditional fundamental metrics like P/E ratios."
                    ),
                    button_label="🤖  Generate Crypto AI Analysis"
                )
        else:
            st.info("Click 'Refresh Crypto Data' above to load.")
