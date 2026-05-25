# ================================================================
# WALL STREET AI DASHBOARD — Production Build v2.0
# Multi-indicator technical analysis | Gemini 2.5 Pro | Supabase
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

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card { background: #1a1a2e; border-radius: 8px; padding: 12px; border-left: 3px solid #4fc3f7; }
    .signal-green { color: #4caf50; font-weight: bold; }
    .signal-red { color: #f44336; font-weight: bold; }
    .signal-neutral { color: #90a4ae; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# SECRETS — stored in .streamlit/secrets.toml, never in code
# ================================================================
def _get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return ""

GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")
SUPABASE_URL   = _get_secret("SUPABASE_URL")
SUPABASE_KEY   = _get_secret("SUPABASE_KEY")

# ── Gemini Client ────────────────────────────────────────────────
AI_AVAILABLE = bool(GEMINI_API_KEY)
gemini_client = None
if AI_AVAILABLE:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        AI_AVAILABLE = False

# ── Supabase Client ──────────────────────────────────────────────
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
GEMINI_MODEL = "gemini-2.5-pro"

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

# ================================================================
# INDICATOR EDUCATION PANEL
# ================================================================
def render_indicator_guide():
    with st.expander("📚  Indicator Guide — What does everything mean?  (click to expand)", expanded=False):
        st.markdown("#### Learn the indicators used in every chart and analysis report")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("---")
            st.markdown("### 📈  Moving Average (SMA / WMA)")
            st.info(
                "**What it is:** The average closing price over the chosen number of days, "
                "updated every session. SMA weights each day equally; WMA gives more weight to "
                "recent days, making it react faster.\n\n"
                "**Analogy:** Think of it as the stock's 'center of gravity.' Prices naturally "
                "drift back toward it. When price punches above, buyers are in control. When it "
                "falls below, sellers have taken the wheel.\n\n"
                "🟢 **Price above MA** → Bullish zone\n"
                "🔴 **Price below MA** → Bearish zone\n"
                "🔺 **Green triangle** → Price just crossed above (Buy signal)\n"
                "🔻 **Red triangle**  → Price just crossed below (Sell signal)"
            )

            st.markdown("---")
            st.markdown("### 📉  RSI — Relative Strength Index")
            st.info(
                "**What it is:** A 0–100 scale that measures how fast and aggressively price "
                "has been moving. Built on 14 days of price changes by default.\n\n"
                "**Analogy:** Imagine a runner's fatigue meter. A runner sprinting flat-out "
                "(RSI above 70) will eventually slow down — the stock is 'overbought' and "
                "a pullback may come. A runner who's been crawling exhausted (RSI below 30) "
                "is 'oversold' and may burst into a rally.\n\n"
                "🔴 **Above 70** → Overbought — pullback risk\n"
                "🟢 **Below 30** → Oversold   — bounce candidate\n"
                "⚪ **30–70**    → Neutral territory, no extreme pressure"
            )

            st.markdown("---")
            st.markdown("### 📊  Volume")
            st.info(
                "**What it is:** The total number of shares (or coins) traded in a session. "
                "It tells you whether a price move has real conviction behind it.\n\n"
                "**Analogy:** Think of volume as votes. A stock rising on heavy volume means "
                "thousands of investors are backing the move. Rising on thin volume is like "
                "winning an election with almost no one showing up — easy to reverse.\n\n"
                "🟢 **Rising price + High volume** → Strong, confirmed move\n"
                "🔴 **Rising price + Low volume**  → Weak, suspect move\n"
                "📌 Volume bars are green when the candle closed up, red when it closed down"
            )

        with c2:
            st.markdown("---")
            st.markdown("### ⚡  MACD — Moving Average Convergence Divergence")
            st.info(
                "**What it is:** Compares a fast 12-day exponential average to a slow 26-day "
                "average. The gap between them (the MACD line) is plotted against a 9-day "
                "'signal' line. The histogram shows how wide the gap is.\n\n"
                "**Analogy:** Picture two pace cars — one fast, one slow. When the fast car "
                "pulls ahead of the slow car (MACD crosses above Signal), buying momentum is "
                "accelerating. When the slow car takes the lead, sellers are in command. The "
                "histogram bars show how far apart the cars are.\n\n"
                "🟢 **MACD above Signal line** → Bullish momentum building\n"
                "🔴 **MACD below Signal line** → Bearish momentum building\n"
                "📊 **Histogram growing**       → Momentum is strengthening\n"
                "📊 **Histogram shrinking**     → Momentum may be reversing"
            )

            st.markdown("---")
            st.markdown("### 🎯  Bollinger Bands")
            st.info(
                "**What it is:** Three lines drawn around price — a 20-day moving average "
                "in the middle, flanked by upper and lower bands set 2 standard deviations "
                "away. The bands widen when volatility is high and squeeze when it's calm.\n\n"
                "**Analogy:** Like lane markers on a highway. Price tends to stay between "
                "the outer lanes. When it drifts to the edge, it usually snaps back toward "
                "the center. A 'squeeze' (narrow bands) often precedes a large breakout "
                "move — it's the calm before the storm.\n\n"
                "🔴 **Price touching upper band** → Potentially overbought\n"
                "🟢 **Price touching lower band** → Potentially oversold\n"
                "⚠️  **Bands squeezing tightly**  → Big directional move is loading"
            )

            st.markdown("---")
            st.markdown("### 🔀  Reading Multiple Signals Together")
            st.info(
                "**No single indicator is a guarantee.** Professional traders look for "
                "'confluence' — multiple indicators agreeing at the same time.\n\n"
                "**Strongest buy setup example:**\n"
                "- Price crosses above MA 🔺\n"
                "- RSI is between 40–60 (not already overbought)\n"
                "- MACD just crossed above Signal line\n"
                "- Volume is above average on the breakout day\n"
                "- Price bouncing off lower Bollinger Band\n\n"
                "The AI analysis report synthesizes all of these for you automatically."
            )

# ================================================================
# PORTFOLIO AUTH — Supabase
# ================================================================
def hash_pin(pin: str) -> str:
    return hashlib.sha256(pin.encode()).hexdigest()

def load_portfolio_from_db(user_id: str, pin: str):
    """Returns portfolio dict, empty dict (new user), or None (wrong PIN)."""
    try:
        result = db.table("portfolios").select("*").eq("user_id", user_id).execute()
        rows = result.data
        if not rows:
            return {}  # New user — account created on first save
        if rows[0]["pin_hash"] != hash_pin(pin):
            return None  # Wrong PIN
        return {
            row["ticker"]: {"shares": row["shares"], "cost": row["cost"]}
            for row in rows
        }
    except Exception as e:
        st.error(f"Database error: {e}")
        return None

def save_position_to_db(user_id: str, pin: str, ticker: str, shares: float, cost: float) -> bool:
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
# TICKER UNIVERSE LOADERS (cached 24 hours)
# ================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers() -> list:
    try:
        table = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
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
            df = pd.read_html(url)[0]
            tickers.update(df[col].str.replace(".", "-", regex=False).tolist())
        except Exception:
            pass
    return sorted(list(tickers))

@st.cache_data(ttl=86400, show_spinner=False)
def get_russell2000_tickers() -> list:
    # Attempt 1: iShares IWM ETF holdings (live source)
    try:
        url = (
            "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"
            "/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
        )
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, headers=headers, timeout=20)
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), skiprows=9)
        df = df[df.get("Asset Class", df.columns[0]) == "Equity"]
        tickers = df["Ticker"].dropna().str.strip().tolist()
        valid = sorted([t for t in tickers if t and t != "-" and len(t) <= 6])
        if len(valid) > 500:
            return valid
    except Exception:
        pass

    # Attempt 2: Wikipedia S&P SmallCap 600 as proxy
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")[0]
        st.caption("ℹ️ Russell 2000 live feed unavailable — showing S&P SmallCap 600 as proxy.")
        return sorted(df["Ticker"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        pass

    st.warning("Russell 2000 data unavailable. Try S&P 500 instead.")
    return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_us_equities() -> list:
    """NASDAQ screener: all US equities with avg volume > 500K."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept":     "application/json, text/plain, */*",
        }
        url = (
            "https://api.nasdaq.com/api/screener/stocks"
            "?tableonly=true&limit=5000&offset=0&download=true"
        )
        r = requests.get(url, headers=headers, timeout=25)
        data = r.json()
        df = pd.DataFrame(data["data"]["rows"])
        df["volume"] = pd.to_numeric(
            df["volume"].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )
        df = df[df["volume"] > 500_000]
        tickers = sorted(df["symbol"].str.strip().tolist())
        if len(tickers) > 100:
            return tickers
    except Exception:
        pass

    # Fallback: S&P 1500
    st.caption("ℹ️ Full equity list unavailable — showing S&P 1500 as broad market proxy.")
    return get_sp1500_tickers()

# ================================================================
# CORE TECHNICAL ANALYSIS ENGINE
# ================================================================
def fetch_technical_data(symbol: str, period_window: int, calc_type: str):
    """
    Returns: (is_bullish: bool, metrics: dict, fig: Figure | None, price: float)
    """
    try:
        ticker  = yf.Ticker(symbol)
        lookback = INTERVAL_MAP[period_window]["history"]
        hist    = ticker.history(period=lookback)

        min_bars = max(period_window, 26, 20) + 5
        if hist.empty or len(hist) < min_bars:
            return False, {}, None, 0.0

        close = hist["Close"]

        # ── Moving Average ─────────────────────────────────────
        if "Simple" in calc_type:
            hist["MA"]  = close.rolling(period_window).mean()
            ma_label    = f"SMA-{period_window}"
        else:
            weights     = np.arange(1, period_window + 1)
            hist["MA"]  = close.rolling(period_window).apply(
                lambda p: np.dot(p, weights) / weights.sum(), raw=True
            )
            ma_label = f"WMA-{period_window}"

        # ── Bollinger Bands (20-day, ±2σ) ─────────────────────
        bb_mid          = close.rolling(20).mean()
        bb_std          = close.rolling(20).std()
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

        # ── MA Crossover Signals ───────────────────────────────
        prev_close = close.shift(1)
        prev_ma    = hist["MA"].shift(1)
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

        # ── Signal State Labels ────────────────────────────────
        buys  = hist["Buy"].dropna()
        sells = hist["Sell"].dropna()

        if not buys.empty and (sells.empty or buys.index[-1] > sells.index[-1]):
            ma_signal = f"🟢 BUY  ({buys.index[-1].strftime('%m/%d')})"
            is_bullish = True
        elif not sells.empty and (buys.empty or sells.index[-1] > buys.index[-1]):
            ma_signal  = f"🔴 SELL ({sells.index[-1].strftime('%m/%d')})"
            is_bullish = False
        else:
            ma_signal  = "⚪ Neutral"
            is_bullish = momentum > 0

        rsi_label = (
            "🔴 Overbought (>70)" if cur_rsi > 70 else
            "🟢 Oversold  (<30)" if cur_rsi < 30 else
            "⚪ Neutral"
        )
        macd_label = (
            "🟢 Bullish cross" if cur_macd > cur_sig else "🔴 Bearish cross"
        )
        bb_label = (
            "🔴 At upper band" if cur_price >= cur_bbu * 0.99 else
            "🟢 At lower band" if cur_price <= cur_bbl * 1.01 else
            "⚪ Within bands"
        )
        vol_label = (
            "🟢 High volume"   if vol_ratio > 1.5 else
            "🔴 Light volume"  if vol_ratio < 0.5 else
            "⚪ Average volume"
        )

        metrics = {
            "Price":            f"${cur_price:.2f}",
            "1-Mo Momentum":   f"{momentum:+.1f}%",
            f"{ma_label}":     f"${cur_ma:.2f}",
            "MA Signal":       ma_signal,
            "RSI (14)":        f"{cur_rsi:.1f} — {rsi_label}",
            "MACD":            macd_label,
            "Bollinger":       bb_label,
            "Volume vs 20-avg": f"{vol_ratio:.2f}x — {vol_label}",
        }

        # ── Build Chart ────────────────────────────────────────
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            row_heights=[0.50, 0.17, 0.17, 0.16],
            vertical_spacing=0.025,
            subplot_titles=[
                f"{symbol} — Price, {ma_label} & Bollinger Bands",
                "Volume",
                "MACD  (12 / 26 / 9)",
                "RSI  (14-period)  |  70 = Overbought · 30 = Oversold",
            ],
        )

        # Row 1: Bollinger fill + price + MA + signals
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["BB_Upper"], name="BB Upper",
            line=dict(color="rgba(120,120,255,0.35)", width=1),
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["BB_Lower"], name="BB Lower",
            fill="tonexty", fillcolor="rgba(120,120,255,0.07)",
            line=dict(color="rgba(120,120,255,0.35)", width=1),
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["BB_Mid"], name="BB Mid",
            line=dict(color="rgba(180,180,255,0.4)", width=1, dash="dot"),
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=close, name="Price",
            line=dict(color="#4fc3f7", width=2)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["MA"], name=ma_label,
            line=dict(color="#ffa726", width=1.8, dash="dot")
        ), row=1, col=1)
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

        # Row 2: Volume bars coloured green/red by session direction
        vol_colors = [
            "#4caf50" if float(c) >= float(o) else "#f44336"
            for c, o in zip(hist["Close"], hist["Open"])
        ]
        fig.add_trace(go.Bar(
            x=hist.index, y=hist["Volume"], name="Volume",
            marker_color=vol_colors, showlegend=False
        ), row=2, col=1)

        # Row 3: MACD histogram + lines
        hist_colors = [
            "#4caf50" if float(v) >= 0 else "#f44336"
            for v in hist["MACD_H"]
        ]
        fig.add_trace(go.Bar(
            x=hist.index, y=hist["MACD_H"], name="MACD Histogram",
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

        # Row 4: RSI with threshold lines
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["RSI"], name="RSI",
            line=dict(color="#ce93d8", width=1.8)
        ), row=4, col=1)
        for y_val, color in [(70, "rgba(244,67,54,0.55)"), (30, "rgba(76,175,80,0.55)")]:
            fig.add_hline(y=y_val, line_dash="dash", line_color=color,
                          line_width=1.2, row=4, col=1)

        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            height=800,
            margin=dict(l=10, r=10, t=40, b=10),
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
        fig.update_yaxes(title_text="RSI",      row=4, col=1,
                         range=[0, 100], gridcolor="#1e1e2e")

        return is_bullish, metrics, fig, cur_price

    except Exception:
        return False, {}, None, 0.0

# ================================================================
# AI ANALYSIS — Gemini 2.5 Pro
# ================================================================
def generate_ai_analysis(
    symbol: str,
    metrics: dict,
    period: int,
    method: str,
    extra_context: str = ""
) -> str:
    if not AI_AVAILABLE:
        return "⚠️ AI analysis unavailable — GEMINI_API_KEY not configured in secrets."
    try:
        prompt = f"""
You are an elite institutional equity and crypto analyst with 20 years of Wall Street experience.
A client — who may be a complete beginner — is looking at {symbol}.

Live market snapshot:
{json.dumps(metrics, indent=2)}

Analysis framework: {period}-day {method}
{f"Additional context: {extra_context}" if extra_context else ""}

Produce a structured report with EXACTLY these five sections. Write clearly enough that a novice
trader can understand, but with the depth a professional would respect.

---

## 📋 Quantitative Tear Sheet
A clean markdown table listing every metric above with three columns:
| Metric | Value | What It Means (plain English) |

---

## 🌊 Trend Structure & Elliott Wave Framework
Analyse the price action relative to the {period}-day {method}.
- Describe the most likely Elliott Wave position (impulsive wave vs corrective wave)
- Identify key psychological price levels (round numbers, MA value, Bollinger boundaries)
- State what price needs to do to confirm a bullish continuation or signal a breakdown
- Keep one sentence explanations for each wave term used

---

## 🔀 Multi-Indicator Confluence Analysis
Synthesise what RSI, MACD, Bollinger Bands, and Volume are saying *together*.
- Where indicators agree: explain what that consensus means
- Where indicators conflict: explain the tension and which signal takes priority and why
- Rate the overall signal strength: Strong / Moderate / Weak, and justify it

---

## ⚠️ Risk Assessment
- Primary risk to the bullish thesis
- Primary risk to the bearish thesis
- What specific price level or event would invalidate the current setup
- Suggested stop-loss zone (as a price range, not just a concept)

---

## 🎯 Portfolio Strategy Suggestion
**Bold, specific, and actionable.**
State: suggested action (buy / hold / sell / avoid), key entry price zone, target price zone,
stop-loss level, and recommended position size tier (aggressive / moderate / conservative)
relative to a typical portfolio. End with one sentence summarising the overall risk/reward.

---

Important: Define any jargon the first time you use it. This report should educate as it advises.
"""
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text

    except Exception as e:
        return f"⚠️ AI analysis error: {e}\n\nCheck that your Gemini API key has access to {GEMINI_MODEL}."

# ================================================================
# BATCH SCANNER — threaded
# ================================================================
def scan_tickers(
    ticker_list: list,
    period: int,
    calc_type: str,
    max_workers: int = 15
):
    """
    Scans a list of tickers concurrently.
    Returns (results: list[dict], figs: dict[str, Figure])
    """
    results = []
    figs    = {}

    progress_bar = st.progress(0.0, text="Preparing scan…")
    total = len(ticker_list)
    done  = 0

    def _scan_one(sym):
        return sym, *fetch_technical_data(sym, period, calc_type)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_scan_one, t): t for t in ticker_list}
        for future in as_completed(futures):
            done += 1
            pct   = done / total
            sym, bullish, metrics, fig, price = future.result()
            progress_bar.progress(pct, text=f"Scanning … {done}/{total}")
            if bullish and metrics:
                results.append({"Ticker": sym, **metrics})
                if fig is not None:
                    figs[sym] = fig

    progress_bar.empty()
    return results, figs

# ================================================================
# DISPLAY HELPERS
# ================================================================
def display_metrics_grid(metrics: dict):
    """Show metric dict in a clean 4-col responsive grid."""
    items = list(metrics.items())
    row1  = items[:4]
    row2  = items[4:]
    cols  = st.columns(len(row1))
    for col, (k, v) in zip(cols, row1):
        col.metric(k, v)
    if row2:
        cols2 = st.columns(len(row2))
        for col, (k, v) in zip(cols2, row2):
            col.metric(k, v)

# ================================================================
# MAIN APP
# ================================================================
st.title("📊  Wall Street AI Dashboard")
st.caption("Institutional-grade technical analysis · Gemini 2.5 Pro · Real-time data via yfinance")

render_indicator_guide()
st.divider()

# ── Sidebar Controls ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Analysis Settings")

    ma_type = st.radio(
        "Moving Average Type",
        ["Simple Moving Average (SMA)", "Weighted Moving Average (WMA)"],
        help="WMA reacts faster to recent price action. SMA is smoother and slower."
    )
    sma_period = st.selectbox(
        "Lookback Period",
        options=list(INTERVAL_MAP.keys()),
        index=1,
        format_func=lambda x: INTERVAL_MAP[x]["label"],
        help="Longer windows = slower signals, fewer false alarms. Shorter = faster but noisier."
    )

    st.divider()

    if not AI_AVAILABLE:
        st.error("⚠️ GEMINI_API_KEY not found in secrets. AI analysis disabled.")
    else:
        st.success("✅ Gemini 2.5 Pro connected")

    if SUPABASE_AVAILABLE:
        st.success("✅ Cloud portfolio storage connected")
    else:
        st.warning("⚠️ Supabase not configured — portfolios won't persist between sessions")

    st.divider()
    st.caption("🔒 All users share the same institutional API. No personal key required.")

# ── Mode Selector ────────────────────────────────────────────────
mode = st.radio(
    "Select Mode",
    ["💼  My Portfolio", "🔍  Analyze Single Asset", "🌐  Market Scanner"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

# ================================================================
# MODE 1: PORTFOLIO DASHBOARD
# ================================================================
if mode == "💼  My Portfolio":
    st.header("💼  Portfolio Dashboard")

    # ── Session state fallback when Supabase is off ───────────────
    if not SUPABASE_AVAILABLE:
        for k, v in [
            ("user_portfolio", {}),
            ("auth_user", "local_session"),
            ("auth_pin", ""),
        ]:
            if k not in st.session_state:
                st.session_state[k] = v
        st.info(
            "☁️ Cloud storage is not configured. Your portfolio is saved for this "
            "browser session only. Contact the app owner to enable persistent portfolios."
        )

    # ── Login Form (Supabase mode) ────────────────────────────────
    if SUPABASE_AVAILABLE and "auth_user" not in st.session_state:
        st.subheader("🔐  Access Your Portfolio")
        st.caption(
            "Enter your username and 4-digit PIN. "
            "First-time users: choose any username and PIN to create your account automatically."
        )

        with st.form("login_form"):
            uid  = st.text_input("Username / Investor ID", placeholder="e.g. john_trader")
            pin  = st.text_input("4-Digit PIN", type="password", max_chars=4,
                                 placeholder="e.g. 4821")
            submitted = st.form_submit_button("Access Portfolio →", use_container_width=True)

        if submitted:
            uid = uid.strip().lower()
            if not uid or not pin:
                st.error("Please enter both a username and PIN.")
            elif len(pin) != 4 or not pin.isdigit():
                st.error("PIN must be exactly 4 digits (numbers only).")
            else:
                with st.spinner("Authenticating…"):
                    portfolio = load_portfolio_from_db(uid, pin)
                if portfolio is None:
                    st.error("❌ Incorrect PIN for this username.")
                else:
                    st.session_state["auth_user"]       = uid
                    st.session_state["auth_pin"]        = pin
                    st.session_state["user_portfolio"]  = portfolio
                    st.rerun()
        st.stop()

    # ── Sidebar Position Manager ──────────────────────────────────
    with st.sidebar:
        st.subheader("🛠️  Position Manager")
        if SUPABASE_AVAILABLE:
            st.caption(f"Logged in as: **{st.session_state.get('auth_user', '—')}**")

        with st.form("position_form"):
            new_ticker = st.text_input(
                "Ticker Symbol",
                placeholder="AAPL, TSLA, BTC-USD …"
            ).upper().strip()
            new_shares = st.number_input("Shares / Units Owned", min_value=0.0, step=0.01,
                                         help="Set to 0 to remove this position.")
            new_cost   = st.number_input("Average Purchase Price ($)", min_value=0.0, step=0.01)
            save_btn   = st.form_submit_button("💾  Save Position", use_container_width=True)

        if save_btn and new_ticker:
            uid = st.session_state.get("auth_user", "local_session")
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

        # Export fallback for session-only users
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
                restored = json.load(uploaded)
                st.session_state["user_portfolio"] = restored
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
            "Your portfolio is empty. Use the **Position Manager** in the sidebar "
            "to add holdings (stocks or crypto like BTC-USD)."
        )
    else:
        total_value, total_cost = 0.0, 0.0
        rows, charts = [], {}

        with st.spinner("Fetching live quotes and calculating indicators…"):
            for sym, pos in list(portfolio.items()):
                _, metrics, fig, price = fetch_technical_data(sym, sma_period, ma_type)
                if price > 0:
                    pos_cost  = pos["shares"] * pos["cost"]
                    pos_value = pos["shares"] * price
                    pos_gain  = pos_value - pos_cost
                    pos_pct   = (pos_gain / pos_cost * 100) if pos_cost > 0 else 0.0
                    total_value += pos_value
                    total_cost  += pos_cost
                    if fig:
                        charts[sym] = fig
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
                        "MACD":          metrics.get("MACD", "—"),
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

        st.subheader("📈  Deep-Dive Chart")
        chosen = st.selectbox("Select a holding", list(portfolio.keys()))

        if chosen in charts:
            st.plotly_chart(charts[chosen], use_container_width=True)

            if st.button("🤖  Generate AI Analyst Report", use_container_width=False):
                chosen_metrics = next((r for r in rows if r["Asset"] == chosen), {})
                pos_detail     = portfolio.get(chosen, {})
                context        = (
                    f"Held position: {pos_detail.get('shares',0):.4f} units "
                    f"at ${pos_detail.get('cost',0):.2f} average cost. "
                    f"Current P&L: {chosen_metrics.get('Return (%)', 'unknown')}."
                )
                with st.spinner("Gemini 2.5 Pro is analysing…"):
                    report = generate_ai_analysis(chosen, chosen_metrics, sma_period, ma_type, context)
                st.markdown(report)
        else:
            st.warning("Chart unavailable for this asset — data may be insufficient.")

# ================================================================
# MODE 2: SINGLE ASSET ANALYSIS
# ================================================================
elif mode == "🔍  Analyze Single Asset":
    st.header("🔍  Single Asset Analysis")
    st.caption("Enter any US stock ticker or crypto symbol (e.g. AAPL, TSLA, BTC-USD, XRP-USD)")

    c1, c2 = st.columns([4, 1])
    with c1:
        symbol_input = st.text_input(
            "Ticker symbol", label_visibility="collapsed",
            placeholder="e.g. NVDA  or  ETH-USD"
        ).upper().strip()
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        go_btn = st.button("Analyse →", use_container_width=True)

    if go_btn and symbol_input:
        with st.spinner(f"Fetching {symbol_input}…"):
            _, metrics, fig, price = fetch_technical_data(symbol_input, sma_period, ma_type)

        if fig is None or price == 0.0:
            st.error(
                f"Could not retrieve data for **{symbol_input}**. "
                "Double-check the ticker. Crypto must use the format BTC-USD, ETH-USD, etc."
            )
        else:
            display_metrics_grid(metrics)
            st.plotly_chart(fig, use_container_width=True)

            col_ai, _ = st.columns([2, 5])
            with col_ai:
                if st.button("🤖  Generate Full AI Report", use_container_width=True):
                    with st.spinner("Gemini 2.5 Pro is analysing…"):
                        report = generate_ai_analysis(symbol_input, metrics, sma_period, ma_type)
                    st.markdown(report)

# ================================================================
# MODE 3: MARKET SCANNER
# ================================================================
elif mode == "🌐  Market Scanner":
    st.header("🌐  Market Scanner")
    st.caption(
        "Scans the selected universe and returns every ticker currently showing a bullish "
        "MA crossover signal combined with positive 1-month momentum."
    )

    tab_stocks, tab_crypto = st.tabs(["📈  Stocks", "₿  Crypto"])

    # ── Stocks Tab ────────────────────────────────────────────────
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
                help=(
                    "Larger universes find more opportunities but take longer. "
                    "Start with S&P 500 to verify the scanner works, then expand."
                )
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            scan_btn = st.button("🚀  Launch Scan", use_container_width=True)

        if "S&P 500" in universe:
            st.info("ℹ️ Scanning ~500 large-cap stocks. Expected time: **2–4 minutes**.")
        elif "Russell" in universe:
            st.warning(
                "⚠️ Scanning ~2,000 small-cap stocks. Expected time: **10–20 minutes**. "
                "Consider running during off-hours."
            )
        else:
            st.warning(
                "⚠️ Scanning the full liquid US equity market. Expected time: **15–30 minutes**. "
                "Only tickers with >500K average daily volume are included."
            )

        if scan_btn:
            with st.spinner("Loading ticker universe…"):
                if "S&P 500" in universe:
                    tickers = get_sp500_tickers()
                elif "Russell" in universe:
                    tickers = get_russell2000_tickers()
                else:
                    tickers = get_all_us_equities()

            if not tickers:
                st.error("Failed to load ticker list. Check your internet connection and try again.")
            else:
                st.info(f"Loaded **{len(tickers)}** tickers. Starting scan…")
                results, figs = scan_tickers(tickers, sma_period, ma_type)

                if results:
                    st.success(f"✅ Scan complete — **{len(results)}** stocks with active buy signals")
                    st.dataframe(
                        pd.DataFrame(results), use_container_width=True, hide_index=True
                    )

                    st.subheader("📊  Chart Deep-Dive")
                    triggered = [r["Ticker"] for r in results]
                    view_sym  = st.selectbox("Select a triggered stock", triggered)

                    if view_sym in figs:
                        st.plotly_chart(figs[view_sym], use_container_width=True)
                        stock_metrics = next(
                            (r for r in results if r["Ticker"] == view_sym), {}
                        )
                        if st.button("🤖  AI Analysis for this stock"):
                            with st.spinner("Analysing…"):
                                report = generate_ai_analysis(
                                    view_sym, stock_metrics, sma_period, ma_type
                                )
                            st.markdown(report)
                else:
                    st.warning(
                        "No stocks currently meet the buy signal criteria for the chosen "
                        "MA period. Try a different lookback window in the sidebar."
                    )

    # ── Crypto Tab ────────────────────────────────────────────────
    with tab_crypto:
        st.subheader("₿  Major Crypto Dashboard")
        st.caption(
            "Tracking: Bitcoin · Ethereum · XRP · Solana  |  "
            "All tickers shown regardless of signal direction."
        )

        crypto_btn = st.button("📡  Refresh Crypto Data")

        if crypto_btn or "crypto_data" not in st.session_state:
            crypto_rows, crypto_figs = [], {}
            with st.spinner("Fetching crypto data…"):
                for name, sym in CRYPTO_TICKERS.items():
                    _, metrics, fig, _ = fetch_technical_data(sym, sma_period, ma_type)
                    if metrics:
                        crypto_rows.append({"Asset": f"{name} ({sym})", **metrics})
                    if fig:
                        crypto_figs[sym] = fig
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
                st.plotly_chart(crypto_figs[chosen_crypto], use_container_width=True)

                if st.button("🤖  AI Crypto Analysis"):
                    crypto_metrics = next(
                        (r for r in crypto_rows if chosen_crypto in r["Asset"]), {}
                    )
                    with st.spinner("Analysing…"):
                        report = generate_ai_analysis(
                            chosen_crypto, crypto_metrics, sma_period, ma_type,
                            extra_context="This is a cryptocurrency. Factor in 24/7 trading, "
                                          "higher volatility norms, and absence of traditional "
                                          "fundamental metrics."
                        )
                    st.markdown(report)
        else:
            st.info("Click 'Refresh Crypto Data' above to load the dashboard.")
