# ================================================================
# WALL STREET AI DASHBOARD — Production Build v2.12
# v2.12 patches (incremental on v2.11):
#   NEW: 📈 Options Lab — pure educational sandbox for options trading
#        • 4 single-leg strategies: Long Call, Long Put, Cash-Secured Put, Covered Call
#        • Liquidity filter: OI ≥100, vol ≥10, spread ≤10%, DTE 14-90
#        • Full Greek suite (Δ, Γ, Θ, V, Ρ) computed via Black-Scholes (no scipy)
#        • Three sub-tabs: Chain Explorer · Strategy Analyzer · Paper Trades
#        • AI is an ANALYZER (not recommender) — explains user-selected trades
#          in context of the underlying's actual technical setup
#        • Paper trades stored in Supabase, auto-revalued daily, auto-expire
#        • Expiring-tomorrow banner + auto-close at intrinsic value on expiry
#   NEW: 👁️ Watch List — track tickers without owning them
#        • New 'watchlists' table in Supabase, soft cap 50 tickers
#        • Sidebar form parallel to Quick Add Position
#        • Fresh signals from watch list appear in same notification box,
#          sectioned under '💼 Your Holdings' and '👁️ Watch List' subheaders
#        • Auto-removes from watch list when ticker added to portfolio
#          (via Quick Add, Buy trade, or portfolio editor save)
# ================================================================
# Previous (v2.11):
#   NEW: OpenInsider is now the PRIMARY insider data source (per-ticker)
#        — scrapes SEC Form 4 directly, more reliable than yfinance
#   NEW: get_insider_data() unified helper (OpenInsider → yfinance fallback)
#   FIX: Elliott Wave AI prompt overhauled — rigorous 9-wave framework,
#        confidence levels, alternative scenarios, eliminates "always W3" bias
# ================================================================
# Previous (v2.10):
# v2.10 patches (incremental on v2.9):
#   FIX: Scanner dropdown no longer clears results — scan persisted to session_state
#   FIX: Insider transactions hardened — defensive column detection, retry, fallback
#   NEW: get_insider_summary_fallback() uses yfinance insider_purchases endpoint
#        as backstop when the detailed table is empty
#   NEW: "Clear Scan" button + cached-scan timestamp + settings-drift warning
# ================================================================
# Previous (v2.9):
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
import hashlib, json, requests, re, time, math
from datetime import datetime, timedelta
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

# ── Watch list ────────────────────────────────────────────────
WATCHLIST_CAP = 50  # soft cap on watchlist size

# ── Options Lab ───────────────────────────────────────────────
RISK_FREE_RATE     = 0.045   # ~1-year US Treasury rate, 2026
MIN_OPTION_OI      = 100     # liquidity filter: open interest
MIN_OPTION_VOL     = 10      # liquidity filter: daily volume
MAX_OPTION_SPREAD  = 0.10    # max bid-ask spread as % of mid (10%)
MIN_OPTION_DTE     = 14      # min days to expiration
MAX_OPTION_DTE     = 90      # max days to expiration

OPTIONS_STRATEGIES = {
    "Long Call": {
        "type": "call",  "direction": "long",
        "description": "Buy a call option. Bullish — profits if stock rises significantly before expiration.",
        "max_loss": "Premium paid (limited)",
        "max_profit": "Unlimited",
        "best_for": "Strong directional uptrend with clear catalyst or fresh BUY crossover",
        "worst_for": "Sideways or declining stock; time decay erodes value daily",
        "complexity": "Beginner",
    },
    "Long Put": {
        "type": "put",   "direction": "long",
        "description": "Buy a put option. Bearish — profits if stock falls significantly before expiration.",
        "max_loss": "Premium paid (limited)",
        "max_profit": "(Strike × 100) minus premium (if stock goes to $0)",
        "best_for": "Strong directional downtrend or as portfolio hedge",
        "worst_for": "Sideways or rising stock; time decay works against you",
        "complexity": "Beginner",
    },
    "Cash-Secured Put": {
        "type": "put",   "direction": "short",
        "description": "Sell a put while holding enough cash to buy 100 shares at strike if assigned. Income strategy for stocks you'd like to own.",
        "max_loss": "(Strike − premium) × 100 (if stock goes to $0)",
        "max_profit": "Premium received (if put expires worthless)",
        "best_for": "Stocks you'd be happy to own at strike price; sideways-to-up markets; high IV environments",
        "worst_for": "Sharp downturns — you'll be assigned shares below current market",
        "complexity": "Intermediate",
    },
    "Covered Call": {
        "type": "call",  "direction": "short",
        "description": "Sell a call against 100 shares you already own. Generates income but caps upside.",
        "max_loss": "Stock can still fall to $0; premium provides a small buffer",
        "max_profit": "(Strike − cost basis) × 100 + premium received",
        "best_for": "Stocks you're holding through sideways or mild-up moves; want income; not expecting a breakout",
        "worst_for": "Stocks about to rally hard — you cap your upside at the strike",
        "complexity": "Intermediate",
    },
}

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
# OPTIONS MATH — Black-Scholes-Merton Greeks (no scipy required)
# ================================================================
def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)

def bs_greeks(S, K, T, r, sigma, opt_type="call"):
    """
    Black-Scholes-Merton Greeks for a European option.

    S      : spot price
    K      : strike price
    T      : time to expiration in years (DTE / 365)
    r      : risk-free rate (decimal, e.g. 0.045 for 4.5%)
    sigma  : implied volatility (decimal, e.g. 0.30 for 30%)
    opt_type: 'call' or 'put'

    Returns dict with delta, gamma, theta (per day), vega (per 1% IV), rho (per 1% rate).
    Falls back to zero values for invalid inputs (T<=0, sigma<=0, etc.) rather than raising.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    sqrt_T = math.sqrt(T)
    try:
        d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    except (ValueError, ZeroDivisionError):
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    d2 = d1 - sigma * sqrt_T

    pdf_d1 = _norm_pdf(d1)
    gamma  = pdf_d1 / (S * sigma * sqrt_T)
    vega   = S * pdf_d1 * sqrt_T / 100.0   # per 1% IV change

    if opt_type == "call":
        delta     = _norm_cdf(d1)
        theta_yr  = -(S * pdf_d1 * sigma) / (2 * sqrt_T) - r * K * math.exp(-r*T) * _norm_cdf(d2)
        rho       = K * T * math.exp(-r*T) * _norm_cdf(d2) / 100.0
    else:  # put
        delta     = _norm_cdf(d1) - 1.0
        theta_yr  = -(S * pdf_d1 * sigma) / (2 * sqrt_T) + r * K * math.exp(-r*T) * _norm_cdf(-d2)
        rho       = -K * T * math.exp(-r*T) * _norm_cdf(-d2) / 100.0

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 4),
        "theta": round(theta_yr / 365.0, 4),  # per day
        "vega":  round(vega, 4),
        "rho":   round(rho, 4),
    }

def prob_above(S, K, T, r, sigma, target):
    """Probability stock price at expiration > target (lognormal)."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > target else 0.0
    try:
        d = (math.log(S/target) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    except (ValueError, ZeroDivisionError):
        return 0.5
    return _norm_cdf(d)

def strategy_metrics(strategy_name, S, K, premium, T, sigma, r=RISK_FREE_RATE, stock_cost_basis=None):
    """
    Compute max profit / max loss / breakeven / probability of profit for a single-leg strategy.
    All amounts per single contract (100 shares).
    Returns dict.
    """
    info = OPTIONS_STRATEGIES[strategy_name]
    opt_type = info["type"]; direction = info["direction"]

    cost = premium * 100  # per contract

    if strategy_name == "Long Call":
        breakeven  = K + premium
        max_profit = float("inf")
        max_loss   = cost
        pop        = prob_above(S, K, T, r, sigma, breakeven)  # P(S_T > BE)
        capital    = cost
    elif strategy_name == "Long Put":
        breakeven  = K - premium
        max_profit = max(K - premium, 0) * 100
        max_loss   = cost
        pop        = 1.0 - prob_above(S, K, T, r, sigma, breakeven)  # P(S_T < BE)
        capital    = cost
    elif strategy_name == "Cash-Secured Put":
        breakeven  = K - premium   # effective cost basis if assigned
        max_profit = cost          # credit received
        max_loss   = (K - premium) * 100
        pop        = prob_above(S, K, T, r, sigma, K)  # keep credit if S_T > K
        capital    = K * 100       # cash reserved
    elif strategy_name == "Covered Call":
        if stock_cost_basis is None: stock_cost_basis = S  # assume current price if not specified
        breakeven  = stock_cost_basis - premium
        max_profit = (K - stock_cost_basis) * 100 + cost
        # Max loss = full stock loss minus premium received (stock can go to 0)
        max_loss   = stock_cost_basis * 100 - cost
        pop        = 1.0 - prob_above(S, K, T, r, sigma, K)  # keep stock + premium if S_T < K
        capital    = stock_cost_basis * 100
    else:
        return None

    return {
        "strategy":   strategy_name,
        "breakeven":  breakeven,
        "max_profit": max_profit,
        "max_loss":   max_loss,
        "pop":        pop,
        "capital":    capital,
        "premium":    premium,
        "strike":     K,
        "spot":       S,
        "dte_days":   T * 365,
        "iv":         sigma,
    }

def strategy_payoff_at_expiry(strategy_name, K, premium, S_T, stock_cost_basis=None):
    """P&L per contract (100 shares) at expiration given final stock price S_T."""
    if strategy_name == "Long Call":
        return (max(S_T - K, 0) - premium) * 100
    elif strategy_name == "Long Put":
        return (max(K - S_T, 0) - premium) * 100
    elif strategy_name == "Cash-Secured Put":
        return (premium - max(K - S_T, 0)) * 100
    elif strategy_name == "Covered Call":
        if stock_cost_basis is None: stock_cost_basis = K
        # Long stock P&L (capped at strike) + premium collected
        stock_pnl = (min(S_T, K) - stock_cost_basis) * 100
        return stock_pnl + premium * 100
    return 0.0

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
# WATCHLIST DB
# ================================================================
def load_watchlist_from_db(user_id):
    """Returns list of watchlist tickers for the user (no PIN check needed — non-sensitive)."""
    if not SUPABASE_AVAILABLE: return []
    try:
        result = db.table("watchlists").select("ticker").eq("user_id", user_id).execute()
        return [row["ticker"] for row in (result.data or [])]
    except Exception as e:
        # Table may not exist yet — silent for graceful degradation
        return []

def save_watchlist_ticker(user_id, pin, ticker):
    if not SUPABASE_AVAILABLE: return False
    try:
        db.table("watchlists").upsert({
            "user_id": user_id, "pin_hash": hash_pin(pin), "ticker": ticker
        }, on_conflict="user_id,ticker").execute()
        return True
    except Exception as e:
        st.error(f"Watchlist save error: {e}"); return False

def remove_watchlist_ticker(user_id, ticker):
    if not SUPABASE_AVAILABLE: return False
    try:
        db.table("watchlists").delete().eq("user_id", user_id).eq("ticker", ticker).execute()
        return True
    except Exception:
        return False

def auto_remove_from_watchlist(user_id, ticker):
    """Called when a ticker is added to portfolio — removes from watch list if present."""
    if not SUPABASE_AVAILABLE: return
    wl = st.session_state.get("user_watchlist", [])
    if ticker in wl:
        wl.remove(ticker)
        st.session_state["user_watchlist"] = wl
        remove_watchlist_ticker(user_id, ticker)

# ================================================================
# PAPER OPTIONS DB
# ================================================================
def load_paper_options_from_db(user_id):
    if not SUPABASE_AVAILABLE: return []
    try:
        result = db.table("paper_options").select("*").eq("user_id", user_id).order("entry_date", desc=True).execute()
        return result.data or []
    except Exception:
        return []

def save_paper_option_to_db(user_id, pin, trade: dict):
    if not SUPABASE_AVAILABLE: return False
    try:
        row = {
            "user_id":       user_id,
            "pin_hash":      hash_pin(pin),
            "ticker":        trade["ticker"],
            "strategy":      trade["strategy"],
            "strike":        float(trade["strike"]),
            "expiration":    trade["expiration"],
            "entry_premium": float(trade["entry_premium"]),
            "contracts":     int(trade.get("contracts", 1)),
            "status":        trade.get("status", "open"),
            "notes":         trade.get("notes", ""),
        }
        db.table("paper_options").insert(row).execute()
        return True
    except Exception as e:
        st.error(f"Paper trade save error: {e}"); return False

def close_paper_option_in_db(trade_id, close_premium, status="closed"):
    if not SUPABASE_AVAILABLE: return False
    try:
        db.table("paper_options").update({
            "status":         status,
            "close_premium":  float(close_premium),
            "close_date":     datetime.utcnow().isoformat(),
        }).eq("id", trade_id).execute()
        return True
    except Exception as e:
        st.error(f"Paper trade close error: {e}"); return False

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
    """
    Hardened insider-transaction fetch.
    - Defensive column detection (yfinance changes names between versions)
    - Distinguishes 'no data' from 'data exists but filtered out'
    - Always returns DataFrame or None (never raises)
    """
    try:
        ticker_obj = yf.Ticker(symbol)
        # Brief retry for transient yfinance hiccups
        raw = None
        for attempt in range(2):
            try:
                raw = ticker_obj.insider_transactions
                if raw is not None and not raw.empty:
                    break
            except Exception:
                if attempt == 0: time.sleep(0.3)

        if raw is None or raw.empty:
            return None

        df = raw.copy()

        # Date column — try known names first, then any 'date'-like column
        date_col = None
        for cand in ["Start Date", "Date", "TransactionDate", "Transaction Date"]:
            if cand in df.columns:
                date_col = cand; break
        if not date_col:
            date_col = next((c for c in df.columns if "date" in str(c).lower()), None)

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)
            df_filtered = df[df[date_col] >= cutoff].copy()
            if df_filtered.empty and not df.empty:
                # All filings older than 12mo — return None; caller shows helpful message
                return None
            df = df_filtered
            df["Date"] = df[date_col].dt.strftime("%m/%d/%Y")

        # Description column — most discriminating for transaction type
        text_col = None
        for cand in ["Text", "Description", "Transaction", "transaction_text"]:
            if cand in df.columns:
                text_col = cand; break
        if not text_col:
            text_col = next(
                (c for c in df.columns if str(c).lower() in ["text","description","transaction"]),
                None
            )

        if text_col:
            df["Transaction Type"] = df[text_col].apply(_categorize_tx)
        else:
            df["Transaction Type"] = "⚪ Unknown"

        def _find(patterns):
            for p in patterns:
                col = next((c for c in df.columns if p in str(c).lower()), None)
                if col: return col
            return None

        name_col  = _find(["insider","name"])
        pos_col   = _find(["position","title","role"])
        share_col = _find(["share"])
        val_col   = _find(["value"])

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
        if text_col:
            clean["Description"] = df[text_col]

        return clean if not clean.empty else None
    except Exception:
        return None

@st.cache_data(ttl=600, show_spinner=False)
def get_insider_summary_fallback(symbol):
    """
    Fallback when detailed insider_transactions returns nothing.
    yfinance also exposes insider_purchases — a small summary table of
    net purchases over 3/6/12 months. Often works when the full table fails.
    Returns a markdown-ready string or None.
    """
    try:
        purchases = yf.Ticker(symbol).insider_purchases
        if purchases is None or purchases.empty:
            return None
        df = purchases.copy()
        # Try to find a row labelled with "Net" insider activity
        net_val = None
        first_col = df.columns[0] if len(df.columns) > 0 else None
        if first_col is not None:
            for label_pattern in ["Net Shares Purchased", "Net Insider", "Net Activity"]:
                matches = df[df[first_col].astype(str).str.contains(label_pattern, case=False, na=False)]
                if not matches.empty:
                    # Scan remaining columns for the first numeric value
                    for vc in df.columns[1:]:
                        try:
                            val = pd.to_numeric(matches[vc].iloc[0], errors="coerce")
                            if pd.notna(val):
                                net_val = int(val); break
                        except Exception: continue
                    if net_val is not None: break

        if net_val is None:
            # Couldn't parse a specific number — show the raw summary
            return f"_yfinance summary endpoint returned data but in an unrecognized format:_\n\n```\n{df.to_string(index=False)[:500]}\n```"

        if net_val > 0:
            return f"🟢 **Net insider buying:** {net_val:,} shares purchased on net (yfinance 6-month summary)."
        elif net_val < 0:
            return f"🔴 **Net insider selling:** {abs(net_val):,} shares sold on net (yfinance 6-month summary)."
        else:
            return "⚪ Net insider activity is zero over the recent summary period."
    except Exception:
        return None

def get_insider_data(symbol):
    """
    Unified insider data getter — tries OpenInsider first, yfinance second.
    This is the recommended entry point for AI context / summary calls.
    Defined as a regular function (not cached) so it composes cached calls.
    """
    if not symbol or symbol.endswith("-USD"):
        return None
    df = get_openinsider_per_ticker(symbol)
    if df is None or df.empty:
        df = get_insider_transactions(symbol)
    return df

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
            # Try OpenInsider first — scrapes SEC Form 4 directly, most reliable
            df = get_openinsider_per_ticker(symbol)
            data_source = "OpenInsider (SEC Form 4 direct)"
            # Fall back to yfinance only if OpenInsider returns nothing
            if df is None:
                df = get_insider_transactions(symbol)
                data_source = "yfinance (fallback)"
        if df is None:
            # Last resort: try yfinance's lightweight insider_purchases summary
            fallback = get_insider_summary_fallback(symbol)
            if fallback:
                st.info(fallback)
                st.caption("Detailed transaction-level data unavailable from either OpenInsider or yfinance — showing yfinance summary only.")
            else:
                st.info("No recent insider data found. Neither OpenInsider nor yfinance returned transactions for this ticker in the last 12 months.")
            return
        st.caption(f"Source: **{data_source}**. Open Market = personal cash = genuine conviction.")
        show_noise = st.toggle("Show Non-Market entries (awards, exercises, gifts)",
                                value=False, key=f"noise_{symbol}")
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

# ----------------------------------------------------------------
# OpenInsider per-ticker: primary insider data source
# Scrapes SEC Form 4 filings directly. More reliable than yfinance.
# Transaction codes:  P = Purchase  S = Sale  A = Award  M = Exercise
#                     G = Gift      F = Tax   D = Sale to Issuer
# ----------------------------------------------------------------
def _map_openinsider_trade_type(raw):
    """Convert OpenInsider trade type string to our standard categories."""
    t = str(raw).strip().lower()
    # Sales involving option exercise are noise (not a discretionary signal)
    if "+oe" in t or "option" in t and "exercise" in t:
        return "⚪ Non-Market"
    # Standard purchase
    if t.startswith("p ") or t.startswith("p-") or "purchase" in t:
        return "🟢 Open Market Buy"
    # Standard sale
    if t.startswith("s ") or t.startswith("s-") or (t.startswith("sale") and "oe" not in t):
        return "🔴 Open Market Sale"
    # Non-market actions
    if any(k in t for k in ("award","grant","stock award","compensation")):
        return "⚪ Non-Market"
    if t.startswith("a ") or t.startswith("a-"):  return "⚪ Non-Market"
    if t.startswith("m ") or t.startswith("m-") or "exercise" in t: return "⚪ Non-Market"
    if t.startswith("g ") or t.startswith("g-") or "gift" in t:     return "⚪ Non-Market"
    if t.startswith("f ") or t.startswith("f-") or "tax" in t:      return "⚪ Non-Market"
    if t.startswith("d ") or "sale to issuer" in t:                 return "⚪ Non-Market"
    return "⚪ Other"

@st.cache_data(ttl=600, show_spinner=False)
def get_openinsider_per_ticker(symbol):
    """
    Fetch per-ticker insider transactions from OpenInsider (last 365 days).
    Returns a cleaned DataFrame matching the same column schema as
    get_insider_transactions() for drop-in compatibility, or None if no data.
    """
    if not symbol or symbol.endswith("-USD"):
        return None
    url = (
        f"https://openinsider.com/screener?s={symbol.upper()}"
        "&o=&pl=&ph=&ll=&lh=&fd=365&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp="
        "&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih="
        "&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=100&page=1"
    )
    try:
        for attempt in range(2):
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code == 200: break
            except Exception:
                if attempt == 0: time.sleep(0.4)
        else:
            return None
        if r.status_code != 200: return None

        dfs = pd.read_html(StringIO(r.text), flavor="lxml")
        if not dfs: return None

        # Find the transaction table — has Insider + Trade Type columns
        target = None
        for df in dfs:
            cols_l = [str(c).strip().lower() for c in df.columns]
            has_insider = any("insider" in c or "name" in c for c in cols_l)
            has_type    = any("trade type" in c or "tradetype" in c or "type" == c for c in cols_l)
            if has_insider and has_type and len(df) > 0:
                target = df; break

        if target is None or target.empty:
            return None

        df = target.copy()
        df.columns = [str(c).strip() for c in df.columns]

        def _col(patterns):
            for p in patterns:
                for c in df.columns:
                    if p in c.lower(): return c
            return None

        date_col  = _col(["filing date", "trade date", "date"])
        name_col  = _col(["insider name", "insider", "name"])
        title_col = _col(["title", "role", "position"])
        type_col  = _col(["trade type", "tradetype"])
        qty_col   = _col(["qty", "shares", "quantity"])
        price_col = _col(["price"])
        value_col = _col(["value"])

        clean = pd.DataFrame()

        if date_col:
            dt = pd.to_datetime(df[date_col], errors="coerce")
            clean["Date"] = dt.dt.strftime("%m/%d/%Y")
        if name_col:
            clean["Insider"] = df[name_col].astype(str).str.strip()
        if title_col:
            clean["Role"] = df[title_col].astype(str).str.strip()

        if type_col:
            clean["Transaction Type"] = df[type_col].apply(_map_openinsider_trade_type)
        else:
            clean["Transaction Type"] = "⚪ Unknown"

        if qty_col:
            qty_num = pd.to_numeric(
                df[qty_col].astype(str).str.replace(",","").str.replace("+","").str.replace("$",""),
                errors="coerce"
            )
            clean["Shares"] = qty_num.abs().apply(
                lambda x: f"{int(x):,}" if pd.notna(x) else "—"
            )

        if value_col:
            val_num = pd.to_numeric(
                df[value_col].astype(str).str.replace(",","").str.replace("+","").str.replace("$",""),
                errors="coerce"
            )
            clean["Est. Value ($)"] = val_num.abs().apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "—"
            )
            clean["_raw_value"] = val_num.abs()

        # Build a Description column for AI context
        if type_col and price_col:
            try:
                clean["Description"] = (
                    df[type_col].astype(str) + " @ $" + df[price_col].astype(str)
                ).str.strip()
            except Exception:
                clean["Description"] = df[type_col].astype(str)
        elif type_col:
            clean["Description"] = df[type_col].astype(str)

        return clean if not clean.empty else None
    except Exception:
        return None

# ================================================================
# OPTIONS DATA — chains, liquidity filter, Greeks enrichment
# ================================================================
@st.cache_data(ttl=600, show_spinner=False)
def get_option_expirations(symbol: str):
    """
    Returns list of (expiration_str, dte) tuples for expirations between
    MIN_OPTION_DTE and MAX_OPTION_DTE days. Empty list if none qualify.
    """
    try:
        ticker = yf.Ticker(symbol)
        all_exps = ticker.options
        if not all_exps:
            return []
        today = datetime.now().date()
        filtered = []
        for exp in all_exps:
            try:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if MIN_OPTION_DTE <= dte <= MAX_OPTION_DTE:
                    filtered.append((exp, dte))
            except Exception:
                continue
        return filtered
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def get_option_chain_raw(symbol: str, expiration: str):
    """Returns (calls_df, puts_df) for a ticker/expiration. Empty DFs on failure."""
    try:
        chain = yf.Ticker(symbol).option_chain(expiration)
        return chain.calls.copy(), chain.puts.copy()
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def filter_liquid_options(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply liquidity filter to an options chain DataFrame.
    Returns filtered DF; original df is preserved.
    """
    if df is None or df.empty:
        return df
    f = df.copy()
    # Normalize: yfinance sometimes has NaN volume / openInterest
    for col in ["volume", "openInterest", "bid", "ask"]:
        if col in f.columns:
            f[col] = pd.to_numeric(f[col], errors="coerce").fillna(0)

    # Filter: OI ≥ 100, volume ≥ 10, bid > 0, spread within tolerance
    mask = (
        (f.get("openInterest", 0) >= MIN_OPTION_OI) &
        (f.get("volume",       0) >= MIN_OPTION_VOL) &
        (f.get("bid",          0) >  0) &
        (f.get("ask",          0) >  0)
    )
    f = f[mask].copy()

    # Bid-ask spread check
    f["mid"] = (f["bid"] + f["ask"]) / 2.0
    f["spread_pct"] = (f["ask"] - f["bid"]) / f["mid"].replace(0, np.nan)
    # Allow either pct spread OR absolute ≤ $0.10 for cheap contracts
    spread_ok = (f["spread_pct"] <= MAX_OPTION_SPREAD) | ((f["ask"] - f["bid"]) <= 0.10)
    f = f[spread_ok].copy()
    return f

def enrich_chain_with_greeks(df: pd.DataFrame, S: float, T_years: float, opt_type: str):
    """Add delta, gamma, theta, vega, rho columns to a filtered chain DataFrame."""
    if df is None or df.empty:
        return df
    rows = []
    for _, row in df.iterrows():
        K     = float(row.get("strike", 0))
        sigma = float(row.get("impliedVolatility", 0))
        g = bs_greeks(S, K, T_years, RISK_FREE_RATE, sigma, opt_type)
        rows.append(g)
    g_df = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, g_df], axis=1)

# ================================================================
# OPTIONS PAYOFF DIAGRAM
# ================================================================
def render_payoff_diagram(strategy_name, S, K, premium, breakeven, max_profit, max_loss, cost_basis=None):
    """Plotly payoff diagram at expiration. Returns figure."""
    # Generate price range: ±40% from spot, centered on max(K, S)
    center = max(K, S)
    lo = max(center * 0.5, 0.01)
    hi = center * 1.5
    prices = np.linspace(lo, hi, 200)

    payoffs = np.array([
        strategy_payoff_at_expiry(strategy_name, K, premium, p, cost_basis)
        for p in prices
    ])

    fig = go.Figure()

    # Zero P&L horizontal reference
    fig.add_hline(y=0, line_color="rgba(180,180,180,0.4)", line_width=1)

    # Profit region (green fill above zero)
    fig.add_trace(go.Scatter(
        x=prices, y=np.where(payoffs > 0, payoffs, 0),
        fill="tozeroy", fillcolor="rgba(76,175,80,0.18)",
        line=dict(color="rgba(76,175,80,0)"), showlegend=False, hoverinfo="skip"
    ))
    # Loss region (red fill below zero)
    fig.add_trace(go.Scatter(
        x=prices, y=np.where(payoffs < 0, payoffs, 0),
        fill="tozeroy", fillcolor="rgba(244,67,54,0.18)",
        line=dict(color="rgba(244,67,54,0)"), showlegend=False, hoverinfo="skip"
    ))
    # P&L line itself
    fig.add_trace(go.Scatter(
        x=prices, y=payoffs, name="P&L at Expiration",
        line=dict(color="#4fc3f7", width=2.5),
        hovertemplate="Stock @ $%{x:.2f}<br>P&L: $%{y:,.0f}<extra></extra>"
    ))

    # Reference lines
    fig.add_vline(x=S, line_dash="dash", line_color="#4fc3f7", line_width=1.5,
                  annotation_text=f"Current ${S:.2f}", annotation_position="top",
                  annotation_font_size=10, annotation_font_color="#4fc3f7")
    fig.add_vline(x=K, line_dash="dot", line_color="#ffa726", line_width=1.5,
                  annotation_text=f"Strike ${K:.2f}", annotation_position="top right",
                  annotation_font_size=10, annotation_font_color="#ffa726")
    if breakeven and lo <= breakeven <= hi:
        fig.add_vline(x=breakeven, line_dash="dash", line_color="#ffd54f", line_width=1.2,
                      annotation_text=f"BE ${breakeven:.2f}", annotation_position="bottom",
                      annotation_font_size=10, annotation_font_color="#ffd54f")

    # Horizontal max profit / max loss lines (only if finite & visible)
    if max_profit not in (None, float("inf")) and abs(max_profit) < 1e7:
        fig.add_hline(y=max_profit, line_dash="dot", line_color="rgba(76,175,80,0.6)",
                      annotation_text=f"Max Profit ${max_profit:,.0f}",
                      annotation_position="right", annotation_font_size=10,
                      annotation_font_color="rgba(76,175,80,0.9)")
    if max_loss not in (None, float("inf")):
        fig.add_hline(y=-abs(max_loss), line_dash="dot", line_color="rgba(244,67,54,0.6)",
                      annotation_text=f"Max Loss -${abs(max_loss):,.0f}",
                      annotation_position="right", annotation_font_size=10,
                      annotation_font_color="rgba(244,67,54,0.9)")

    fig.update_layout(
        template="plotly_dark", height=420,
        title=f"{strategy_name} — Payoff at Expiration  (per 1 contract = 100 shares)",
        xaxis_title="Stock Price at Expiration ($)",
        yaxis_title="Profit / Loss ($)",
        margin=dict(l=40, r=60, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,25,1)",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#1e1e2e")
    fig.update_yaxes(gridcolor="#1e1e2e", zerolinecolor="rgba(180,180,180,0.4)")
    return fig

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

**IMPORTANT — Apply Elliott Wave theory rigorously. DO NOT default to "wave 3 impulse"
just because the stock is in an uptrend. Wave 3 is the EASIEST wave to mis-call.**

Identify the current wave position using these specific markers:

| Wave | Tell-tale Markers |
|------|------------------|
| **Wave 1** (new impulse) | Just emerging from base/consolidation; modest volume; no prior clear structure; often goes unnoticed |
| **Wave 2** (pullback) | Sharp 50–78.6% retracement of W1; sentiment fearful again; volume dries up |
| **Wave 3** (strongest) | REQUIRES clear W1-W2 setup before this; HIGHEST volume; price gaps and breaks decisively above W1 high; momentum/RSI peaking |
| **Wave 4** (consolidation) | Shallow 23.6–38.2% pullback from W3 peak; sideways triangle or flat; RSI cools to 40-55 |
| **Wave 5** (final push) | Extended rally; RSI/MACD shows NEGATIVE DIVERGENCE (lower momentum highs vs higher price highs); volume often lower than W3 |
| **Wave A** (correction start) | Sharp decline from W5 peak |
| **Wave B** (counter-rally) | Partial recovery, 50–78.6% of A; often deceptive ("dead cat bounce") |
| **Wave C** (final decline) | Second-leg drop, often equal in length to wave A |
| **Uncertain** | Acknowledge when data is insufficient or the pattern is ambiguous — this is a legitimate answer |

Specifically reference the provided data:
- **Volume vs 20-avg** → distinguishes W3 (high volume) from W5 (lower volume)
- **RSI level + Signal Age** → fresh signal at RSI 40-55 suggests W1 or W3 start; extended rally with RSI >70 + old signal suggests W5
- **Fibonacci Zone proximity** → "above 23.6%" after a rally = possible W4; "above 50/61.8%" after decline = possible Wave B
- **1-Mo Momentum** → very strong (>10%) suggests W3 in progress; modest with extended history suggests W5

Then provide:
1. **Most likely wave count** with explicit reasoning citing the markers above
2. **Confidence level**: High / Moderate / Low — be honest about uncertainty
3. **Alternative wave count** if the primary is not high-confidence (e.g., "Could also be early W5 if volume divergence confirms")
4. **Specific Fibonacci-based price targets** (W3 = 1.618×W1, W5 = 1.0×W1 measured from W4 low, etc.)
5. **Clear invalidation level** — the exact price at which your wave count would be wrong

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
# OPTIONS AI ANALYZER — analyzer, not recommender
# ================================================================
def generate_options_ai_analysis(symbol, strategy_name, metrics_options, technical_metrics, greeks):
    """
    Educational AI analyzer for a user-selected options strategy.
    Receives BOTH the options-specific metrics (strike, premium, POP, etc.)
    AND the underlying technical setup (confluence score, RSI, MA signal, etc.)
    so the AI can ground its explanation in the actual chart context.

    The AI's role is to EXPLAIN, NOT RECOMMEND. It walks through what the
    trade is betting on, how the technical setup either supports or argues
    against that bet, what conditions favor it, and what would kill it.
    """
    if not AI_AVAILABLE: return "⚠️ AI unavailable — GEMINI_API_KEY not in Secrets."

    info = OPTIONS_STRATEGIES.get(strategy_name, {})
    opt_type = info.get("type", "?"); direction = info.get("direction", "?")

    # Strip private keys before sending technical context
    tech_public = {k: v for k, v in technical_metrics.items() if not k.startswith("_")}

    # Format greeks for readability
    greeks_str = (
        f"Delta: {greeks['delta']:+.4f}  "
        f"Gamma: {greeks['gamma']:.4f}  "
        f"Theta: {greeks['theta']:+.4f}/day  "
        f"Vega: {greeks['vega']:+.4f}/1% IV  "
        f"Rho: {greeks['rho']:+.4f}/1% rate"
    )

    prompt = f"""
You are an options education specialist. The user is a comfortable equity trader new to options.
Explain — DO NOT RECOMMEND — the trade they're considering.

This is a PURELY EDUCATIONAL SANDBOX. No real money is involved. The user wants to UNDERSTAND
this trade, not be told whether to enter it.

═══ TRADE BEING ANALYZED ═══
Underlying: {symbol}
Strategy: {strategy_name}  (option type: {opt_type}, direction: {direction})
Strategy description: {info.get('description', '')}

Trade-specific metrics:
{json.dumps(metrics_options, indent=2, default=str)}

Greeks at current spot:
{greeks_str}

═══ UNDERLYING TECHNICAL SETUP (from app's analysis) ═══
{json.dumps(tech_public, indent=2)}

═══ OUTPUT FORMAT (5 SECTIONS, BE CONCISE BUT THOROUGH) ═══

## What This Trade Is Actually Betting On
Plain English: what scenario makes money, what scenario loses, what's needed to break even.
Use the actual numbers from the metrics provided. Avoid jargon; define terms on first use.

## How the Underlying Technical Setup Supports or Argues Against This Bet
This is the most important section. Reference the SPECIFIC technical readings:
- Signal Strength score, MA Signal, Signal Age, RSI, MACD, Bollinger, Volume, Momentum, Fibonacci.
- Does the chart's directional bias match what this strategy needs to profit?
- Does the confluence score support the assumed direction?
- A long call on a Strong Buy 9/14 setup is a very different story than a long call on a Watch List 5/14 setup. SAY WHICH ONE THIS IS.

## Greek Breakdown (Plain English)
Explain each Greek as it applies to THIS specific trade:
- **Delta** — how much you gain/lose per $1 move in the stock (in dollar terms, not abstract decimals).
- **Gamma** — how fast delta changes; high gamma means trade behavior shifts quickly.
- **Theta** — how much value the position loses (long) or gains (short) per day from time decay alone. State the DOLLAR amount per day.
- **Vega** — how much value moves per 1% change in implied volatility. Explain if vega is helping or hurting given current IV environment.
- **Rho** — sensitivity to interest rate changes. For short-dated retail trades this is usually small; explain its real-world relevance honestly (often minimal for trades under 90 DTE).

## Conditions That Favor This Trade
3-4 specific conditions, ideally referencing the actual current setup. Be honest if conditions are NOT favorable.

## Conditions That Would Kill This Trade
3-4 specific failure modes. Time decay, IV crush, directional move against, assignment risk for shorts, etc.
For Cash-Secured Puts: explain what happens if assigned. For Covered Calls: explain what happens if called away.

CRITICAL FORMATTING RULES:
- Use clean markdown only. Never nest bold markers.
- When writing dollar amounts, separate them with words ("the $105 strike", "premium of $2.50"), never run them together.
- Do not tell the user whether to enter this trade. EXPLAIN, do not RECOMMEND.
- Be honest about uncertainty. If the technical setup doesn't support this strategy, SAY SO clearly.
"""
    errors = []
    for m in GEMINI_MODEL_CANDIDATES:
        try:
            resp = gemini_client.models.generate_content(model=m, contents=prompt)
            return f"*Educational analysis · Model: `{m}`*\n\n" + resp.text
        except Exception as e: errors.append(f"**{m}:** {str(e)[:120]}")
    return "⚠️ AI analysis failed.\n\n" + "\n".join(f"- {e}" for e in errors)

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
def _classify_alerts(charts: dict):
    """Helper: returns (fresh_buys, fresh_sells) lists from charts dict."""
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
    fresh_buys.sort(key=lambda x: x[5])
    fresh_sells.sort(key=lambda x: x[5])
    return fresh_buys, fresh_sells

def render_portfolio_alerts(portfolio_charts: dict, watchlist_charts: dict = None):
    """
    Show fresh signals (≤ FRESH_SIGNAL_WINDOW trading days) from both portfolio
    and watch list. Two sectioned subheaders inside a single bordered container.
    Hides entirely if nothing fresh in either source.
    """
    watchlist_charts = watchlist_charts or {}
    p_buys, p_sells = _classify_alerts(portfolio_charts)
    w_buys, w_sells = _classify_alerts(watchlist_charts)

    total = len(p_buys) + len(p_sells) + len(w_buys) + len(w_sells)
    if total == 0:
        return  # hide entirely — no notification clutter

    with st.container(border=True):
        st.markdown(f"### 🔔  Fresh Signals — Last {FRESH_SIGNAL_WINDOW} Trading Days")
        st.caption(f"{total} active signal(s). Older signals are filtered out.")

        # ── Section 1: Your Holdings ───────────────────────
        if p_buys or p_sells:
            st.markdown("**💼 Your Holdings**")
            for label, signal, price, strength, age, _bars in p_buys:
                st.markdown(f"🟢 **{label}** — {signal} · {age} · {price} · {strength}")
            for label, signal, price, strength, age, _bars in p_sells:
                st.markdown(f"🔴 **{label}** — {signal} · {age} · {price} · {strength}")

        # ── Section 2: Watch List ──────────────────────────
        if w_buys or w_sells:
            if p_buys or p_sells:
                st.markdown("")  # visual gap
            st.markdown("**👁️ Watch List**")
            for label, signal, price, strength, age, _bars in w_buys:
                st.markdown(f"🟢 **{label}** — {signal} · {age} · {price} · {strength}")
            for label, signal, price, strength, age, _bars in w_sells:
                st.markdown(f"🔴 **{label}** — {signal} · {age} · {price} · {strength}")

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
            # Auto-remove newly added/updated tickers from watch list
            if is_buy and ticker in new_portfolio:
                auto_remove_from_watchlist(uid, ticker)
            st.rerun()

# ================================================================
# WATCH LIST UI
# ================================================================
def render_watchlist_form(watchlist: list, uid: str, pin: str):
    """Sidebar UI for adding/removing tickers from watch list."""
    with st.expander("👁️  Watch List", expanded=False):
        st.caption(
            f"Track tickers you're monitoring (not owned). Fresh signals on these "
            f"appear in the notification box alongside your holdings. "
            f"Soft cap: {WATCHLIST_CAP} tickers."
        )
        # Add form
        with st.form("watchlist_form", clear_on_submit=True):
            wl_input = st.text_input(
                "Add Ticker", placeholder="AAPL · NVDA · ETH · BTC",
                help="Crypto auto-detected — no suffix needed.",
            ).strip()
            wl_add = st.form_submit_button("➕ Add to Watch List", use_container_width=True)
        if wl_add and wl_input:
            new_tkr = normalize_ticker(wl_input)
            if not new_tkr:
                st.warning("Enter a ticker symbol.")
            elif new_tkr in watchlist:
                st.info(f"{ticker_label(new_tkr)} is already on your watch list.")
            elif len(watchlist) >= WATCHLIST_CAP:
                st.warning(
                    f"You have {WATCHLIST_CAP} watch list items — consider removing "
                    f"inactive ones before adding more."
                )
            else:
                # Check if already in portfolio
                portfolio = st.session_state.get("user_portfolio", {})
                if new_tkr in portfolio:
                    st.info(f"{ticker_label(new_tkr)} is already in your portfolio — no need to watch it.")
                else:
                    watchlist.append(new_tkr)
                    st.session_state["user_watchlist"] = watchlist
                    if SUPABASE_AVAILABLE:
                        save_watchlist_ticker(uid, pin, new_tkr)
                    st.success(f"✅ Added {ticker_label(new_tkr)} to watch list")
                    st.rerun()

        # Current list with remove buttons
        if watchlist:
            st.markdown(f"**Currently watching ({len(watchlist)}/{WATCHLIST_CAP}):**")
            for tkr in sorted(watchlist):
                cols = st.columns([4, 1])
                cols[0].markdown(f"👁️ {ticker_label(tkr)}")
                if cols[1].button("✕", key=f"rmwl_{tkr}", help=f"Remove {ticker_label(tkr)}"):
                    watchlist.remove(tkr)
                    st.session_state["user_watchlist"] = watchlist
                    if SUPABASE_AVAILABLE:
                        remove_watchlist_ticker(uid, tkr)
                    st.rerun()
        else:
            st.caption("_Watch list is empty._")

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
            added   = set(new_portfolio.keys()) - set(portfolio.keys())
            if SUPABASE_AVAILABLE:
                for t,pos in new_portfolio.items(): save_position_to_db(uid,pin,t,pos["shares"],pos["cost"])
                for t in deleted: save_position_to_db(uid,pin,t,0,0)
            st.session_state["user_portfolio"] = new_portfolio
            # Auto-remove newly added tickers from watch list
            for t in added:
                auto_remove_from_watchlist(uid, t)
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

mode = st.radio("Mode", ["💼  My Portfolio","🔍  Analyze Single Asset","🌐  Market Scanner","📈  Options Lab"],
                horizontal=True, label_visibility="collapsed")
st.markdown("<br>", unsafe_allow_html=True)

# ================================================================
# MODE 1 — PORTFOLIO
# ================================================================
if mode == "💼  My Portfolio":
    st.header("💼  Portfolio Dashboard")

    if not SUPABASE_AVAILABLE:
        for k,v in [("user_portfolio",{}),("auth_user","local"),("auth_pin",""),("user_watchlist",[])]:
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
                    # Load watchlist + paper options alongside portfolio
                    watchlist_db = load_watchlist_from_db(uid)
                    st.session_state.update({
                        "auth_user": uid, "auth_pin": pin,
                        "user_portfolio": portfolio_db,
                        "user_watchlist": watchlist_db,
                    })
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
                # Auto-remove from watch list since it's now a holding
                auto_remove_from_watchlist(uid_, new_ticker)
                st.success(f"✅ {new_ticker} added")

        st.divider()
        render_trade_form(
            st.session_state.get("user_portfolio",{}),
            st.session_state.get("auth_user","local"),
            st.session_state.get("auth_pin","")
        )

        st.divider()
        render_watchlist_form(
            st.session_state.get("user_watchlist", []),
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
                for k in ["auth_user","auth_pin","user_portfolio","user_watchlist","paper_options"]:
                    st.session_state.pop(k,None)
                st.rerun()

    portfolio = st.session_state.get("user_portfolio", {})
    watchlist = st.session_state.get("user_watchlist", [])
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

        # ── Fetch watch list signals for the notification box ──
        watchlist_charts = {}
        if watchlist:
            with st.spinner(f"Checking watch list signals ({len(watchlist)} tickers)…"):
                for wsym in watchlist:
                    _, w_metrics, w_fig, w_price, w_fib, w_score, w_err = fetch_technical_data(
                        wsym, sma_period, ma_type
                    )
                    if not w_err and w_metrics:
                        watchlist_charts[wsym] = (w_fig, w_fib, w_metrics)

        # Fresh notifications — dual source (holdings + watch list)
        if charts or watchlist_charts:
            render_portfolio_alerts(charts, watchlist_charts)

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
                ins_df = get_insider_data(chosen) if not chosen.endswith("-USD") else None
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
        ins_df = get_insider_data(sym) if not sym.endswith("-USD") else None
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
                    # ── Persist scan to session state so dropdown changes
                    #    don't clear the results on rerun
                    st.session_state["scanner_state"] = {
                        "results":   results,
                        "figs":      figs,
                        "universe":  universe,
                        "period":    sma_period,
                        "method":    ma_type,
                        "threshold": min_score,
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    }

        # ── Display from session state (survives reruns triggered by dropdown) ──
        scanner_state = st.session_state.get("scanner_state")
        if scanner_state:
            results = scanner_state["results"]
            figs    = scanner_state["figs"]
            cached_univ   = scanner_state.get("universe", "?")
            cached_period = scanner_state.get("period", "?")
            cached_method = scanner_state.get("method", "?")
            cached_thresh = scanner_state.get("threshold", "?")
            cached_time   = scanner_state.get("timestamp", "?")

            # Notice if current sidebar settings differ from cached scan
            settings_drifted = (
                cached_period != sma_period or
                cached_method != ma_type or
                cached_thresh != min_score
            )
            label_col, clear_col = st.columns([5,1])
            with label_col:
                if settings_drifted:
                    st.warning(
                        f"⚠️ Showing cached scan from **{cached_time}** "
                        f"({cached_method[:3]} · {cached_period}d · ≥{cached_thresh}/14). "
                        f"Sidebar settings have changed — click **Launch Scan** to refresh."
                    )
                else:
                    st.caption(
                        f"📋 Last scan: **{cached_time}** · "
                        f"{cached_method[:3]} · {cached_period}d · ≥{cached_thresh}/14"
                    )
            with clear_col:
                if st.button("🗑️ Clear", key="clear_scan", use_container_width=True):
                    del st.session_state["scanner_state"]
                    # Also clear any per-scanner AI reports
                    for k in [k for k in list(st.session_state.keys()) if k.startswith(("rpt_scanner_","btn_scanner_","clr_scanner_"))]:
                        st.session_state.pop(k, None)
                    st.rerun()

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
                view_sym  = st.selectbox("Select a ticker (sorted by score)", triggered, key="scanner_chart_pick")

                if view_sym in figs:
                    fig, fib, breakdown = _unpack_chart_entry(figs[view_sym])
                    if fig: st.plotly_chart(fig, use_container_width=True)
                    render_signal_breakdown(breakdown)
                    if not view_sym.endswith("-USD"): render_insider_section(view_sym)
                    stock_metrics = next((r for r in results if r["Ticker"]==view_sym),{})
                    ins_df = get_insider_data(view_sym) if not view_sym.endswith("-USD") else None
                    show_ai_report(f"scanner_{view_sym}", view_sym, stock_metrics,
                                   cached_period, cached_method, fib,
                                   extra_context=insider_summary(ins_df),
                                   button_label="🤖  AI Analysis for this stock")
            else:
                st.warning(f"No stocks scored ≥{cached_thresh}/14 in the cached scan. Try lowering the threshold or relaunching.")

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

# ================================================================
# MODE 4 — OPTIONS LAB (Educational Sandbox)
# ================================================================
elif mode == "📈  Options Lab":
    st.header("📈  Options Lab — Educational Sandbox")
    st.warning(
        "🎓 **Pure educational sandbox.** No real money. No broker connection. "
        "No trade recommendations — the AI here is an **analyzer**, not a recommender. "
        "Use this to learn how options strategies behave before risking real capital."
    )

    # Initialize paper options state
    if "paper_options" not in st.session_state:
        uid_pap = st.session_state.get("auth_user", "local")
        st.session_state["paper_options"] = load_paper_options_from_db(uid_pap) if SUPABASE_AVAILABLE else []

    tab_chain, tab_strategy, tab_paper = st.tabs([
        "🔍  Chain Explorer",
        "🎯  Strategy Analyzer",
        "📋  Paper Trades",
    ])

    # ──────────────────────────────────────────────────────────────
    # TAB 1: CHAIN EXPLORER
    # ──────────────────────────────────────────────────────────────
    with tab_chain:
        st.markdown("Browse a ticker's option chain — calls and puts with Greeks, "
                    f"filtered for liquidity (OI ≥{MIN_OPTION_OI}, vol ≥{MIN_OPTION_VOL}, "
                    f"spread ≤{MAX_OPTION_SPREAD*100:.0f}%, DTE {MIN_OPTION_DTE}-{MAX_OPTION_DTE}d).")

        ch_c1, ch_c2 = st.columns([3, 1])
        with ch_c1:
            chain_ticker = st.text_input(
                "Ticker", placeholder="AAPL · NVDA · SPY · QQQ",
                key="chain_ticker_input"
            ).strip().upper()
        with ch_c2:
            st.markdown("<br>", unsafe_allow_html=True)
            chain_go = st.button("Load Chain →", use_container_width=True, key="chain_load")

        if chain_go and chain_ticker:
            st.session_state["chain_loaded_ticker"] = chain_ticker

        loaded_tk = st.session_state.get("chain_loaded_ticker")
        if loaded_tk:
            with st.spinner(f"Loading {loaded_tk} options data…"):
                # Get current price
                hist_for_price = _cached_history(loaded_tk, "1mo")
                if hist_for_price.empty:
                    st.error(f"No price data for **{loaded_tk}**. Check the ticker.")
                else:
                    spot = float(hist_for_price["Close"].iloc[-1])
                    expirations = get_option_expirations(loaded_tk)

            if not hist_for_price.empty:
                if not expirations:
                    st.error(f"No option expirations between {MIN_OPTION_DTE}-{MAX_OPTION_DTE} DTE for {loaded_tk}. "
                             "This ticker may not have actively-traded options.")
                else:
                    st.success(f"✅ **{loaded_tk}** · Spot: ${spot:.2f} · {len(expirations)} valid expiration(s)")

                    ec1, ec2 = st.columns([2, 1])
                    with ec1:
                        exp_choice = st.selectbox(
                            "Expiration",
                            options=[e[0] for e in expirations],
                            format_func=lambda x: f"{x} ({next(d for e,d in expirations if e==x)}d)"
                        )
                    with ec2:
                        opt_side = st.radio("Type", ["Calls", "Puts"], horizontal=True, key="chain_side")

                    chosen_dte = next(d for e, d in expirations if e == exp_choice)
                    T_years = chosen_dte / 365.0

                    with st.spinner("Fetching chain…"):
                        calls_df, puts_df = get_option_chain_raw(loaded_tk, exp_choice)

                    if opt_side == "Calls":
                        raw_df = calls_df; opt_type_str = "call"
                    else:
                        raw_df = puts_df; opt_type_str = "put"

                    if raw_df.empty:
                        st.warning(f"No {opt_side.lower()} returned by data provider.")
                    else:
                        before_n = len(raw_df)
                        filt_df = filter_liquid_options(raw_df)
                        after_n = len(filt_df)
                        hidden = before_n - after_n
                        if filt_df.empty:
                            st.warning(f"All {before_n} contracts failed the liquidity filter. "
                                       "Try a different expiration or ticker.")
                        else:
                            enriched = enrich_chain_with_greeks(filt_df, spot, T_years, opt_type_str)

                            # Build display
                            disp = pd.DataFrame()
                            disp["Strike"]    = enriched["strike"].apply(lambda x: f"${x:.2f}")
                            disp["ITM"]       = enriched.apply(
                                lambda r: "🟢 ITM" if (
                                    (opt_type_str == "call" and r["strike"] < spot) or
                                    (opt_type_str == "put"  and r["strike"] > spot)
                                ) else "⚪ OTM", axis=1
                            )
                            disp["Bid"]       = enriched["bid"].apply(lambda x: f"${x:.2f}")
                            disp["Ask"]       = enriched["ask"].apply(lambda x: f"${x:.2f}")
                            disp["Mid"]       = enriched["mid"].apply(lambda x: f"${x:.2f}")
                            disp["Volume"]    = enriched["volume"].astype(int).apply(lambda x: f"{x:,}")
                            disp["OI"]        = enriched["openInterest"].astype(int).apply(lambda x: f"{x:,}")
                            disp["IV"]        = (enriched["impliedVolatility"] * 100).apply(lambda x: f"{x:.1f}%")
                            disp["Δ Delta"]    = enriched["delta"].apply(lambda x: f"{x:+.3f}")
                            disp["Γ Gamma"]    = enriched["gamma"].apply(lambda x: f"{x:.4f}")
                            disp["Θ Theta/d"]  = enriched["theta"].apply(lambda x: f"{x:+.3f}")
                            disp["V Vega"]    = enriched["vega"].apply(lambda x: f"{x:+.3f}")
                            disp["P Rho"]     = enriched["rho"].apply(lambda x: f"{x:+.3f}")

                            st.dataframe(disp, use_container_width=True, hide_index=True)
                            if hidden > 0:
                                st.caption(f"_Hidden: {hidden} illiquid contracts (failed OI/vol/spread filter)._")
                            st.caption(
                                "**Δ Delta**: $ gain per $1 stock move (×100 per contract) · "
                                "**Γ Gamma**: how fast delta changes · "
                                "**Θ Theta**: daily $ time decay (×100) · "
                                "**V Vega**: $ per 1% IV change (×100) · "
                                "**P Rho**: $ per 1% rate change (×100)"
                            )

    # ──────────────────────────────────────────────────────────────
    # TAB 2: STRATEGY ANALYZER
    # ──────────────────────────────────────────────────────────────
    with tab_strategy:
        st.markdown(
            "Pick a strategy and a specific contract, then get a personalised educational "
            "walkthrough that ties the trade to your underlying ticker's actual technical setup. "
            "**The AI is an analyzer, not a recommender** — it explains the trade you select, "
            "it does not tell you which trade to make."
        )

        sa_c1, sa_c2 = st.columns([2, 2])
        with sa_c1:
            sa_ticker = st.text_input(
                "Underlying Ticker", placeholder="NVDA · AAPL · SPY",
                key="sa_ticker_input"
            ).strip().upper()
        with sa_c2:
            sa_strategy = st.selectbox(
                "Strategy",
                options=list(OPTIONS_STRATEGIES.keys()),
                key="sa_strategy_pick"
            )

        # Strategy description card
        info = OPTIONS_STRATEGIES[sa_strategy]
        with st.container(border=True):
            st.markdown(f"**{sa_strategy}** · _{info['complexity']}_")
            st.markdown(info["description"])
            cinfo1, cinfo2 = st.columns(2)
            with cinfo1:
                st.markdown(f"**Max Loss:** {info['max_loss']}")
                st.markdown(f"**Best for:** {info['best_for']}")
            with cinfo2:
                st.markdown(f"**Max Profit:** {info['max_profit']}")
                st.markdown(f"**Worst for:** {info['worst_for']}")

        if sa_ticker:
            with st.spinner(f"Loading {sa_ticker} data…"):
                hist_sa = _cached_history(sa_ticker, "1mo")
                expirations_sa = get_option_expirations(sa_ticker) if not hist_sa.empty else []

            if hist_sa.empty:
                st.error(f"No price data for **{sa_ticker}**.")
            elif not expirations_sa:
                st.error(f"No liquid expirations between {MIN_OPTION_DTE}-{MAX_OPTION_DTE} DTE for {sa_ticker}.")
            else:
                spot_sa = float(hist_sa["Close"].iloc[-1])
                st.info(f"Spot: **${spot_sa:.2f}** · {len(expirations_sa)} valid expiration(s) available")

                sa_e1, sa_e2 = st.columns([2, 1])
                with sa_e1:
                    sa_exp = st.selectbox(
                        "Expiration",
                        options=[e[0] for e in expirations_sa],
                        format_func=lambda x: f"{x} ({next(d for e,d in expirations_sa if e==x)}d)",
                        key="sa_exp_pick"
                    )
                with sa_e2:
                    if sa_strategy == "Covered Call":
                        sa_cost_basis = st.number_input(
                            "Your stock cost basis ($)",
                            min_value=0.01, value=float(spot_sa), step=0.01,
                            help="What you paid per share. Used for max profit / breakeven calc."
                        )
                    else:
                        sa_cost_basis = None

                sa_dte = next(d for e, d in expirations_sa if e == sa_exp)
                T_sa = sa_dte / 365.0

                # Fetch + filter the relevant chain
                with st.spinner("Loading filtered chain…"):
                    calls_sa, puts_sa = get_option_chain_raw(sa_ticker, sa_exp)
                    chain_sa = calls_sa if info["type"] == "call" else puts_sa
                    chain_sa = filter_liquid_options(chain_sa)

                if chain_sa.empty:
                    st.warning("No liquid contracts in this expiration. Try a different one.")
                else:
                    chain_sa = enrich_chain_with_greeks(chain_sa, spot_sa, T_sa, info["type"])
                    chain_sa = chain_sa.sort_values("strike").reset_index(drop=True)

                    strike_options = chain_sa["strike"].tolist()
                    # Default to ATM strike (closest to spot)
                    default_idx = min(range(len(strike_options)),
                                       key=lambda i: abs(strike_options[i] - spot_sa))

                    sa_strike = st.selectbox(
                        "Strike",
                        options=strike_options,
                        index=default_idx,
                        format_func=lambda x: f"${x:.2f}  ({('ITM' if (info['type']=='call' and x<spot_sa) or (info['type']=='put' and x>spot_sa) else 'OTM')})",
                        key="sa_strike_pick"
                    )

                    row = chain_sa[chain_sa["strike"] == sa_strike].iloc[0]
                    sa_premium = float(row["mid"])
                    sa_iv      = float(row["impliedVolatility"])
                    greeks_sa  = {
                        "delta": float(row.get("delta", 0)),
                        "gamma": float(row.get("gamma", 0)),
                        "theta": float(row.get("theta", 0)),
                        "vega":  float(row.get("vega", 0)),
                        "rho":   float(row.get("rho", 0)),
                    }

                    metrics_sa = strategy_metrics(
                        sa_strategy, spot_sa, sa_strike, sa_premium,
                        T_sa, sa_iv, RISK_FREE_RATE,
                        stock_cost_basis=sa_cost_basis,
                    )

                    # Key trade metrics
                    st.subheader("Trade Metrics")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Premium (mid)", f"${sa_premium:.2f}")
                    m2.metric("Breakeven", f"${metrics_sa['breakeven']:.2f}")
                    mp = metrics_sa["max_profit"]
                    m3.metric("Max Profit",
                              "Unlimited" if mp == float("inf") else f"${mp:,.0f}")
                    m4.metric("Max Loss", f"${metrics_sa['max_loss']:,.0f}")
                    m5.metric("Prob of Profit", f"{metrics_sa['pop']*100:.1f}%")

                    # Capital required
                    cap = metrics_sa["capital"]
                    st.caption(
                        f"💰 **Capital required per contract:** ${cap:,.0f}  ·  "
                        f"**IV:** {sa_iv*100:.1f}%  ·  "
                        f"**DTE:** {sa_dte} days"
                    )

                    # Greeks display
                    st.subheader("Greeks (per contract = 100 shares)")
                    g1, g2, g3, g4, g5 = st.columns(5)
                    g1.metric("Δ Delta", f"{greeks_sa['delta']:+.3f}",
                              help=f"$ gain per $1 stock move: ${greeks_sa['delta']*100:+.2f}")
                    g2.metric("Γ Gamma", f"{greeks_sa['gamma']:.4f}",
                              help="How fast Delta itself changes per $1 stock move")
                    g3.metric("Θ Theta/day", f"{greeks_sa['theta']*100:+.2f}",
                              help="Daily time decay in $ — negative means losing value daily")
                    g4.metric("V Vega/1% IV", f"{greeks_sa['vega']*100:+.2f}",
                              help="$ change per 1% IV move — positive = benefits from rising IV")
                    g5.metric("P Rho/1% rate", f"{greeks_sa['rho']*100:+.3f}",
                              help="$ change per 1% interest rate move — usually small for retail trades")

                    # Payoff diagram
                    st.subheader("Payoff at Expiration")
                    payoff_fig = render_payoff_diagram(
                        sa_strategy, spot_sa, sa_strike, sa_premium,
                        metrics_sa["breakeven"], metrics_sa["max_profit"],
                        metrics_sa["max_loss"], cost_basis=sa_cost_basis,
                    )
                    st.plotly_chart(payoff_fig, use_container_width=True)

                    # AI Analysis + Paper Trade button
                    ai_col, paper_col = st.columns([3, 1])
                    with ai_col:
                        # Need to also fetch the technical setup to give AI context
                        ai_key_options = f"options_ai_{sa_ticker}_{sa_strategy}_{sa_strike}_{sa_exp}"
                        if st.button("🎓  Generate Educational Analysis", key=f"btn_{ai_key_options}",
                                     use_container_width=True):
                            with st.spinner("Loading technical context + analysing trade…"):
                                # Get the technical setup for context
                                _, tech_metrics, _, _, _, _, tech_err = fetch_technical_data(
                                    sa_ticker, sma_period, ma_type
                                )
                                if tech_err: tech_metrics = {"note": "technical data unavailable"}
                                opt_metrics_for_ai = {
                                    "strategy":   sa_strategy,
                                    "spot":       f"${spot_sa:.2f}",
                                    "strike":     f"${sa_strike:.2f}",
                                    "premium":    f"${sa_premium:.2f}",
                                    "dte":        f"{sa_dte} days",
                                    "iv":         f"{sa_iv*100:.1f}%",
                                    "breakeven":  f"${metrics_sa['breakeven']:.2f}",
                                    "max_profit": ("Unlimited" if mp == float("inf") else f"${mp:,.0f}"),
                                    "max_loss":   f"${metrics_sa['max_loss']:,.0f}",
                                    "pop":        f"{metrics_sa['pop']*100:.1f}%",
                                    "capital_required": f"${cap:,.0f}",
                                }
                                st.session_state[ai_key_options] = generate_options_ai_analysis(
                                    sa_ticker, sa_strategy, opt_metrics_for_ai, tech_metrics, greeks_sa
                                )
                        if ai_key_options in st.session_state:
                            st.markdown("---")
                            st.markdown(_sanitize_ai_markdown(st.session_state[ai_key_options]))
                            if st.button("🗑️ Clear Analysis", key=f"clr_{ai_key_options}"):
                                del st.session_state[ai_key_options]; st.rerun()

                    with paper_col:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("📋  Log Paper Trade", use_container_width=True,
                                     key=f"paper_log_{sa_ticker}_{sa_strategy}_{sa_strike}_{sa_exp}"):
                            trade_record = {
                                "ticker":        sa_ticker,
                                "strategy":      sa_strategy,
                                "strike":        sa_strike,
                                "expiration":    sa_exp,
                                "entry_premium": sa_premium,
                                "contracts":     1,
                                "status":        "open",
                                "notes":         f"DTE entry: {sa_dte}, IV: {sa_iv*100:.1f}%",
                            }
                            if SUPABASE_AVAILABLE:
                                uid_pap = st.session_state.get("auth_user","local")
                                pin_pap = st.session_state.get("auth_pin","")
                                ok = save_paper_option_to_db(uid_pap, pin_pap, trade_record)
                                if ok:
                                    st.session_state["paper_options"] = load_paper_options_from_db(uid_pap)
                                    st.success(f"✅ Paper trade logged: {sa_strategy} {sa_ticker} ${sa_strike} exp {sa_exp}")
                                    st.rerun()
                            else:
                                # Local session only
                                trade_record["id"] = len(st.session_state.get("paper_options", []))
                                trade_record["entry_date"] = datetime.utcnow().isoformat()
                                st.session_state.setdefault("paper_options", []).append(trade_record)
                                st.success(f"✅ Paper trade logged (session only — cloud not configured)")
                                st.rerun()

    # ──────────────────────────────────────────────────────────────
    # TAB 3: PAPER TRADES
    # ──────────────────────────────────────────────────────────────
    with tab_paper:
        st.markdown(
            "Track hypothetical options trades over time. Positions are revalued daily "
            "using live IV — you'll feel theta decay as your contracts age."
        )

        paper = st.session_state.get("paper_options", [])

        if not paper:
            st.info("No paper trades yet. Use the **Strategy Analyzer** tab to log your first hypothetical trade.")
        else:
            today = datetime.now().date()

            # Compute current values for open positions
            open_rows = []; closed_rows = []
            expiring_tomorrow = []

            for p in paper:
                status = p.get("status", "open")
                ticker = p.get("ticker"); strategy = p.get("strategy")
                strike = float(p.get("strike", 0))
                entry_prem = float(p.get("entry_premium", 0))
                contracts = int(p.get("contracts", 1))
                exp_str = str(p.get("expiration", ""))[:10]
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                except Exception:
                    continue
                dte_remaining = (exp_date - today).days
                info_p = OPTIONS_STRATEGIES.get(strategy, {})

                row = {
                    "id":            p.get("id"),
                    "Ticker":        ticker,
                    "Strategy":      strategy,
                    "Strike":        f"${strike:.2f}",
                    "Expiration":    exp_str,
                    "DTE":           dte_remaining,
                    "Entry Premium": f"${entry_prem:.2f}",
                    "Contracts":     contracts,
                }

                if status == "open" and dte_remaining > 0:
                    # Look up current premium from live chain
                    current_prem = None
                    try:
                        calls_p, puts_p = get_option_chain_raw(ticker, exp_str)
                        cdf = calls_p if info_p.get("type") == "call" else puts_p
                        if not cdf.empty:
                            match = cdf[cdf["strike"] == strike]
                            if not match.empty:
                                bid_p = float(match.iloc[0].get("bid", 0))
                                ask_p = float(match.iloc[0].get("ask", 0))
                                current_prem = (bid_p + ask_p) / 2.0 if (bid_p + ask_p) > 0 else None
                    except Exception:
                        current_prem = None

                    row["Current Premium"] = f"${current_prem:.2f}" if current_prem else "—"
                    if current_prem is not None:
                        # P&L direction depends on long/short
                        if info_p.get("direction") == "long":
                            pnl = (current_prem - entry_prem) * 100 * contracts
                        else:  # short — profits when premium decays
                            pnl = (entry_prem - current_prem) * 100 * contracts
                        row["P&L ($)"] = f"${pnl:+,.0f}"
                        row["P&L (%)"] = f"{pnl / (entry_prem * 100 * contracts) * 100:+.1f}%"
                    else:
                        row["P&L ($)"] = "—"; row["P&L (%)"] = "—"

                    if dte_remaining == 1:
                        expiring_tomorrow.append(row)

                    open_rows.append(row)
                elif dte_remaining <= 0 and status == "open":
                    # Auto-expire: assume contract expired at zero (worst case for long, best for short)
                    # Use spot at expiration for more accurate resolution
                    try:
                        spot_hist = _cached_history(ticker, "5d")
                        spot_at_exp = float(spot_hist["Close"].iloc[-1]) if not spot_hist.empty else strike
                    except Exception:
                        spot_at_exp = strike
                    info_p = OPTIONS_STRATEGIES.get(strategy, {})
                    if info_p.get("type") == "call":
                        intrinsic = max(spot_at_exp - strike, 0)
                    else:
                        intrinsic = max(strike - spot_at_exp, 0)

                    # Auto-close at intrinsic value
                    if p.get("id") and SUPABASE_AVAILABLE:
                        close_paper_option_in_db(p["id"], intrinsic, status="expired")
                    p["status"] = "expired"; p["close_premium"] = intrinsic
                    row["Current Premium"] = f"${intrinsic:.2f} (expired)"
                    if info_p.get("direction") == "long":
                        pnl = (intrinsic - entry_prem) * 100 * contracts
                    else:
                        pnl = (entry_prem - intrinsic) * 100 * contracts
                    row["P&L ($)"] = f"${pnl:+,.0f}"
                    row["P&L (%)"] = f"{pnl / (entry_prem * 100 * contracts) * 100:+.1f}%"
                    closed_rows.append(row)
                else:
                    # Already closed
                    close_prem = float(p.get("close_premium", 0))
                    info_p = OPTIONS_STRATEGIES.get(strategy, {})
                    if info_p.get("direction") == "long":
                        pnl = (close_prem - entry_prem) * 100 * contracts
                    else:
                        pnl = (entry_prem - close_prem) * 100 * contracts
                    row["Current Premium"] = f"${close_prem:.2f}"
                    row["P&L ($)"] = f"${pnl:+,.0f}"
                    row["P&L (%)"] = f"{pnl / (entry_prem * 100 * contracts) * 100:+.1f}%"
                    closed_rows.append(row)

            # Expiration warning banner
            if expiring_tomorrow:
                with st.container(border=True):
                    st.warning(f"⏰  **{len(expiring_tomorrow)} position(s) expire tomorrow:**")
                    for r in expiring_tomorrow:
                        st.markdown(f"- **{r['Ticker']}** {r['Strategy']} ${r['Strike']} (exp {r['Expiration']})")

            # Open positions
            if open_rows:
                st.subheader(f"🟢 Open Positions ({len(open_rows)})")
                # Display
                disp_cols = ["Ticker","Strategy","Strike","Expiration","DTE",
                             "Entry Premium","Current Premium","P&L ($)","P&L (%)","Contracts"]
                df_open = pd.DataFrame(open_rows)[disp_cols]
                st.dataframe(df_open, use_container_width=True, hide_index=True)

                # Close position UI
                with st.expander("Close a Position"):
                    close_options = {
                        f"{r['Ticker']} {r['Strategy']} ${r['Strike']} exp {r['Expiration']}": r["id"]
                        for r in open_rows if r.get("id") is not None
                    }
                    if close_options:
                        close_pick = st.selectbox("Position to close", list(close_options.keys()),
                                                  key="paper_close_pick")
                        close_prem_input = st.number_input(
                            "Close at premium ($)", min_value=0.0, step=0.01,
                            key="paper_close_premium"
                        )
                        if st.button("Confirm Close", key="paper_close_confirm"):
                            tid = close_options[close_pick]
                            if SUPABASE_AVAILABLE:
                                ok = close_paper_option_in_db(tid, close_prem_input)
                                if ok:
                                    uid_pap = st.session_state.get("auth_user","local")
                                    st.session_state["paper_options"] = load_paper_options_from_db(uid_pap)
                                    st.success(f"✅ Closed: {close_pick}")
                                    st.rerun()
                            else:
                                for p in st.session_state["paper_options"]:
                                    if p.get("id") == tid:
                                        p["status"] = "closed"
                                        p["close_premium"] = close_prem_input
                                        p["close_date"] = datetime.utcnow().isoformat()
                                st.success(f"✅ Closed: {close_pick}")
                                st.rerun()
                    else:
                        st.caption("_No positions with IDs available to close (session-only trades cannot be closed via this UI yet)._")
            else:
                st.info("No open paper trades currently.")

            # Closed/expired positions
            if closed_rows:
                with st.expander(f"🗂️  Closed & Expired Positions ({len(closed_rows)})", expanded=False):
                    disp_cols_c = ["Ticker","Strategy","Strike","Expiration",
                                   "Entry Premium","Current Premium","P&L ($)","P&L (%)","Contracts"]
                    df_closed = pd.DataFrame(closed_rows)[disp_cols_c]
                    st.dataframe(df_closed, use_container_width=True, hide_index=True)
