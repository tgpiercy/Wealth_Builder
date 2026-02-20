# FILE: titan_streamlit.py
# ROLE: Live Institutional Command Deck + ADX Chop Radar
# ARCHITECTURE: Titan Ultimate Engine + Macro Plumbing

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- UI CONFIGURATION ---
st.set_page_config(page_title="TITAN Command Deck", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.big-font {font-size:30px !important; font-weight: bold;} .metric-value {font-size:24px; color:#4CAF50;}</style>", unsafe_allow_html=True)

st.title("🦅 TITAN ENGINE: LIVE COMMAND DECK")
st.markdown("---")

# --- DATA INGESTION ---
@st.cache_data(ttl=3600)
def load_data():
    tickers = ["SPY", "SSO", "IEF", "DX-Y.NYB", "^TNX", "^VIX", "HYG"]
    end_date = datetime.today()
    start_date = end_date - timedelta(days=400) 
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    close_px = data['Close'].ffill()
    high_px = data['High'].ffill()
    low_px = data['Low'].ffill()
    return close_px, high_px, low_px

with st.spinner('Downloading live market physics...'):
    close_px, high_px, low_px = load_data()

# --- ENGINE PHYSICS (VECTORIZED) ---
spy_close = close_px['SPY']
ief_close = close_px['IEF']

# 1. Trapdoor & Depth
sma_100 = spy_close.rolling(window=100).mean()
dist_to_100 = (spy_close - sma_100) / sma_100

# 2. ATR (Volatility Stop)
tr = np.maximum(high_px['SPY'] - low_px['SPY'], 
                np.maximum(abs(high_px['SPY'] - spy_close.shift(1)), 
                           abs(low_px['SPY'] - spy_close.shift(1))))
atr_14 = tr.rolling(window=14).mean()

# 3. RRG Momentum (The Whip)
rs = spy_close / ief_close
rrg_ratio = 100 * (rs / rs.rolling(window=10).mean())
rrg_mom = 100 + rrg_ratio.pct_change() * 100
rrg_whip = rrg_mom - rrg_mom.shift(1)

# 4. ADX (Trend Strength / Chop Radar)
plus_dm = high_px['SPY'].diff()
minus_dm = -low_px['SPY'].diff()
plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

tr_14_sum = tr.rolling(14).sum()
plus_di_14 = 100 * (pd.Series(plus_dm, index=spy_close.index).rolling(14).sum() / tr_14_sum)
minus_di_14 = 100 * (pd.Series(minus_dm, index=spy_close.index).rolling(14).sum() / tr_14_sum)
dx = 100 * (abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14))
adx = dx.rolling(14).mean()

# Extract Today's Data
today_spy = spy_close.iloc[-1]
today_sma = sma_100.iloc[-1]
today_dist = dist_to_100.iloc[-1] * 100
today_atr = atr_14.iloc[-1]
today_whip = rrg_whip.iloc[-1]
today_adx = adx.iloc[-1]

# Macro Data
today_dxy = close_px['DX-Y.NYB'].iloc[-1]
today_tnx = close_px['^TNX'].iloc[-1]
today_vix = close_px['^VIX'].iloc[-1]
dxy_trend = today_dxy > close_px['DX-Y.NYB'].rolling(50).mean().iloc[-1]

# --- THE STATE MACHINE (SIGNAL LOGIC) ---
current_gear = "UNKNOWN"
action_text = ""
gear_color = "gray"

if today_spy > today_sma:
    current_gear = "GEAR 2: NUCLEAR (SSO)"
    action_text = "The market is in a structural uptrend above the 100 SMA. Hold 2x Leverage."
    gear_color = "#00FF00" 
else:
    if today_dist <= -8.0 and today_whip >= 1.25:
        current_gear = "GEAR 1: PHOENIX CATCH (SPY)"
        stop_level = today_spy - (3.5 * today_atr)
        action_text = f"Knives caught. RRG Whip confirmed. Hold 1x SPY. Strict Trailing Stop at ${stop_level:.2f}"
        gear_color = "#FFA500" 
    else:
        current_gear = "GEAR 0: CASH (SLEEPING)"
        action_text = "Market is broken (Below 100 SMA). Waiting for -8% depth and institutional RRG snap."
        gear_color = "#FF0000" 

# --- DASHBOARD RENDER ---
st.markdown(f"<p class='big-font' style='color:{gear_color};'>CURRENT STATE: {current_gear}</p>", unsafe_allow_html=True)
st.write(f"**Directive:** {action_text}")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("⚙️ Engine Telemetry")
    st.metric(label="SPY Price", value=f"${today_spy:.2f}")
    st.metric(label="100-Day SMA", value=f"${today_sma:.2f}")
    st.metric(label="Distance to SMA", value=f"{today_dist:.2f}%", delta=f"{today_dist:.2f}%", delta_color="normal")

with col2:
    st.subheader("🎯 Trigger Metrics")
    st.metric(label="Depth Filter (-8% Req)", value=f"{today_dist:.2f}%")
    st.metric(label="RRG Whip (1.25 Req)", value=f"{today_whip:.2f}")
    st.metric(label="14-Day ATR", value=f"${today_atr:.2f}")

with col3:
    st.subheader("🚰 Macro Plumbing")
    st.metric(label="US Dollar ($DXY)", value=f"{today_dxy:.2f}", delta="Uptrend" if dxy_trend else "Downtrend", delta_color="inverse")
    st.metric(label="10-Year Yield", value=f"{today_tnx:.2f}%")
    st.metric(label="VIX (Fear Gauge)", value=f"{today_vix:.2f}")

with col4:
    st.subheader("📡 ADX Chop Radar")
    adx_status = "EXTREME CHOP (Whipsaw Risk)" if today_adx < 20 else ("BUILDING TREND" if today_adx < 25 else "STRONG TREND")
    st.metric(label="ADX Score (Strength)", value=f"{today_adx:.1f}")
    st.write(f"**Radar Status:** {adx_status}")
    
    if current_gear == "GEAR 2: NUCLEAR (SSO)" and today_adx < 20:
        st.warning("⚠️ ENGINE IS DEPLOYED BUT ADX IS LOW. Expect sideways paper-cuts.")
    elif current_gear == "GEAR 0: CASH (SLEEPING)":
        st.info("Capital preserved. Let the market bleed.")
    else:
        st.success("Conditions Nominal.")

st.markdown("---")

# --- VISUALIZATION (PLOTLY) ---
st.subheader("📈 The Trapdoor Matrix")
fig = go.Figure()

fig.add_trace(go.Scatter(x=spy_close.index[-200:], y=spy_close.iloc[-200:], mode='lines', name='SPY (S&P 500)', line=dict(color='white', width=2)))
fig.add_trace(go.Scatter(x=sma_100.index[-200:], y=sma_100.iloc[-200:], mode='lines', name='100 SMA Trapdoor', line=dict(color='blue', width=2, dash='dot')))

fig.update_layout(
    plot_bgcolor='#1E1E1E',
    paper_bgcolor='#1E1E1E',
    font=dict(color='white'),
    xaxis_title="Date",
    yaxis_title="Price ($)",
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
