import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# --- SECURITY CONFIGURATION ---
CREDENTIALS = {
    "dad": "1234",
    "son": "1234"
}

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Titan Strategy", layout="wide")

# --- AUTHENTICATION LOGIC ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None

def check_login():
    username = st.session_state.username_input
    password = st.session_state.password_input
    
    if username in CREDENTIALS and CREDENTIALS[username] == password:
        st.session_state.authenticated = True
        st.session_state.user = username
    else:
        st.error("Incorrect Username or Password")

def logout():
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()

# --- LOGIN SCREEN ---
if not st.session_state.authenticated:
    st.title("🛡️ Titan Strategy Login")
    st.write("Please sign in to access your portfolio.")
    
    with st.form("login_form"):
        st.text_input("Username", key="username_input")
        st.text_input("Password", type="password", key="password_input")
        st.form_submit_button("Login", on_click=check_login)
    
    st.stop() 

# ==============================================================================
#  TITAN STRATEGY APP
# ==============================================================================

current_user = st.session_state.user
PORTFOLIO_FILE = f"portfolio_{current_user}.csv"

st.sidebar.write(f"👤 Logged in as: **{current_user.upper()}**")
if st.sidebar.button("Log Out"):
    logout()

st.title(f"🛡️ Titan Strategy v51.6 ({current_user.upper()})")
st.caption("Institutional Protocol: DLR.TO Exception Logic")

RISK_UNIT = 2300  

# --- DATA MAP ---
# Format: Ticker: [Category, Benchmark, Description]
# DLR.TO Benchmark is set to None to skip Ratio calculations.

DATA_MAP = {
    # --- 00. INDICES (LOCKED TOP) ---
    "DIA": ["00. INDICES", "SPY", "Dow Jones"],
    "QQQ": ["00. INDICES", "SPY", "Nasdaq 100"],
    "IWM": ["00. INDICES", "SPY", "Russell 2000"],
    "IWC": ["00. INDICES", "SPY", "Micro-Cap"],
    "HXT.TO": ["00. INDICES", "SPY", "TSX 60 Index"], 
    "^VIX": ["00. INDICES", "SPY", "VIX Volatility"],
    "SPY": ["00. INDICES", "SPY", "S&P 500 Base"],

    # --- 01. MATERIALS (XLB) ---
    "XLB": ["01. MATERIALS (XLB)", "SPY", "Materials Sector"],
    "GLD": ["01. MATERIALS (XLB)", "SPY", "Gold Bullion"],
    "SLV": ["01. MATERIALS (XLB)", "SPY", "Silver Bullion"],
    "GDX": ["01. MATERIALS (XLB)", "SPY", "Gold Miners"],
    "SILJ": ["01. MATERIALS (XLB)", "SPY", "Junior Silver"], 
    "COPX": ["01. MATERIALS (XLB)", "SPY", "Copper Miners"],
    "REMX": ["01. MATERIALS (XLB)", "SPY", "Rare Earths"],
    "NTR.TO": ["01. MATERIALS (XLB)", "HXT.TO", "Nutrien"],
    "TECK-B.TO": ["01. MATERIALS (XLB)", "HXT.TO", "Teck Resources"],

    # --- 02. ENERGY (XLE) ---
    "XLE": ["02. ENERGY (XLE)", "SPY", "Energy Sector"],
    "XOP": ["02. ENERGY (XLE)", "SPY", "Oil & Gas Exp"],
    "OIH": ["02. ENERGY (XLE)", "SPY", "Oil Services"],
    "MLPX": ["02. ENERGY (XLE)", "SPY", "MLP Infra"],
    "URA": ["02. ENERGY (XLE)", "SPY", "Uranium"],
    "NLR": ["02. ENERGY (XLE)", "SPY", "Nuclear"],
    "ICLN": ["02. ENERGY (XLE)", "SPY", "Clean Energy"],
    "TAN": ["02. ENERGY (XLE)", "SPY", "Solar Energy"],
    "CNQ.TO": ["02. ENERGY (XLE)", "HXT.TO", "Cdn Natural Res"],

    # --- 03. FINANCIALS (XLF) ---
    "XLF": ["03. FINANCIALS (XLF)", "SPY", "Financials Sector"],
    "KBE": ["03. FINANCIALS (XLF)", "SPY", "Bank ETF"],
    "KRE": ["03. FINANCIALS (XLF)", "SPY", "Regional Banks"],
    "IAK": ["03. FINANCIALS (XLF)", "SPY", "Insurance"],

    # --- 04. INDUSTRIALS (XLI) ---
    "XLI": ["04. INDUSTRIALS (XLI)", "SPY", "Industrials Sector"],
    "ITA": ["04. INDUSTRIALS (XLI)", "SPY", "Aerospace & Def"],
    "IYT": ["04. INDUSTRIALS (XLI)", "SPY", "Transport"],
    "PAVE": ["04. INDUSTRIALS (XLI)", "SPY", "Infrastructure"],
    "BOTZ": ["04. INDUSTRIALS (XLI)", "SPY", "Robotics & AI"],
    "CP.TO": ["04. INDUSTRIALS (XLI)", "HXT.TO", "CP KC Rail"],
    "WSP.TO": ["04. INDUSTRIALS (XLI)", "HXT.TO", "WSP Global"],
    "CSU.TO": ["04. INDUSTRIALS (XLI)", "HXT.TO", "Constellation Soft"], 

    # --- 05. TECHNOLOGY (XLK) ---
    "XLK": ["05. TECHNOLOGY (XLK)", "SPY", "Technology Sector"],
    "AAPL": ["05. TECHNOLOGY (XLK)", "QQQ", "Apple Inc"], 
    "MSFT": ["05. TECHNOLOGY (XLK)", "QQQ", "Microsoft"],
    "NVDA": ["05. TECHNOLOGY (XLK)", "QQQ", "Nvidia"],
    "SMH": ["05. TECHNOLOGY (XLK)", "SPY", "Semiconductors"],
    "XSD": ["05. TECHNOLOGY (XLK)", "SPY", "Semi SPDR"], 
    "AIQ": ["05. TECHNOLOGY (XLK)", "SPY", "Artificial Intel"],
    "IGV": ["05. TECHNOLOGY (XLK)", "SPY", "Tech Software"],
    "SMCI": ["05. TECHNOLOGY (XLK)", "QQQ", "Super Micro"],
    "DELL": ["05. TECHNOLOGY (XLK)", "QQQ", "Dell Tech"],
    "WDC": ["05. TECHNOLOGY (XLK)", "QQQ", "Western Digital"],
    "PSTG": ["05. TECHNOLOGY (XLK)", "QQQ", "Pure Storage"],
    "ANET": ["05. TECHNOLOGY (XLK)", "QQQ", "Arista Networks"],
    "SHOP.TO": ["05. TECHNOLOGY (XLK)", "HXT.TO", "Shopify"],

    # --- 06. COMM SERVICES (XLC) ---
    "XLC": ["06. COMM SVC (XLC)", "SPY", "Comm Services"],
    "META": ["06. COMM SVC (XLC)", "QQQ", "Meta Platforms"],
    "GOOGL": ["06. COMM SVC (XLC)", "QQQ", "Alphabet Inc"],

    # --- 07. HEALTH CARE (XLV) ---
    "XLV": ["07. HEALTH CARE (XLV)", "SPY", "Health Care Sector"],
    "IBB": ["07. HEALTH CARE (XLV)", "SPY", "Biotech Core"],
    "XBI": ["07. HEALTH CARE (XLV)", "SPY", "Biotech SPDR"],
    "ARKG": ["07. HEALTH CARE (XLV)", "SPY", "Genomics"],
    "PPH": ["07. HEALTH CARE (XLV)", "SPY", "Pharma"],
    "IHI": ["07. HEALTH CARE (XLV)", "SPY", "Med Devices"],

    # --- 08. CONS DISCRET (XLY) ---
    "XLY": ["08. CONS DISCRET (XLY)", "SPY", "Cons Discret Sector"],
    "AMZN": ["08. CONS DISCRET (XLY)", "QQQ", "Amazon"],
    "ITB": ["08. CONS DISCRET (XLY)", "SPY", "Home Construction"],

    # --- 09. CONS STAPLES (XLP) ---
    "XLP": ["09. CONS STAPLES (XLP)", "SPY", "Cons Staples Sector"],
    "MOO": ["09. CONS STAPLES (XLP)", "SPY", "Agribusiness"],

    # --- 10. UTILITIES / REAL ESTATE ---
    "XLU": ["10. UTIL / RE (XLU)", "SPY", "Utilities Sector"],
    "XLRE": ["10. UTIL / RE (XLU)", "SPY", "Real Estate Sector"],

    # --- 11. TREASURY / CURRENCY ---
    "IEF": ["11. BONDS/FX", "SPY", "7-10 Year Treasuries"],
    "DLR.TO": ["11. BONDS/FX", None, "USD/CAD Currency"], # EXCEPTION: No Benchmark
    
    # --- MANUAL ---
    "MANL": ["99. MANUAL", "SPY", "Manual / Spy Proxy"]
}

# --- CALCULATIONS ---
def calc_sma(series, length):
    return series.rolling(window=length).mean()

def calc_ad(high, low, close, volume):
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)
    mfv = mfm * volume
    return mfv.cumsum()

def calc_ichimoku(high, low, close):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return span_a, span_b

def calc_atr(high, low, close, length=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean()

# --- STYLING ---
def style_final(styler):
    def color_pct(val):
        if isinstance(val, str) and '%' in val:
            try:
                num = float(val.strip('%'))
                return 'color: #00ff00; font-weight: bold' if num >= 0 else 'color: #ff0000; font-weight: bold'
            except: return ''
        return ''

    return styler.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#111'), ('color', 'white'), ('font-size', '12px'), ('vertical-align', 'top')]}, 
        {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '14px'), ('padding', '8px')]}
    ]).set_properties(**{'background-color': '#222', 'color': 'white', 'border-color': '#444'})\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if v in ["BUY", "STRONG BUY"] else ('color: #00ffff; font-weight: bold' if "SCOUT" in v else ('color: #ffaa00; font-weight: bold' if v in ["SOON", "CAUTION"] else 'color: white')), subset=["Action"])\
      .map(lambda v: 'color: #ff00ff; font-weight: bold' if "SPIKE" in v else ('color: #00ff00' if "HIGH" in v else 'color: #ccc'), subset=["Volume"])\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if "STRONG" in v else ('color: #ff0000' if "WEAK" in v else 'color: #ffaa00'), subset=["A/D Breadth"])\
      .map(lambda v: 'color: #ff0000; font-weight: bold' if "FAIL" in v or "NO" in v else 'color: #00ff00', subset=["Ichimoku<br>Cloud", "Weekly<br>SMA8"])\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if "GOOD" in v else ('color: #ffaa00; font-weight: bold' if "WEAK" in v else 'color: #ff0000; font-weight: bold'), subset=["Weekly<br>Impulse"])\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if v >= 4 else ('color: #ffaa00; font-weight: bold' if v == 3 else 'color: #ff0000; font-weight: bold'), subset=["Weekly<br>Score", "Daily<br>Score"])\
      .map(lambda v: 'color: #ff0000; font-weight: bold' if "BELOW 18" in v else 'color: #00ff00', subset=["Structure"])\
      .map(color_pct, subset=["4W %", "2W %"])\
      .hide(axis='index')

def style_market(styler):
    return styler.set_table_styles([
         {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#333'), ('color', 'white')]},
         {'selector': 'td', 'props': [('text-align', 'center'), ('font-weight', 'bold')]}
    ]).map(lambda v: 'color: #00ff00' if v in ["BULLISH", "RISK ON", "CALM"] else ('color: #ffaa00
