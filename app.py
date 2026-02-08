import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# --- SAFE IMPORTS ---
try:
    import yfinance as yf
except ImportError:
    st.error("⚠️ YFinance missing.")
    st.stop()

# --- IMPORT CONFIG ---
try:
    import titan_config as tc
except ImportError:
    st.error("⚠️ titan_config.py is missing. Please create it!")
    st.stop()

# --- SETUP ---
st.set_page_config(page_title="Titan Strategy", layout="wide")
CREDENTIALS = {"dad": "1234", "son": "1234"}

# --- HELPER: BUILD DATA MAP FROM CONFIG ---
DATA_MAP = {}
for k, v in tc.INDICES.items(): DATA_MAP[k] = ["00. INDICES", "SPY", v]
for k, v in tc.SECTORS.items(): DATA_MAP[k] = ["01. SECTORS", "SPY", v]
for k, v in tc.THEMES.items(): DATA_MAP[k] = ["02. THEMES", "SPY", v]
for sec, ind_dict in tc.INDUSTRY_MAP.items():
    bench = tc.BENCHMARK_CA if "Canada" in sec else "SPY"
    cat = "04. CANADA" if "Canada" in sec else f"03. {sec} IND"
    for k, v in ind_dict.items():
        DATA_MAP[k] = [cat, bench, v]

# --- HELPER: CALCULATIONS ---
def calc_rsi(series, length=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- AUTHENTICATION ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if not st.session_state.authenticated:
    st.title("🛡️ Titan Strategy Login")
    with st.form("login"):
        u = st.text_input("Username"); p = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if u in CREDENTIALS and CREDENTIALS[u] == p:
                st.session_state.authenticated = True; st.session_state.user = u
                st.rerun()
            else: st.error("Access Denied")
    st.stop()

# --- APP UI ---
user = st.session_state.user
PORTFOLIO_FILE = f"portfolio_{user}.csv"
st.sidebar.write(f"👤 **{user.upper()}**")
if st.sidebar.button("Logout"): st.session_state.authenticated = False; st.rerun()

st.title(f"🛡️ Titan Strategy v58.1")
st.caption("Modularized Engine | Auto-Fix Enabled")

tab1, tab2, tab3 = st.tabs(["💼 Portfolio", "🏥 Market Health", "🔍 Scanner"])

# --- TAB 1: PORTFOLIO ---
with tab1:
    # 1. Load or Create File
    if not os.path.exists(PORTFOLIO_FILE):
        # Default to 'Cost_Basis' for compatibility
        pd.DataFrame(columns=["ID","Ticker","Shares","Cost_Basis","Date","Status"]).to_csv(PORTFOLIO_FILE, index=False)
    
    pf = pd.read_csv(PORTFOLIO_FILE)

    # 2. DATA REPAIR (Fixes the KeyError)
    # If the file has 'Cost' but not 'Cost_Basis', rename it
    if 'Cost' in pf.columns and 'Cost_Basis' not in pf.columns:
        pf.rename(columns={'Cost': 'Cost_Basis'}, inplace=True)
        pf.to_csv(PORTFOLIO_FILE, index=False)
    
    # If 'Cost_Basis' is missing entirely, create it
    if 'Cost_Basis' not in pf.columns:
        pf['Cost_Basis'] = 0.0

    # 3. Buy Logic
    with st.expander("➕ Record Trade"):
        with st.form("buy"):
            tk = st.selectbox("Ticker", list(DATA_MAP.keys()))
            sh = st.number_input("Shares", min_value=1)
            pr = st.number_input("Price", min_value=0.01)
            if st.form_submit_button("Execute"):
                new_row = pd.DataFrame([{
                    "ID": len(pf)+1, 
                    "Ticker": tk, 
                    "Shares": sh, 
                    "Cost_Basis": pr, # Use consistent name
                    "Date": str(datetime.now().date()), 
                    "Status": "OPEN"
                }])
                pf = pd.concat([pf, new_row], ignore_index=True)
                pf.to_csv(PORTFOLIO_FILE, index=False)
                st.success(f"Added {tk}")
                st.rerun()
    
    # 4. Display Active Positions
    active = pf[pf['Status']=="OPEN"]
    if not active.empty:
        # Calculate Value safely
        active['Shares'] = pd.to_numeric(active['Shares'], errors='coerce').fillna(0)
        active['Cost_Basis'] = pd.to_numeric(active['Cost_Basis'], errors='coerce').fillna(0)
        
        val = (active['Shares'] * active['Cost_Basis']).sum()
        
        st.metric("Total Invested Capital", f"${val:,.2f}")
        st.dataframe(active[["ID", "Ticker", "Shares", "Cost_Basis", "Date"]])
    else:
        st.info("Portfolio Empty")

# --- TAB 2: HEALTH ---
with tab2:
    if st.button("Check Vitals"):
        with st.spinner("Checking VIX & SPY..."):
            try:
                vix = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
                spy = yf.Ticker("SPY").history(period="1mo")['Close']
                if not spy.empty:
                    sma = spy.rolling(20).mean().iloc[-1]
                    curr = spy.iloc[-1]
                    
                    c1, c2 = st.columns(2)
                    c1.metric("VIX Level", f"{vix:.2f}", delta="-Bullish" if vix < 20 else "+Bearish", delta_color="inverse")
                    c2.metric("SPY vs 20SMA", f"${curr:.2f}", delta=f"{curr-sma:.2f}")
                    
                    if vix < 20 and curr > sma: st.success("✅ MARKET STATUS: GREEN LIGHT")
                    elif vix > 25 or curr < sma: st.error("🛑 MARKET STATUS: RED LIGHT (CASH/DEFENSE)")
                    else: st.warning("⚠️ MARKET STATUS: CAUTION")
                else:
                    st.error("Could not fetch SPY data.")
            except Exception as e:
                st.error(f"Data Error: {e}")

# --- TAB 3: SCANNER ---
with tab3:
    st.write("Scans utilizing `titan_config` universe.")
    if st.button("Run Full Scan"):
        with st.spinner(f"Scanning {len(DATA_MAP)} Tickers..."):
            results = []
            tickers = list(DATA_MAP.keys())
            batch_size = 20
            
            # Batch Processing
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i+batch_size]
                try:
                    data = yf.download(batch, period="6mo", progress=False)['Close']
                    if data.empty: continue
                    
                    for tk in batch:
                        if tk not in data.columns: continue
                        prices = data[tk].dropna()
                        if len(prices) < 50: continue
                        
                        curr = prices.iloc[-1]
                        sma20 = prices.rolling(20).mean().iloc[-1]
                        sma50 = prices.rolling(50).mean().iloc[-1]
                        rsi = calc_rsi(prices).iloc[-1]
                        
                        action = "WAIT"
                        if curr > sma20 and sma20 > sma50: action = "BUY"
                        if rsi < 30: action = "OVERSOLD"
                        if rsi > 70: action = "OVERBOUGHT"
                        
                        cat = DATA_MAP[tk][0]
                        name = DATA_MAP[tk][2]
                        
                        results.append({
                            "Category": cat, "Ticker": tk, "Name": name,
                            "Price": curr, "Action": action, "RSI": rsi
                        })
                except Exception as e:
                    print(f"Batch failed: {e}")
            
            if results:
                df_scan = pd.DataFrame(results).sort_values(["Category", "Ticker"])
                st.dataframe(df_scan.style.applymap(lambda x: "color: green" if x=="BUY" else "color: red" if x=="OVERBOUGHT" else "", subset=["Action"]))
            else:
                st.warning("No data returned.")
