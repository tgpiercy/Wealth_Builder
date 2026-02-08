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
PORTFOLIO_FILE = "portfolio_user.csv" # Simplified for demo

# --- HELPER: BUILD DATA MAP FROM CONFIG ---
# This converts the config lists into the format the Scanner expects
DATA_MAP = {}
# 1. Indices
for k, v in tc.INDICES.items(): DATA_MAP[k] = ["00. INDICES", "SPY", v]
# 2. Sectors
for k, v in tc.SECTORS.items(): DATA_MAP[k] = ["01. SECTORS", "SPY", v]
# 3. Themes
for k, v in tc.THEMES.items(): DATA_MAP[k] = ["02. THEMES", "SPY", v]
# 4. Industries
for sec, ind_dict in tc.INDUSTRY_MAP.items():
    bench = tc.BENCHMARK_CA if "Canada" in sec else "SPY"
    cat = "04. CANADA" if "Canada" in sec else f"03. {sec} IND"
    for k, v in ind_dict.items():
        DATA_MAP[k] = [cat, bench, v]

# --- HELPER: CALCULATIONS ---
def calc_sma(s, l): return s.rolling(l).mean()
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

st.title(f"🛡️ Titan Strategy v58.0")
st.caption("Modularized Engine | Powered by titan_config.py")

tab1, tab2, tab3 = st.tabs(["💼 Portfolio", "🏥 Market Health", "🔍 Scanner"])

# --- TAB 1: PORTFOLIO ---
with tab1:
    if not os.path.exists(PORTFOLIO_FILE):
        pd.DataFrame(columns=["ID","Ticker","Shares","Cost","Date","Status"]).to_csv(PORTFOLIO_FILE, index=False)
    pf = pd.read_csv(PORTFOLIO_FILE)
    
    # Simple Buy Logic
    with st.expander("➕ Record Trade"):
        with st.form("buy"):
            tk = st.selectbox("Ticker", list(DATA_MAP.keys()))
            sh = st.number_input("Shares", min_value=1)
            pr = st.number_input("Price", min_value=0.01)
            if st.form_submit_button("Execute"):
                new_row = pd.DataFrame([{"ID": len(pf)+1, "Ticker": tk, "Shares": sh, "Cost": pr, "Date": str(datetime.now().date()), "Status": "OPEN"}])
                pf = pd.concat([pf, new_row], ignore_index=True)
                pf.to_csv(PORTFOLIO_FILE, index=False)
                st.success(f"Added {tk}")
                st.rerun()
    
    # Display
    active = pf[pf['Status']=="OPEN"]
    if not active.empty:
        st.dataframe(active)
        val = (active['Shares'] * active['Cost']).sum()
        st.metric("Total Equity Cost", f"${val:,.2f}")
    else:
        st.info("Portfolio Empty")

# --- TAB 2: HEALTH ---
with tab2:
    if st.button("Check Vitals"):
        with st.spinner("Checking VIX & SPY..."):
            vix = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
            spy = yf.Ticker("SPY").history(period="1mo")['Close']
            sma = spy.rolling(20).mean().iloc[-1]
            curr = spy.iloc[-1]
            
            c1, c2 = st.columns(2)
            c1.metric("VIX Level", f"{vix:.2f}", delta="-Bullish" if vix < 20 else "+Bearish", delta_color="inverse")
            c2.metric("SPY vs 20SMA", f"${curr:.2f}", delta=f"{curr-sma:.2f}")
            
            if vix < 20 and curr > sma: st.success("✅ MARKET STATUS: GREEN LIGHT")
            elif vix > 25 or curr < sma: st.error("🛑 MARKET STATUS: RED LIGHT (CASH/DEFENSE)")
            else: st.warning("⚠️ MARKET STATUS: CAUTION")

# --- TAB 3: SCANNER ---
with tab3:
    st.write("Scans utilizing `titan_config` universe.")
    if st.button("Run Full Scan"):
        with st.spinner(f"Scanning {len(DATA_MAP)} Tickers..."):
            results = []
            # Batch fetch for speed
            tickers = list(DATA_MAP.keys())
            # Fetch small batches to prevent memory crash
            batch_size = 20
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i+batch_size]
                try:
                    data = yf.download(batch, period="6mo", progress=False)['Close']
                    if data.empty: continue
                    
                    # Process Batch
                    for tk in batch:
                        if tk not in data.columns: continue
                        prices = data[tk].dropna()
                        if len(prices) < 50: continue
                        
                        curr = prices.iloc[-1]
                        sma20 = prices.rolling(20).mean().iloc[-1]
                        sma50 = prices.rolling(50).mean().iloc[-1]
                        rsi = calc_rsi(prices).iloc[-1]
                        
                        # Logic
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
