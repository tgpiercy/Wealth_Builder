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
PORTFOLIO_FILE = f"portfolio_{st.session_state.get('user', 'dad')}.csv"

# ==============================================================================
#  HELPER FUNCTIONS (Restored Logic)
# ==============================================================================

def calc_sma(series, length): return series.rolling(window=length).mean()
def calc_atr(high, low, close, length=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(length).mean()

def calc_rsi(series, length):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_ichimoku(high, low, close):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return pd.concat([span_a, span_b], axis=1).max(axis=1) # Cloud Top

def calc_structure(df, deviation_pct=0.035):
    # ZIG ZAG LOGIC (Simplified for Speed)
    if len(df) < 50: return "None"
    pivots = []
    trend = 1
    last_val = df['Close'].iloc[0]
    pivots.append((0, last_val, 1))
    
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        if trend == 1:
            if price > last_val:
                last_val = price
                if pivots[-1][2] == 1: pivots[-1] = (i, price, 1)
                else: pivots.append((i, price, 1))
            elif price < last_val * (1 - deviation_pct):
                trend = -1
                last_val = price
                pivots.append((i, price, -1))
        else:
            if price < last_val:
                last_val = price
                if pivots[-1][2] == -1: pivots[-1] = (i, price, -1)
                else: pivots.append((i, price, -1))
            elif price > last_val * (1 + deviation_pct):
                trend = 1
                last_val = price
                pivots.append((i, price, 1))
                
    if len(pivots) < 3: return "Range"
    # Compare last pivot to previous same-type pivot
    curr, prev = pivots[-1], pivots[-3]
    if curr[2] == 1: return "HH" if curr[1] > prev[1] else "LH"
    return "LL" if curr[1] < prev[1] else "HL"

def style_portfolio(styler):
    def color_pl(val):
        if isinstance(val, str) and '$' in val:
            color = '#ff4444' if '-' in val else '#00ff00'
            return f'color: {color}; font-weight: bold'
        return ''
    return styler.map(color_pl, subset=["Gain/Loss ($)", "% Return"])

def style_scanner(styler):
    def color_action(val):
        if val == "BUY": return 'color: #00ff00; font-weight: bold'
        if "SCOUT" in val: return 'color: #00ffff; font-weight: bold'
        if val == "AVOID": return 'color: #ff4444'
        return ''
    return styler.map(color_action, subset=["Action"])

# ==============================================================================
#  MAIN APP LOGIC
# ==============================================================================

# --- AUTH ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if not st.session_state.authenticated:
    st.title("🛡️ Titan Strategy Login")
    with st.form("login"):
        u = st.text_input("Username"); p = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if u in CREDENTIALS and CREDENTIALS[u] == p:
                st.session_state.authenticated = True; st.session_state.user = u; st.rerun()
            else: st.error("Access Denied")
    st.stop()

user = st.session_state.user
PORTFOLIO_FILE = f"portfolio_{user}.csv"

# --- SIDEBAR ---
st.sidebar.title(f"👤 {user.upper()}")
if st.sidebar.button("Log Out"): st.session_state.authenticated = False; st.rerun()

st.title(f"🛡️ Titan Strategy v59.0")
st.caption("Restored Core Logic | Modular Data")

tab1, tab2, tab3 = st.tabs(["💼 Portfolio Manager", "🏥 Market Health", "🔍 Deep Scanner"])

# --- TAB 1: PORTFOLIO MANAGER ---
with tab1:
    if not os.path.exists(PORTFOLIO_FILE):
        pd.DataFrame(columns=["ID","Ticker","Date","Shares","Cost_Basis","Status","Exit_Price","Realized_PL","Type"]).to_csv(PORTFOLIO_FILE, index=False)
    pf_df = pd.read_csv(PORTFOLIO_FILE)
    
    # 1. Active Holdings Logic
    active_rows = []
    total_equity = 0.0
    total_cost = 0.0
    
    # Get Current Prices
    open_pos = pf_df[(pf_df['Status']=='OPEN') & (pf_df['Ticker']!='CASH')]
    if not open_pos.empty:
        tickers = open_pos['Ticker'].unique().tolist()
        prices = yf.download(tickers, period="1d", progress=False)['Close'].iloc[-1]
        
        for idx, row in open_pos.iterrows():
            t = row['Ticker']
            # Handle Single Ticker vs Series
            curr_p = prices[t] if isinstance(prices, pd.Series) else prices
            val = row['Shares'] * curr_p
            cost = row['Shares'] * row['Cost_Basis']
            
            total_equity += val
            total_cost += cost
            pl = val - cost
            pl_pct = (pl / cost) * 100 if cost != 0 else 0
            
            active_rows.append({
                "Ticker": t, "Shares": row['Shares'], 
                "Avg Cost": f"${row['Cost_Basis']:.2f}", "Current": f"${curr_p:.2f}",
                "Gain/Loss ($)": f"${pl:+.2f}", "% Return": f"{pl_pct:+.2f}%"
            })

    # Cash Logic
    cash_df = pf_df[(pf_df['Ticker']=='CASH')]
    cash_bal = cash_df['Shares'].sum()
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Net Worth", f"${(cash_bal + total_equity):,.2f}")
    c2.metric("Cash Available", f"${cash_bal:,.2f}")
    c3.metric("Unrealized P/L", f"${(total_equity - total_cost):+,.2f}")
    
    if active_rows:
        st.subheader("Active Holdings")
        st.dataframe(pd.DataFrame(active_rows).style.pipe(style_portfolio))
    
    # 2. Closed Performance
    closed = pf_df[pf_df['Status']=='CLOSED']
    if not closed.empty:
        st.subheader("Closed Performance")
        wins = closed[closed['Realized_PL'] > 0]
        win_rate = len(wins) / len(closed) * 100
        total_realized = closed['Realized_PL'].sum()
        
        c1, c2 = st.columns(2)
        c1.metric("Win Rate", f"{win_rate:.0f}%")
        c2.metric("Total Realized P/L", f"${total_realized:+,.2f}")
        st.dataframe(closed[["Ticker", "Exit_Price", "Realized_PL"]])

    # 3. Trade Entry (Simplified for space)
    with st.expander("➕ Record Trade"):
        with st.form("trade"):
            t_tk = st.text_input("Ticker").upper()
            t_sh = st.number_input("Shares", min_value=1)
            t_pr = st.number_input("Price", min_value=0.01)
            if st.form_submit_button("Buy"):
                nid = pf_df['ID'].max() + 1 if not pf_df.empty else 1
                # Stock Row
                r1 = {"ID":nid, "Ticker":t_tk, "Date":datetime.today(), "Shares":t_sh, "Cost_Basis":t_pr, "Status":"OPEN", "Type":"STOCK"}
                # Cash Debit
                r2 = {"ID":nid+1, "Ticker":"CASH", "Date":datetime.today(), "Shares":-(t_sh*t_pr), "Cost_Basis":1, "Status":"OPEN", "Type":"TRADE_CASH"}
                pf_df = pd.concat([pf_df, pd.DataFrame([r1, r2])], ignore_index=True)
                pf_df.to_csv(PORTFOLIO_FILE, index=False)
                st.success(f"Bought {t_tk}")
                st.rerun()

# --- TAB 2: MARKET HEALTH ---
with tab2:
    if st.button("Check Vitals"):
        with st.spinner("Analyzing Macro Environment..."):
            vix = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
            spy = yf.Ticker("SPY").history(period="3mo")['Close']
            rsp = yf.Ticker("RSP").history(period="3mo")['Close']
            
            spy_sma20 = spy.rolling(20).mean().iloc[-1]
            rsp_sma20 = rsp.rolling(20).mean().iloc[-1]
            
            score = 0
            if vix < 20: score += 1
            if spy.iloc[-1] > spy_sma20: score += 1
            if rsp.iloc[-1] > rsp_sma20: score += 1
            
            st.metric("VIX Volatility", f"{vix:.2f}", delta="Bullish" if vix<20 else "Bearish", delta_color="inverse")
            
            if score == 3:
                st.success("🟢 MARKET STATUS: AGGRESSIVE BUY (100% Risk)")
            elif score == 2:
                st.warning("🟡 MARKET STATUS: CAUTIOUS BUY (Manage Risk)")
            else:
                st.error("🔴 MARKET STATUS: DEFENSIVE / CASH (Stop Buys)")

# --- TAB 3: DEEP SCANNER ---
with tab3:
    st.write("Full Titan Strategy Logic (ZigZag, Cloud, Impulse)")
    
    # Combine all Config Data
    scan_list = []
    # Add Indices
    for k,v in tc.INDICES.items(): scan_list.append((k, "INDEX", "SPY"))
    # Add Sectors
    for k,v in tc.SECTORS.items(): scan_list.append((k, "SECTOR", "SPY"))
    # Add Industries
    for sec, ind_map in tc.INDUSTRY_MAP.items():
        bench = tc.BENCHMARK_CA if "Canada" in sec else "SPY"
        for k,v in ind_map.items(): scan_list.append((k, f"IND: {sec}", bench))
            
    if st.button("RUN FULL SCAN"):
        results = []
        tickers = [x[0] for x in scan_list]
        meta = {x[0]: {'cat': x[1], 'bench': x[2]} for x in scan_list}
        
        # Batch Fetch to prevent memory crash (25 at a time)
        batch_size = 25
        progress = st.progress(0)
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            progress.progress(i / len(tickers))
            
            try:
                data = yf.download(batch, period="1y", progress=False)['Close']
                if data.empty: continue
                
                for tk in batch:
                    if tk not in data.columns: continue
                    prices = data[tk].dropna()
                    if len(prices) < 60: continue
                    
                    # 1. Calculations
                    curr = prices.iloc[-1]
                    sma18 = calc_sma(prices, 18).iloc[-1]
                    sma50 = calc_sma(prices, 50).iloc[-1]
                    rsi5 = calc_rsi(prices, 5).iloc[-1]
                    cloud = calc_ichimoku(prices, prices, prices).iloc[-1] # Approx using close
                    structure = calc_structure(prices.to_frame())
                    
                    # 2. Logic (The Titan Brain)
                    score = 0
                    if curr > sma18: score += 1
                    if sma18 > sma50: score += 1
                    if curr > cloud: score += 1
                    
                    action = "AVOID"
                    if score >= 3:
                        if structure == "HH" or structure == "HL":
                            action = "BUY" if rsi5 < 70 else "WATCH (Extended)"
                        else:
                            action = "SCOUT (Trend OK/Struct Weak)"
                    elif score == 2:
                        action = "WATCH"
                        
                    results.append({
                        "Category": meta[tk]['cat'],
                        "Ticker": tk,
                        "Price": curr,
                        "Action": action,
                        "Structure": structure,
                        "Score": f"{score}/3",
                        "RSI (5)": f"{rsi5:.1f}"
                    })
                    
            except Exception as e:
                print(f"Batch Error: {e}")
                
        progress.empty()
        
        if results:
            df_res = pd.DataFrame(results).sort_values(["Category", "Score"], ascending=[True, False])
            st.dataframe(df_res.style.pipe(style_scanner))
        else:
            st.warning("Scan complete but no results found.")
