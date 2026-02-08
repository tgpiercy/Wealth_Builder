import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# --- SAFE IMPORTS ---
try:
    import yfinance as yf
except ImportError:
    st.error("⚠️ YFinance missing. pip install yfinance")
    st.stop()

# --- IMPORT CONFIG (Modularization) ---
try:
    import titan_config as tc
except ImportError:
    # Fallback if config file is missing (Safe Mode)
    st.warning("⚠️ titan_config.py missing. Using internal backup.")
    class tc:
        BENCHMARK_US = "SPY"
        BENCHMARK_CA = "HXT.TO"
        INDICES = {"QQQ":"Nasdaq", "SPY":"S&P500", "IWM":"Russell2000"}
        SECTORS = {"XLK":"Tech", "XLF":"Financials"}
        THEMES = {"BOTZ":"AI", "GDX":"Gold"}
        INDUSTRY_MAP = {"XLK": {"NVDA":"Nvidia", "MSFT":"Microsoft"}}

# --- CONFIGURATION ---
st.set_page_config(page_title="Titan Strategy", layout="wide")
CREDENTIALS = {"dad": "1234", "son": "1234"}

# --- AUTHENTICATION ---
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

# ==============================================================================
#  HELPER FUNCTIONS (Core Logic)
# ==============================================================================

def safe_download(tickers, period):
    try:
        data = yf.download(tickers, period=period, group_by='ticker', progress=False)
        if len(tickers) == 1: return {tickers[0]: data}
        return data
    except: return pd.DataFrame()

def get_price_series(data, ticker):
    try:
        if isinstance(data, pd.DataFrame):
            if isinstance(data.columns, pd.MultiIndex) and ticker in data.columns.levels[0]:
                return data[ticker]['Close']
            elif ticker in data.columns: return data[ticker]
            elif ticker == data.columns[0]: return data['Close']
        if ticker in data: return data[ticker]['Close']
    except: pass
    return pd.Series(dtype=float)

def calc_sma(s, l): return s.rolling(l).mean()

def calc_atr(high, low, close, length=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def calc_rsi(series, length=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_ichimoku(high, low):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return pd.concat([span_a, span_b], axis=1).max(axis=1)

def calc_structure(df, deviation_pct=0.035):
    if len(df) < 50: return "None"
    pivots = []; trend = 1; last_val = df['Close'].iloc[0]
    pivots.append((0, last_val, 1))
    
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        if trend == 1:
            if price > last_val:
                last_val = price; pivots[-1] = (i, price, 1) if pivots[-1][2]==1 else (i, price, 1)
            elif price < last_val * (1 - deviation_pct):
                trend = -1; last_val = price; pivots.append((i, price, -1))
        else:
            if price < last_val:
                last_val = price; pivots[-1] = (i, price, -1) if pivots[-1][2]==-1 else (i, price, -1)
            elif price > last_val * (1 + deviation_pct):
                trend = 1; last_val = price; pivots.append((i, price, 1))
                
    if len(pivots) < 3: return "Range"
    curr, prev = pivots[-1], pivots[-3]
    if curr[2] == 1: return "HH" if curr[1] > prev[1] else "LH"
    return "LL" if curr[1] < prev[1] else "HL"

def calc_ad(high, low, close, volume):
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)
    mfv = mfm * volume
    return mfv.cumsum()

# --- STYLING ---
def style_portfolio(styler):
    def color_pl(val):
        if isinstance(val, str):
            if '-' in val: return 'color: #FF4444; font-weight: bold'
            elif '$' in val: return 'color: #00FF00; font-weight: bold'
        return ''
    return styler.map(color_pl, subset=["Gain/Loss ($)", "% Return"])

def style_scanner(styler):
    def color_action(val):
        if val == "BUY": return 'color: #00FF00; font-weight: bold'
        if "SCOUT" in val: return 'color: #00BFFF; font-weight: bold'
        if val == "AVOID": return 'color: #FF4444'
        return ''
    return styler.map(color_action, subset=["Action"])

def style_daily_health(styler):
    def color_status(v):
        if "PASS" in v or "NORMAL" in v or "RISING" in v or "AGGRESSIVE" in v: return 'color: #00ff00; font-weight: bold'
        if "FAIL" in v or "PANIC" in v or "FALLING" in v or "CASH" in v: return 'color: #ff4444; font-weight: bold'
        return 'color: white; font-weight: bold'
    return styler.map(color_status, subset=['Status']).hide(axis='index')

# ==============================================================================
#  MAIN APP
# ==============================================================================

st.sidebar.write(f"👤 **{user.upper()}**")
if st.sidebar.button("Logout"): st.session_state.authenticated = False; st.rerun()

st.title(f"🛡️ Titan Strategy v55.9 (Modular)")
st.caption("Institutional Protocol | ATR & 3-Factor Model Active")

tab1, tab2, tab3 = st.tabs(["💼 Portfolio Manager", "🏥 Market Health", "🔍 Deep Scanner"])

# --- TAB 1: PORTFOLIO MANAGER ---
with tab1:
    if not os.path.exists(PORTFOLIO_FILE):
        pd.DataFrame(columns=["ID","Ticker","Date","Shares","Cost_Basis","Status"]).to_csv(PORTFOLIO_FILE, index=False)
    pf_df = pd.read_csv(PORTFOLIO_FILE)
    
    # Auto-Fix Columns
    if 'Cost' in pf_df.columns and 'Cost_Basis' not in pf_df.columns: pf_df.rename(columns={'Cost': 'Cost_Basis'}, inplace=True)
    if 'Cost_Basis' not in pf_df.columns: pf_df['Cost_Basis'] = 0.0

    # 1. Active Holdings
    active_rows = []; total_equity = 0.0; total_cost = 0.0
    open_pos = pf_df[(pf_df['Status']=='OPEN') & (pf_df['Ticker']!='CASH')]
    
    if not open_pos.empty:
        tickers = open_pos['Ticker'].unique().tolist()
        try:
            prices_data = safe_download(tickers, "1d")
            for idx, row in open_pos.iterrows():
                t = row['Ticker']; shares = float(row['Shares']); cost_base = float(row['Cost_Basis'])
                curr_p = cost_base
                p_ser = get_price_series(prices_data, t)
                if not p_ser.empty: curr_p = float(p_ser.iloc[-1])

                val = shares * curr_p; cost = shares * cost_base
                total_equity += val; total_cost += cost
                pl = val - cost
                pl_pct = (pl / cost) * 100 if cost != 0 else 0
                
                active_rows.append({
                    "Ticker": t, "Shares": shares, 
                    "Avg Cost": f"${cost_base:.2f}", "Current": f"${curr_p:.2f}",
                    "Gain/Loss ($)": f"${pl:+.2f}", "% Return": f"{pl_pct:+.2f}%"
                })
        except: pass

    # Cash & Metrics
    cash_bal = pf_df[(pf_df['Ticker']=='CASH') & (pf_df['Status']=='OPEN')]['Shares'].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Net Worth", f"${(cash_bal + total_equity):,.2f}")
    c2.metric("Cash Available", f"${cash_bal:,.2f}")
    c3.metric("Unrealized P/L", f"${(total_equity - total_cost):+,.2f}")
    
    if active_rows:
        st.subheader("Active Holdings")
        st.dataframe(pd.DataFrame(active_rows).style.pipe(style_portfolio), use_container_width=True)
    else: st.info("No active positions.")

    # 2. Buy Logic
    with st.expander("➕ Record Trade"):
        with st.form("trade"):
            # Build Ticker List from Config
            all_tickers = list(tc.SECTORS.keys()) + list(tc.THEMES.keys())
            for v in tc.INDUSTRY_MAP.values(): all_tickers.extend(v.keys())
            
            t_tk = st.selectbox("Ticker", sorted(list(set(all_tickers))))
            t_sh = st.number_input("Shares", min_value=1)
            t_pr = st.number_input("Price", min_value=0.01)
            
            if st.form_submit_button("Execute Buy"):
                nid = pf_df['ID'].max() + 1 if not pf_df.empty else 1
                r1 = {"ID":nid, "Ticker":t_tk, "Date":str(datetime.now().date()), "Shares":t_sh, "Cost_Basis":t_pr, "Status":"OPEN", "Type":"STOCK"}
                r2 = {"ID":nid+1, "Ticker":"CASH", "Date":str(datetime.now().date()), "Shares":-(t_sh*t_pr), "Cost_Basis":1, "Status":"OPEN", "Type":"DEBIT"}
                pf_df = pd.concat([pf_df, pd.DataFrame([r1, r2])], ignore_index=True)
                pf_df.to_csv(PORTFOLIO_FILE, index=False)
                st.success(f"Bought {t_tk}")
                st.rerun()

# --- TAB 2: MARKET HEALTH ---
with tab2:
    if st.button("Check Vitals"):
        with st.spinner("Diagnosing..."):
            data = safe_download(["^VIX", "SPY", "RSP"], "3mo")
            vix = get_price_series(data, "^VIX")
            spy = get_price_series(data, "SPY")
            
            if not spy.empty and not vix.empty:
                v_cur = vix.iloc[-1]
                s_cur = spy.iloc[-1]
                s_sma = calc_sma(spy, 20).iloc[-1]
                
                score = 0; sigs = []
                
                if v_cur < 20: 
                    score += 1; sigs.append(["Volatility", f"{v_cur:.2f}", "✅ PASS (<20)"])
                else: sigs.append(["Volatility", f"{v_cur:.2f}", "❌ FAIL (>20)"])
                    
                if s_cur > s_sma: 
                    score += 1; sigs.append(["Trend (SPY)", f"${s_cur:.2f}", "✅ PASS (>20SMA)"])
                else: sigs.append(["Trend (SPY)", f"${s_cur:.2f}", "❌ FAIL (<20SMA)"])
                
                if score == 2: st.success("🟢 MARKET GREEN (Aggressive)")
                elif score == 1: st.warning("🟡 MARKET YELLOW (Caution)")
                else: st.error("🔴 MARKET RED (Cash/Defensive)")
                
                st.table(pd.DataFrame(sigs, columns=["Metric", "Level", "Status"]).style.pipe(style_daily_health))

# --- TAB 3: DEEP SCANNER (Legacy Logic Restored) ---
with tab3:
    st.write("Full Titan Logic (ZigZag, Impulse, Cloud, ATR)")
    
    # 1. Build Scan List from Config
    scan_list = []
    for k,v in tc.INDICES.items(): scan_list.append((k, "INDEX", "SPY"))
    for k,v in tc.SECTORS.items(): scan_list.append((k, "SECTOR", "SPY"))
    for sec, ind_map in tc.INDUSTRY_MAP.items():
        bench = tc.BENCHMARK_CA if "Canada" in sec else "SPY"
        for k,v in ind_map.items(): scan_list.append((k, f"IND: {sec}", bench))
            
    if st.button("RUN FULL SCAN"):
        results = []
        tickers = [x[0] for x in scan_list]
        meta = {x[0]: {'cat': x[1]} for x in scan_list}
        
        # 2. Batch Processing
        batch_size = 20
        prog_bar = st.progress(0)
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            prog_bar.progress(min(i / len(tickers), 1.0))
            
            try:
                data = safe_download(batch, "1y")
                for tk in batch:
                    prices = get_price_series(data, tk).dropna()
                    if len(prices) < 60: continue
                    
                    # 3. Calculations
                    curr = prices.iloc[-1]
                    sma18 = calc_sma(prices, 18).iloc[-1]
                    sma50 = calc_sma(prices, 50).iloc[-1]
                    rsi = calc_rsi(prices, 5).iloc[-1]
                    cloud = calc_ichimoku(prices, prices).iloc[-1]
                    struct = calc_structure(prices.to_frame(name='Close'))
                    atr = calc_atr(prices, prices, prices).iloc[-1]
                    
                    # 4. Scoring Logic (3-Factor)
                    score = 0
                    if curr > sma18: score += 1
                    if sma18 > sma50: score += 1
                    if curr > cloud: score += 1
                    
                    action = "AVOID"
                    if score >= 3:
                        if struct in ["HH", "HL"]: action = "BUY" if rsi < 70 else "WATCH"
                        else: action = "SCOUT"
                    elif score == 2: action = "WATCH"
                    
                    results.append({
                        "Category": meta[tk]['cat'], "Ticker": tk, "Price": curr,
                        "Action": action, "Structure": struct, "Score": f"{score}/3",
                        "ATR": f"{atr:.2f}"
                    })
            except: pass
            
        prog_bar.empty()
        if results:
            df = pd.DataFrame(results).sort_values(["Category", "Score"], ascending=[True, False])
            st.dataframe(df.style.pipe(style_scanner), use_container_width=True)
        else: st.warning("No results found.")
