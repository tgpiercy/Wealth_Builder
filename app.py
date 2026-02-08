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

# --- IMPORT CONFIG ---
try:
    import titan_config as tc
except ImportError:
    st.error("⚠️ titan_config.py is missing!")
    st.stop()

# --- SETUP ---
st.set_page_config(page_title="Titan Strategy", layout="wide")
CREDENTIALS = {"dad": "1234", "son": "1234"}
PORTFOLIO_FILE = f"portfolio_{st.session_state.get('user', 'dad')}.csv"

# ==============================================================================
#  HELPER FUNCTIONS
# ==============================================================================

def safe_download(tickers, period):
    """Robust downloader that handles yfinance MultiIndex issues"""
    try:
        data = yf.download(tickers, period=period, group_by='ticker', progress=False)
        # If only 1 ticker, yfinance returns different shape. Force standard.
        if len(tickers) == 1:
            # Reconstruct dict-like structure for consistency
            return {tickers[0]: data}
        return data
    except Exception as e:
        return pd.DataFrame()

def get_price_series(data, ticker):
    """Extracts Close price series safely from complex yfinance structures"""
    try:
        if isinstance(data, pd.DataFrame):
            # Check if multi-index (Ticker, PriceType)
            if isinstance(data.columns, pd.MultiIndex):
                if ticker in data.columns.levels[0]:
                    return data[ticker]['Close']
            # Check if simple columns
            elif ticker in data.columns:
                return data[ticker]
            # Check if columns are Close only but named by ticker
            elif ticker == data.columns[0]: 
                return data['Close']
        # If dict (from group_by='ticker')
        if ticker in data:
            return data[ticker]['Close']
    except:
        pass
    return pd.Series(dtype=float)

def calc_sma(series, length): return series.rolling(window=length).mean()
def calc_rsi(series, length=14):
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
    return pd.concat([span_a, span_b], axis=1).max(axis=1)

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
    curr, prev = pivots[-1], pivots[-3]
    if curr[2] == 1: return "HH" if curr[1] > prev[1] else "LH"
    return "LL" if curr[1] < prev[1] else "HL"

def style_portfolio(styler):
    def color_pl(val):
        # Check if value is string with $ and -
        if isinstance(val, str):
            if '-' in val: return 'color: #FF4444; font-weight: bold' # Red
            elif '$' in val: return 'color: #00FF00; font-weight: bold' # Green
        # Check if numeric
        elif isinstance(val, (int, float)):
            if val < 0: return 'color: #FF4444; font-weight: bold'
            elif val > 0: return 'color: #00FF00; font-weight: bold'
        return ''
    return styler.map(color_pl, subset=["Gain/Loss ($)", "% Return", "Realized_PL"])

def style_scanner(styler):
    def color_action(val):
        if val == "BUY": return 'color: #00FF00; font-weight: bold' # Green
        if "SCOUT" in val: return 'color: #00BFFF; font-weight: bold' # Blue
        if "WATCH" in val: return 'color: #FFFF00' # Yellow
        if val == "AVOID": return 'color: #FF4444' # Red
        return ''
    return styler.map(color_action, subset=["Action"])

# ==============================================================================
#  MAIN APP
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

st.title(f"🛡️ Titan Strategy v59.1")

tab1, tab2, tab3 = st.tabs(["💼 Portfolio Manager", "🏥 Market Health", "🔍 Deep Scanner"])

# --- TAB 1: PORTFOLIO MANAGER ---
with tab1:
    # 1. Load File
    if not os.path.exists(PORTFOLIO_FILE):
        pd.DataFrame(columns=["ID","Ticker","Date","Shares","Cost_Basis","Status","Exit_Price","Realized_PL","Type"]).to_csv(PORTFOLIO_FILE, index=False)
    pf_df = pd.read_csv(PORTFOLIO_FILE)
    
    # Ensure Columns (Fixing the previous crash)
    if 'Cost_Basis' not in pf_df.columns and 'Cost' in pf_df.columns: pf_df.rename(columns={'Cost': 'Cost_Basis'}, inplace=True)
    if 'Realized_PL' not in pf_df.columns: pf_df['Realized_PL'] = 0.0

    # 2. Process Active Holdings
    active_rows = []
    total_equity = 0.0
    total_cost = 0.0
    
    open_pos = pf_df[(pf_df['Status']=='OPEN') & (pf_df['Ticker']!='CASH')]
    if not open_pos.empty:
        tickers = open_pos['Ticker'].unique().tolist()
        try:
            # Robust Download
            prices_data = safe_download(tickers, "1d")
            
            for idx, row in open_pos.iterrows():
                t = row['Ticker']
                shares = float(row['Shares'])
                cost_base = float(row['Cost_Basis'])
                
                # Extract Price
                curr_p = cost_base # Default to cost if fetch fails
                try:
                    p_series = get_price_series(prices_data, t)
                    if not p_series.empty: curr_p = float(p_series.iloc[-1])
                except: pass

                val = shares * curr_p
                cost = shares * cost_base
                total_equity += val
                total_cost += cost
                pl = val - cost
                pl_pct = (pl / cost) * 100 if cost != 0 else 0
                
                active_rows.append({
                    "Ticker": t, "Shares": shares, 
                    "Avg Cost": f"${cost_base:.2f}", "Current": f"${curr_p:.2f}",
                    "Gain/Loss ($)": f"${pl:+.2f}", "% Return": f"{pl_pct:+.2f}%"
                })
        except Exception as e:
            st.error(f"Price Fetch Error: {e}")

    # Cash Logic
    cash_df = pf_df[(pf_df['Ticker']=='CASH') & (pf_df['Status']=='OPEN')]
    cash_bal = cash_df['Shares'].sum() if not cash_df.empty else 0
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Net Worth", f"${(cash_bal + total_equity):,.2f}")
    c2.metric("Cash Available", f"${cash_bal:,.2f}")
    c3.metric("Unrealized P/L", f"${(total_equity - total_cost):+,.2f}")
    
    st.subheader("Active Holdings")
    if active_rows:
        df_active = pd.DataFrame(active_rows)
        st.dataframe(df_active.style.pipe(style_portfolio), use_container_width=True)
    else:
        st.info("No active stock positions.")

    # 3. Closed Performance
    closed = pf_df[pf_df['Status']=='CLOSED']
    if not closed.empty:
        st.divider()
        st.subheader("Closed Performance")
        wins = closed[closed['Realized_PL'] > 0]
        win_rate = len(wins) / len(closed) * 100
        total_realized = closed['Realized_PL'].sum()
        
        c1, c2 = st.columns(2)
        c1.metric("Win Rate", f"{win_rate:.0f}%")
        c2.metric("Total Realized P/L", f"${total_realized:+,.2f}")
        
        # Display Table with Styling
        closed_disp = closed[["Ticker", "Exit_Price", "Realized_PL"]].copy()
        st.dataframe(closed_disp.style.pipe(style_portfolio), use_container_width=True)

    # 4. Simple Buy
    with st.expander("➕ Record Trade"):
        with st.form("trade"):
            t_tk = st.text_input("Ticker").upper()
            t_sh = st.number_input("Shares", min_value=1)
            t_pr = st.number_input("Price", min_value=0.01)
            if st.form_submit_button("Execute Buy"):
                nid = 1
                if not pf_df.empty and 'ID' in pf_df.columns:
                     nid = pf_df['ID'].max() + 1
                
                r1 = {"ID":nid, "Ticker":t_tk, "Date":str(datetime.today().date()), "Shares":t_sh, "Cost_Basis":t_pr, "Status":"OPEN", "Type":"STOCK", "Realized_PL":0}
                r2 = {"ID":nid+1, "Ticker":"CASH", "Date":str(datetime.today().date()), "Shares":-(t_sh*t_pr), "Cost_Basis":1, "Status":"OPEN", "Type":"DEBIT", "Realized_PL":0}
                
                pf_df = pd.concat([pf_df, pd.DataFrame([r1, r2])], ignore_index=True)
                pf_df.to_csv(PORTFOLIO_FILE, index=False)
                st.success(f"Bought {t_tk}")
                st.rerun()

# --- TAB 2: MARKET HEALTH ---
with tab2:
    if st.button("Check Vitals (Run Diagnosis)"):
        with st.spinner("Analyzing Macro Environment..."):
            # Fetch Data
            tickers = ["^VIX", "SPY", "RSP"]
            data = safe_download(tickers, "3mo")
            
            # Extract Series
            vix = get_price_series(data, "^VIX")
            spy = get_price_series(data, "SPY")
            rsp = get_price_series(data, "RSP")
            
            if not spy.empty and not vix.empty:
                vix_cur = vix.iloc[-1]
                spy_cur = spy.iloc[-1]
                rsp_cur = rsp.iloc[-1]
                
                spy_sma = spy.rolling(20).mean().iloc[-1]
                rsp_sma = rsp.rolling(20).mean().iloc[-1]
                
                # Logic
                score = 0
                signals = []
                
                # 1. Volatility
                if vix_cur < 20: 
                    score += 1
                    signals.append(["Volatility (VIX)", f"{vix_cur:.2f}", "PASS (<20)", "✅"])
                else:
                    signals.append(["Volatility (VIX)", f"{vix_cur:.2f}", "FAIL (>20)", "❌"])
                    
                # 2. Trend (SPY)
                if spy_cur > spy_sma:
                    score += 1
                    signals.append(["Trend (SPY)", f"${spy_cur:.2f}", "PASS (>20SMA)", "✅"])
                else:
                    signals.append(["Trend (SPY)", f"${spy_cur:.2f}", "FAIL (<20SMA)", "❌"])

                # 3. Breadth (RSP)
                if rsp_cur > rsp_sma:
                    score += 1
                    signals.append(["Breadth (RSP)", f"${rsp_cur:.2f}", "PASS (>20SMA)", "✅"])
                else:
                    signals.append(["Breadth (RSP)", f"${rsp_cur:.2f}", "FAIL (<20SMA)", "❌"])
                
                # Output
                if score == 3:
                    st.success("🟢 MARKET STATUS: GREEN LIGHT (Aggressive)")
                elif score == 2:
                    st.warning("🟡 MARKET STATUS: YELLOW LIGHT (Caution)")
                else:
                    st.error("🔴 MARKET STATUS: RED LIGHT (Defensive/Cash)")
                
                # RESTORED TABLE
                df_sigs = pd.DataFrame(signals, columns=["Metric", "Level", "Condition", "Status"])
                st.table(df_sigs)

# --- TAB 3: DEEP SCANNER ---
with tab3:
    st.write("Full Titan Strategy Logic (ZigZag, Cloud, Impulse)")
    
    # 1. Prepare Scan List from Config
    scan_list = []
    for k,v in tc.INDICES.items(): scan_list.append((k, "INDEX", "SPY"))
    for k,v in tc.SECTORS.items(): scan_list.append((k, "SECTOR", "SPY"))
    for sec, ind_map in tc.INDUSTRY_MAP.items():
        bench = tc.BENCHMARK_CA if "Canada" in sec else "SPY"
        for k,v in ind_map.items(): scan_list.append((k, f"IND: {sec}", bench))
            
    if st.button("RUN FULL SCAN"):
        results = []
        tickers = [x[0] for x in scan_list]
        meta = {x[0]: {'cat': x[1], 'bench': x[2]} for x in scan_list}
        
        # 2. Batch Processing
        batch_size = 20
        progress = st.progress(0)
        status_txt = st.empty()
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            progress.progress(min(i / len(tickers), 1.0))
            status_txt.caption(f"Scanning batch {i}-{i+len(batch)}...")
            
            try:
                # Robust Download
                data = safe_download(batch, "1y")
                
                for tk in batch:
                    # Robust Price Extraction
                    prices = get_price_series(data, tk)
                    if prices.empty or len(prices) < 60: continue
                    
                    # 3. Calculations
                    prices = prices.dropna()
                    curr = prices.iloc[-1]
                    sma18 = calc_sma(prices, 18).iloc[-1]
                    sma50 = calc_sma(prices, 50).iloc[-1]
                    rsi5 = calc_rsi(prices, 5).iloc[-1]
                    
                    # Approximating Cloud (High/Low not always avail in flat series, use Close)
                    cloud = calc_ichimoku(prices, prices, prices).iloc[-1]
                    structure = calc_structure(prices.to_frame(name='Close'))
                    
                    # 4. Scoring Logic
                    score = 0
                    if curr > sma18: score += 1
                    if sma18 > sma50: score += 1
                    if curr > cloud: score += 1
                    
                    action = "AVOID"
                    if score >= 3:
                        if structure in ["HH", "HL"]:
                            action = "BUY" if rsi5 < 70 else "WATCH (Overbought)"
                        else:
                            action = "SCOUT (Trend OK)"
                    elif score == 2:
                        action = "WATCH"
                        
                    results.append({
                        "Category": meta[tk]['cat'],
                        "Ticker": tk,
                        "Price": curr,
                        "Action": action,
                        "Structure": structure,
                        "Score": f"{score}/3",
                        "RSI": f"{rsi5:.1f}"
                    })
                    
            except Exception as e:
                print(f"Batch Error: {e}")
                
        progress.empty()
        status_txt.empty()
        
        if results:
            df_res = pd.DataFrame(results).sort_values(["Category", "Score"], ascending=[True, False])
            # Apply Styling
            st.dataframe(df_res.style.pipe(style_scanner), use_container_width=True)
        else:
            st.error("Scan complete but NO results found. Check connection or tickers.")
