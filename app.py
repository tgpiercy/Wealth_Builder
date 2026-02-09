import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

# --- IMPORT MODULES ---
try:
    import titan_config as tc
    import titan_math as tm
    import titan_rrg as tr
    import titan_style as ts
except ImportError as e:
    st.error(f"⚠️ CRITICAL ERROR: Missing Module. {e}")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Titan Strategy", layout="wide")

# --- AUTHENTICATION ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None

def check_login():
    username = st.session_state.username_input
    password = st.session_state.password_input
    if username in tc.CREDENTIALS and tc.CREDENTIALS[username] == password:
        st.session_state.authenticated = True
        st.session_state.user = username
    else:
        st.error("Incorrect Username or Password")

def logout():
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()

if not st.session_state.authenticated:
    st.title("🛡️ Titan Strategy Login")
    with st.form("login_form"):
        st.text_input("Username", key="username_input")
        st.text_input("Password", type="password", key="password_input")
        st.form_submit_button("Login", on_click=check_login)
    st.stop() 

# ==============================================================================
#  TITAN STRATEGY APP (v63.5 UI Cleanup)
# ==============================================================================

current_user = st.session_state.user
PORTFOLIO_FILE = f"portfolio_{current_user}.csv"

# --- GLOBAL SETTINGS (SIDEBAR) ---
st.sidebar.write(f"👤 Logged in as: **{current_user.upper()}**")

if "is_dark" not in st.session_state:
    st.session_state.is_dark = True
st.sidebar.toggle("🌙 Dark Mode", key="is_dark")

if st.sidebar.button("Log Out"):
    logout()

st.title(f"🛡️ Titan Strategy v63.5 ({current_user.upper()})")
st.caption("Institutional Protocol: Performance Dashboard Restored")

# --- UNIFIED DATA ENGINE ---
@st.cache_data(ttl=3600) 
def fetch_master_data(ticker_list):
    unique_tickers = sorted(list(set(ticker_list))) 
    data_map = {}
    for t in unique_tickers:
        try:
            fetch_sym = "SPY" if t == "MANL" else t 
            tk = yf.Ticker(fetch_sym)
            # 10Y History for robust Moving Average calculation
            df = tk.history(period="10y", interval="1d")
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if not df.empty and 'Close' in df.columns:
                data_map[t] = df
        except: continue
    return data_map

# --- PORTFOLIO ENGINE ---
def load_portfolio():
    cols = ["ID", "Ticker", "Date", "Shares", "Cost_Basis", "Status", "Exit_Date", "Exit_Price", "Return", "Realized_PL", "SPY_Return", "Type", "Shadow_SPY"]
    if not os.path.exists(PORTFOLIO_FILE): pd.DataFrame(columns=cols).to_csv(PORTFOLIO_FILE, index=False)
    df = pd.read_csv(PORTFOLIO_FILE)
    if 'Cost' in df.columns: df.rename(columns={'Cost': 'Cost_Basis'}, inplace=True)
    if 'Cost_Basis' not in df.columns: df['Cost_Basis'] = 0.0
    if "ID" not in df.columns: df["ID"] = range(1, len(df) + 1)
    if 'Shadow_SPY' not in df.columns: df['Shadow_SPY'] = 0.0
    df['Shadow_SPY'] = pd.to_numeric(df['Shadow_SPY'], errors='coerce').fillna(0.0)
    return df

def save_portfolio(df):
    dollar_cols = ['Cost_Basis', 'Exit_Price', 'Realized_PL', 'Return', 'SPY_Return']
    for col in dollar_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
    def clean_shares(row): return round(row['Shares'], 2) if row['Ticker'] == 'CASH' else int(row['Shares'])
    if not df.empty: df['Shares'] = df.apply(clean_shares, axis=1)
    df.to_csv(PORTFOLIO_FILE, index=False)

# --- SIDEBAR: MANAGER ---
st.sidebar.header("💼 Portfolio Manager")
pf_df = load_portfolio()
cash_rows = pf_df[(pf_df['Ticker'] == 'CASH') & (pf_df['Status'] == 'OPEN')]
current_cash = cash_rows['Shares'].sum() if not cash_rows.empty else 0.0
st.sidebar.metric("Cash Available", f"${current_cash:,.2f}")

tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["🟢 Buy", "🔴 Sell", "💵 Cash", "🧮 Calc", "🛠️ Fix"])

with tab1:
    with st.form("buy_trade"):
        b_tick = st.selectbox("Ticker", list(tc.DATA_MAP.keys()))
        b_date = st.date_input("Buy Date")
        b_shares = st.number_input("Shares", min_value=1, value=100)
        b_price = st.number_input("Price", min_value=0.01, value=100.00)
        if st.form_submit_button("Execute Buy"):
            new_id = 1 if pf_df.empty else pf_df["ID"].max() + 1
            pf_df = pd.concat([pf_df, pd.DataFrame([{"ID": new_id, "Ticker": b_tick, "Date": b_date, "Shares": b_shares, "Cost_Basis": b_price, "Status": "OPEN", "Type": "STOCK", "Shadow_SPY": 0.0}])], ignore_index=True)
            if current_cash >= (b_shares * b_price):
                 pf_df = pd.concat([pf_df, pd.DataFrame([{"ID": pf_df["ID"].max()+1, "Ticker": "CASH", "Date": b_date, "Shares": -(b_shares * b_price), "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRADE_CASH", "Shadow_SPY": 0.0}])], ignore_index=True)
            save_portfolio(pf_df); st.success(f"Bought {b_tick}"); st.rerun()

with tab2:
    open_trades = pf_df[(pf_df['Status'] == 'OPEN') & (pf_df['Ticker'] != 'CASH')]
    if not open_trades.empty:
        trade_map = {}
        opts = []
        for idx, row in open_trades.iterrows():
            label = f"ID:{row['ID']} | {row['Ticker']} | {int(row['Shares'])} shares | {row['Date']}"
            trade_map[label] = {'id': row['ID'], 'max_shares': int(row['Shares']), 'idx': idx}
            opts.append(label)
        selected_trade_str = st.selectbox("Select Position", opts)
        if selected_trade_str:
            sel_data = trade_map[selected_trade_str]; sel_id = sel_data['id']; max_qty = sel_data['max_shares']
            with st.form("sell_trade"):
                s_shares = st.number_input("Shares to Sell", min_value=1, max_value=max_qty, value=max_qty, step=1)
                s_date = st.date_input("Date"); s_price = st.number_input("Price", 0.01, value=100.00)
                if st.form_submit_button("Execute Sell"):
                    row_idx = sel_data['idx']; buy_price = float(pf_df.at[row_idx, 'Cost_Basis']); buy_date_str = pf_df.at[row_idx, 'Date']
                    ret_pct = ((s_price - buy_price) / buy_price) * 100; pl_dollars = (s_price - buy_price) * s_shares
                    cash_id = pf_df["ID"].max() + 1
                    cash_row = pd.DataFrame([{"ID": cash_id, "Ticker": "CASH", "Date": s_date, "Shares": (s_price * s_shares), "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRADE_CASH", "Shadow_SPY": 0.0}])
                    pf_df = pd.concat([pf_df, cash_row], ignore_index=True)
                    if s_shares < max_qty:
                        pf_df.at[row_idx, 'Shares'] -= s_shares
                        new_id = pf_df["ID"].max() + 1
                        new_closed_row = pd.DataFrame([{"ID": new_id, "Ticker": pf_df.at[row_idx, 'Ticker'], "Date": buy_date_str, "Shares": s_shares, "Cost_Basis": buy_price, "Status": "CLOSED", "Exit_Date": s_date, "Exit_Price": s_price, "Return": ret_pct, "Realized_PL": pl_dollars, "SPY_Return": 0.0, "Type": "STOCK", "Shadow_SPY": 0.0}])
                        pf_df = pd.concat([pf_df, new_closed_row], ignore_index=True)
                    else:
                        pf_df.at[row_idx, 'Status'] = 'CLOSED'; pf_df.at[row_idx, 'Exit_Date'] = s_date; pf_df.at[row_idx, 'Exit_Price'] = s_price; pf_df.at[row_idx, 'Return'] = ret_pct; pf_df.at[row_idx, 'Realized_PL'] = pl_dollars
                    save_portfolio(pf_df); st.success(f"Sold {s_shares} shares. P&L: ${pl_dollars:+.2f}"); st.rerun()
    else: st.info("No Open Positions")

with tab3:
    with st.form("cash"):
        op = st.radio("Op", ["Deposit", "Withdraw"]); amt = st.number_input("Amt", 100.00); dt = st.date_input("Date")
        if st.form_submit_button("Execute"):
            shares = 0.0
            try: 
                spy = yf.Ticker("SPY").history(start=dt, end=dt+timedelta(days=5))
                if not spy.empty: shares = amt / spy['Close'].iloc[0]
            except: pass
            final = amt if op == "Deposit" else -amt; s_shares = shares if op == "Deposit" else -shares
            pf_df = pd.concat([pf_df, pd.DataFrame([{"ID": pf_df["ID"].max()+1, "Ticker": "CASH", "Date": dt, "Shares": final, "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRANSFER", "Shadow_SPY": s_shares}])], ignore_index=True)
            save_portfolio(pf_df); st.success("Done"); st.rerun()

with tab4:
    st.subheader("Calculator"); RISK_UNIT_BASE = st.number_input("Risk Unit", 100, value=2300); tk = st.text_input("Ticker").upper()
    if tk:
        try:
            d = yf.Ticker(tk).history("1mo"); c = d['Close'].iloc[-1]; atr = tm.calc_atr(d['High'], d['Low'], d['Close']).iloc[-1]
            stop = tm.round_to_03_07(c - 2.618*atr)
            if c > stop:
                sh = int(RISK_UNIT_BASE / (c - stop)) 
                st.info(f"Entry: ${c:.2f} | Stop: ${stop:.2f} | Shares: {sh} | Cap: ${sh*c:,.0f}")
        except: st.error("Error")

with tab5:
    st.write("### 🛠️ Data Management")
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "rb") as file:
            st.download_button("Download Portfolio CSV", file, PORTFOLIO_FILE, "text/csv")
    else: st.warning("No portfolio file found.")
    st.write("---")
    uploaded_file = st.file_uploader("Restore .csv", type=["csv"])
    if uploaded_file is not None and st.button("CONFIRM RESTORE"):
        try:
            pd.read_csv(uploaded_file).to_csv(PORTFOLIO_FILE, index=False)
            st.success("Data Restored!"); st.rerun()
        except: st.error("Error")
    st.write("---")
    
    action_type = st.radio("Advanced Tools", ["Delete Trade", "Edit Trade", "⚠️ FACTORY RESET", "Rebuild Benchmark History"])
    if action_type == "⚠️ FACTORY RESET" and st.button("CONFIRM RESET"):
        if os.path.exists(PORTFOLIO_FILE): os.remove(PORTFOLIO_FILE)
        st.success("Reset!"); st.rerun()
    elif action_type == "Rebuild Benchmark History" and st.button("RUN REBUILD"):
         with st.spinner("Rebuilding..."):
             try:
                 spy_hist = yf.Ticker("SPY").history(period="10y")
                 for idx, row in pf_df.iterrows():
                     if row['Type'] == 'TRANSFER' and row['Ticker'] == 'CASH':
                         t_date = pd.to_datetime(row['Date'])
                         idx_loc = spy_hist.index.searchsorted(t_date)
                         price = spy_hist.iloc[idx_loc]['Close'] if idx_loc < len(spy_hist) else spy_hist.iloc[-1]['Close']
                         if price > 0: pf_df.at[idx, 'Shadow_SPY'] = float(row['Shares']) / price
                 save_portfolio(pf_df); st.success("Done!"); st.rerun()
             except: st.error("Error")

# --- HTML CACHING ---
@st.cache_data
def generate_scanner_html(results_df):
    if results_df.empty: return ""
    return results_df.style.pipe(ts.style_final).to_html(escape=False)

# --- MAIN EXECUTION ---
if "run_analysis" not in st.session_state: st.session_state.run_analysis = False
if st.button("RUN ANALYSIS", type="primary"): st.session_state.run_analysis = True; st.rerun()

if st.session_state.run_analysis:
    if st.button("⬅️ Back to Menu"): st.session_state.run_analysis = False; st.rerun()
    
    # --- UNIFIED LIST GENERATION ---
    pf_tickers = pf_df['Ticker'].unique().tolist() if not pf_df.empty else []
    pf_tickers = [x for x in pf_tickers if x != "CASH"]
    all_tickers = list(tc.DATA_MAP.keys()) + pf_tickers + list(tc.RRG_SECTORS.keys()) + list(tc.RRG_INDICES.keys()) + list(tc.RRG_THEMES.keys()) + ["CAD=X", "IEF", "RSP", "SPY", "^VIX"] 
    for v in tc.RRG_INDUSTRY_MAP.values(): all_tickers.extend(list(v.keys()))
    
    # --- MASTER DATA FETCH & RRG CALC ---
    with st.spinner('Downloading Unified Market Data...'):
        master_data = fetch_master_data(all_tickers)
        rrg_snapshot = tr.generate_full_rrg_snapshot(master_data, "SPY")

    # --- STATE-BASED NAVIGATION ---
    mode = st.radio("Navigation", ["Scanner", "Sector Rotation"], horizontal=True, key="main_nav")
    
    if mode == "Scanner":
        # 1. HOLDINGS & PERFORMANCE CALCULATION
        open_pos = pf_df[(pf_df['Status'] == 'OPEN') & (pf_df['Ticker'] != 'CASH')]
        eq_val = 0.0; total_cost_basis = 0.0; pf_rows = []
        
        if not open_pos.empty:
            for idx, row in open_pos.iterrows():
                t = row['Ticker']; shares = row['Shares']; cost = row['Cost_Basis']
                curr_price = cost 
                if t in master_data and not master_data[t].empty: curr_price = master_data[t]['Close'].iloc[-1]
                pos_val = shares * curr_price; eq_val += pos_val
                total_cost_basis += (shares * cost)
                pf_rows.append({"Ticker": t, "Shares": int(shares), "Avg Cost": f"${cost:.2f}", "Current": f"${curr_price:.2f}", "Gain/Loss ($)": f"${(pos_val - (shares * cost)):+.2f}", "% Return": f"{((curr_price - cost) / cost) * 100:+.2f}%", "Audit Action": "HOLD"})
        
        total_net_worth = current_cash + eq_val
        cad_data = master_data.get("CAD=X")
        if cad_data is not None and not cad_data.empty:
            rate = cad_data['Close'].iloc[-1]
            if rate < 1.0: rate = 1.0 / rate 
            cad_rate = rate
        else:
            cad_rate = 1.40 
            
        total_nw_cad = total_net_worth * cad_rate
        open_pl_val = eq_val - total_cost_basis
        open_pl_cad = open_pl_val * cad_rate
        
        # SHADOW SPY BENCHMARK
        shadow_spy_qty = pf_df['Shadow_SPY'].sum() if not pf_df.empty else 0.0
        spy_price = master_data['SPY']['Close'].iloc[-1] if 'SPY' in master_data else 0.0
        shadow_equity = shadow_spy_qty * spy_price
        alpha_dollars = total_net_worth - shadow_equity
        
        # --- SECTION 1: PERFORMANCE DASHBOARD ---
        st.subheader("📊 Performance Dashboard")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(f"Net Worth (CAD)", f"${total_nw_cad:,.2f}", ts.fmt_delta(open_pl_cad))
        c2.metric("Net Worth (USD)", f"${total_net_worth:,.2f}", ts.fmt_delta(open_pl_val))
        c3.metric("Benchmark (Shadow SPY)", f"${shadow_equity:,.2f}", f"{alpha_dollars:+.2f} Alpha")
        c4.metric("Cash", f"${current_cash:,.2f}")
        c5.metric("Equity", f"${eq_val:,.2f}")
        st.write("---")

        # --- SECTION 2: CURRENT HOLDINGS ---
        st.subheader("💼 Current Holdings")
        if pf_rows: st.markdown(pd.DataFrame(pf_rows).style.pipe(ts.style_portfolio).to_html(), unsafe_allow_html=True)
        else: st.info("No active trades.")
        st.write("---")

        # --- SECTION 3: MARKET HEALTH ---
        spy = master_data.get("SPY"); vix = master_data.get("^VIX"); rsp = master_data.get("RSP")
        mkt_score = 0; h_rows = []
        if spy is not None:
            if vix is not None:
                v = vix.iloc[-1]['Close']
                s = "<span style='color:#00ff00'>NORMAL</span>" if v < 17 else ("<span style='color:#ffaa00'>CAUTIOUS</span>" if v < 20 else "<span style='color:#ff4444'>PANIC</span>")
                mkt_score += 9 if v < 17 else (6 if v < 20 else (3 if v < 25 else 0))
                h_rows.append({"Indicator": f"VIX Level ({v:.2f})", "Status": s})
            
            sc = spy.iloc[-1]['Close']; s18 = tm.calc_sma(spy['Close'], 18).iloc[-1]; s8 = tm.calc_sma(spy['Close'], 8).iloc[-1]
            if sc > s18: mkt_score += 1
            h_rows.append({"Indicator": "SPY Price > SMA18", "Status": "<span style='color:#00ff00'>PASS</span>" if sc > s18 else "<span style='color:#ff4444'>FAIL</span>"})
            
            if s18 >= tm.calc_sma(spy['Close'], 18).iloc[-2]: mkt_score += 1
            h_rows.append({"Indicator": "SPY SMA18 Rising", "Status": "<span style='color:#00ff00'>RISING</span>" if s18 >= tm.calc_sma(spy['Close'], 18).iloc[-2] else "<span style='color:#ff4444'>FALLING</span>"})
            
            if s8 > tm.calc_sma(spy['Close'], 8).iloc[-2]: mkt_score += 1
            h_rows.append({"Indicator": "SPY SMA8 Rising", "Status": "<span style='color:#00ff00'>RISING</span>" if s8 > tm.calc_sma(spy['Close'], 8).iloc[-2] else "<span style='color:#ff4444'>FALLING</span>"})

            if rsp is not None:
                rc = rsp.iloc[-1]['Close']; r18 = tm.calc_sma(rsp['Close'], 18).iloc[-1]
                if rc > r18: mkt_score += 1
                h_rows.append({"Indicator": "RSP Price > SMA18", "Status": "<span style='color:#00ff00'>PASS</span>" if rc > r18 else "<span style='color:#ff4444'>FAIL</span>"})
            
            col = "#00ff00" if mkt_score >= 8 else ("#ffaa00" if mkt_score >= 5 else "#ff4444")
            msg = "AGGRESSIVE" if mkt_score >= 10 else ("CAUTIOUS" if mkt_score >= 8 else "DEFENSIVE")
            
            BASE_RISK = 2300 
            risk_per_trade = BASE_RISK if mkt_score >= 8 else (BASE_RISK * 0.5 if mkt_score >= 5 else 0)
            
            h_rows.append({"Indicator": "TOTAL SCORE", "Status": f"<span style='color:{col}'><b>{mkt_score}/11</b></span>"})
            h_rows.append({"Indicator": "STRATEGY MODE", "Status": f"<span style='color:{col}'><b>{msg}</b></span>"})
            st.subheader("🏥 Daily Market Health")
            st.markdown(pd.DataFrame(h_rows).style.pipe(ts.style_daily_health).to_html(escape=False), unsafe_allow_html=True)
            st.write("---")

        # --- SECTION 4: SCANNER LOOP (GW2 SCORECARD PARITY) ---
        results = []
        scan_list = list(set(list(tc.DATA_MAP.keys()) + pf_tickers))
        analysis_db = {}
        
        for t in scan_list:
            if t not in master_data or len(master_data[t]) < 50: continue
            df = master_data[t].copy()
            
            # --- DAILY CALCULATIONS ---
            df['SMA8'] = tm.calc_sma(df['Close'], 8)
            df['SMA18'] = tm.calc_sma(df['Close'], 18)
            df['SMA40'] = tm.calc_sma(df['Close'], 40)
            df['AD'] = tm.calc_ad(df['High'], df['Low'], df['Close'], df['Volume'])
            ad_sma18 = tm.calc_sma(df['AD'], 18); ad_sma40 = tm.calc_sma(df['AD'], 40)
            df['VolSMA'] = tm.calc_sma(df['Volume'], 18)
            df['RSI5'] = tm.calc_rsi(df['Close'], 5); df['RSI20'] = tm.calc_rsi(df['Close'], 20)
            
            # --- RS CALC (DAILY) ---
            bench_ticker = "SPY"
            if t in tc.DATA_MAP and tc.DATA_MAP[t][1]: bench_ticker = tc.DATA_MAP[t][1]
            
            rs_score_ok = False
            if bench_ticker in master_data:
                bench_series = master_data[bench_ticker]['Close']
                common_idx = df.index.intersection(bench_series.index)
                rs_series = df.loc[common_idx, 'Close'] / bench_series.loc[common_idx]
                rs_sma18 = tm.calc_sma(rs_series, 18)
                
                # SCORECARD LOGIC: In Zone + Not Down (Strict 1-bar check)
                if len(rs_series) > 2 and len(rs_sma18) > 2:
                    curr_rs = rs_series.iloc[-1]; curr_rs_sma = rs_sma18.iloc[-1]
                    lower_band = curr_rs_sma - (abs(curr_rs_sma) * 0.005) # 0.5% tolerance
                    rs_not_down = curr_rs_sma >= rs_sma18.iloc[-2] # Strict 1-bar stability check
                    rs_in_zone = curr_rs >= lower_band
                    
                    if rs_in_zone and rs_not_down: rs_score_ok = True
            else:
                rs_score_ok = True 
            
            # --- WEEKLY CALCULATIONS ---
            df_w = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
            df_w.dropna(inplace=True)
            if len(df_w) < 5: continue
            
            df_w['SMA8'] = tm.calc_sma(df_w['Close'], 8)
            df_w['SMA18'] = tm.calc_sma(df_w['Close'], 18)
            df_w['SMA40'] = tm.calc_sma(df_w['Close'], 40)
            df_w['AD'] = tm.calc_ad(df_w['High'], df_w['Low'], df_w['Close'], df_w['Volume'])
            w_ad_sma18 = tm.calc_sma(df_w['AD'], 18)
            w_ad_sma40 = tm.calc_sma(df_w['AD'], 40)
            span_a, span_b = tm.calc_ichimoku(df_w['High'], df_w['Low'], df_w['Close'])
            df_w['Cloud_Top'] = pd.concat([span_a, span_b], axis=1).max(axis=1)

            dc = df.iloc[-1]; wc = df_w.iloc[-1]
            inst_activity = tm.calc_structure(df)
            
            # --- DAILY SCORE (5 Pts - STRICT) ---
            # 1. Breadth (A/D)
            ad_score_ok = False
            if len(ad_sma18) > 2:
                ad_val = df['AD'].iloc[-1]; ad18 = ad_sma18.iloc[-1]
                ad_lower_band = ad18 - (abs(ad18) * 0.005)
                ad_not_down = ad18 >= ad_sma18.iloc[-2] # Strict 1-bar stability check
                ad_in_zone = ad_val >= ad_lower_band
                
                if ad_in_zone and ad_not_down: ad_score_ok = True
            
            d_chk = 0
            if ad_score_ok: d_chk += 1
            if rs_score_ok: d_chk += 1
            if dc['Close'] > df['SMA18'].iloc[-1]: d_chk += 1 # Trend
            if tm.calc_rising(df['SMA18'], 2): d_chk += 1     # Momentum (Strict 2-bar rise)
            if df['SMA18'].iloc[-1] > df['SMA40'].iloc[-1]: d_chk += 1 # Structure

            # --- WEEKLY SCORE (5 Pts - STRICT) ---
            # 1. Breadth (Weekly A/D)
            w_ad_score_ok = False
            if len(w_ad_sma18) > 2:
                w_ad_val = df_w['AD'].iloc[-1]; w_ad18 = w_ad_sma18.iloc[-1]
                w_ad_lower = w_ad18 - (abs(w_ad18) * 0.005)
                w_ad_not_down = w_ad18 >= w_ad_sma18.iloc[-2] # Strict 1-bar stability check
                w_ad_in_zone = w_ad_val >= w_ad_lower
                
                if w_ad_in_zone and w_ad_not_down: w_ad_score_ok = True
            
            # 2. RS (Weekly)
            w_rs_score_ok = False
            if bench_ticker in master_data:
                bench_w = master_data[bench_ticker].resample('W-FRI').last()['Close']
                common_w = df_w.index.intersection(bench_w.index)
                w_rs = df_w.loc[common_w, 'Close'] / bench_w.loc[common_w]
                w_rs_sma18 = tm.calc_sma(w_rs, 18)
                if len(w_rs) > 2 and len(w_rs_sma18) > 2:
                    w_curr_rs = w_rs.iloc[-1]; w_curr_rs_sma = w_rs_sma18.iloc[-1]
                    w_lower_band = w_curr_rs_sma - (abs(w_curr_rs_sma) * 0.005)
                    w_rs_not_down = w_curr_rs_sma >= w_rs_sma18.iloc[-2] # Strict 1-bar stability check
                    w_rs_in_zone = w_curr_rs >= w_lower_band
                    
                    if w_rs_in_zone and w_rs_not_down: w_rs_score_ok = True
            else: w_rs_score_ok = True

            w_score = 0
            if w_ad_score_ok: w_score += 1
            if w_rs_score_ok: w_score += 1
            if wc['Close'] > wc['SMA18']: w_score += 1 # Trend
            if tm.calc_rising(df_w['SMA18'], 2): w_score += 1 # Momentum (Strict 2-bar rise)
            if wc['SMA18'] > wc['SMA40']: w_score += 1 # Structure

            # --- DECISION LOGIC ---
            w_pulse = "GOOD" if (wc['Close'] > wc['SMA18']) and (dc['Close'] > df['SMA18'].iloc[-1]) else "NO"
            
            vol_msg = "NORMAL"
            if df['Volume'].iloc[-1] > (df['VolSMA'].iloc[-1] * 1.5): vol_msg = "SPIKE (Live)"
            elif df['Volume'].iloc[-2] > (df['VolSMA'].iloc[-2] * 1.5): vol_msg = "SPIKE (Prev)"
            elif df['Volume'].iloc[-1] > df['VolSMA'].iloc[-1]: vol_msg = "HIGH (Live)"
            
            r5 = df['RSI5'].iloc[-1]; r20 = df['RSI20'].iloc[-1] if not pd.isna(df['RSI20'].iloc[-1]) else 50
            r5_prev = df['RSI5'].iloc[-2]; is_rising = r5 > r5_prev
            
            final_inst_msg = inst_activity
            if "SPIKE" in vol_msg:
                if inst_activity == "HL": final_inst_msg = "ACCUMULATION (HL)" if is_rising else final_inst_msg
                if inst_activity == "HH": final_inst_msg = "BREAKOUT (HH)" if is_rising else "DISTRIBUTION (HH)"
                if inst_activity == "LL": final_inst_msg = "CAPITULATION (LL)" if is_rising else "LIQUIDATION (LL)"
                if inst_activity == "LH": final_inst_msg = "SELLING (LH)"

            decision = "AVOID"; reason = "Low Score"
            if w_score >= 4:
                if d_chk == 5: decision = "BUY"; reason = "Score 5/5" 
                elif d_chk == 4: decision = "SCOUT"; reason = "D-Score 4"
                elif d_chk == 3: decision = "SCOUT"; reason = "Dip Buy"
                else: decision = "WATCH"; reason = "Daily Weak"
            else: decision = "AVOID"; reason = "Weekly Weak"

            # Re-introduced Cloud check as a hard filter instead of score
            if not (wc['Close'] > wc['Cloud_Top']): decision = "AVOID"; reason = "Below Cloud"
            elif "NO" in w_pulse: decision = "AVOID"; reason = "Impulse NO"
            if risk_per_trade == 0 and "BUY" in decision: decision = "CAUTION"; reason = "VIX Lock"

            atr = tm.calc_atr(df['High'], df['Low'], df['Close']).iloc[-1]
            raw_stop = dc['Close'] - (2.618 * atr); smart_stop_val = tm.round_to_03_07(raw_stop)
            stop_dist = dc['Close'] - smart_stop_val; stop_pct = (stop_dist / dc['Close']) * 100 if dc['Close'] else 0
            
            if r5 >= r20:
                n_c = "#00BFFF" if (r20 > 50 and is_rising) else ("#00FF00" if is_rising else "#FF4444")
            else:
                n_c = "#FFA500" if r20 > 50 else "#FF4444"
            a_c = "#00FF00" if is_rising else "#FF4444"; arrow = "↑" if is_rising else "↓"
            rsi_msg = f"<span style='color:{n_c}'><b>{int(r5)}/{int(r20)}</b></span> <span style='color:{a_c}'><b>{arrow}</b></span>"
            
            rrg_phase = rrg_snapshot.get(t, "unknown").upper()
            if "WEAKENING" in rrg_phase and "BUY" in decision: decision = "CAUTION"; reason = "Rotation Weak"
            
            analysis_db[t] = {"Decision": decision, "Reason": reason, "Price": dc['Close'], "Stop": smart_stop_val, "StopPct": stop_pct, "RRG": rrg_phase, "W_SMA8_Pass": (wc['Close']>wc['SMA8']), "W_Pulse": w_pulse, "W_Score": w_score, "D_Score": d_chk, "D_Chk_Price": (dc['Close'] > df['SMA18'].iloc[-1]), "W_Cloud": (wc['Close']>wc['Cloud_Top']), "AD_Pass": ad_score_ok, "Vol_Msg": vol_msg, "RSI_Msg": rsi_msg, "Inst_Act": final_inst_msg}

        for t in scan_list:
            cat_name = tc.DATA_MAP.get(t, ["OTHER"])[0]
            if "99. DATA" in cat_name: continue
            if t not in analysis_db: continue
            is_scanner = t in tc.DATA_MAP and (tc.DATA_MAP[t][0] != "BENCH" or t in ["DIA", "QQQ", "IWM", "IWC", "HXT.TO"])
            if not is_scanner: continue
            
            db = analysis_db[t]
            final_decision = db['Decision']; final_reason = db['Reason']
            if cat_name in tc.SECTOR_PARENTS:
                parent = tc.SECTOR_PARENTS[cat_name]
                if parent in analysis_db and "AVOID" in analysis_db[parent]['Decision']:
                    if t != parent: final_decision = "AVOID"; final_reason = "Sector Lock"

            is_blue_spike = ("#00BFFF" in db['RSI_Msg']) and ("SPIKE" in db['Vol_Msg'])
            final_risk = risk_per_trade / 3 if "SCOUT" in final_decision else risk_per_trade
            if is_blue_spike: final_risk = risk_per_trade
            
            if "AVOID" in final_decision and not is_blue_spike: disp_stop = ""; disp_shares = ""
            else:
                shares = int(final_risk / (db['Price'] - db['Stop'])) if (db['Price'] - db['Stop']) > 0 else 0
                disp_stop = f"${db['Stop']:.2f} (-{db['StopPct']:.1f}%)"; disp_shares = f"{shares} shares"

            row = {
                "Sector": cat_name, "Ticker": t, "Rank": (0 if "00." in cat_name else 1), "Rotation": db['RRG'],
                "Weekly<br>SMA8": "PASS" if db['W_SMA8_Pass'] else "FAIL", "Weekly<br>Impulse": db['W_Pulse'], 
                "Weekly<br>Score": db['W_Score'], "Daily<br>Score": db['D_Score'],
                "Structure": "ABOVE 18" if db['D_Chk_Price'] else "BELOW 18",
                "Ichimoku<br>Cloud": "PASS" if db['W_Cloud'] else "FAIL", "A/D Breadth": "STRONG" if db['AD_Pass'] else "WEAK",
                "Volume": db['Vol_Msg'], "Dual RSI": db['RSI_Msg'], "Institutional<br>Activity": db['Inst_Act'],
                "Action": final_decision, "Reasoning": final_reason, "Stop Price": disp_stop, "Position Size": disp_shares
            }
            results.append(row)
            if t == "HXT.TO": row_cad = row.copy(); row_cad["Sector"] = "15. CANADA (HXT)"; row_cad["Rank"] = 0; results.append(row_cad)
            if t in tc.SECTOR_ETFS: row_sec = row.copy(); row_sec["Sector"] = "02. SECTORS (SUMMARY)"; row_sec["Rank"] = 0; results.append(row_sec)

        if results:
            df_final = pd.DataFrame(results).sort_values(["Sector", "Rank", "Ticker"], ascending=[True, True, True])
            df_final["Sector"] = df_final["Sector"].apply(lambda x: x.split(". ", 1)[1].replace("(SUMMARY)", "").strip() if ". " in x else x)
            cols = ["Sector", "Ticker", "Rotation", "Weekly<br>SMA8", "Weekly<br>Impulse", "Weekly<br>Score", "Daily<br>Score", "Structure", "Ichimoku<br>Cloud", "A/D Breadth", "Volume", "Dual RSI", "Institutional<br>Activity", "Action", "Reasoning", "Stop Price", "Position Size"]
            st.markdown(generate_scanner_html(df_final[cols]), unsafe_allow_html=True)
        else:
            st.warning("Scanner returned no results.")

    if mode == "Sector Rotation":
        st.subheader("🔄 Relative Rotation Graphs (RRG)")
        is_dark = st.session_state.get('is_dark', True)
        rrg_mode = st.radio("View:", ["Indices", "Sectors", "Drill-Down", "Themes"], horizontal=True, key="rrg_nav")
        
        if rrg_mode == "Indices":
            c1, c2 = st.columns([1,3])
            with c1: bench_sel = st.selectbox("Benchmark", ["SPY", "IEF"], key="bench_idx")
            tgt = "IEF" if bench_sel == "IEF" else "SPY"
            idx_list = list(tc.RRG_INDICES.keys())
            if tgt == "IEF": idx_list.append("SPY")
            elif "SPY" in idx_list: idx_list.remove("SPY")
            
            if st.button("Run Indices"):
                wide_df = tr.prepare_rrg_inputs(master_data, idx_list, tgt)
                r, m = tr.calculate_rrg_math(wide_df, tgt)
                lbls = tc.RRG_INDICES.copy(); lbls['SPY'] = 'S&P 500 (Eq)'
                st.session_state['fig_idx'] = tr.plot_rrg_chart(r, m, lbls, f"Indices vs {tgt}", is_dark)
            if 'fig_idx' in st.session_state: st.plotly_chart(st.session_state['fig_idx'], use_container_width=True)

        elif rrg_mode == "Sectors":
            if st.button("Run Sectors"):
                wide_df = tr.prepare_rrg_inputs(master_data, list(tc.RRG_SECTORS.keys()), "SPY")
                r, m = tr.calculate_rrg_math(wide_df, "SPY")
                st.session_state['fig_sec'] = tr.plot_rrg_chart(r, m, tc.RRG_SECTORS, "Sectors vs SPY", is_dark)
            if 'fig_sec' in st.session_state: st.plotly_chart(st.session_state['fig_sec'], use_container_width=True)

        elif rrg_mode == "Drill-Down":
            c1, c2 = st.columns([1,3])
            with c1:
                def fmt(x): return f"{x} - {tc.RRG_SECTORS[x]}" if x in tc.RRG_SECTORS else x
                opts = list(tc.RRG_SECTORS.keys()) + ["Canada (TSX)"]
                sec_key = st.selectbox("Select Sector", opts, format_func=fmt, key="dd_sel")
            if sec_key == "Canada (TSX)": bench_dd = "HXT.TO"; name_dd = "Canadian Titans"
            else: bench_dd = sec_key; name_dd = tc.RRG_SECTORS[sec_key]
            
            if st.button(f"Run {name_dd}"):
                comp_list = list(tc.RRG_INDUSTRY_MAP.get(sec_key, {}).keys())
                wide_df = tr.prepare_rrg_inputs(master_data, comp_list, bench_dd)
                r, m = tr.calculate_rrg_math(wide_df, bench_dd)
                all_labels = {**tc.RRG_INDUSTRY_MAP.get(sec_key, {}), **tc.RRG_SECTORS}
                st.session_state['fig_dd'] = tr.plot_rrg_chart(r, m, all_labels, f"{name_dd} vs {bench_dd}", is_dark)
            if 'fig_dd' in st.session_state: st.plotly_chart(st.session_state['fig_dd'], use_container_width=True)

        elif rrg_mode == "Themes":
            if st.button("Run Themes"):
                wide_df = tr.prepare_rrg_inputs(master_data, list(tc.RRG_THEMES.keys()), "SPY")
                r, m = tr.calculate_rrg_math(wide_df, "SPY")
                st.session_state['fig_thm'] = tr.plot_rrg_chart(r, m, tc.RRG_THEMES, "Themes vs SPY", is_dark)
            if 'fig_thm' in st.session_state: st.plotly_chart(st.session_state['fig_thm'], use_container_width=True)
