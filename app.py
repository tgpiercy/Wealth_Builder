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
#  TITAN STRATEGY APP (v68.0 VSA Upgrade)
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

st.title(f"🛡️ Titan Strategy v68.0 ({current_user.upper()})")
st.caption("Institutional Protocol: VSA Engine")

# --- UNIFIED DATA ENGINE (CACHED) ---
@st.cache_data(ttl=3600, show_spinner="Downloading Unified Market Data...") 
def fetch_master_data(ticker_list):
    unique_tickers = sorted(list(set(ticker_list))) 
    data_map = {}
    for t in unique_tickers:
        try:
            fetch_sym = "SPY" if t == "MANL" else t 
            tk = yf.Ticker(fetch_sym)
            df = tk.history(period="10y", interval="1d")
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if not df.empty and 'Close' in df.columns:
                data_map[t] = df
        except: continue
    return data_map

# --- STRATEGY ENGINE (CACHED) ---
@st.cache_data(show_spinner="Running Quantitative Analysis...")
def run_strategy_engine(master_data, scan_list, risk_per_trade, rrg_snapshot):
    """
    Core Logic Engine (Pass 1 & Pass 2).
    """
    calculation_db = {}
    unique_scan_list = sorted(list(set(scan_list)))
    
    # PASS 1: Calculate
    for t in unique_scan_list:
        if t not in master_data or len(master_data[t]) < 50: continue
        df = master_data[t].copy()
        
        df['SMA8'] = tm.calc_sma(df['Close'], 8)
        df['SMA18'] = tm.calc_sma(df['Close'], 18)
        df['SMA40'] = tm.calc_sma(df['Close'], 40)
        df['AD'] = tm.calc_ad(df['High'], df['Low'], df['Close'], df['Volume'])
        ad_sma18 = tm.calc_sma(df['AD'], 18); ad_sma40 = tm.calc_sma(df['AD'], 40)
        df['VolSMA'] = tm.calc_sma(df['Volume'], 18)
        df['RSI5'] = tm.calc_rsi(df['Close'], 5); df['RSI20'] = tm.calc_rsi(df['Close'], 20)
        
        bench_ticker = "SPY"
        if t in tc.DATA_MAP and tc.DATA_MAP[t][1]: bench_ticker = tc.DATA_MAP[t][1]
        
        rs_score_ok = False
        if bench_ticker in master_data:
            bench_series = master_data[bench_ticker]['Close']
            common_idx = df.index.intersection(bench_series.index)
            rs_series = df.loc[common_idx, 'Close'] / bench_series.loc[common_idx]
            rs_sma18 = tm.calc_sma(rs_series, 18)
            
            if len(rs_series) > 2 and len(rs_sma18) > 2:
                curr_rs = rs_series.iloc[-1]; curr_rs_sma = rs_sma18.iloc[-1]
                lower_band = curr_rs_sma - (abs(curr_rs_sma) * 0.005) 
                rs_not_down = curr_rs_sma >= rs_sma18.iloc[-2] 
                rs_in_zone = curr_rs >= lower_band
                if rs_in_zone and rs_not_down: rs_score_ok = True
        else: rs_score_ok = True 
        
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
        
        # --- NEW VSA CALL ---
        inst_activity = tm.calc_smart_money(df)
        
        ad_score_ok = False
        ad_val = 0; ad18 = 0
        if len(ad_sma18) > 2:
            ad_val = df['AD'].iloc[-1]; ad18 = ad_sma18.iloc[-1]
            ad_lower_band = ad18 - (abs(ad18) * 0.005)
            ad_not_down = ad18 >= ad_sma18.iloc[-2]
            ad_in_zone = ad_val >= ad_lower_band
            if ad_in_zone and ad_not_down: ad_score_ok = True
        
        d_chk = 0
        if ad_score_ok: d_chk += 1
        if rs_score_ok: d_chk += 1
        if dc['Close'] > df['SMA18'].iloc[-1]: d_chk += 1 
        if tm.calc_rising(df['SMA18'], 2): d_chk += 1     
        if df['SMA18'].iloc[-1] > df['SMA40'].iloc[-1]: d_chk += 1 

        w_score = 0
        if len(w_ad_sma18) > 2 and (df_w['AD'].iloc[-1] >= (w_ad_sma18.iloc[-1] * 0.99)): w_score += 1
        if rs_score_ok: w_score += 1
        if wc['Close'] > wc['SMA18']: w_score += 1
        if tm.calc_rising(df_w['SMA18'], 2): w_score += 1
        if wc['SMA18'] > wc['SMA40']: w_score += 1

        w_pulse = "GOOD" if (wc['Close'] > wc['SMA18']) and (dc['Close'] > df['SMA18'].iloc[-1]) else "NO"
        
        vol_msg = "NORMAL"
        if df['Volume'].iloc[-1] > (df['VolSMA'].iloc[-1] * 1.5): vol_msg = "SPIKE (Live)"
        elif df['Volume'].iloc[-2] > (df['VolSMA'].iloc[-2] * 1.5): vol_msg = "SPIKE (Prev)"
        elif df['Volume'].iloc[-1] > df['VolSMA'].iloc[-1]: vol_msg = "HIGH (Live)"
        
        r5 = df['RSI5'].iloc[-1]; r20 = df['RSI20'].iloc[-1] if not pd.isna(df['RSI20'].iloc[-1]) else 50
        r5_prev = df['RSI5'].iloc[-2]; is_rising = r5 > r5_prev
        
        if r5 >= r20:
            n_c = "#00BFFF" if (r20 > 50 and is_rising) else ("#00FF00" if is_rising else "#FF4444")
        else:
            n_c = "#FFA500" if r20 > 50 else "#FF4444"
        a_c = "#00FF00" if is_rising else "#FF4444"; arrow = "↑" if is_rising else "↓"
        rsi_msg = f"<span style='color:{n_c}'><b>{int(r5)}/{int(r20)}</b></span> <span style='color:{a_c}'><b>{arrow}</b></span>"
        
        final_inst_msg = inst_activity # Now using the VSA Result

        decision = "AVOID"; reason = "Low Score"
        if w_score >= 4:
            if d_chk == 5: decision = "BUY"; reason = "Score 5/5" 
            elif d_chk == 4: decision = "SCOUT"; reason = "D-Score 4"
            elif d_chk == 3: decision = "SCOUT"; reason = "Dip Buy"
            else: decision = "WATCH"; reason = "Daily Weak"
        else: decision = "AVOID"; reason = "Weekly Weak"

        if not (wc['Close'] > wc['Cloud_Top']): decision = "AVOID"; reason = "Below Cloud"
        elif "NO" in w_pulse: decision = "AVOID"; reason = "Impulse NO"
        if risk_per_trade == 0 and "BUY" in decision: decision = "CAUTION"; reason = "VIX Lock"

        rrg_phase = rrg_snapshot.get(t, "UNKNOWN ➡️").upper()
        if "WEAKENING" in rrg_phase and "BUY" in decision: decision = "CAUTION"; reason = "Rotation Weak"
        
        struct_pass = df['SMA18'].iloc[-1] > df['SMA40'].iloc[-1]
        
        if ad_val > ad18: ad_msg = "ACCUMULATION"
        elif ad_score_ok: ad_msg = "NEUTRAL"
        else: ad_msg = "DISTRIBUTION"

        calculation_db[t] = {"Decision": decision, "Reason": reason, "RRG": rrg_phase, "W_Score": w_score, "D_Score": d_chk, "Structure": "BULLISH" if struct_pass else "", "A/D": ad_msg, "Vol": vol_msg, "RSI": rsi_msg, "Inst": final_inst_msg, "Pulse": w_pulse, "W_SMA8": "PASS" if (wc['Close']>wc['SMA8']) else "FAIL"}

    # PASS 2: Enforce Parent Lock
    results = []
    TICKER_PRIORITY = {t: i for i, t in enumerate(tc.DATA_MAP.keys())}
    
    for t in unique_scan_list:
        if t not in calculation_db: continue
        data = calculation_db[t]
        final_decision = data["Decision"]; final_reason = data["Reason"]
        
        parent_ticker = tc.DATA_MAP.get(t, [None, None])[1]
        if parent_ticker and parent_ticker != "SPY" and parent_ticker != "HXT.TO":
            if parent_ticker in calculation_db:
                if "AVOID" in calculation_db[parent_ticker]["Decision"]:
                    final_decision = "LOCKED"; final_reason = f"Parent ({parent_ticker}) Weak"

        cat_name = tc.DATA_MAP.get(t, ["OTHER"])[0]
        if "99. DATA" in cat_name: continue
        is_scanner = t in tc.DATA_MAP and (tc.DATA_MAP[t][0] != "BENCH" or t in ["DIA", "QQQ", "IWM", "IWC", "HXT.TO", "SPY", "RSP", "IEF", "^VIX"])
        
        if is_scanner:
            disp_action = final_decision if "AVOID" not in final_decision else ""
            sort_priority = TICKER_PRIORITY.get(t, 9999)
            
            row = {"Sector": cat_name, "Ticker": t, "Rank": (0 if "00." in cat_name else 1), "Rotation": data["RRG"], "Weekly<br>SMA8": data["W_SMA8"], "Weekly<br>Impulse": data["Pulse"], "Weekly<br>Score": data["W_Score"], "Daily<br>Score": data["D_Score"], "Structure": data["Structure"], "A/D Breadth": data["A/D"], "Volume": data["Vol"], "Dual RSI": data["RSI"], "Institutional<br>Activity": data["Inst"], "Action": disp_action, "Reasoning": final_reason, "Priority": sort_priority}
            results.append(row)

    return results, calculation_db

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

# --- SIDEBAR & MAIN EXECUTION ---
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
    
    action_type = st.radio("Advanced Tools", ["Delete Trade", "Edit Trade", "⚠️ CLEAR CACHE & REBOOT", "Rebuild Benchmark History"])
    if action_type == "⚠️ CLEAR CACHE & REBOOT" and st.button("EXECUTE CLEAR"):
        st.cache_data.clear()
        st.success("Cache Cleared! Rebooting..."); time.sleep(1); st.rerun()
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
    # DEDUPLICATION FOR DATA FETCHING
    all_tickers = sorted(list(set(all_tickers)))
    
    # --- STEP 1: FETCH DATA (Cached) ---
    master_data = fetch_master_data(all_tickers)
    
    # --- STEP 2: CALCULATE METRICS (Fast) ---
    spy = master_data.get("SPY"); vix = master_data.get("^VIX")
    mkt_score = 0
    if spy is not None and vix is not None:
        v = vix.iloc[-1]['Close']
        mkt_score += 9 if v < 17 else (6 if v < 20 else (3 if v < 25 else 0))
        sc = spy.iloc[-1]['Close']; s18 = tm.calc_sma(spy['Close'], 18).iloc[-1]; s8 = tm.calc_sma(spy['Close'], 8).iloc[-1]
        if sc > s18: mkt_score += 1
        if s18 >= tm.calc_sma(spy['Close'], 18).iloc[-2]: mkt_score += 1
        if s8 > tm.calc_sma(spy['Close'], 8).iloc[-2]: mkt_score += 1
    
    BASE_RISK = 2300 
    risk_per_trade = BASE_RISK if mkt_score >= 8 else (BASE_RISK * 0.5 if mkt_score >= 5 else 0)
    
    # --- STEP 3: RUN STRATEGY ENGINE (Cached) ---
    rrg_snapshot = tr.generate_full_rrg_snapshot(master_data, "SPY")
    
    # Deduped list is passed to engine (redundant but safe)
    scan_results, analysis_db = run_strategy_engine(master_data, all_tickers, risk_per_trade, rrg_snapshot)

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
        c1.metric(f"Net Worth (CAD @ {cad_rate:.2f})", f"${total_nw_cad:,.2f}", f"{open_pl_cad:+.2f}")
        c2.metric("Net Worth (USD)", f"${total_net_worth:,.2f}", f"{open_pl_val:+.2f}")
        c3.metric("Benchmark (Shadow SPY)", f"${shadow_equity:,.2f}", f"{alpha_dollars:+.2f}")
        c4.metric("Cash", f"${current_cash:,.2f}")
        c5.metric("Equity", f"${eq_val:,.2f}")
        st.write("---")

        # --- SECTION 2: CURRENT HOLDINGS ---
        st.subheader("💼 Current Holdings")
        if pf_rows: st.markdown(pd.DataFrame(pf_rows).style.pipe(ts.style_portfolio).to_html(), unsafe_allow_html=True)
        else: st.info("No active trades.")
        st.write("---")

        # --- SECTION 3: MARKET HEALTH ---
        rsp = master_data.get("RSP")
        h_rows = []
        if spy is not None and vix is not None:
            v = vix.iloc[-1]['Close']
            s = "<span style='color:#00ff00'>NORMAL</span>" if v < 17 else ("<span style='color:#ffaa00'>CAUTIOUS</span>" if v < 20 else "<span style='color:#ff4444'>PANIC</span>")
            h_rows.append({"Indicator": f"VIX Level ({v:.2f})", "Status": s})
            
            sc = spy.iloc[-1]['Close']; s18 = tm.calc_sma(spy['Close'], 18).iloc[-1]; s8 = tm.calc_sma(spy['Close'], 8).iloc[-1]
            h_rows.append({"Indicator": "SPY Price > SMA18", "Status": "<span style='color:#00ff00'>PASS</span>" if sc > s18 else "<span style='color:#ff4444'>FAIL</span>"})
            h_rows.append({"Indicator": "SPY SMA18 Rising", "Status": "<span style='color:#00ff00'>RISING</span>" if s18 >= tm.calc_sma(spy['Close'], 18).iloc[-2] else "<span style='color:#ff4444'>FALLING</span>"})
            h_rows.append({"Indicator": "SPY SMA8 Rising", "Status": "<span style='color:#00ff00'>RISING</span>" if s8 > tm.calc_sma(spy['Close'], 8).iloc[-2] else "<span style='color:#ff4444'>FALLING</span>"})

            if rsp is not None:
                rc = rsp.iloc[-1]['Close']; r18 = tm.calc_sma(rsp['Close'], 18).iloc[-1]
                h_rows.append({"Indicator": "RSP Price > SMA18", "Status": "<span style='color:#00ff00'>PASS</span>" if rc > r18 else "<span style='color:#ff4444'>FAIL</span>"})
            
            col = "#00ff00" if mkt_score >= 8 else ("#ffaa00" if mkt_score >= 5 else "#ff4444")
            msg = "AGGRESSIVE" if mkt_score >= 10 else ("CAUTIOUS" if mkt_score >= 8 else "DEFENSIVE")
            
            h_rows.append({"Indicator": "TOTAL SCORE", "Status": f"<span style='color:{col}'><b>{mkt_score}/11</b></span>"})
            h_rows.append({"Indicator": "STRATEGY MODE", "Status": f"<span style='color:{col}'><b>{msg}</b></span>"})
            st.subheader("🏥 Daily Market Health")
            st.markdown(pd.DataFrame(h_rows).style.pipe(ts.style_daily_health).to_html(escape=False), unsafe_allow_html=True)
            st.write("---")

        # --- SECTION 4: SCANNER RESULTS (Cached) ---
        if scan_results:
            df_final = pd.DataFrame(scan_results).sort_values("Priority", ascending=True)
            df_final["Sector"] = df_final["Sector"].apply(lambda x: x.split(". ", 1)[1].replace("(SUMMARY)", "").strip() if ". " in x else x)
            cols = ["Sector", "Ticker", "Rotation", "Weekly<br>SMA8", "Weekly<br>Impulse", "Weekly<br>Score", "Daily<br>Score", "Structure", "A/D Breadth", "Volume", "Dual RSI", "Institutional<br>Activity", "Action", "Reasoning"]
            st.markdown(generate_scanner_html(df_final[cols]), unsafe_allow_html=True)
        else:
            st.warning("Scanner returned no results.")

    if mode == "Sector Rotation":
        st.subheader("🔄 Relative Rotation Graphs (RRG)")
        is_dark = st.session_state.get('is_dark', True)
        rrg_mode = st.radio("View:", ["Indices", "Sectors", "Drill-Down", "Themes", "Metals"], horizontal=True, key="rrg_nav")
        
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

        elif rrg_mode == "Metals":
            if st.button("Run Precious Metals"):
                metals = ["GLD", "SLV"]
                wide_df = tr.prepare_rrg_inputs(master_data, metals, "SPY")
                r, m = tr.calculate_rrg_math(wide_df, "SPY")
                st.session_state['fig_met'] = tr.plot_rrg_chart(r, m, {k:k for k in metals}, "Precious Metals vs SPY", is_dark)
            if 'fig_met' in st.session_state: st.plotly_chart(st.session_state['fig_met'], use_container_width=True)
