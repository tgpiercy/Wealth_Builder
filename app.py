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
#  TITAN STRATEGY APP (v61.8 Fixed Risk Variable)
# ==============================================================================

current_user = st.session_state.user
PORTFOLIO_FILE = f"portfolio_{current_user}.csv"

# --- GLOBAL SETTINGS (SIDEBAR) ---
st.sidebar.write(f"👤 Logged in as: **{current_user.upper()}**")

# Global Dark Mode Logic
if "is_dark" not in st.session_state:
    st.session_state.is_dark = True
st.sidebar.toggle("🌙 Dark Mode", key="is_dark")

if st.sidebar.button("Log Out"):
    logout()

st.title(f"🛡️ Titan Strategy v61.8 ({current_user.upper()})")
st.caption("Institutional Protocol: Modular & Debugged")

# --- UNIFIED DATA ENGINE ---
@st.cache_data(ttl=3600) 
def fetch_master_data(ticker_list):
    """Downloads daily data for ALL tickers once."""
    unique_tickers = sorted(list(set(ticker_list))) 
    data_map = {}
    for t in unique_tickers:
        try:
            # Handle special ticker mapping if needed
            fetch_sym = "SPY" if t == "MANL" else t 
            tk = yf.Ticker(fetch_sym)
            df = tk.history(period="2y", interval="1d")
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
        # 1. HOLDINGS
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
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Net Worth (CAD @ {cad_rate:.2f})", f"${total_nw_cad:,.2f}", ts.fmt_delta(open_pl_cad))
        c2.metric("Net Worth (USD)", f"${total_net_worth:,.2f}", ts.fmt_delta(open_pl_val))
        c3.metric("Cash", f"${current_cash:,.2f}"); c4.metric("Equity", f"${eq_val:,.2f}")
        
        if pf_rows: st.markdown(pd.DataFrame(pf_rows).style.pipe(ts.style_portfolio).to_html(), unsafe_allow_html=True)
        else: st.info("No active trades.")
        st.write("---")

        # 3. BENCHMARK
        shadow_shares_total = pf_df['Shadow_SPY'].sum()
        spy_data = master_data.get("SPY")
        if spy_data is not None:
            curr_spy = spy_data['Close'].iloc[-1]
            bench_val = shadow_shares_total * curr_spy
            alpha = total_net_worth - bench_val
            alpha_pct = ((total_net_worth - bench_val) / bench_val * 100) if bench_val > 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Titan Net Worth", f"${total_net_worth:,.2f}")
            c2.metric("SPY Benchmark", f"${bench_val:,.2f}")
            c3.metric("Alpha (Edge)", f"${alpha:,.2f}", f"{alpha_pct:+.2f}%")
            st.write("---")

        # 4. MARKET HEALTH
        spy = master_data.get("SPY"); vix = master_data.get("^VIX"); rsp = master_data.get("RSP")
        mkt_score = 0; h_rows = []
        if spy is not None:
            if vix is not None:
                v = vix.iloc[-1]['Close']
                s = "<span style='color:#00ff00'>NORMAL</span>" if v < 17 else ("<span style='color:#ffaa00'>CAUTIOUS</span>" if v < 20 else "<span style='color:#ff4
