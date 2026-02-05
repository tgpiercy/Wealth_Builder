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
    ]).map(lambda v: 'color: #00ff00' if v in ["BULLISH", "RISK ON", "CALM"] else ('color: #ffaa00' if v in ["STABLE", "CAUTION"] else 'color: #ff0000'), subset=["Status"])

def color_pl(val):
    if isinstance(val, str) and '%' in val:
        try:
            num = float(val.strip('%').replace('+',''))
            return 'color: #00ff00; font-weight: bold' if num >= 0 else 'color: #ff4444; font-weight: bold'
        except: return ''
    return ''

def color_pl_dol(val):
    if isinstance(val, str) and '$' in val:
        try:
            num = float(val.strip('$').replace('+','').replace(',',''))
            if val.startswith('-'): num = -num 
            return 'color: #00ff00; font-weight: bold' if num >= 0 else 'color: #ff4444; font-weight: bold'
        except: return ''
    return ''

def color_action(val):
    if "EXIT" in val: return 'color: #ff0000; font-weight: bold; background-color: #220000'
    if "HOLD" in val: return 'color: #00ff00; font-weight: bold'
    return 'color: #ffffff'

# --- PORTFOLIO STYLER ---
def style_portfolio(styler):
    return styler.set_table_styles([
         {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#111'), ('color', 'white')]},
         {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '14px')]}
    ]).map(color_pl, subset=["% Return"])\
      .map(color_pl_dol, subset=["Gain/Loss ($)"])\
      .map(color_action, subset=["Audit Action"])\
      .hide(axis='index')

def style_history(styler):
    return styler.set_table_styles([
         {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#111'), ('color', 'white')]},
         {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '14px')]}
    ]).map(color_pl, subset=["% Return", "% Delta vs SPY"])\
      .map(color_pl_dol, subset=["$ P&L"])\
      .hide(axis='index')

# --- PORTFOLIO ENGINE ---
def load_portfolio():
    cols = ["ID", "Ticker", "Date", "Shares", "Cost_Basis", "Status", "Exit_Date", "Exit_Price", "Return", "Realized_PL", "SPY_Return"]
    
    if not os.path.exists(PORTFOLIO_FILE):
        df = pd.DataFrame(columns=cols)
        df.to_csv(PORTFOLIO_FILE, index=False)
        return df
    
    df = pd.read_csv(PORTFOLIO_FILE)
    for c in cols:
        if c not in df.columns: df[c] = None
    
    df['Shares'] = pd.to_numeric(df['Shares'], errors='coerce')
    df['Cost_Basis'] = pd.to_numeric(df['Cost_Basis'], errors='coerce')
    df['Exit_Price'] = pd.to_numeric(df['Exit_Price'], errors='coerce')
    df['Realized_PL'] = pd.to_numeric(df['Realized_PL'], errors='coerce')
    
    for idx, row in df.iterrows():
        if row['Status'] == 'CLOSED' and pd.isna(row['Realized_PL']):
             try:
                 pl = (float(row['Exit_Price']) - float(row['Cost_Basis'])) * float(row['Shares'])
                 df.at[idx, 'Realized_PL'] = pl
             except: df.at[idx, 'Realized_PL'] = 0.0
        if pd.isna(row['SPY_Return']): df.at[idx, 'SPY_Return'] = 0.0

    if "ID" not in df.columns or df["ID"].isnull().all():
        df["ID"] = range(1, len(df) + 1)
        
    return df

def save_portfolio(df):
    dollar_cols = ['Cost_Basis', 'Exit_Price', 'Realized_PL', 'Return', 'SPY_Return']
    for col in dollar_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
            
    def clean_shares(row):
        val = row['Shares']
        if pd.isna(val): return 0
        if row['Ticker'] == 'CASH':
            return round(val, 2)
        else:
            return int(val) 
            
    if not df.empty:
        df['Shares'] = df.apply(clean_shares, axis=1)

    df.to_csv(PORTFOLIO_FILE, index=False)

# --- SIDEBAR: MANAGER ---
st.sidebar.header("💼 Portfolio Manager")
pf_df = load_portfolio()

cash_rows = pf_df[(pf_df['Ticker'] == 'CASH') & (pf_df['Status'] == 'OPEN')]
current_cash = cash_rows['Shares'].sum() if not cash_rows.empty else 0.0

st.sidebar.metric("Cash Available", f"${current_cash:,.2f}")

tab1, tab2, tab3, tab4 = st.sidebar.tabs(["🟢 Buy", "🔴 Sell", "💵 Cash", "🛠️ Fix/Edit"])

with tab1:
    with st.form("buy_trade"):
        st.caption("Record New Position")
        all_options = list(DATA_MAP.keys())
        b_tick = st.selectbox("Ticker", all_options)
        b_date = st.date_input("Buy Date")
        b_shares = st.number_input("Shares", min_value=1, value=100, step=1)
        b_price = st.number_input("Buy Price", min_value=0.01, value=100.00, step=0.01, format="%.2f")
        
        if st.form_submit_button("Execute Buy"):
            new_id = 1 if pf_df.empty else pf_df["ID"].max() + 1
            new_row = pd.DataFrame([{
                "ID": new_id, "Ticker": b_tick, "Date": b_date, "Shares": b_shares, 
                "Cost_Basis": b_price, "Status": "OPEN", "Exit_Date": None, 
                "Exit_Price": None, "Return": 0.0, "Realized_PL": 0.0, "SPY_Return": 0.0
            }])
            pf_df = pd.concat([pf_df, new_row], ignore_index=True)
            
            if current_cash >= (b_shares * b_price):
                 cash_id = pf_df["ID"].max() + 1
                 cash_row = pd.DataFrame([{
                    "ID": cash_id, "Ticker": "CASH", "Date": b_date, "Shares": -(b_shares * b_price), 
                    "Cost_Basis": 1.0, "Status": "OPEN", "Exit_Date": None, 
                    "Exit_Price": None, "Return": 0.0, "Realized_PL": 0.0, "SPY_Return": 0.0
                }])
                 pf_df = pd.concat([pf_df, cash_row], ignore_index=True)

            save_portfolio(pf_df)
            st.success(f"Bought {b_tick}")
            st.rerun()

with tab2:
    st.caption("Close Position")
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
            sel_data = trade_map[selected_trade_str]
            sel_id = sel_data['id']
            max_qty = sel_data['max_shares']
            
            with st.form("sell_trade"):
                s_shares = st.number_input("Shares to Sell", min_value=1, max_value=max_qty, value=max_qty, step=1)
                s_date = st.date_input("Sell Date")
                s_price = st.number_input("Sell Price", min_value=0.01, value=100.00, step=0.01, format="%.2f")
                
                if st.form_submit_button("Execute Sell"):
                    row_idx = sel_data['idx']
                    buy_price = float(pf_df.at[row_idx, 'Cost_Basis'])
                    buy_date_str = pf_df.at[row_idx, 'Date']
                    
                    ret_pct = ((s_price - buy_price) / buy_price) * 100
                    pl_dollars = (s_price - buy_price) * s_shares
                    
                    spy_ret_val = 0.0
                    try:
                        spy_tk = yf.Ticker("SPY")
                        b_dt = pd.to_datetime(buy_date_str)
                        s_dt = pd.to_datetime(s_date)
                        hist = spy_tk.history(start=b_dt, end=s_dt + timedelta(days=5))
                        spy_buy = hist.asof(b_dt)['Close']
                        spy_sell = hist.asof(s_dt)['Close']
                        if not pd.isna(spy_buy) and not pd.isna(spy_sell):
                            spy_ret_val = ((spy_sell - spy_buy) / spy_buy) * 100
                    except: pass

                    if s_shares < max_qty:
                        pf_df.at[row_idx, 'Shares'] -= s_shares
                        new_id = pf_df["ID"].max() + 1
                        new_closed_row = pd.DataFrame([{
                            "ID": new_id, 
                            "Ticker": pf_df.at[row_idx, 'Ticker'], 
                            "Date": buy_date_str, 
                            "Shares": s_shares, 
                            "Cost_Basis": buy_price, 
                            "Status": "CLOSED", 
                            "Exit_Date": s_date, 
                            "Exit_Price": s_price, 
                            "Return": ret_pct, 
                            "Realized_PL": pl_dollars, 
                            "SPY_Return": spy_ret_val
                        }])
                        pf_df = pd.concat([pf_df, new_closed_row], ignore_index=True)
                    else:
                        pf_df.at[row_idx, 'Status'] = 'CLOSED'
                        pf_df.at[row_idx, 'Exit_Date'] = s_date
                        pf_df.at[row_idx, 'Exit_Price'] = s_price
                        pf_df.at[row_idx, 'Return'] = ret_pct
                        pf_df.at[row_idx, 'Realized_PL'] = pl_dollars
                        pf_df.at[row_idx, 'SPY_Return'] = spy_ret_val
                    
                    cash_id = pf_df["ID"].max() + 1
                    cash_row = pd.DataFrame([{
                        "ID": cash_id, "Ticker": "CASH", "Date": s_date, "Shares": (s_price * s_shares), 
                        "Cost_Basis": 1.0, "Status": "OPEN", "Exit_Date": None, 
                        "Exit_Price": None, "Return": 0.0, "Realized_PL": 0.0, "SPY_Return": 0.0
                    }])
                    pf_df = pd.concat([pf_df, cash_row], ignore_index=True)

                    save_portfolio(pf_df)
                    st.success(f"Sold {s_shares} shares. P&L: ${pl_dollars:+.2f}")
                    st.rerun()
    else:
        st.info("No Open Positions")

with tab3:
    st.caption("Deposit / Withdraw")
    with st.form("cash_ops"):
        op_type = st.radio("Operation", ["Deposit", "Withdraw"])
        amount = st.number_input("Amount", min_value=0.01, value=1000.00, step=0.01, format="%.2f")
        c_date = st.date_input("Date")
        
        if st.form_submit_button("Execute"):
            final_amt = amount if op_type == "Deposit" else -amount
            new_id = 1 if pf_df.empty else pf_df["ID"].max() + 1
            new_row = pd.DataFrame([{
                "ID": new_id, "Ticker": "CASH", "Date": c_date, "Shares": final_amt, 
                "Cost_Basis": 1.0, "Status": "OPEN", "Exit_Date": None, 
                "Exit_Price": None, "Return": 0.0, "Realized_PL": 0.0, "SPY_Return": 0.0
            }])
            pf_df = pd.concat([pf_df, new_row], ignore_index=True)
            save_portfolio(pf_df)
            st.success(f"{op_type} ${amount}")
            st.rerun()

with tab4:
    action_type = st.radio("Mode", ["Delete Trade", "Edit Trade"])
    
    if not pf_df.empty:
        opts = pf_df.apply(lambda x: f"ID:{x['ID']} | {x['Ticker']} ({x['Status']})", axis=1).tolist()
        sel_str = st.selectbox("Select Trade", opts)
        
        if sel_str:
            sel_id = int(sel_str.split("|")[0].replace("ID:", "").strip())
            row_idx = pf_df[pf_df['ID'] == sel_id].index[0]
            
            if action_type == "Delete Trade":
                if st.button("Permanently Delete"):
                    pf_df = pf_df[pf_df['ID'] != sel_id]
                    save_portfolio(pf_df)
                    st.warning(f"Deleted ID {sel_id}")
                    st.rerun()
            
            elif action_type == "Edit Trade":
                with st.form("edit_form"):
                    st.subheader(f"Editing ID: {sel_id}")
                    c1, c2 = st.columns(2)
                    cur_status = pf_df.at[row_idx, 'Status']
                    new_status = c1.selectbox("Status", ["OPEN", "CLOSED"], index=0 if cur_status=="OPEN" else 1)
                    new_shares = c2.number_input("Shares", value=float(pf_df.at[row_idx, 'Shares']))
                    
                    c3, c4 = st.columns(2)
                    new_cost = c3.number_input("Cost Basis", value=float(pf_df.at[row_idx, 'Cost_Basis']))
                    cur_exit = pf_df.at[row_idx, 'Exit_Price']
                    new_exit = c4.number_input("Exit Price (0 if Open)", value=float(cur_exit) if pd.notna(cur_exit) else 0.0)
                    
                    if st.form_submit_button("Update Record"):
                        pf_df.at[row_idx, 'Status'] = new_status
                        pf_df.at[row_idx, 'Shares'] = new_shares
                        pf_df.at[row_idx, 'Cost_Basis'] = new_cost
                        if new_status == "OPEN":
                            pf_df.at[row_idx, 'Exit_Price'] = None
                            pf_df.at[row_idx, 'Exit_Date'] = None
                            pf_df.at[row_idx, 'Return'] = 0.0
                            pf_df.at[row_idx, 'Realized_PL'] = 0.0
                        else:
                            pf_df.at[row_idx, 'Exit_Price'] = new_exit
                            if new_exit > 0:
                                ret = ((new_exit - new_cost)/new_cost)*100
                                pl = (new_exit - new_cost) * new_shares
                                pf_df.at[row_idx, 'Return'] = ret
                                pf_df.at[row_idx, 'Realized_PL'] = pl
                                
                        save_portfolio(pf_df)
                        st.success("Record Updated")
                        st.rerun()

    st.write("---")
    st.subheader("⚠️ Danger Zone")
    if st.button("FACTORY RESET (Delete All Data)"):
        if os.path.exists(PORTFOLIO_FILE):
            os.remove(PORTFOLIO_FILE)
            st.success(f"Deleted {PORTFOLIO_FILE}. Please refresh the page.")
            st.rerun()
        else:
            st.error("No data file found to delete.")

# --- MAIN EXECUTION ---
if st.button("RUN ANALYSIS", type="primary"):
    
    with st.spinner('Checking Vitals...'):
        market_tickers = ["SPY", "IEF", "^VIX", "CAD=X", "HXT.TO"]
        market_data = {}
        for t in market_tickers:
            try:
                tk = yf.Ticker(t)
                df = tk.history(period="2y", interval="1d")
                df.index = pd.to_datetime(df.index).tz_localize(None)
                if not df.empty and 'Close' in df.columns: market_data[t] = df
            except: pass
        
        spy = market_data.get("SPY"); ief = market_data.get("IEF"); vix = market_data.get("^VIX")
        cad = market_data.get("CAD=X")
        cad_rate = cad.iloc[-1]['Close'] if cad is not None else 1.40 
        
        mkt_score = 0; total_exp = 0; exposure_rows = []
        
        if spy is not None and ief is not None and vix is not None:
            spy_c = spy.iloc[-1]['Close']; spy_sma18 = calc_sma(spy['Close'], 18).iloc[-1]
            if spy_c > spy_sma18
