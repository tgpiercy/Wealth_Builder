import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# --- IMPORT CONFIG ---
try:
    import titan_config as tc
except ImportError:
    st.error("⚠️ CRITICAL ERROR: `titan_config.py` is missing. Please create it.")
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
#  TITAN STRATEGY APP (v56.1 Restored)
# ==============================================================================

current_user = st.session_state.user
PORTFOLIO_FILE = f"portfolio_{current_user}.csv"

st.sidebar.write(f"👤 Logged in as: **{current_user.upper()}**")
if st.sidebar.button("Log Out"):
    logout()

st.title(f"🛡️ Titan Strategy v56.1 ({current_user.upper()})")
st.caption("Institutional Protocol: Full Logic | Modular Config")

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

def calc_rsi(series, length):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    avg_gain[length] = gain[1:length+1].mean()
    avg_loss[length] = loss[1:length+1].mean()
    for i in range(length + 1, len(series)):
        avg_gain[i] = (avg_gain[i-1] * (length - 1) + gain.iloc[i]) / length
        avg_loss[i] = (avg_loss[i-1] * (length - 1) + loss.iloc[i]) / length
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi_vals = 100 - (100 / (1 + rs))
    return pd.Series(rsi_vals, index=series.index)

# --- ZIG ZAG STRUCTURE ENGINE ---
def calc_structure(df, deviation_pct=0.035):
    if len(df) < 50: return "None"
    pivots = []; trend = 1; last_val = df['Close'].iloc[0]
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
    curr_pivot = pivots[-1]
    prev_same = pivots[-3]
    if curr_pivot[2] == 1: 
        return "HH" if curr_pivot[1] > prev_same[1] else "LH"
    else:
        return "LL" if curr_pivot[1] < prev_same[1] else "HL"

# --- HELPER: SMART STOP ---
def round_to_03_07(price):
    if pd.isna(price): return 0.0
    whole = int(price)
    candidates = [c for c in [whole + 0.03, whole + 0.07, (whole - 1) + 0.97, (whole - 1) + 0.93] if c > 0]
    if not candidates: return price 
    return min(candidates, key=lambda x: abs(x - price))

# --- STYLING ---
def style_final(styler):
    def color_pct(val):
        if isinstance(val, str) and '%' in val:
            try: return 'color: #00ff00; font-weight: bold' if float(val.strip('%')) >= 0 else 'color: #ff0000; font-weight: bold'
            except: return ''
        return ''
    
    def color_rsi(val):
        try:
            parts = val.split()
            if len(parts) < 2: return ''
            r5 = float(parts[0].split('/')[0]); r20 = float(parts[0].split('/')[1]); arrow = parts[1]
            is_rising = (arrow == "↑")
            if r5 >= r20:
                return 'color: #00BFFF; font-weight: bold' if (r20 > 50 and is_rising) else ('color: #00FF00; font-weight: bold' if is_rising else 'color: #FF4444; font-weight: bold')
            elif r20 > 50: return 'color: #FFA500; font-weight: bold'
            return 'color: #FF4444; font-weight: bold'
        except: return ''

    def highlight_ticker_row(row):
        styles = ['' for _ in row.index]
        if 'Ticker' not in row.index: return styles
        idx = row.index.get_loc('Ticker')
        act = str(row.get('Action', '')).upper()
        vol = str(row.get('Volume', '')).upper()
        rsi = str(row.get('Dual RSI', ''))
        
        if "AVOID" in act: pass
        elif "00BFFF" in rsi and "SPIKE" in vol: styles[idx] = 'background-color: #0044CC; color: white; font-weight: bold'
        elif "BUY" in act: styles[idx] = 'background-color: #006600; color: white; font-weight: bold'
        elif "SCOUT" in act: styles[idx] = 'background-color: #005555; color: white; font-weight: bold'
        elif "SOON" in act or "CAUTION" in act: styles[idx] = 'background-color: #CC5500; color: white; font-weight: bold'
        return styles

    return styler.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#111'), ('color', 'white'), ('font-size', '12px')]}
    ]).set_properties(**{'background-color': '#222', 'color': 'white', 'border-color': '#444'})\
      .apply(highlight_ticker_row, axis=1)\
      .map(color_pct, subset=["4W %", "2W %"])\
      .map(color_rsi, subset=["Dual RSI"])\
      .hide(axis='index')

def style_daily_health(styler):
    def color_status(v):
        if "PASS" in v or "NORMAL" in v or "RISING" in v or "AGGRESSIVE" in v: return 'color: #00ff00; font-weight: bold'
        if "FAIL" in v or "PANIC" in v or "FALLING" in v or "CASH" in v: return 'color: #ff4444; font-weight: bold'
        return 'color: white; font-weight: bold'

    return styler.set_table_styles([
         {'selector': 'th', 'props': [('text-align', 'left'), ('background-color', '#111'), ('color', 'white'), ('font-size', '14px')]}, 
         {'selector': 'td', 'props': [('text-align', 'left'), ('font-size', '14px'), ('padding', '8px')]}
    ]).set_properties(**{'background-color': '#222', 'border-color': '#444'})\
      .set_properties(subset=['Indicator'], **{'color': 'white', 'font-weight': 'bold'})\
      .map(color_status, subset=['Status'])\
      .hide(axis='index')

def style_portfolio(styler):
    def color_pl(val):
        if isinstance(val, str) and ('$' in val or '%' in val):
            return 'color: #FF4444; font-weight: bold' if '-' in val else 'color: #00FF00; font-weight: bold'
        return ''
    return styler.map(color_pl, subset=["Gain/Loss ($)", "% Return"]).hide(axis='index')

def style_history(styler):
    def color_pl(val):
        if isinstance(val, str) and ('$' in val or '%' in val):
            return 'color: #FF4444; font-weight: bold' if '-' in val else 'color: #00FF00; font-weight: bold'
        return ''
    return styler.map(color_pl, subset=["P/L", "% Return"]).hide(axis='index')

def fmt_delta(val): return f"-${abs(val):,.2f}" if val < 0 else f"${val:,.2f}"

# --- PORTFOLIO ENGINE ---
def load_portfolio():
    cols = ["ID", "Ticker", "Date", "Shares", "Cost_Basis", "Status", "Exit_Date", "Exit_Price", "Return", "Realized_PL", "SPY_Return", "Type", "Shadow_SPY"]
    
    if not os.path.exists(PORTFOLIO_FILE):
        pd.DataFrame(columns=cols).to_csv(PORTFOLIO_FILE, index=False)
    
    df = pd.read_csv(PORTFOLIO_FILE)
    
    # 1. AUTO-ADD MISSING COLUMNS
    for c in cols:
        if c not in df.columns: df[c] = None
    
    # 2. RENAME LEGACY
    if 'Cost' in df.columns and 'Cost_Basis' not in df.columns: df.rename(columns={'Cost': 'Cost_Basis'}, inplace=True)
    
    # 3. FILL DEFAULTS
    if 'Cost_Basis' not in df.columns: df['Cost_Basis'] = 0.0
    df['Shadow_SPY'] = pd.to_numeric(df['Shadow_SPY'], errors='coerce').fillna(0.0)
    
    if "ID" not in df.columns or df["ID"].isnull().all(): df["ID"] = range(1, len(df) + 1)
    
    return df

def save_portfolio(df):
    dollar_cols = ['Cost_Basis', 'Exit_Price', 'Realized_PL', 'Return', 'SPY_Return']
    for col in dollar_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
    def clean_shares(row):
        val = row['Shares']
        if pd.isna(val): return 0
        return round(val, 2) if row['Ticker'] == 'CASH' else int(val)
    if not df.empty: df['Shares'] = df.apply(clean_shares, axis=1)
    df.to_csv(PORTFOLIO_FILE, index=False)

# --- SIDEBAR MANAGER ---
st.sidebar.header("💼 Portfolio Manager")
pf_df = load_portfolio()
cash_rows = pf_df[(pf_df['Ticker'] == 'CASH') & (pf_df['Status'] == 'OPEN')]
current_cash = cash_rows['Shares'].sum() if not cash_rows.empty else 0.0
st.sidebar.metric("Cash Available", f"${current_cash:,.2f}")

tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["🟢 Buy", "🔴 Sell", "💵 Cash", "🧮 Calc", "🛠️ Fix"])

with tab1:
    with st.form("buy_trade"):
        st.caption("Record New Position")
        # MODULAR: Using tc.DATA_MAP
        all_options = list(tc.DATA_MAP.keys())
        b_tick = st.selectbox("Ticker", all_options)
        b_date = st.date_input("Buy Date")
        b_shares = st.number_input("Shares", min_value=1, value=100, step=1)
        b_price = st.number_input("Buy Price", min_value=0.01, value=100.00, step=0.01, format="%.2f")
        
        if st.form_submit_button("Execute Buy"):
            new_id = 1 if pf_df.empty else pf_df["ID"].max() + 1
            # New Stock Row
            new_row = pd.DataFrame([{
                "ID": new_id, "Ticker": b_tick, "Date": b_date, "Shares": b_shares, 
                "Cost_Basis": b_price, "Status": "OPEN", "Exit_Date": None, 
                "Exit_Price": None, "Return": 0.0, "Realized_PL": 0.0, "SPY_Return": 0.0,
                "Type": "STOCK", "Shadow_SPY": 0.0
            }])
            pf_df = pd.concat([pf_df, new_row], ignore_index=True)
            
            # Cash Deduction Row (Marked as TRADE_CASH, ignores Shadow)
            if current_cash >= (b_shares * b_price):
                 cash_id = pf_df["ID"].max() + 1
                 cash_row = pd.DataFrame([{
                    "ID": cash_id, "Ticker": "CASH", "Date": b_date, "Shares": -(b_shares * b_price), 
                    "Cost_Basis": 1.0, "Status": "OPEN", "Exit_Date": None, 
                    "Exit_Price": None, "Return": 0.0, "Realized_PL": 0.0, "SPY_Return": 0.0,
                    "Type": "TRADE_CASH", "Shadow_SPY": 0.0
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
                s_shares = st.number_input("Shares to Sell
