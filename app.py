import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# --- IMPORT CONFIG ---
# This matches the 'tc' variable to the titan_config.py file
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
    # Access CREDENTIALS from the tc module
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
#  TITAN STRATEGY APP (v56.6 Stable)
# ==============================================================================

current_user = st.session_state.user
PORTFOLIO_FILE = f"portfolio_{current_user}.csv"

st.sidebar.write(f"👤 Logged in as: **{current_user.upper()}**")
if st.sidebar.button("Log Out"):
    logout()

st.title(f"🛡️ Titan Strategy v56.6 ({current_user.upper()})")
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

# --- ZIG ZAG ENGINE ---
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
        {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#111'), ('color', 'white'), ('font-size', '12px'), ('vertical-align', 'top')]}, 
        {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '14px'), ('padding', '8px')]}
    ]).set_properties(**{'background-color': '#222', 'color': 'white', 'border-color': '#444'})\
      .apply(highlight_ticker_row, axis=1)\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if v in ["BUY", "STRONG BUY"] else ('color: #00ffff; font-weight: bold' if "SCOUT" in v else ('color: #ffaa00; font-weight: bold' if v in ["SOON", "CAUTION"] else 'color: white')), subset=["Action"])\
      .map(lambda v: 'color: #ff00ff; font-weight: bold' if "SPIKE" in v else ('color: #00ff00' if "HIGH" in v else 'color: #ccc'), subset=["Volume"])\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if "STRONG" in v else ('color: #ff0000' if "WEAK" in v else 'color: #ffaa00'), subset=["A/D Breadth"])\
      .map(lambda v: 'color: #ff0000; font-weight: bold' if "FAIL" in v or "NO" in v else 'color: #00ff00', subset=["Ichimoku<br>Cloud", "Weekly<br>SMA8"])\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if "GOOD" in v else ('color: #ffaa00; font-weight: bold' if "WEAK" in v else 'color: #ff0000; font-weight: bold'), subset=["Weekly<br>Impulse"])\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if v >= 4 else ('color: #ffaa00; font-weight: bold' if v == 3 else 'color: #ff0000; font-weight: bold'), subset=["Weekly<br>Score", "Daily<br>Score"])\
      .map(lambda v: 'color: #ff0000; font-weight: bold' if "BELOW 18" in v else 'color: #00ff00', subset=["Structure"])\
      .map(color_pct, subset=["4W %", "2W %"])\
      .map(color_rsi, subset=["Dual RSI"])\
      .map(color_inst, subset=["Institutional<br>Activity"])\
      .hide(axis='index')

def style_daily_health(styler):
    def color_status(v):
        if "PASS" in v or "NORMAL" in v or "CAUTIOUS" in v or "RISING" in v or "AGGRESSIVE" in v: return 'color: #00ff00; font-weight: bold'
        if "FAIL" in v or "PANIC" in v or "DEFENSIVE" in v or "FALLING" in v or "CASH" in v: return 'color: #ff4444; font-weight: bold'
        return 'color: white; font-weight: bold'

    return styler.set_table_styles([
         {'selector': 'th', 'props': [('text-align', 'left'), ('background-color', '#111'), ('color', 'white'), ('font-size', '14px')]}, 
         {'selector': 'td', 'props': [('text-align', 'left'), ('font-size', '14px'), ('padding', '8px')]}
    ]).set_properties(**{'background-color': '#222', 'border-color': '#444'})\
      .set_properties(subset=['Indicator'], **{'color': 'white', 'font-weight': 'bold'})\
      .map(color_status, subset=['Status'])\
      .hide(axis='index')

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
    ]).map(color_pl, subset=["% Return"])\
      .map(color_pl_dol, subset=["P/L"])\
      .hide(axis='index')

# --- HELPER: FMT DELTA (Global) ---
def fmt_delta(val):
    return f"-${abs(val):,.2f}" if val < 0 else f"${val:,.2f}"

# --- PORTFOLIO ENGINE ---
def load_portfolio():
    cols = ["ID", "Ticker", "Date", "Shares", "Cost_Basis", "Status", "Exit_Date", "Exit_Price", "Return", "Realized_PL", "SPY_Return", "Type", "Shadow_SPY"]
    
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
    df['Shadow_SPY'] = pd.to_numeric(df['Shadow_SPY'], errors='coerce').fillna(0.0)

    # Auto-Correction
    if df['Type'].isnull().any() and not df.empty:
        df.loc[(df['Ticker'] != 'CASH') & (df['Type'].isnull()), 'Type'] = "STOCK"
        cash_mask = (df['Ticker'] == 'CASH') & (df['Type'].isnull())
        if cash_mask.any():
            for idx, row in df[cash_mask].iterrows():
                c_date = row['Date']
                buys = df[(df['Ticker']!='CASH') & (df['Date']==c_date)]
                sells = df[(df['Ticker']!='CASH') & (df['Exit_Date']==c_date)]
                
                if not buys.empty or not sells.empty:
                     df.at[idx, 'Type'] = "TRADE_CASH"
                else:
                     df.at[idx, 'Type'] = "TRANSFER"
        df.to_csv(PORTFOLIO_FILE, index=False)
                    
    for idx, row in df.iterrows():
        if row['Status'] == 'CLOSED' and pd.isna(row['Realized_PL']):
             try:
                 pl = (float(row['Exit_Price']) - float(row['Cost_Basis'])) * float(row['Shares'])
                 df.at[idx, 'Realized_PL'] = pl
             except: df.at[idx, 'Realized_PL'] = 0.0

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
                s_shares = st.number_input("Shares to Sell", min_value=1, max_value=max_qty, value=max_qty, step=1)
                s_date = st.date_input("Sell Date")
                s_price = st.number_input("Sell Price", min_value=0.01, value=100.00, step=0.01, format="%.2f")
                
                if st.form_submit_button("Execute Sell"):
                    row_idx = sel_data['idx']
                    buy_price = float(pf_df.at[row_idx, 'Cost_Basis'])
                    buy_date_str = pf_df.at[row_idx, 'Date']
                    
                    ret_pct = ((s_price - buy_price) / buy_price) * 100
                    pl_dollars = (s_price - buy_price) * s_shares
                    
                    # --- RESTORED SPY RETURN CALCULATION ---
                    spy_ret_val = 0.0
                    try:
                        spy_tk = yf.Ticker("SPY")
                        b_dt = pd.to_datetime(buy_date_str)
                        s_dt = pd.to_datetime(s_date)
                        hist = spy_tk.history(start=b_dt, end=s_dt + timedelta(days=5))
                        if not hist.empty:
                            spy_buy = hist.asof(b_dt)['Close']
                            spy_sell = hist.asof(s_dt)['Close']
                            if not pd.isna(spy_buy) and not pd.isna(spy_sell):
                                spy_ret_val = ((spy_sell - spy_buy) / spy_buy) * 100
                    except: 
                        pass
                    # ----------------------------------------

                    # Cash Addition Row (TRADE_CASH, ignores Shadow)
                    cash_id = pf_df["ID"].max() + 1
                    cash_row = pd.DataFrame([{
                        "ID": cash_id, "Ticker": "CASH", "Date": s_date, "Shares": (s_price * s_shares), 
                        "Cost_Basis": 1.0, "Status": "OPEN", "Exit_Date": None, 
                        "Exit_Price": None, "Return": 0.0, "Realized_PL": 0.0, "SPY_Return": 0.0,
                        "Type": "TRADE_CASH", "Shadow_SPY": 0.0
                    }])
                    pf_df = pd.concat([pf_df, cash_row], ignore_index=True)
                    
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
                            "SPY_Return": spy_ret_val,
                            "Type": "STOCK", "Shadow_SPY": 0.0
                        }])
                        pf_df = pd.concat([pf_df, new_closed_row], ignore_index=True)
                    else:
                        pf_df.at[row_idx, 'Status'] = 'CLOSED'
                        pf_df.at[row_idx, 'Exit_Date'] = s_date
                        pf_df.at[row_idx, 'Exit_Price'] = s_price
                        pf_df.at[row_idx, 'Return'] = ret_pct
                        pf_df.at[row_idx, 'Realized_PL'] = pl_dollars
                        pf_df.at[row_idx, 'SPY_Return'] = spy_ret_val

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
            # SHADOW BENCHMARK CALCULATION
            # Fetch SPY price for the deposit date to simulate buying the index
            shadow_shares = 0.0
            try:
                spy_obj = yf.Ticker("SPY")
                start_d = pd.to_datetime(c_date)
                # Look for price in a 5-day window in case of weekend/holiday
                end_d = start_d + timedelta(days=5)
                hist = spy_obj.history(start=start_d, end=end_d)
                
                ref_price = 0.0
                if not hist.empty:
                    ref_price = hist['Close'].iloc[0]
                else:
                    # Fallback to current price if data missing or future date
                    curr = spy_obj.history(period="1d")
                    if not curr.empty: ref_price = curr['Close'].iloc[-1]
                
                if ref_price > 0:
                    shadow_shares = amount / ref_price
            except:
                st.warning("Benchmark Calc Failed. Using 0.")

            final_amt = amount if op_type == "Deposit" else -amount
            final_shadow = shadow_shares if op_type == "Deposit" else -shadow_shares
            
            new_id = 1 if pf_df.empty else pf_df["ID"].max() + 1
            new_row = pd.DataFrame([{
                "ID": new_id, "Ticker": "CASH", "Date": c_date, "Shares": final_amt, 
                "Cost_Basis": 1.0, "Status": "OPEN", "Exit_Date": None, 
                "Exit_Price": None, "Return": 0.0, "Realized_PL": 0.0, "SPY_Return": 0.0,
                "Type": "TRANSFER", "Shadow_SPY": final_shadow
            }])
            pf_df = pd.concat([pf_df, new_row], ignore_index=True)
            save_portfolio(pf_df)
            st.success(f"{op_type} ${amount} (Shadow: {final_shadow:.2f} SPY)")
            st.rerun()

with tab4:
    st.subheader("🧮 Smart Risk Calculator")
    
    # Global setting inside Tab 4 (Calc)
    RISK_UNIT_BASE = st.number_input("Risk Unit ($)", min_value=100, value=2300, step=100)
    
    calc_ticker = st.text_input("Ticker", "").upper()
    
    if calc_ticker:
        try:
            tk = yf.Ticker(calc_ticker)
            df = tk.history(period="1mo")
            if not df.empty:
                curr_p = df['Close'].iloc[-1]
                tr1 = df['High'] - df['Low']
                tr2 = abs(df['High'] - df['Close'].shift(1))
                tr3 = abs(df['Low'] - df['Close'].shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_val = tr.rolling(14).mean().iloc[-1]
                
                raw_stop = curr_p - (2.618 * atr_val)
                smart_stop = round_to_03_07(raw_stop)
                
                if curr_p > smart_stop:
                    risk_per_share = curr_p - smart_stop
                    
                    # 100% Risk Calculation
                    shares_100 = int(RISK_UNIT_BASE / risk_per_share)
                    cap_100 = shares_100 * curr_p
                    
                    # 50% Risk Calculation
                    shares_50 = int((RISK_UNIT_BASE * 0.5) / risk_per_share)
                    cap_50 = shares_50 * curr_p
                    
                    # HTML CARD
                    st.markdown(f"""
                    <div style="background-color: #1E1E1E; border-radius: 10px; padding: 15px; border: 1px solid #333;">
                        <h3 style="margin: 0; color: #FFFFFF;">{calc_ticker}</h3>
                        <div style="display: flex; justify-content: space-between; margin-top: 5px; margin-bottom: 10px; font-size: 14px;">
                            <span style="color: #FFFFFF;">Entry: <b>${curr_p:.2f}</b></span>
                            <span style="color: #FFFFFF;">Stop: <b>${smart_stop:.2f}</b></span>
                        </div>
                        <hr style="margin: 5px 0; border-color: #444;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                            <span style="color: #00FF00; font-weight: bold; font-size: 14px;">STANDARD (100%)</span>
                            <div style="text-align: right;">
                                <div style="font-size: 18px; font-weight: bold; color: #FFF;">{shares_100}</div>
                                <div style="font-size: 12px; color: #AAA;">${cap_100:,.0f}</div>
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                            <span style="color: #FFA500; font-weight: bold; font-size: 14px;">DEFENSIVE (50%)</span>
                            <div style="text-align: right;">
                                <div style="font-size: 18px; font-weight: bold; color: #FFF;">{shares_50}</div>
                                <div style="font-size: 12px; color: #AAA;">${cap_50:,.0f}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if cap_100 > current_cash:
                         st.markdown(f"<div style='color: #FF4444; font-weight: bold; font-size: 14px; margin-top: 10px;'>⚠️ Cash Short (100%): -${cap_100 - current_cash:,.0f}</div>", unsafe_allow_html=True)
                else:
                    st.error("Stop Price calculated above Entry.")
            else:
                st.error("Ticker not found.")
        except:
            st.error("Could not fetch ticker data.")

with tab5:
    action_type = st.radio("Mode", ["Delete Trade", "Edit Trade", "⚠️ FACTORY RESET", "Rebuild Benchmark History"])
    
    if action_type == "⚠️ FACTORY RESET":
        st.error("This will permanently delete all data.")
        if st.button("CONFIRM RESET"):
            if os.path.exists(PORTFOLIO_FILE):
                os.remove(PORTFOLIO_FILE)
                st.success("Reset Complete. Please Refresh.")
                st.cache_data.clear()
                st.rerun()

    elif action_type == "Rebuild Benchmark History":
        st.info("Recalculate Shadow SPY shares for all past Deposits.")
        if st.button("RUN REBUILD"):
             with st.spinner("Fetching history..."):
                 try:
                     spy_hist = yf.Ticker("SPY").history(period="10y")
                     spy_hist.index = pd.to_datetime(spy_hist.index).tz_localize(None)
                     
                     for idx, row in pf_df.iterrows():
                         if row['Type'] == 'TRANSFER' and row['Ticker'] == 'CASH':
                             t_date = pd.to_datetime(row['Date'])
                             amt = float(row['Shares'])
                             
                             idx_loc = spy_hist.index.searchsorted(t_date)
                             price = 0
                             if idx_loc < len(spy_hist):
                                 price = spy_hist.iloc[idx_loc]['Close']
                             else:
                                 price = spy_hist.iloc[-1]['Close']
                                 
                             if price > 0:
                                 pf_df.at[idx, 'Shadow_SPY'] = amt / price
                                 
                     save_portfolio(pf_df)
                     st.success("Benchmark Rebuilt!")
                     st.rerun()
                 except Exception as e:
                     st.error(f"Error: {e}")

    elif not pf_df.empty:
        opts = pf_df.apply(lambda x: f"ID:{x['ID']} | {x['Ticker']} ({x['Status']})", axis=1).tolist()
        sel_str = st.selectbox("Select Trade", opts)
        if sel_str:
            sel_id = int(sel_str.split("|")[0].replace("ID:", "").strip())
            row_idx = pf_df[pf_df['ID'] == sel_id].index[0]
            if action_type == "Delete Trade":
                if st.button("Permanently Delete"):
                    pf_df = pf_df[pf_df['ID'] != sel_id]
                    save_portfolio(pf_df)
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

# --- MAIN EXECUTION ---
if st.button("RUN ANALYSIS", type="primary"):
    
    # Init vars for scope safety
    results = []
    mkt_score = 0
    health_rows = []
    pf_tickers = pf_df['Ticker'].unique().tolist() if not pf_df.empty else []
    pf_tickers = [x for x in pf_tickers if x != "CASH"]
    
    # Initialize cache_d here to ensure it exists
    cache_d = {}

    with st.spinner('Checking Vitals...'):
        market_tickers = ["SPY", "IEF", "^VIX", "CAD=X", "HXT.TO", "RSP"] 
        market_data = {}
        for t in market_tickers:
            try:
                tk = yf.Ticker(t)
                df = tk.history(period="10y", interval="1d") 
                df.index = pd.to_datetime(df.index).tz_localize(None)
                if not df.empty and 'Close' in df.columns: market_data[t] = df
            except: pass
        
        spy = market_data.get("SPY"); ief = market_data.get("IEF"); vix = market_data.get("^VIX")
        rsp = market_data.get("RSP")
        cad = market_data.get("CAD=X")
        cad_rate = cad.iloc[-1]['Close'] if cad is not None else 1.40 

        # ----------------------------------------
        # 1. BUILD ACTIVE HOLDINGS (Safely)
        # ----------------------------------------
        open_pos = pf_df[(pf_df['Status'] == 'OPEN') & (pf_df['Ticker'] != 'CASH')]
        eq_val = 0.0
        total_cost_basis = 0.0 
        pf_rows = []
        
        if not open_pos.empty:
            # Need live prices for holdings
            holding_tickers = open_pos['Ticker'].unique().tolist()
            # Add to cache if missing
            for t in holding_tickers:
                if t not in market_data:
                    try:
                        tk = yf.Ticker(t)
                        hist = tk.history(period="5d")
                        if not hist.empty: market_data[t] = hist
                    except: pass
            
            for idx, row in open_pos.iterrows():
                t = row['Ticker']
                shares = row['Shares']
                cost = row['Cost_Basis']
                
                curr_price = cost 
                if t in market_data and not market_data[t].empty:
                    curr_price = market_data[t]['Close'].iloc[-1]
                
                pos_val = shares * curr_price
                eq_val += pos_val
                total_cost_basis += (shares * cost)
                
                pl = pos_val - (shares * cost)
                pl_pct = ((curr_price - cost) / cost) * 100
                
                pf_rows.append({
                    "Ticker": t, "Shares": int(shares), 
                    "Avg Cost": f"${cost:.2f}", "Current": f"${curr_price:.2f}",
                    "Gain/Loss ($)": f"${pl:+.2f}", "% Return": f"{pl_pct:+.2f}%",
                    "Audit Action": "HOLD"
                })
        
        total_net_worth = current_cash + eq_val
        total_nw_cad = total_net_worth * cad_rate
        open_pl_val = eq_val - total_cost_basis
        open_pl_cad = open_pl_val * cad_rate
        
        st.subheader("💼 Active Holdings")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net Worth (CAD)", f"${total_nw_cad:,.2f}", fmt_delta(open_pl_cad))
        c2.metric("Net Worth (USD)", f"${total_net_worth:,.2f}", fmt_delta(open_pl_val))
        c3.metric("Cash", f"${current_cash:,.2f}")
        c4.metric("Equity", f"${eq_val:,.2f}")
        
        if pf_rows:
            st.markdown(pd.DataFrame(pf_rows).style.pipe(style_portfolio).to_html(), unsafe_allow_html=True)
        else:
            st.info("No active trades.")
        st.write("---")

        # ----------------------------------------
        # 2. CLOSED PERFORMANCE (Safely)
        # ----------------------------------------
        closed_trades = pf_df[(pf_df['Status'] == 'CLOSED') & (pf_df['Ticker'] != 'CASH')]
        if not closed_trades.empty:
            st.subheader("📜 Closed Performance")
            wins = closed_trades[closed_trades['Return'] > 0]
            win_rate = (len(wins) / len(closed_trades)) * 100
            total_pl = closed_trades['Realized_PL'].sum()
            c1, c2 = st.columns(2)
            c1.metric("Win Rate", f"{win_rate:.0f}%")
            c2.metric("Total P&L", f"${total_pl:,.2f}")
            
            hist_view = closed_trades[["Ticker", "Cost_Basis", "Exit_Price", "Realized_PL", "Return"]].copy()
            hist_view["Open Position"] = hist_view["Cost_Basis"].apply(lambda x: f"${x:,.2f}")
            hist_view["Close Position"]
