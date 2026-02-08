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
    st.error("⚠️ titan_config.py is missing! Please create it.")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Titan Strategy", layout="wide")

# --- AUTHENTICATION LOGIC ---
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

# --- LOGIN SCREEN ---
if not st.session_state.authenticated:
    st.title("🛡️ Titan Strategy Login")
    with st.form("login_form"):
        st.text_input("Username", key="username_input")
        st.text_input("Password", type="password", key="password_input")
        st.form_submit_button("Login", on_click=check_login)
    st.stop() 

# ==============================================================================
#  TITAN STRATEGY APP (v56.0 Modular)
# ==============================================================================

current_user = st.session_state.user
PORTFOLIO_FILE = f"portfolio_{current_user}.csv"

st.sidebar.write(f"👤 Logged in as: **{current_user.upper()}**")
if st.sidebar.button("Log Out"):
    logout()

st.title(f"🛡️ Titan Strategy v56.0 ({current_user.upper()})")
st.caption("Institutional Protocol: Modular Architecture")

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

# --- PRECISE TRADINGVIEW RSI ---
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
    pivots = []; trend = 1; last_pivot_val = df['Close'].iloc[0]
    pivots.append((0, last_pivot_val, 1)) 
    
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        if trend == 1:
            if price > last_pivot_val:
                last_pivot_val = price
                if pivots[-1][2] == 1: pivots[-1] = (i, price, 1)
                else: pivots.append((i, price, 1))
            elif price < last_pivot_val * (1 - deviation_pct):
                trend = -1; last_pivot_val = price; pivots.append((i, price, -1))
        else:
            if price < last_pivot_val:
                last_pivot_val = price
                if pivots[-1][2] == -1: pivots[-1] = (i, price, -1)
                else: pivots.append((i, price, -1))
            elif price > last_pivot_val * (1 + deviation_pct):
                trend = 1; last_pivot_val = price; pivots.append((i, price, 1))

    if len(pivots) < 3: return "Range"
    curr_pivot = pivots[-1]
    prev_same = pivots[-3]
    if curr_pivot[2] == 1: 
        return "HH" if curr_pivot[1] > prev_same[1] else "LH"
    else:
        return "LL" if curr_pivot[1] < prev_same[1] else "HL"

# --- SMART STOP HELPER ---
def round_to_03_07(price):
    if pd.isna(price): return 0.0
    whole = int(price)
    c1 = whole + 0.03; c2 = whole + 0.07
    c3 = (whole - 1) + 0.97 if whole > 0 else 0.0
    c4 = (whole - 1) + 0.93 if whole > 0 else 0.0
    candidates = [c for c in [c1, c2, c3, c4] if c > 0]
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
                if r20 > 50: return 'color: #00BFFF; font-weight: bold' if is_rising else 'color: #FF4444; font-weight: bold'
                else: return 'color: #00FF00; font-weight: bold' if is_rising else 'color: #FF4444; font-weight: bold'
            elif r20 > 50: return 'color: #FFA500; font-weight: bold' 
            else: return 'color: #FF4444; font-weight: bold' 
        except: return ''
            
    def color_inst(val):
        if "ACCUMULATION" in val or "BREAKOUT" in val: return 'color: #00FF00; font-weight: bold' 
        if "CAPITULATION" in val: return 'color: #00BFFF; font-weight: bold'       
        if "DISTRIBUTION" in val or "LIQUIDATION" in val: return 'color: #FF4444; font-weight: bold' 
        if "SELLING" in val: return 'color: #FFA500; font-weight: bold'      
        if "HH" in val: return 'color: #CCFFCC'
        if "LL" in val: return 'color: #FFCCCC'
        return 'color: #888888'

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
        try: return 'color: #00ff00; font-weight: bold' if float(val.strip('%').replace('+','')) >= 0 else 'color: #ff4444; font-weight: bold'
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

    if df['Type'].isnull().any() and not df.empty:
        df.loc[(df['Ticker'] != 'CASH') & (df['Type'].isnull()), 'Type'] = "STOCK"
        cash_mask = (df['Ticker'] == 'CASH') & (df['Type'].isnull())
        if cash_mask.any():
            for idx, row in df[cash_mask].iterrows():
                c_date = row['Date']
                if not df[(df['Ticker']!='CASH') & ((df['Date']==c_date) | (df['Exit_Date']==c_date))].empty:
                     df.at[idx, 'Type'] = "TRADE_CASH"
                else:
                     df.at[idx, 'Type'] = "TRANSFER"
        df.to_csv(PORTFOLIO_FILE, index=False)
    if "ID" not in df.columns or df["ID"].isnull().all(): df["ID"] = range(1, len(df) + 1)
    return df

def save_portfolio(df):
    dollar_cols = ['Cost_Basis', 'Exit_Price', 'Realized_PL', 'Return', 'SPY_Return']
    for col in dollar_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
    def clean_shares(row):
        val = row['Shares']; return round(val, 2) if row['Ticker'] == 'CASH' else int(val) if not pd.isna(val) else 0
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
        all_options = list(tc.DATA_MAP.keys())
        b_tick = st.selectbox("Ticker", all_options)
        b_date = st.date_input("Buy Date")
        b_shares = st.number_input("Shares", min_value=1, value=100)
        b_price = st.number_input("Buy Price", min_value=0.01, value=100.00)
        if st.form_submit_button("Execute Buy"):
            new_id = 1 if pf_df.empty else pf_df["ID"].max() + 1
            new_row = pd.DataFrame([{"ID": new_id, "Ticker": b_tick, "Date": b_date, "Shares": b_shares, "Cost_Basis": b_price, "Status": "OPEN", "Type": "STOCK"}])
            pf_df = pd.concat([pf_df, new_row], ignore_index=True)
            if current_cash >= (b_shares * b_price):
                 cash_row = pd.DataFrame([{"ID": pf_df["ID"].max() + 1, "Ticker": "CASH", "Date": b_date, "Shares": -(b_shares * b_price), "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRADE_CASH"}])
                 pf_df = pd.concat([pf_df, cash_row], ignore_index=True)
            save_portfolio(pf_df); st.success(f"Bought {b_tick}"); st.rerun()

with tab2:
    open_trades = pf_df[(pf_df['Status'] == 'OPEN') & (pf_df['Ticker'] != 'CASH')]
    if not open_trades.empty:
        opts = open_trades.apply(lambda x: f"ID:{x['ID']} | {x['Ticker']} | {int(x['Shares'])} sh", axis=1).tolist()
        sel_str = st.selectbox("Select Position", opts)
        if sel_str:
            sel_id = int(sel_str.split("|")[0].replace("ID:", "").strip())
            with st.form("sell_trade"):
                s_shares = st.number_input("Shares to Sell", min_value=1)
                s_date = st.date_input("Sell Date")
                s_price = st.number_input("Sell Price", min_value=0.01)
                if st.form_submit_button("Execute Sell"):
                    row_idx = pf_df[pf_df['ID']==sel_id].index[0]
                    buy_price = float(pf_df.at[row_idx, 'Cost_Basis'])
                    pl_dollars = (s_price - buy_price) * s_shares
                    
                    cash_row = pd.DataFrame([{"ID": pf_df["ID"].max() + 1, "Ticker": "CASH", "Date": s_date, "Shares": (s_price * s_shares), "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRADE_CASH"}])
                    pf_df = pd.concat([pf_df, cash_row], ignore_index=True)
                    
                    if s_shares < pf_df.at[row_idx, 'Shares']:
                        pf_df.at[row_idx, 'Shares'] -= s_shares
                        closed_row = pd.DataFrame([{"ID": pf_df["ID"].max() + 1, "Ticker": pf_df.at[row_idx, 'Ticker'], "Date": pf_df.at[row_idx, 'Date'], "Shares": s_shares, "Cost_Basis": buy_price, "Status": "CLOSED", "Exit_Date": s_date, "Exit_Price": s_price, "Realized_PL": pl_dollars, "Type": "STOCK"}])
                        pf_df = pd.concat([pf_df, closed_row], ignore_index=True)
                    else:
                        pf_df.at[row_idx, 'Status'] = 'CLOSED'
                        pf_df.at[row_idx, 'Exit_Date'] = s_date
                        pf_df.at[row_idx, 'Exit_Price'] = s_price
                        pf_df.at[row_idx, 'Realized_PL'] = pl_dollars
                    
                    save_portfolio(pf_df); st.success(f"Sold. P&L: ${pl_dollars:+.2f}"); st.rerun()

with tab3:
    with st.form("cash_ops"):
        op_type = st.radio("Operation", ["Deposit", "Withdraw"])
        amount = st.number_input("Amount", min_value=0.01)
        c_date = st.date_input("Date")
        if st.form_submit_button("Execute"):
            final_amt = amount if op_type == "Deposit" else -amount
            # Shadow SPY Logic
            try:
                spy_hist = yf.Ticker("SPY").history(start=c_date, end=c_date + timedelta(days=5))
                ref_price = spy_hist['Close'].iloc[0] if not spy_hist.empty else 0
                shadow = (amount / ref_price) * (1 if op_type=="Deposit" else -1) if ref_price > 0 else 0
            except: shadow = 0
            
            new_row = pd.DataFrame([{"ID": pf_df["ID"].max()+1, "Ticker": "CASH", "Date": c_date, "Shares": final_amt, "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRANSFER", "Shadow_SPY": shadow}])
            pf_df = pd.concat([pf_df, new_row], ignore_index=True)
            save_portfolio(pf_df); st.success("Updated"); st.rerun()

with tab4:
    RISK_UNIT_BASE = st.number_input("Risk Unit ($)", min_value=100, value=2300, step=100)
    calc_ticker = st.text_input("Ticker", "").upper()
    if calc_ticker:
        try:
            df = yf.Ticker(calc_ticker).history(period="1mo")
            if not df.empty:
                curr_p = df['Close'].iloc[-1]
                atr_val = calc_atr(df['High'], df['Low'], df['Close']).iloc[-1]
                smart_stop = round_to_03_07(curr_p - (2.618 * atr_val))
                risk_per_share = curr_p - smart_stop
                if risk_per_share > 0:
                    sh_100 = int(RISK_UNIT_BASE / risk_per_share)
                    st.success(f"Entry: ${curr_p:.2f} | Stop: ${smart_stop:.2f}")
                    st.info(f"100% Risk Size: {sh_100} shares (${sh_100*curr_p:,.0f})")
        except: st.error("Ticker error")

with tab5:
    if st.button("Rebuild Benchmark History"):
        with st.spinner("Rebuilding..."):
            spy_hist = yf.Ticker("SPY").history(period="10y")
            for idx, row in pf_df.iterrows():
                if row['Type'] == 'TRANSFER' and row['Ticker'] == 'CASH':
                    try:
                        price = spy_hist.loc[pd.to_datetime(row['Date']):].iloc[0]['Close']
                        pf_df.at[idx, 'Shadow_SPY'] = float(row['Shares']) / price
                    except: pass
            save_portfolio(pf_df); st.success("Done"); st.rerun()

# --- MAIN EXECUTION ---
if st.button("RUN ANALYSIS", type="primary"):
    cache_d = {}
    with st.spinner('Checking Vitals...'):
        market_data = {}
        for t in ["SPY", "IEF", "^VIX", "CAD=X", "HXT.TO", "RSP"]:
            try: market_data[t] = yf.Ticker(t).history(period="10y")
            except: pass
        
        spy = market_data.get("SPY"); vix = market_data.get("^VIX"); rsp = market_data.get("RSP")
        cad_rate = market_data.get("CAD=X").iloc[-1]['Close'] if "CAD=X" in market_data else 1.40 

        # 1. ACTIVE HOLDINGS
        open_pos = pf_df[(pf_df['Status'] == 'OPEN') & (pf_df['Ticker'] != 'CASH')]
        eq_val = 0.0; total_cost_basis = 0.0; pf_rows = []
        
        if not open_pos.empty:
            tickers = open_pos['Ticker'].unique().tolist()
            # Fetch live prices
            live_data = {}
            for t in tickers: 
                try: live_data[t] = yf.Ticker(t).history(period="5d")['Close'].iloc[-1]
                except: live_data[t] = 0
            
            for idx, row in open_pos.iterrows():
                t = row['Ticker']; shares = row['Shares']; cost = row['Cost_Basis']
                curr = live_data.get(t, cost)
                val = shares * curr; eq_val += val; total_cost_basis += (shares * cost)
                pf_rows.append({
                    "Ticker": t, "Shares": int(shares), "Avg Cost": f"${cost:.2f}", "Current": f"${curr:.2f}",
                    "Gain/Loss ($)": f"${(val - (shares*cost)):+.2f}", "% Return": f"{((curr-cost)/cost)*100:+.2f}%",
                    "Audit Action": "HOLD"
                })
        
        total_nw = current_cash + eq_val
        st.subheader("💼 Active Holdings")
        c1, c2, c3 = st.columns(3)
        c1.metric("Net Worth (CAD)", f"${total_nw*cad_rate:,.2f}")
        c2.metric("Net Worth (USD)", f"${total_nw:,.2f}")
        c3.metric("Equity", f"${eq_val:,.2f}")
        if pf_rows: st.markdown(pd.DataFrame(pf_rows).style.pipe(style_portfolio).to_html(), unsafe_allow_html=True)
        else: st.info("No active trades.")
        st.write("---")

        # 2. BENCHMARK
        st.subheader("📈 Performance vs SPY")
        if spy is not None:
            bench_val = pf_df['Shadow_SPY'].sum() * spy['Close'].iloc[-1]
            st.metric("Alpha", f"${(total_nw - bench_val):,.2f}")
        st.write("---")

        # 3. MARKET HEALTH
        if spy is not None and vix is not None and rsp is not None:
            vix_c = vix['Close'].iloc[-1]; score = 0; rows = []
            if vix_c < 20: score += 1; rows.append(["VIX", "PASS"])
            else: rows.append(["VIX", "FAIL"])
            
            spy_sma = calc_sma(spy['Close'], 20).iloc[-1]
            if spy['Close'].iloc[-1] > spy_sma: score += 1; rows.append(["SPY Trend", "PASS"])
            else: rows.append(["SPY Trend", "FAIL"])
            
            rsp_sma = calc_sma(rsp['Close'], 20).iloc[-1]
            if rsp['Close'].iloc[-1] > rsp_sma: score += 1; rows.append(["Breadth", "PASS"])
            else: rows.append(["Breadth", "FAIL"])
            
            st.subheader("🏥 Market Health")
            st.markdown(pd.DataFrame(rows, columns=["Indicator","Status"]).style.pipe(style_daily_health).to_html(escape=False), unsafe_allow_html=True)
            st.write("---")

    # 4. SCANNER
    with st.spinner('Running Scanner...'):
        all_tickers = list(set(list(tc.DATA_MAP.keys()) + [x for x in pf_tickers if x != "CASH"]))
        analysis_db = {}
        # Batch Fetch
        for t in all_tickers:
            if t not in cache_d:
                try: cache_d[t] = yf.Ticker("SPY" if t=="MANL" else t).history(period="2y")
                except: pass
        
        results = []
        for t in all_tickers:
            if t not in cache_d or len(cache_d[t]) < 20: continue
            df = cache_d[t]
            
            # Calcs
            df['SMA18'] = calc_sma(df['Close'], 18)
            df['SMA50'] = calc_sma(df['Close'], 50)
            df['ATR'] = calc_atr(df['High'], df['Low'], df['Close'])
            cloud = calc_ichimoku(df['High'], df['Low'], df['Close'])[0]
            struct = calc_structure(df)
            
            curr = df['Close'].iloc[-1]
            sma18 = df['SMA18'].iloc[-1]
            sma50 = df['SMA50'].iloc[-1]
            cld = cloud.iloc[-1]
            
            score = 0
            if curr > sma18: score += 1
            if sma18 > sma50: score += 1
            if curr > cld: score += 1
            
            action = "AVOID"
            if score == 3: action = "BUY" if struct in ["HH", "HL"] else "SCOUT"
            elif score == 2: action = "WATCH"
            
            smart_stop = round_to_03_07(curr - (2.618 * df['ATR'].iloc[-1]))
            
            cat = tc.DATA_MAP[t][0] if t in tc.DATA_MAP else "OTHER"
            if "99. DATA" in cat: continue
            
            results.append({
                "Sector": cat, "Ticker": t, "Action": action, 
                "Weekly<br>Score": score, "Structure": struct,
                "Stop Price": f"${smart_stop:.2f}"
            })
            
        if results:
            df_final = pd.DataFrame(results).sort_values(["Sector", "Weekly<br>Score"], ascending=[True, False])
            cols = ["Sector", "Ticker", "Action", "Weekly<br>Score", "Structure", "Stop Price"]
            st.markdown(df_final[cols].style.pipe(style_final).to_html(escape=False), unsafe_allow_html=True)
        else:
            st.warning("Scanner returned no results.")
