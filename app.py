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

st.title(f"🛡️ Titan Strategy v50.9 ({current_user.upper()})")
st.caption("Institutional Protocol: Clean Split Metrics")

RISK_UNIT = 2300  

# --- DATA MAP ---
DATA_MAP = {
    "MANL": ["BENCH", "SPY", "Manual / Spy Proxy"],
    "VOO": ["BENCH", "SPY", "Vanguard S&P 500"],
    "SPY": ["BENCH", "SPY", "S&P 500"],
    "DIA": ["BENCH", "SPY", "Dow Jones"],
    "QQQ": ["BENCH", "SPY", "Nasdaq 100"],
    "IWM": ["BENCH", "SPY", "Russell 2000"],
    "IWC": ["BENCH", "SPY", "Micro-Cap"],
    "HXT.TO": ["CANADA", "SPY", "TSX 60 Index"],
    "IBB": ["THEME", "SPY", "Biotech Core"],
    "XLB": ["SECTOR", "SPY", "Materials"],
    "XLC": ["SECTOR", "SPY", "Comm Services"],
    "XLE": ["SECTOR", "SPY", "Energy"],
    "XLF": ["SECTOR", "SPY", "Financials"],
    "XLI": ["SECTOR", "SPY", "Industrials"],
    "XLK": ["SECTOR", "SPY", "Technology"],
    "XLV": ["SECTOR", "SPY", "Health Care"],
    "XLY": ["SECTOR", "SPY", "Cons Discret"],
    "XLP": ["SECTOR", "SPY", "Cons Staples"],
    "XLRE": ["SECTOR", "SPY", "Real Estate"],
    "XLU": ["SECTOR", "SPY", "Utilities"],
    "GLD": ["COMMODITY", "SPY", "Gold Bullion"],
    "SLV": ["COMMODITY", "SPY", "Silver Bullion"],
    "BTC-USD": ["COMMODITY", "SPY", "Bitcoin (USD)"],
    "BOTZ": ["THEME", "SPY", "Robotics & AI"],
    "XBI":  ["THEME", "SPY", "Biotechnology"],
    "ICLN": ["THEME", "SPY", "Clean Energy"],
    "REMX": ["THEME", "SPY", "Rare Earth Metals"],
    "GDX":  ["THEME", "SPY", "Gold Miners"],
    "IEF": ["BENCH", "SPY", "7-10 Year Treasuries"],
    "RSP": ["BENCH", "SPY", "S&P 500 Equal Weight"], 
    "^VIX": ["BENCH", "SPY", "VIX Volatility Index"]
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
        market_tickers = ["SPY", "IEF", "^VIX", "CAD=X"]
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
            if spy_c > spy_sma18: status = "BULLISH"; total_exp += 40
            else: status = "BEARISH"
            exposure_rows.append({"Metric": "Trend (SPY > SMA18)", "Value": f"${spy_c:.2f}", "Status": status})

            aligned = pd.concat([spy['Close'], ief['Close']], axis=1, join='inner')
            ratio = aligned.iloc[:,0] / aligned.iloc[:,1]
            ratio_c = ratio.iloc[-1]; ratio_sma18 = calc_sma(ratio, 18).iloc[-1]
            dist_pct = ((ratio_c - ratio_sma18) / ratio_sma18) * 100
            if ratio_c > ratio_sma18: status = "RISK ON"; total_exp += 40
            elif ratio_c >= ratio_sma18 * 0.99: status = "STABLE"; total_exp += 40
            else: status = "RISK OFF"
            exposure_rows.append({"Metric": "Power (SPY:IEF)", "Value": f"{ratio_c:.3f} ({dist_pct:+.2f}%)", "Status": status})

            vix_c = vix.iloc[-1]['Close']
            if vix_c < 20: status = "CALM"; total_exp += 20
            elif vix_c < 25: status = "CAUTION"
            else: status = "PANIC"
            exposure_rows.append({"Metric": "VIX (<20)", "Value": f"{vix_c:.2f}", "Status": status})
            
            exposure_rows.append({"Metric": "TOTAL EXPOSURE", "Value": "---", "Status": f"{total_exp}%"})
            
            risk_per_trade = RISK_UNIT * 0.5 if total_exp <= 40 else RISK_UNIT
            if total_exp == 0: risk_per_trade = 0
            
            df_mkt = pd.DataFrame(exposure_rows)
            st.subheader(f"📊 Market Health: {total_exp}% Invested")
            st.markdown(df_mkt.style.pipe(style_market).to_html(), unsafe_allow_html=True)
            st.write("---")
        else:
            risk_per_trade = 0
            st.error("Market Data Failed to Load")

    with st.spinner('Running Titan Protocol...'):
        tickers = list(DATA_MAP.keys())
        pf_tickers = pf_df['Ticker'].unique().tolist() if not pf_df.empty else []
        pf_tickers = [x for x in pf_tickers if x != "CASH"]
        all_tickers = list(set(tickers + pf_tickers))
        
        cache_d = {}
        cache_d.update(market_data)
        
        for t in all_tickers:
            if t in cache_d: continue
            try:
                fetch_sym = "SPY" if t == "MANL" else t
                tk = yf.Ticker(fetch_sym)
                df = tk.history(period="10y", interval="1d") 
                df.index = pd.to_datetime(df.index).tz_localize(None)
                if not df.empty and 'Close' in df.columns: cache_d[t] = df
            except: pass

        analysis_db = {}
        results = []
        
        for t in all_tickers:
            if t not in cache_d: continue
            is_scanner = t in DATA_MAP and (DATA_MAP[t][0] != "BENCH" or t in ["DIA", "QQQ", "IWM", "IWC", "HXT.TO"])
            
            df_d = cache_d[t].copy()
            df_d['SMA18'] = calc_sma(df_d['Close'], 18)
            df_d['SMA40'] = calc_sma(df_d['Close'], 40)
            df_d['SMA200'] = calc_sma(df_d['Close'], 200)
            df_d['AD'] = calc_ad(df_d['High'], df_d['Low'], df_d['Close'], df_d['Volume'])
            df_d['AD_SMA18'] = calc_sma(df_d['AD'], 18)
            df_d['AD_SMA40'] = calc_sma(df_d['AD'], 40)
            df_d['VolSMA'] = calc_sma(df_d['Volume'], 18)
            
            logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df_w = df_d.resample('W-FRI').agg(logic)
            df_w.dropna(subset=['Close'], inplace=True)
            df_w['SMA8'] = calc_sma(df_w['Close'], 8)
            df_w['SMA18'] = calc_sma(df_w['Close'], 18)
            df_w['SMA40'] = calc_sma(df_w['Close'], 40)
            span_a, span_b = calc_ichimoku(df_w['High'], df_w['Low'], df_w['Close'])
            df_w['Cloud_Top'] = pd.concat([span_a, span_b], axis=1).max(axis=1)

            mom_4w = ""; mom_2w = ""
            if len(df_w) >= 5:
                curr = df_w.iloc[-1]['Close']; prev2 = df_w.iloc[-3]['Close']; prev4 = df_w.iloc[-5]['Close']
                mom_2w = f"{((curr/prev2)-1)*100:.1f}%"
                mom_4w = f"{((curr/prev4)-1)*100:.1f}%"

            dc = df_d.iloc[-1]; dp = df_d.iloc[-2]; wc = df_w.iloc[-1]; wp = df_w.iloc[-2]

            bench_ticker = "SPY"
            if t in DATA_MAP: bench_ticker = DATA_MAP[t][1]
            if t == "MANL": bench_ticker = "SPY"
            
            rs_score_pass = False; rs_breakdown = False
            if bench_ticker in cache_d:
                bench_df = cache_d[bench_ticker]
                aligned = pd.concat([df_d['Close'], bench_df['Close']], axis=1, join='inner')
                rs_series = aligned.iloc[:,0] / aligned.iloc[:,1]
                rs_sma18 = calc_sma(rs_series, 18)
                c_rs = rs_series.iloc[-1]; c_rs_sma = rs_sma18.iloc[-1]; p_rs_sma = rs_sma18.iloc[-2]
                rs_score_pass = (c_rs >= rs_sma18.iloc[-1] * 0.995) and (c_rs_sma >= p_rs_sma)
                pp_rs_sma = rs_sma18.iloc[-3]
                rs_breakdown = (c_rs < c_rs_sma) and (c_rs_sma < p_rs_sma) and (p_rs_sma < pp_rs_sma)

            ad_pass = False
            if not pd.isna(dc['AD_SMA18']):
                ad_pass = (dc['AD'] >= dc['AD_SMA18'] * 0.995) and (dc['AD_SMA18'] >= dp['AD_SMA18'])
            
            vol_msg = "NORMAL"
            if dc['Volume'] > (dc['VolSMA'] * 1.5): vol_msg = "SPIKE (Live)"
            elif dp['Volume'] > (dp['VolSMA'] * 1.5): vol_msg = "SPIKE (Prev)"
            elif dc['Volume'] > dc['VolSMA']: vol_msg = "HIGH (Live)"
            elif dp['Volume'] > dp['VolSMA']: vol_msg = "HIGH (Prev)"

            w_score = 0
            if wc['Close'] > wc['SMA18']: w_score += 1
            if wc['SMA18'] > wp['SMA18']: w_score += 1
            if wc['SMA18'] > wc['SMA40']: w_score += 1
            if wc['Close'] > wc['Cloud_Top']: w_score += 1
            if wc['Close'] > wc['SMA8']: w_score += 1 
            
            d_chk = {'Price': dc['Close'] > dc['SMA18'], 'Trend': dc['SMA18'] >= dp['SMA18'], 'Align': dc['SMA18'] > dc['SMA40'], 'A/D': ad_pass, 'RS': rs_score_pass}
            d_score = sum(d_chk.values())
            
            w_uptrend = (wc['Close'] > wc['SMA18']) and (wc['SMA18'] > wc['SMA40']) and (wc['SMA18'] > wp['SMA18'])
            d_health_ok = (dc['Close'] > dc['SMA18']) and (dc['SMA18'] >= dp['SMA18']) and ad_pass
            w_pulse = "NO"; w_pulse = "GOOD" if w_uptrend and d_health_ok else ("WEAK" if w_uptrend else "NO")

            decision = "AVOID"; reason = "Low Score"
            if w_score >= 4:
                if d_score == 5: decision = "BUY"; reason = "Score 5/5" if w_score==5 else "Score 4/5"
                elif d_score == 4: decision = "SOON" if w_score==5 else "SCOUT"; reason = "D-Score 4"
                elif d_score == 3: decision = "SCOUT"; reason = "Dip Buy"
                else: decision = "WATCH"; reason = "Daily Weak"
            else: decision = "AVOID"; reason = "Weekly Weak"

            w_sma8_pass = wc['Close'] > wc['SMA8']
            w_cloud_pass = wc['Close'] > wc['Cloud_Top']
            
            if not w_sma8_pass: decision = "AVOID"; reason = "BELOW W-SMA8"
            elif "NO" in w_pulse: decision = "AVOID"; reason = "Impulse NO"
            elif "SCOUT" in decision and "WEAK" in w_pulse: decision = "WATCH"; reason = "Impulse Weak"
            elif "BUY" in decision and not (dc['Close'] > dc['SMA200']): decision = "SCOUT"; reason = "Below 200MA"
            elif not w_cloud_pass and "BUY" in decision: decision = "WATCH"; reason = "Cloud Fail"
            elif "SCOUT" in decision and not d_chk['Price']: decision = "WATCH"; reason = "Price Low"
            elif rs_breakdown: decision = "WATCH"; reason = "RS BREAK"
            elif risk_per_trade == 0 and ("BUY" in decision or "SCOUT" in decision): 
                decision = "CAUTION"; reason = "VIX Lock"

            atr = calc_atr(df_d['High'], df_d['Low'], df_d['Close']).iloc[-1]
            stop_dist = 2.618 * atr
            stop_price = dc['Close'] - stop_dist
            
            analysis_db[t] = {
                "Price": dc['Close'],
                "Stop": stop_price,
                "Decision": decision,
                "Reason": reason,
                "ATR": atr
            }

            if is_scanner and t not in ["MANL", "VOO"]:
                final_risk = risk_per_trade / 3 if "SCOUT" in decision else risk_per_trade
                shares = int(final_risk / stop_dist) if stop_dist > 0 and ("BUY" in decision or "SCOUT" in decision) else 0
                stop_pct = (stop_dist / dc['Close']) * 100 if dc['Close'] else 0
                
                row = {
                    "Sector": DATA_MAP[t][0] if t in DATA_MAP else "OTHER", "Ticker": t,
                    "4W %": mom_4w, "2W %": mom_2w,
                    "Weekly<br>SMA8": "PASS" if w_sma8_pass else "FAIL", 
                    "Weekly<br>Impulse": w_pulse, 
                    "Weekly<br>Score": w_score, "Daily<br>Score": d_score,
                    "Structure": "ABOVE 18" if d_chk['Price'] else "BELOW 18",
                    "Ichimoku<br>Cloud": "PASS" if w_cloud_pass else "FAIL", "A/D Breadth": "STRONG" if ad_pass else "WEAK",
                    "Volume": vol_msg, "Action": decision, "Reasoning": reason,
                    "Stop Price": f"${stop_price:.2f} (-{stop_pct:.1f}%)", "Position Size": f"{shares} shares"
                }
                results.append(row)

    if not pf_df.empty:
        open_trades = pf_df[(pf_df['Status'] == 'OPEN') & (pf_df['Ticker'] != 'CASH')]
        
        # AGGREGATE LOGIC
        agg_trades = {}
        for index, row in open_trades.iterrows():
            t = row['Ticker']
            s = row['Shares']
            c = row['Cost_Basis']
            
            if t not in agg_trades: agg_trades[t] = {'Shares': 0, 'TotalCost': 0.0}
            agg_trades[t]['Shares'] += s
            agg_trades[t]['TotalCost'] += (s * c)
            
        equity_val = 0.0
        total_active_cost = 0.0
        pf_rows = []
        
        for t, data in agg_trades.items():
            if t not in analysis_db: continue
            
            total_shares = data['Shares']
            avg_cost = data['TotalCost'] / total_shares if total_shares > 0 else 0
            curr_price = analysis_db[t]['Price']
            
            pos_val = total_shares * curr_price
            equity_val += pos_val
            total_active_cost += data['TotalCost']
            
            pl_pct = ((curr_price - avg_cost) / avg_cost) * 100
            
            decision = analysis_db[t]['Decision']
            stop_price = analysis_db[t]['Stop']
            
            action = "HOLD"
            if "AVOID" in decision: action = "EXIT (Signal Break)"
            elif curr_price < stop_price: action = "EXIT (Stop Hit)"
            elif "WATCH" in decision: action = "CAUTION / HOLD"
            elif "CAUTION" in decision: action = "CAUTION / HOLD"
            
            pf_rows.append({
                "Ticker": t, "Shares": int(total_shares), 
                "Avg Cost": f"${avg_cost:.2f}", "Current": f"${curr_price:.2f}",
                "Position ($)": f"${pos_val:,.2f}",
                "% Return": f"{pl_pct:+.2f}%", 
                "Titan Status": decision, "Audit Action": action
            })

        total_acct = current_cash + equity_val
        cash_pct = (current_cash / total_acct * 100) if total_acct > 0 else 0
        invested_pct = 100 - cash_pct
        total_acct_cad = total_acct * cad_rate
        
        open_pl_val = equity_val - total_active_cost
        open_pl_cad = open_pl_val * cad_rate

        st.subheader("💼 Active Holdings")
        
        # SPLIT INTO 4 COLUMNS
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net Worth (USD)", f"${total_acct:,.2f}", f"${open_pl_val:,.2f}")
        c2.metric("Net Worth (CAD)", f"${total_acct_cad:,.2f}", f"${open_pl_cad:,.2f}")
        c3.metric("Cash Balance", f"${current_cash:,.2f}", f"{cash_pct:.1f}%")
        c4.metric("Invested Equity", f"${equity_val:,.2f}", f"{invested_pct:.1f}%")

        if pf_rows:
            df_pf = pd.DataFrame(pf_rows)
            st.markdown(df_pf.style.pipe(style_portfolio).to_html(), unsafe_allow_html=True)
            st.write("---")

    closed_trades = pf_df[(pf_df['Status'] == 'CLOSED') & (pf_df['Ticker'] != 'CASH')]
    if not closed_trades.empty:
        st.subheader("📜 Closed Performance")
        
        wins = closed_trades[closed_trades['Return'] > 0]
        win_rate = (len(wins) / len(closed_trades)) * 100
        avg_ret = closed_trades['Return'].mean()
        total_pl_dollars = closed_trades['Realized_PL'].sum()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Win Rate", f"{win_rate:.0f}%", f"{len(wins)}/{len(closed_trades)} Trades")
        c2.metric("Cumulative P&L", f"${total_pl_dollars:,.2f}", delta="Net Profit")
        c3.metric("Avg Return %", f"{avg_ret:+.2f}%")
        
        hist_view = closed_trades[["Ticker", "Date", "Exit_Date", "Cost_Basis", "Exit_Price", "Return", "Realized_PL", "SPY_Return"]].copy()
        
        hist_view["% Delta vs SPY"] = hist_view["Return"] - hist_view["SPY_Return"]
        
        hist_view["% Delta vs SPY"] = hist_view["% Delta vs SPY"].apply(lambda x: f"{x:+.2f}%")
        hist_view["Return"] = hist_view["Return"].apply(lambda x: f"{x:+.2f}%")
        hist_view["Realized_PL"] = hist_view["Realized_PL"].apply(lambda x: f"${x:+.2f}")
        
        hist_view.rename(columns={
            "Realized_PL": "$ P&L", 
            "Return": "% Return",
            "Cost_Basis": "Buy Price", 
            "Exit_Price": "Sell Price"
        }, inplace=True)
        
        hist_view = hist_view.drop(columns=["SPY_Return"])
        
        st.dataframe(hist_view.style.pipe(style_history))
        st.write("---")

    st.subheader("🔍 Master Scanner")
    df_final = pd.DataFrame(results).sort_values(["Sector", "Action"], ascending=[True, True])
    cols = ["Sector", "Ticker", "4W %", "2W %", "Weekly<br>SMA8", "Weekly<br>Impulse", "Weekly<br>Score", "Daily<br>Score", "Structure", "Ichimoku<br>Cloud", "A/D Breadth", "Volume", "Action", "Reasoning", "Stop Price", "Position Size"]
    st.markdown(df_final[cols].style.pipe(style_final).to_html(), unsafe_allow_html=True)
