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

st.title(f"🛡️ Titan Strategy v54.10 ({current_user.upper()})")
st.caption("Institutional Protocol: Streamlined Interface")

# --- SECTOR PARENT MAP ---
SECTOR_PARENTS = {
    "04. MATERIALS": "XLB",
    "05. ENERGY": "XLE",
    "06. FINANCIALS": "XLF",
    "07. INDUSTRIALS": "XLI",
    "08. TECHNOLOGY": "XLK",
    "09. COMM SERVICES": "XLC",
    "10. HEALTH CARE": "XLV",
    "11. CONS DISCRET": "XLY",
    "12. CONS STAPLES": "XLP",
    "13. UTILITIES / RE": "XLU",
    "15. CANADA (HXT)": "HXT.TO",
    "03. THEMES": "SPY"
}

SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLC", "XLV", "XLY", "XLP", "XLU", "XLRE"]

# --- DATA MAP ---
DATA_MAP = {
    # --- 00. INDICES ---
    "DIA": ["00. INDICES", "SPY", "Dow Jones"],
    "QQQ": ["00. INDICES", "SPY", "Nasdaq 100"],
    "IWM": ["00. INDICES", "SPY", "Russell 2000"],
    "IWC": ["00. INDICES", "SPY", "Micro-Cap"],
    "SPY": ["00. INDICES", "SPY", "S&P 500 Base"],
    "HXT.TO": ["00. INDICES", "SPY", "TSX 60 Index"], 
    
    # Hidden VIX & RSP & VOO
    "^VIX": ["99. DATA", "SPY", "VIX Volatility"],
    "RSP": ["99. DATA", "SPY", "S&P 500 Equal Weight"],
    "VOO": ["99. DATA", "SPY", "Vanguard S&P 500"],

    # --- 01. BONDS/FX ---
    "IEF": ["01. BONDS/FX", "SPY", "7-10 Year Treasuries"],
    "DLR.TO": ["01. BONDS/FX", None, "USD/CAD Currency"],

    # --- 03. THEMES ---
    "BOTZ": ["03. THEMES", "SPY", "Robotics & AI"],
    "AIQ": ["03. THEMES", "SPY", "Artificial Intel"],
    "ARKG": ["03. THEMES", "SPY", "Genomics"],
    "ICLN": ["03. THEMES", "SPY", "Clean Energy"],
    "TAN": ["03. THEMES", "SPY", "Solar Energy"],
    "NLR": ["03. THEMES", "SPY", "Nuclear"],
    "URA": ["03. THEMES", "SPY", "Uranium"],
    "GDX": ["03. THEMES", "SPY", "Gold Miners"],
    "SILJ": ["03. THEMES", "SPY", "Junior Silver"], 
    "COPX": ["03. THEMES", "SPY", "Copper Miners"],
    "REMX": ["03. THEMES", "SPY", "Rare Earths"],
    "PAVE": ["03. THEMES", "SPY", "Infrastructure"],

    # --- 04. MATERIALS ---
    "XLB": ["04. MATERIALS", "SPY", "Materials Sector"],
    "GLD": ["04. MATERIALS", "SPY", "Gold Bullion"],
    "SLV": ["04. MATERIALS", "SPY", "Silver Bullion"],

    # --- 05. ENERGY ---
    "XLE": ["05. ENERGY", "SPY", "Energy Sector"],
    "XOP": ["05. ENERGY", "SPY", "Oil & Gas Exp"],
    "OIH": ["05. ENERGY", "SPY", "Oil Services"],
    "MLPX": ["05. ENERGY", "SPY", "MLP Infra"],

    # --- 06. FINANCIALS ---
    "XLF": ["06. FINANCIALS", "SPY", "Financials Sector"],
    "KBE": ["06. FINANCIALS", "SPY", "Bank ETF"],
    "KRE": ["06. FINANCIALS", "SPY", "Regional Banks"],
    "IAK": ["06. FINANCIALS", "SPY", "Insurance"],

    # --- 07. INDUSTRIALS ---
    "XLI": ["07. INDUSTRIALS", "SPY", "Industrials Sector"],
    "ITA": ["07. INDUSTRIALS", "SPY", "Aerospace & Def"],
    "IYT": ["07. INDUSTRIALS", "SPY", "Transport"],

    # --- 08. TECHNOLOGY ---
    "XLK": ["08. TECHNOLOGY", "SPY", "Technology Sector"],
    "AAPL": ["08. TECHNOLOGY", "QQQ", "Apple Inc"], 
    "MSFT": ["08. TECHNOLOGY", "QQQ", "Microsoft"],
    "NVDA": ["08. TECHNOLOGY", "QQQ", "Nvidia"],
    "SMH": ["08. TECHNOLOGY", "SPY", "Semiconductors"],
    "XSD": ["08. TECHNOLOGY", "SPY", "Semi SPDR"], 
    "IGV": ["08. TECHNOLOGY", "SPY", "Tech Software"],
    "SMCI": ["08. TECHNOLOGY", "QQQ", "Super Micro"],
    "DELL": ["08. TECHNOLOGY", "QQQ", "Dell Tech"],
    "WDC": ["08. TECHNOLOGY", "QQQ", "Western Digital"],
    "PSTG": ["08. TECHNOLOGY", "QQQ", "Pure Storage"],
    "ANET": ["08. TECHNOLOGY", "QQQ", "Arista Networks"],

    # --- 09. COMM SERVICES ---
    "XLC": ["09. COMM SERVICES", "SPY", "Comm Services"],
    "META": ["09. COMM SERVICES", "QQQ", "Meta Platforms"],
    "GOOGL": ["09. COMM SERVICES", "QQQ", "Alphabet Inc"],

    # --- 10. HEALTH CARE ---
    "XLV": ["10. HEALTH CARE", "SPY", "Health Care Sector"],
    "IBB": ["10. HEALTH CARE", "SPY", "Biotech Core"],
    "XBI": ["10. HEALTH CARE", "SPY", "Biotech SPDR"],
    "PPH": ["10. HEALTH CARE", "SPY", "Pharma"],
    "IHI": ["10. HEALTH CARE", "SPY", "Med Devices"],

    # --- 11. CONS DISCRET ---
    "XLY": ["11. CONS DISCRET", "SPY", "Cons Discret Sector"],
    "AMZN": ["11. CONS DISCRET", "QQQ", "Amazon"],
    "ITB": ["11. CONS DISCRET", "SPY", "Home Construction"],

    # --- 12. CONS STAPLES ---
    "XLP": ["12. CONS STAPLES", "SPY", "Cons Staples Sector"],
    "MOO": ["12. CONS STAPLES", "SPY", "Agribusiness"],

    # --- 13. UTILITIES / RE ---
    "XLU": ["13. UTIL / RE", "SPY", "Utilities Sector"],
    "XLRE": ["13. UTIL / RE", "SPY", "Real Estate Sector"],
    
    # --- 15. CANADA (HXT) ---
    "CNQ.TO": ["15. CANADA (HXT)", "HXT.TO", "Cdn Natural Res"],
    "CP.TO": ["15. CANADA (HXT)", "HXT.TO", "CP KC Rail"],
    "WSP.TO": ["15. CANADA (HXT)", "HXT.TO", "WSP Global"],
    "SHOP.TO": ["15. CANADA (HXT)", "HXT.TO", "Shopify"],
    "CSU.TO": ["15. CANADA (HXT)", "HXT.TO", "Constellation Soft"],
    "NTR.TO": ["15. CANADA (HXT)", "HXT.TO", "Nutrien"],
    "TECK-B.TO": ["15. CANADA (HXT)", "HXT.TO", "Teck Resources"],

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

def calc_rsi(series, length):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- SMART STOP HELPER ---
def round_to_03_07(price):
    if pd.isna(price): return 0.0
    whole = int(price)
    # Candidates for this whole number
    c1 = whole + 0.03
    c2 = whole + 0.07
    # Candidates for previous whole number (if price is like 100.01)
    c3 = (whole - 1) + 0.97 if whole > 0 else 0.0
    c4 = (whole - 1) + 0.93 if whole > 0 else 0.0
    
    # Find closest
    candidates = [c1, c2, c3, c4]
    candidates = [c for c in candidates if c > 0]
    
    if not candidates: return price 
    
    best_c = min(candidates, key=lambda x: abs(x - price))
    return best_c

# --- STYLING ---
def style_final(styler):
    def color_pct(val):
        if isinstance(val, str) and '%' in val:
            try:
                num = float(val.strip('%'))
                return 'color: #00ff00; font-weight: bold' if num >= 0 else 'color: #ff0000; font-weight: bold'
            except: return ''
        return ''

    def color_rsi(val):
        try:
            parts = val.split()
            if len(parts) < 2: return ''
            nums = parts[0].split('/')
            r5 = float(nums[0])
            r20 = float(nums[1])
            arrow = parts[1]
            is_rising = (arrow == "↑")
            
            if r5 >= r20:
                if r20 > 50:
                    if is_rising: return 'color: #00BFFF; font-weight: bold' 
                    else: return 'color: #FF4444; font-weight: bold' 
                else:
                    return 'color: #00FF00; font-weight: bold' 
            elif r20 > 50:
                return 'color: #FFA500; font-weight: bold' 
            else:
                return 'color: #FF4444; font-weight: bold' 
        except:
            return ''

    # ROW-WISE TICKER HIGHLIGHTING
    def highlight_ticker_row(row):
        styles = ['' for _ in row.index]
        if 'Ticker' not in row.index: return styles
        
        ticker_idx = row.index.get_loc('Ticker')
        action = str(row.get('Action', '')).upper()
        vol = str(row.get('Volume', '')).upper()
        rsi_html = str(row.get('Dual RSI', ''))
        
        if "AVOID" in action:
             pass 
        elif "00BFFF" in rsi_html and "SPIKE" in vol:
             styles[ticker_idx] = 'background-color: #0044CC; color: white; font-weight: bold' 
             return styles

        if "BUY" in action:
            styles[ticker_idx] = 'background-color: #006600; color: white; font-weight: bold' 
        elif "SCOUT" in action:
            styles[ticker_idx] = 'background-color: #005555; color: white; font-weight: bold' 
        elif "SOON" in action or "CAUTION" in action:
            styles[ticker_idx] = 'background-color: #CC5500; color: white; font-weight: bold' 
            
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
    # Defined Schema with Shadow Tracking
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

    # --- AGGRESSIVE AUTO-CORRECTION FOR TYPES ---
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

tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["🟢 Buy", "🔴 Sell", "💵 Cash", "🛠️ Fix", "🧮 Calc"])

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

with tab5:
    st.subheader("🧮 Smart Risk Calculator")
    
    # Moved global setting inside the tab
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
                
                st.write("---")
                
                # Logic strictly uses calculated values
                if curr_p > smart_stop:
                    risk_per_share = curr_p - smart_stop
                    shares = int(RISK_UNIT_BASE / risk_per_share)
                    capital = shares * curr_p
                    
                    st.markdown(f"**Price:** ${curr_p:.2f}")
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <div><div style="font-size: 12px; color: #888;">Shares</div><div style="font-size: 18px; font-weight: bold; color: #FFF;">{shares}</div></div>
                        <div><div style="font-size: 12px; color: #888;">Capital</div><div style="font-size: 18px; font-weight: bold; color: #FFF;">${capital:,.2f}</div></div>
                        <div><div style="font-size: 12px; color: #888;">Stop</div><div style="font-size: 18px; font-weight: bold; color: #FFF;">${smart_stop:.2f}</div></div>
                    </div>
                    """, unsafe_allow_html=True)

                    if capital > current_cash:
                         st.markdown(f"<div style='color: #FF4444; font-weight: bold; font-size: 14px;'>⚠️ Over Budget by ${capital - current_cash:,.2f}</div>", unsafe_allow_html=True)
                    else:
                         st.markdown(f"<div style='color: #00FF00; font-weight: bold; font-size: 14px;'>✅ Trade Approved</div>", unsafe_allow_html=True)
                else:
                    st.error("Stop Price calculated above Entry (Short trend?).")
            else:
                st.error("Ticker not found.")
        except:
            st.error("Could not fetch ticker data.")

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
            hist_view["Close Position"] = hist_view["Exit_Price"].apply(lambda x: f"${x:,.2f}")
            hist_view["P/L"] = hist_view["Realized_PL"].apply(lambda x: f"${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}")
            hist_view["% Return"] = hist_view["Return"].apply(lambda x: f"{x:+.2f}%")
            st.dataframe(hist_view[["Ticker", "Open Position", "Close Position", "P/L", "% Return"]].style.pipe(style_history))
            st.write("---")

        # ----------------------------------------
        # 3. BENCHMARK TRACKER (Shadow SPY)
        # ----------------------------------------
        st.subheader("📈 Performance vs SPY Benchmark")
        shadow_shares_total = pf_df['Shadow_SPY'].sum()
        if spy is not None:
            curr_spy = spy['Close'].iloc[-1]
            bench_val = shadow_shares_total * curr_spy
            alpha = total_net_worth - bench_val
            alpha_pct = ((total_net_worth - bench_val) / bench_val * 100) if bench_val > 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Titan Net Worth", f"${total_net_worth:,.2f}")
            c2.metric("SPY Benchmark", f"${bench_val:,.2f}")
            c3.metric("Alpha (Edge)", f"${alpha:,.2f}", f"{alpha_pct:+.2f}%")
        else:
            st.warning("Waiting for SPY data...")
        st.write("---")

        # ----------------------------------------
        # 4. MARKET HEALTH
        # ----------------------------------------
        if spy is not None and ief is not None and vix is not None and rsp is not None:
            vix_c = vix.iloc[-1]['Close']
            if vix_c < 17: v_pts=9; v_s="<span style='color:#00ff00'>NORMAL</span>"
            elif vix_c < 20: v_pts=6; v_s="<span style='color:#00ff00'>CAUTIOUS</span>"
            elif vix_c < 25: v_pts=3; v_s="<span style='color:#ffaa00'>DEFENSIVE</span>"
            else: v_pts=0; v_s="<span style='color:#ff4444'>PANIC</span>"
            mkt_score += v_pts
            health_rows.append({"Indicator": f"VIX Level ({vix_c:.2f})", "Status": v_s})
            
            s_c = spy.iloc[-1]['Close']; s_sma18 = calc_sma(spy['Close'], 18); s_sma8 = calc_sma(spy['Close'], 8)
            s_18c = s_sma18.iloc[-1]; s_18p = s_sma18.iloc[-2]
            s_8c = s_sma8.iloc[-1]; s_8p = s_sma8.iloc[-2]
            cond1 = s_c > s_18c; cond2 = s_18c >= s_18p; cond3 = s_8c > s_8p
            if cond1 and cond2 and cond3: mkt_score += 1
            s_p = "<span style='color:#00ff00'>PASS</span>"; s_f = "<span style='color:#ff4444'>FAIL</span>"
            s_r = "<span style='color:#00ff00'>RISING</span>"; s_d = "<span style='color:#ff4444'>FALLING</span>"
            health_rows.append({"Indicator": "SPY Price > SMA18", "Status": s_p if cond1 else s_f})
            health_rows.append({"Indicator": "SPY SMA18 Rising", "Status": s_r if cond2 else s_d})
            health_rows.append({"Indicator": "SPY SMA8 Rising", "Status": s_r if cond3 else s_d})
            
            r_c = rsp.iloc[-1]['Close']; r_sma18 = calc_sma(rsp['Close'], 18); r_sma8 = calc_sma(rsp['Close'], 8)
            r_18c = r_sma18.iloc[-1]; r_18p = r_sma18.iloc[-2]
            r_8c = r_sma8.iloc[-1]; r_8p = r_sma8.iloc[-2]
            r_cond1 = r_c > r_18c; r_cond2 = r_18c >= r_18p; r_cond3 = r_8c > r_8p
            if r_cond1 and r_cond2 and r_cond3: mkt_score += 1
            health_rows.append({"Indicator": "RSP Price > SMA18", "Status": s_p if r_cond1 else s_f})
            health_rows.append({"Indicator": "RSP SMA18 Rising", "Status": s_r if r_cond2 else s_d})
            health_rows.append({"Indicator": "RSP SMA8 Rising", "Status": s_r if r_cond3 else s_d})
            
            if mkt_score >= 10: msg="AGGRESSIVE (100%)"; cl="#00ff00"; risk_per_trade=RISK_UNIT_BASE
            elif mkt_score >= 8: msg="CAUTIOUS BUY (100%)"; cl="#00ff00"; risk_per_trade=RISK_UNIT_BASE
            elif mkt_score >= 5: msg="DEFENSIVE (50%)"; cl="#ffaa00"; risk_per_trade=RISK_UNIT_BASE*0.5
            else: msg="CASH / SAFETY (0%)"; cl="#ff4444"; risk_per_trade=0
            
            tc = "#00ff00" if mkt_score >= 8 else ("#ffaa00" if mkt_score >= 5 else "#ff4444")
            health_rows.append({"Indicator": "TOTAL SCORE", "Status": f"<span style='color:{tc}; font-weight:bold'>TOTAL: {mkt_score}/11</span>"})
            health_rows.append({"Indicator": "STRATEGY MODE", "Status": f"<span style='color:{cl}; font-weight:bold'>{msg}</span>"})
            
            st.subheader("🏥 Daily Market Health")
            st.markdown(pd.DataFrame(health_rows)[["Indicator", "Status"]].style.pipe(style_daily_health).to_html(escape=False), unsafe_allow_html=True)
            st.write("---")

    # ----------------------------------------
    # 5. RUN SCANNER (Using cache_d)
    # ----------------------------------------
    with st.spinner('Running Scanner...'):
        all_tickers = list(set(list(DATA_MAP.keys()) + pf_tickers))
        # Ensure cache_d has everything from market_data
        cache_d.update(market_data)
        
        analysis_db = {}
        for t in all_tickers:
            if t not in cache_d: 
                try:
                    fetch_sym = "SPY" if t == "MANL" else t
                    tk = yf.Ticker(fetch_sym)
                    df = tk.history(period="2y", interval="1d") 
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    if not df.empty and 'Close' in df.columns: 
                        cache_d[t] = df
                        df_d = df
                    else: continue
                except: continue
            else:
                df_d = cache_d[t].copy()

            # --- DATA SAFETY CHECK ---
            if len(df_d) < 20: continue

            # Calculations
            df_d['SMA18'] = calc_sma(df_d['Close'], 18)
            df_d['SMA40'] = calc_sma(df_d['Close'], 40)
            df_d['SMA200'] = calc_sma(df_d['Close'], 200)
            df_d['AD'] = calc_ad(df_d['High'], df_d['Low'], df_d['Close'], df_d['Volume'])
            df_d['AD_SMA18'] = calc_sma(df_d['AD'], 18)
            df_d['AD_SMA40'] = calc_sma(df_d['AD'], 40)
            df_d['VolSMA'] = calc_sma(df_d['Volume'], 18)
            df_d['RSI5'] = calc_rsi(df_d['Close'], 5)
            df_d['RSI20'] = calc_rsi(df_d['Close'], 20)
            
            logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df_w = df_d.resample('W-FRI').agg(logic)
            
            # --- WEEKLY DATA SAFETY CHECK ---
            if len(df_w) < 2: continue
            
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
            
            if bench_ticker is None:
                rs_score_pass = True 
            elif bench_ticker in cache_d:
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
            
            # --- SMART STOP CALCULATION IN SCANNER ---
            raw_stop = dc['Close'] - (2.618 * atr)
            smart_stop_val = round_to_03_07(raw_stop)
            stop_dist = dc['Close'] - smart_stop_val
            stop_pct = (stop_dist / dc['Close']) * 100 if dc['Close'] else 0
            
            # --- DUAL RSI LOGIC (HTML Construction) ---
            r5 = df_d['RSI5'].iloc[-1] if not pd.isna(df_d['RSI5'].iloc[-1]) else 50
            r5_prev = df_d['RSI5'].iloc[-2] if len(df_d) > 1 and not pd.isna(df_d['RSI5'].iloc[-2]) else r5
            r20 = df_d['RSI20'].iloc[-1] if not pd.isna(df_d['RSI20'].iloc[-1]) else 50
            
            is_rising = r5 > r5_prev
            arrow = "↑" if is_rising else "↓"
            arrow_col = "#00FF00" if is_rising else "#FF4444"
            
            num_col = "#FF4444"
            if r5 >= r20:
                if r20 > 50:
                    if is_rising: num_col = "#00BFFF" 
                    else: num_col = "#FFA500" 
                else:
                    if is_rising: num_col = "#00FF00"
                    else: num_col = "#FF4444" 
            elif r20 > 50:
                num_col = "#FFA500"
            
            rsi_msg = f"<span style='color:{num_col}'><b>{int(r5)}/{int(r20)}</b></span> <span style='color:{arrow_col}'><b>{arrow}</b></span>"
            
            analysis_db[t] = {
                "Decision": decision,
                "Reason": reason,
                "Price": dc['Close'],
                "Stop": smart_stop_val, # Use Smart Stop
                "StopPct": stop_pct,
                "ATR": atr,
                "Mom4W": mom_4w,
                "Mom2W": mom_2w,
                "W_SMA8_Pass": w_sma8_pass,
                "W_Pulse": w_pulse,
                "W_Score": w_score,
                "D_Score": d_score,
                "D_Chk_Price": d_chk['Price'],
                "W_Cloud": w_cloud_pass,
                "AD_Pass": ad_pass,
                "Vol_Msg": vol_msg,
                "RSI_Msg": rsi_msg
            }

        # --- PASS 2: Build Results & Apply Sector Lock ---
        results = []
        for t in all_tickers:
            cat_name = DATA_MAP[t][0] if t in DATA_MAP else "OTHER"
            if "99. DATA" in cat_name: continue 
            
            is_scanner = t in DATA_MAP and (DATA_MAP[t][0] != "BENCH" or t in ["DIA", "QQQ", "IWM", "IWC", "HXT.TO"])
            if not is_scanner or t not in analysis_db: continue
            
            db = analysis_db[t]
            
            final_decision = db['Decision']
            final_reason = db['Reason']
            
            if cat_name in SECTOR_PARENTS:
                parent = SECTOR_PARENTS[cat_name]
                if parent in analysis_db and "AVOID" in analysis_db[parent]['Decision']:
                    if t != parent: 
                        final_decision = "AVOID"
                        final_reason = "Sector Lock"

            sort_rank = 1
            if "00. INDICES" in cat_name: sort_rank = 0 
            elif t in ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLV", "XLY", "XLP", "XLU", "XLRE", "HXT.TO"]: sort_rank = 0 

            # SHARES CALC & BLUE SPIKE LOGIC
            rsi_html = db['RSI_Msg']
            vol_str = db['Vol_Msg']
            is_blue_spike = ("00BFFF" in rsi_html) and ("SPIKE" in vol_str)
            
            final_risk = risk_per_trade / 3 if "SCOUT" in final_decision else risk_per_trade
            if is_blue_spike: final_risk = risk_per_trade

            stop_dist_value = db['Price'] - db['Stop']
            
            if "AVOID" in final_decision and not is_blue_spike:
                disp_stop = ""
                disp_shares = ""
            else:
                shares = int(final_risk / stop_dist_value) if stop_dist_value > 0 else 0
                disp_stop = f"${db['Stop']:.2f} (-{db['StopPct']:.1f}%)"
                disp_shares = f"{shares} shares"

            row = {
                "Sector": cat_name, 
                "Ticker": t,
                "Rank": sort_rank, 
                "4W %": db['Mom4W'], "2W %": db['Mom2W'],
                "Weekly<br>SMA8": "PASS" if db['W_SMA8_Pass'] else "FAIL", 
                "Weekly<br>Impulse": db['W_Pulse'], 
                "Weekly<br>Score": db['W_Score'], "Daily<br>Score": db['D_Score'],
                "Structure": "ABOVE 18" if db['D_Chk_Price'] else "BELOW 18",
                "Ichimoku<br>Cloud": "PASS" if db['W_Cloud'] else "FAIL", 
                "A/D Breadth": "STRONG" if db['AD_Pass'] else "WEAK",
                "Volume": db['Vol_Msg'], 
                "Dual RSI": db['RSI_Msg'],
                "Action": final_decision, 
                "Reasoning": final_reason,
                "Stop Price": disp_stop, 
                "Position Size": disp_shares
            }
            results.append(row)
            
            if t == "HXT.TO":
                row_cad = row.copy()
                row_cad["Sector"] = "15. CANADA (HXT)"
                row_cad["Rank"] = 0 
                results.append(row_cad)
            
            if t in SECTOR_ETFS:
                row_sec = row.copy()
                row_sec["Sector"] = "02. SECTORS (SUMMARY)"
                row_sec["Rank"] = 0
                results.append(row_sec)

    if results:
        df_final = pd.DataFrame(results).sort_values(["Sector", "Rank", "Ticker"], ascending=[True, True, True])
        
        def clean_sector_name(val):
            if ". " in val: val = val.split(". ", 1)[1]
            val = val.replace("(SUMMARY)", "").strip()
            return val

        df_final["Sector"] = df_final["Sector"].apply(clean_sector_name)

        cols = ["Sector", "Ticker", "4W %", "2W %", "Weekly<br>SMA8", "Weekly<br>Impulse", "Weekly<br>Score", "Daily<br>Score", "Structure", "Ichimoku<br>Cloud", "A/D Breadth", "Volume", "Dual RSI", "Action", "Reasoning", "Stop Price", "Position Size"]
        
        st.markdown(df_final[cols].style.pipe(style_final).to_html(escape=False), unsafe_allow_html=True)
    else:
        st.warning("Scanner returned no results. Check market data connection.")
