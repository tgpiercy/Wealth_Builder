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
#  TITAN STRATEGY APP (v55.9)
# ==============================================================================

current_user = st.session_state.user
PORTFOLIO_FILE = f"portfolio_{current_user}.csv"

st.sidebar.write(f"👤 Logged in as: **{current_user.upper()}**")
if st.sidebar.button("Log Out"):
    logout()

st.title(f"🛡️ Titan Strategy v55.9 ({current_user.upper()})")
st.caption("Institutional Protocol: Full Logic Restored")

# --- SECTOR PARENT MAP ---
SECTOR_PARENTS = {
    "04. MATERIALS": "XLB", "05. ENERGY": "XLE", "06. FINANCIALS": "XLF",
    "07. INDUSTRIALS": "XLI", "08. TECHNOLOGY": "XLK", "09. COMM SERVICES": "XLC",
    "10. HEALTH CARE": "XLV", "11. CONS DISCRET": "XLY", "12. CONS STAPLES": "XLP",
    "13. UTILITIES / RE": "XLU", "15. CANADA (HXT)": "HXT.TO", "03. THEMES": "SPY"
}

SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLC", "XLV", "XLY", "XLP", "XLU", "XLRE"]

# --- DATA MAP (Full Universe) ---
DATA_MAP = {
    # INDICES
    "DIA": ["00. INDICES", "SPY", "Dow Jones"], "QQQ": ["00. INDICES", "SPY", "Nasdaq 100"],
    "IWM": ["00. INDICES", "SPY", "Russell 2000"], "IWC": ["00. INDICES", "SPY", "Micro-Cap"],
    "SPY": ["00. INDICES", "SPY", "S&P 500 Base"], "HXT.TO": ["00. INDICES", "SPY", "TSX 60 Index"], 
    "^VIX": ["99. DATA", "SPY", "VIX Volatility"], "RSP": ["99. DATA", "SPY", "S&P Equal Weight"],
    
    # BONDS
    "IEF": ["01. BONDS/FX", "SPY", "7-10 Year Treasuries"], "DLR.TO": ["01. BONDS/FX", None, "USD/CAD Currency"],

    # THEMES
    "BOTZ": ["03. THEMES", "SPY", "Robotics & AI"], "AIQ": ["03. THEMES", "SPY", "Artificial Intel"],
    "ARKG": ["03. THEMES", "SPY", "Genomics"], "ICLN": ["03. THEMES", "SPY", "Clean Energy"],
    "TAN": ["03. THEMES", "SPY", "Solar Energy"], "NLR": ["03. THEMES", "SPY", "Nuclear"],
    "URA": ["03. THEMES", "SPY", "Uranium"], "GDX": ["03. THEMES", "SPY", "Gold Miners"],
    "SILJ": ["03. THEMES", "SPY", "Junior Silver"], "COPX": ["03. THEMES", "SPY", "Copper Miners"],
    "REMX": ["03. THEMES", "SPY", "Rare Earths"], "PAVE": ["03. THEMES", "SPY", "Infrastructure"],

    # SECTORS & INDUSTRIES
    "XLB": ["04. MATERIALS", "SPY", "Materials Sector"], "GLD": ["04. MATERIALS", "SPY", "Gold Bullion"],
    "SLV": ["04. MATERIALS", "SPY", "Silver Bullion"], "AA": ["04. MATERIALS", "XLB", "Alcoa"],
    "XLE": ["05. ENERGY", "SPY", "Energy Sector"], "XOP": ["05. ENERGY", "SPY", "Oil & Gas Exp"],
    "OIH": ["05. ENERGY", "SPY", "Oil Services"], "MLPX": ["05. ENERGY", "SPY", "MLP Infra"],
    "XLF": ["06. FINANCIALS", "SPY", "Financials Sector"], "KBE": ["06. FINANCIALS", "SPY", "Bank ETF"],
    "KRE": ["06. FINANCIALS", "SPY", "Regional Banks"], "IAK": ["06. FINANCIALS", "SPY", "Insurance"],
    "XLI": ["07. INDUSTRIALS", "SPY", "Industrials Sector"], "ITA": ["07. INDUSTRIALS", "SPY", "Aerospace"],
    "IYT": ["07. INDUSTRIALS", "SPY", "Transport"],
    
    # TECH
    "XLK": ["08. TECHNOLOGY", "SPY", "Technology Sector"], "AAPL": ["08. TECHNOLOGY", "QQQ", "Apple Inc"], 
    "MSFT": ["08. TECHNOLOGY", "QQQ", "Microsoft"], "NVDA": ["08. TECHNOLOGY", "QQQ", "Nvidia"],
    "SMH": ["08. TECHNOLOGY", "SPY", "Semiconductors"], "IGV": ["08. TECHNOLOGY", "SPY", "Tech Software"],
    "SMCI": ["08. TECHNOLOGY", "QQQ", "Super Micro"], "DELL": ["08. TECHNOLOGY", "QQQ", "Dell Tech"],
    "WDC": ["08. TECHNOLOGY", "QQQ", "Western Digital"], "ANET": ["08. TECHNOLOGY", "QQQ", "Arista"],
    "CIBR": ["08. TECHNOLOGY", "QQQ", "CyberSecurity"],

    "XLC": ["09. COMM SERVICES", "SPY", "Comm Services"], "META": ["09. COMM SERVICES", "QQQ", "Meta"],
    "GOOGL": ["09. COMM SERVICES", "QQQ", "Alphabet"],
    "XLV": ["10. HEALTH CARE", "SPY", "Health Care Sector"], "IBB": ["10. HEALTH CARE", "SPY", "Biotech"],
    "XLY": ["11. CONS DISCRET", "SPY", "Cons Discret"], "AMZN": ["11. CONS DISCRET", "QQQ", "Amazon"],
    "XLP": ["12. CONS STAPLES", "SPY", "Cons Staples"], "MOO": ["12. CONS STAPLES", "SPY", "Agribusiness"],
    "XLU": ["13. UTIL / RE", "SPY", "Utilities"], "XLRE": ["13. UTIL / RE", "SPY", "Real Estate"],
    
    # CANADA
    "CNQ.TO": ["15. CANADA (HXT)", "HXT.TO", "Cdn Natural"], "CP.TO": ["15. CANADA (HXT)", "HXT.TO", "CP Rail"],
    "WSP.TO": ["15. CANADA (HXT)", "HXT.TO", "WSP Global"], "SHOP.TO": ["15. CANADA (HXT)", "HXT.TO", "Shopify"],
    "CSU.TO": ["15. CANADA (HXT)", "HXT.TO", "Constellation"], "NTR.TO": ["15. CANADA (HXT)", "HXT.TO", "Nutrien"],
    "TECK-B.TO": ["15. CANADA (HXT)", "HXT.TO", "Teck Res"], "RY.TO": ["15. CANADA (HXT)", "HXT.TO", "Royal Bank"],
    "BN.TO": ["15. CANADA (HXT)", "HXT.TO", "Brookfield"]
}

# --- CALCULATIONS ---
def calc_sma(series, length): return series.rolling(window=length).mean()
def calc_ad(high, low, close, volume):
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)
    return (mfm * volume).cumsum()
def calc_ichimoku(high, low, close):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return span_a, span_b
def calc_atr(high, low, close, length=14):
    tr1 = high - low; tr2 = abs(high - close.shift(1)); tr3 = abs(low - close.shift(1))
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(length).mean()
def calc_rsi(series, length):
    delta = series.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean(); avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss; return 100 - (100 / (1 + rs))

# --- ZIG ZAG ENGINE ---
def calc_structure(df, deviation_pct=0.035):
    if len(df) < 50: return "None"
    pivots = []; trend = 1; last_val = df['Close'].iloc[0]
    pivots.append((0, last_val, 1))
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        if trend == 1:
            if price > last_val: last_val = price; pivots[-1] = (i, price, 1)
            elif price < last_val * (1 - deviation_pct): trend = -1; last_val = price; pivots.append((i, price, -1))
        else:
            if price < last_val: last_val = price; pivots[-1] = (i, price, -1)
            elif price > last_val * (1 + deviation_pct): trend = 1; last_val = price; pivots.append((i, price, 1))
    if len(pivots) < 3: return "Range"
    curr, prev = pivots[-1], pivots[-3]
    if curr[2] == 1: return "HH" if curr[1] > prev[1] else "LH"
    return "LL" if curr[1] < prev[1] else "HL"

# --- STYLING ---
def style_final(styler):
    def color_pct(val):
        if isinstance(val, str) and '%' in val:
            try: return 'color: #00ff00; font-weight: bold' if float(val.strip('%')) >= 0 else 'color: #ff0000; font-weight: bold'
            except: return ''
        return ''
    def color_action(val):
        if "BUY" in val: return 'color: #00ff00; font-weight: bold'
        if "SCOUT" in val: return 'color: #00ffff; font-weight: bold'
        if "AVOID" in val: return 'color: #ff4444'
        return ''
    def color_inst(val):
        if "ACCUMULATION" in val: return 'color: #00FF00; font-weight: bold'
        if "DISTRIBUTION" in val: return 'color: #FF4444; font-weight: bold'
        return 'color: #888888'
    return styler.set_properties(**{'background-color': '#222', 'color': 'white'})\
      .map(color_action, subset=["Action"])\
      .map(color_pct, subset=["4W %", "2W %"])\
      .map(color_inst, subset=["Institutional Activity"])\
      .hide(axis='index')

def style_portfolio(styler):
    def color_pl(val):
        if isinstance(val, str) and ('$' in val or '%' in val):
            if '-' in val: return 'color: #FF4444; font-weight: bold'
            return 'color: #00FF00; font-weight: bold'
        return ''
    return styler.map(color_pl, subset=["Gain/Loss ($)", "% Return"])

# --- PORTFOLIO ENGINE ---
def load_portfolio():
    cols = ["ID", "Ticker", "Date", "Shares", "Cost_Basis", "Status", "Exit_Date", "Exit_Price", "Return", "Realized_PL", "SPY_Return", "Type", "Shadow_SPY"]
    if not os.path.exists(PORTFOLIO_FILE):
        df = pd.DataFrame(columns=cols)
        df.to_csv(PORTFOLIO_FILE, index=False)
        return df
    df = pd.read_csv(PORTFOLIO_FILE)
    # Auto-Fix Columns
    if 'Cost' in df.columns and 'Cost_Basis' not in df.columns: df.rename(columns={'Cost': 'Cost_Basis'}, inplace=True)
    if 'Cost_Basis' not in df.columns: df['Cost_Basis'] = 0.0
    if 'Realized_PL' not in df.columns: df['Realized_PL'] = 0.0
    if "ID" not in df.columns: df["ID"] = range(1, len(df) + 1)
    return df

def save_portfolio(df): df.to_csv(PORTFOLIO_FILE, index=False)

# --- SIDEBAR MANAGER ---
st.sidebar.header("💼 Portfolio Manager")
pf_df = load_portfolio()
cash_rows = pf_df[(pf_df['Ticker'] == 'CASH') & (pf_df['Status'] == 'OPEN')]
current_cash = cash_rows['Shares'].sum() if not cash_rows.empty else 0.0
st.sidebar.metric("Cash Available", f"${current_cash:,.2f}")

tab1, tab2, tab3 = st.sidebar.tabs(["🟢 Buy", "🔴 Sell", "💵 Cash"])

with tab1:
    with st.form("buy"):
        tk = st.selectbox("Ticker", sorted(list(DATA_MAP.keys())))
        sh = st.number_input("Shares", min_value=1)
        pr = st.number_input("Price", min_value=0.01)
        if st.form_submit_button("Execute Buy"):
            nid = pf_df["ID"].max() + 1 if not pf_df.empty else 1
            r1 = {"ID":nid, "Ticker":tk, "Date":str(datetime.now().date()), "Shares":sh, "Cost_Basis":pr, "Status":"OPEN", "Type":"STOCK"}
            r2 = {"ID":nid+1, "Ticker":"CASH", "Date":str(datetime.now().date()), "Shares":-(sh*pr), "Cost_Basis":1, "Status":"OPEN", "Type":"DEBIT"}
            pf_df = pd.concat([pf_df, pd.DataFrame([r1, r2])], ignore_index=True)
            save_portfolio(pf_df); st.success(f"Bought {tk}"); st.rerun()

with tab2:
    open_ops = pf_df[(pf_df['Status']=='OPEN') & (pf_df['Ticker']!='CASH')]
    if not open_ops.empty:
        opts = open_ops.apply(lambda x: f"ID:{x['ID']} {x['Ticker']} ({int(x['Shares'])})", axis=1).tolist()
        sel = st.selectbox("Select", opts)
        if sel:
            sid = int(sel.split()[0].split(":")[1])
            with st.form("sell"):
                sp = st.number_input("Sell Price", min_value=0.01)
                if st.form_submit_button("Sell"):
                    idx = pf_df[pf_df['ID']==sid].index[0]
                    shares = pf_df.at[idx, 'Shares']; cost = pf_df.at[idx, 'Cost_Basis']
                    pl = (sp - cost) * shares
                    pf_df.at[idx, 'Status'] = 'CLOSED'; pf_df.at[idx, 'Exit_Price'] = sp
                    pf_df.at[idx, 'Realized_PL'] = pl
                    
                    cid = pf_df["ID"].max() + 1
                    r3 = {"ID":cid, "Ticker":"CASH", "Date":str(datetime.now().date()), "Shares":(shares*sp), "Cost_Basis":1, "Status":"OPEN", "Type":"CREDIT"}
                    pf_df = pd.concat([pf_df, pd.DataFrame([r3])], ignore_index=True)
                    save_portfolio(pf_df); st.success("Sold"); st.rerun()

with tab3:
    with st.form("cash"):
        amt = st.number_input("Amount", min_value=1.0); typ = st.radio("Type", ["Deposit", "Withdraw"])
        if st.form_submit_button("Execute"):
            val = amt if typ == "Deposit" else -amt
            nid = pf_df["ID"].max() + 1 if not pf_df.empty else 1
            r = {"ID":nid, "Ticker":"CASH", "Date":str(datetime.now().date()), "Shares":val, "Cost_Basis":1, "Status":"OPEN", "Type":"TRANSFER"}
            pf_df = pd.concat([pf_df, pd.DataFrame([r])], ignore_index=True)
            save_portfolio(pf_df); st.success("Updated"); st.rerun()

# --- MAIN EXECUTION ---
if st.button("RUN ANALYSIS", type="primary"):
    
    # 1. Market Data
    st.info("Fetching Market Data...")
    mkt_tickers = ["SPY", "^VIX", "RSP"]
    mkt_data = {}
    try:
        raw = yf.download(mkt_tickers, period="6mo", group_by='ticker', progress=False)
        for t in mkt_tickers: mkt_data[t] = raw[t] if t in raw else raw
    except: pass
    
    spy = mkt_data.get("SPY"); vix = mkt_data.get("^VIX")
    
    # 2. Portfolio View
    st.subheader("💼 Active Holdings")
    active_rows = []; total_eq = 0.0; total_cost = 0.0
    open_pos = pf_df[(pf_df['Status']=='OPEN') & (pf_df['Ticker']!='CASH')]
    
    if not open_pos.empty:
        tk_list = open_pos['Ticker'].unique().tolist()
        try:
            live_pr = yf.download(tk_list, period="1d", progress=False)['Close'].iloc[-1]
            for i, row in open_pos.iterrows():
                t = row['Ticker']; s = float(row['Shares']); c = float(row['Cost_Basis'])
                curr = live_pr[t] if isinstance(live_pr, pd.Series) else live_pr
                val = s * curr; cost_val = s * c
                total_eq += val; total_cost += cost_val
                active_rows.append({
                    "Ticker": t, "Shares": int(s), "Avg Cost": f"${c:.2f}", 
                    "Current": f"${curr:.2f}", "Gain/Loss ($)": f"${(val-cost_val):+.2f}",
                    "% Return": f"{((curr-c)/c)*100:+.2f}%"
                })
        except: pass
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Net Worth", f"${(current_cash + total_eq):,.2f}")
    col2.metric("Cash", f"${current_cash:,.2f}")
    col3.metric("Unrealized P/L", f"${(total_eq - total_cost):+,.2f}")
    
    if active_rows: st.dataframe(pd.DataFrame(active_rows).style.pipe(style_portfolio))
    
    # 3. Market Health
    st.subheader("🏥 Market Health")
    if spy is not None and vix is not None:
        v_cur = vix.iloc[-1]['Close']; s_cur = spy.iloc[-1]['Close']
        s_sma = calc_sma(spy['Close'], 20).iloc[-1]
        
        score = 0; sigs = []
        if v_cur < 20: score += 1; sigs.append(f"VIX: {v_cur:.2f} (PASS)")
        else: sigs.append(f"VIX: {v_cur:.2f} (FAIL)")
        if s_cur > s_sma: score += 1; sigs.append(f"SPY Trend: BULLISH")
        else: sigs.append(f"SPY Trend: BEARISH")
        
        c_stat = "🟢 AGGRESSIVE" if score == 2 else ("🟡 CAUTION" if score == 1 else "🔴 DEFENSIVE")
        st.write(f"**Status:** {c_stat} | { ' | '.join(sigs) }")
    
    # 4. Scanner
    st.subheader("🔍 Deep Scanner")
    tickers = list(DATA_MAP.keys())
    batch_size = 20
    results = []
    
    prog = st.progress(0)
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        prog.progress(i / len(tickers))
        try:
            data = yf.download(batch, period="1y", group_by='ticker', progress=False)
            for tk in batch:
                if tk not in data and tk != batch[0]: continue # Handle single/multi
                df = data[tk] if len(batch) > 1 else data
                df = df.dropna()
                if len(df) < 50: continue
                
                # Calcs
                curr = df['Close'].iloc[-1]
                sma18 = calc_sma(df['Close'], 18).iloc[-1]
                sma50 = calc_sma(df['Close'], 50).iloc[-1]
                cloud = calc_ichimoku(df['High'], df['Low'], df['Close'])[0].iloc[-1] # Span A approx
                struct = calc_structure(df)
                atr = calc_atr(df['High'], df['Low'], df['Close']).iloc[-1]
                
                # Logic
                score = 0
                if curr > sma18: score += 1
                if sma18 > sma50: score += 1
                if curr > cloud: score += 1
                
                action = "AVOID"
                if score >= 3:
                    if struct in ["HH", "HL"]: action = "BUY"
                    else: action = "SCOUT"
                elif score == 2: action = "WATCH"
                
                stop = curr - (2.618 * atr)
                
                results.append({
                    "Sector": DATA_MAP[tk][0], "Ticker": tk,
                    "Price": curr, "Action": action, "Structure": struct,
                    "Score": f"{score}/3", "Stop": f"${stop:.2f}",
                    "Institutional Activity": "ACCUMULATION" if struct == "HL" and score == 3 else "NORMAL"
                })
        except: pass
        
    prog.empty()
    if results:
        df_res = pd.DataFrame(results).sort_values(["Sector", "Score"], ascending=[True, False])
        st.dataframe(df_res.style.pipe(style_final), use_container_width=True)
    else: st.warning("No results found.")
