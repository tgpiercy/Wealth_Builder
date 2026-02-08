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
#  TITAN STRATEGY APP (v56.5 Syntax Fix)
# ==============================================================================

current_user = st.session_state.user
PORTFOLIO_FILE = f"portfolio_{current_user}.csv"

st.sidebar.write(f"👤 Logged in as: **{current_user.upper()}**")
if st.sidebar.button("Log Out"):
    logout()

st.title(f"🛡️ Titan Strategy v56.5 ({current_user.upper()})")
st.caption("Institutional Protocol: Syntax Error Resolved")

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
            if price > last_val: last_val = price; pivots[-1] = (i, price, 1)
            elif price < last_val * (1 - deviation_pct): trend = -1; last_val = price; pivots.append((i, price, -1))
        else:
            if price < last_val: last_val = price; pivots[-1] = (i, price, -1)
            elif price > last_val * (1 + deviation_pct): trend = 1; last_val = price; pivots.append((i, price, 1))
    if len(pivots) < 3: return "Range"
    curr, prev = pivots[-1], pivots[-3]
    if curr[2] == 1: return "HH" if curr[1] > prev[1] else "LH"
    return "LL" if curr[1] < prev[1] else "HL"

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
    if 'Cost' in df.columns and 'Cost_Basis' not in df.columns: df.rename(columns={'Cost': 'Cost_Basis'}, inplace=True)
    if 'Cost_Basis' not in df.columns: df['Cost_Basis'] = 0.0
    if "ID" not in df.columns or df["ID"].isnull().all(): df["ID"] = range(1, len(df) + 1)
    # Ensure Shadow_SPY exists
    if 'Shadow_SPY' not in df.columns: df['Shadow_SPY'] = 0.0
    df['Shadow_SPY'] = pd.to_numeric(df['Shadow_SPY'], errors='coerce').fillna(0.0)
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
        b_tick = st.selectbox("Ticker", list(tc.DATA_MAP.keys()))
        b_date = st.date_input("Buy Date")
        b_shares = st.number_input("Shares", min_value=1, value=100)
        b_price = st.number_input("Buy Price", min_value=0.01, value=100.00)
        if st.form_submit_button("Execute Buy"):
            nid = 1 if pf_df.empty else pf_df["ID"].max() + 1
            new_row = pd.DataFrame([{"ID": nid, "Ticker": b_tick, "Date": b_date, "Shares": b_shares, "Cost_Basis": b_price, "Status": "OPEN", "Type": "STOCK"}])
            pf_df = pd.concat([pf_df, new_row], ignore_index=True)
            if current_cash >= (b_shares * b_price):
                 pf_df = pd.concat([pf_df, pd.DataFrame([{"ID": nid+1, "Ticker": "CASH", "Date": b_date, "Shares": -(b_shares * b_price), "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRADE_CASH"}])], ignore_index=True)
            save_portfolio(pf_df); st.success(f"Bought {b_tick}"); st.rerun()

with tab2:
    open_trades = pf_df[(pf_df['Status'] == 'OPEN') & (pf_df['Ticker'] != 'CASH')]
    if not open_trades.empty:
        opts = open_trades.apply(lambda x: f"ID:{x['ID']} | {x['Ticker']} ({int(x['Shares'])})", axis=1).tolist()
        sel = st.selectbox("Select", opts)
        if sel:
            sid = int(sel.split("|")[0].replace("ID:", "").strip())
            with st.form("sell_trade"):
                s_shares = st.number_input("Shares to Sell", min_value=1)
                s_date = st.date_input("Sell Date")
                s_price = st.number_input("Sell Price", min_value=0.01)
                if st.form_submit_button("Execute Sell"):
                    idx = pf_df[pf_df['ID']==sid].index[0]
                    shares = pf_df.at[idx, 'Shares']; cost = pf_df.at[idx, 'Cost_Basis']
                    pl = (s_price - cost) * s_shares
                    pf_df.at[idx, 'Status'] = 'CLOSED'; pf_df.at[idx, 'Exit_Price'] = s_price
                    pf_df.at[idx, 'Realized_PL'] = pl
                    pf_df = pd.concat([pf_df, pd.DataFrame([{"ID": pf_df["ID"].max()+1, "Ticker": "CASH", "Date": s_date, "Shares": (s_price*s_shares), "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRADE_CASH"}])], ignore_index=True)
                    save_portfolio(pf_df); st.success(f"Sold. P&L: ${pl:+.2f}"); st.rerun()

with tab3:
    with st.form("cash_ops"):
        op = st.radio("Operation", ["Deposit", "Withdraw"])
        amt = st.number_input("Amount", min_value=0.01)
        dt = st.date_input("Date")
        if st.form_submit_button("Execute"):
            val = amt if op == "Deposit" else -amt
            try:
                spy_hist = yf.Ticker("SPY").history(start=dt, end=dt + timedelta(days=5))
                ref = spy_hist['Close'].iloc[0] if not spy_hist.empty else 0
                shadow = (amt / ref) * (1 if op=="Deposit" else -1) if ref > 0 else 0
            except: shadow = 0
            pf_df = pd.concat([pf_df, pd.DataFrame([{"ID": pf_df["ID"].max()+1, "Ticker": "CASH", "Date": dt, "Shares": val, "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRANSFER", "Shadow_SPY": shadow}])], ignore_index=True)
            save_portfolio(pf_df); st.success("Updated"); st.rerun()

with tab4:
    RISK_UNIT_BASE = st.number_input("Risk Unit ($)", min_value=100, value=2300)
    calc_tk = st.text_input("Ticker", "").upper()
    if calc_tk:
        try:
            df = yf.Ticker(calc_tk).history(period="1mo")
            if not df.empty:
                cp = df['Close'].iloc[-1]
                atr = calc_atr(df['High'], df['Low'], df['Close']).iloc[-1]
                stop = round_to_03_07(cp - (2.618 * atr))
                risk = cp - stop
                if risk > 0:
                    sh = int(RISK_UNIT_BASE / risk)
                    st.success(f"Entry: ${cp:.2f} | Stop: ${stop:.2f}")
                    st.info(f"Size: {sh} shares (${sh*cp:,.0f})")
        except: st.error("Error")

with tab5:
    if st.button("Rebuild Benchmark"):
        spy_hist = yf.Ticker("SPY").history(period="10y")
        for idx, row in pf_df.iterrows():
            if row['Type'] == 'TRANSFER':
                try:
                    price = spy_hist.loc[pd.to_datetime(row['Date']):].iloc[0]['Close']
                    pf_df.at[idx, 'Shadow_SPY'] = float(row['Shares']) / price
                except: pass
        save_portfolio(pf_df); st.success("Done"); st.rerun()

# --- MAIN EXECUTION ---
if st.button("RUN ANALYSIS", type="primary"):
    
    # 1. MARKET DATA
    cache_d = {}
    with st.spinner('Checking Vitals...'):
        mkt_map = {"SPY":None, "^VIX":None, "RSP":None, "CAD=X":None}
        for t in mkt_map:
            try: mkt_map[t] = yf.Ticker(t).history(period="10y")
            except: pass
        
        spy = mkt_map["SPY"]; vix = mkt_map["^VIX"]; rsp = mkt_map["RSP"]
        cad_rate = mkt_map["CAD=X"].iloc[-1]['Close'] if mkt_map["CAD=X"] is not None else 1.40 

        # 2. PORTFOLIO VIEW
        open_pos = pf_df[(pf_df['Status'] == 'OPEN') & (pf_df['Ticker'] != 'CASH')]
        eq_val = 0.0; total_cost = 0.0; pf_rows = []
        
        if not open_pos.empty:
            tickers = open_pos['Ticker'].unique().tolist()
            live_prices = {}
            for t in tickers:
                try: live_prices[t] = yf.Ticker(t).history(period="5d")['Close'].iloc[-1]
                except: live_prices[t] = 0
            
            for idx, row in open_pos.iterrows():
                t = row['Ticker']; s = row['Shares']; c = row['Cost_Basis']
                curr = live_prices.get(t, c)
                val = s * curr; eq_val += val; total_cost += (s * c)
                pf_rows.append({
                    "Ticker": t, "Shares": int(s), "Avg Cost": f"${c:.2f}", "Current": f"${curr:.2f}",
                    "Gain/Loss ($)": f"${(val - (s*c)):+.2f}", "% Return": f"{((curr-c)/c)*100:+.2f}%",
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

        # 3. BENCHMARK
        st.subheader("📈 Performance vs SPY")
        if spy is not None:
            if 'Shadow_SPY' in pf_df.columns:
                bench_val = pf_df['Shadow_SPY'].sum() * spy['Close'].iloc[-1]
                st.metric("Alpha", f"${(total_nw - bench_val):,.2f}")
            else:
                st.warning("Benchmark initialized. Reload to see Alpha.")
        st.write("---")

        # 4. HEALTH (Fixed List-of-Dicts UI)
        if spy is not None and vix is not None and rsp is not None:
            v_cur = vix.iloc[-1]['Close']; s_cur = spy.iloc[-1]['Close']
            s_sma = calc_sma(spy['Close'], 20).iloc[-1]
            
            mkt_score = 0
            health_rows = []
            
            # VIX
            if v_cur < 17: v_stat = "<span style='color:#00ff00'>NORMAL</span>"; mkt_score += 9
            elif v_cur < 20: v_stat = "<span style='color:#00ff00'>CAUTIOUS</span>"; mkt_score += 6
            elif v_cur < 25: v_stat = "<span style='color:#ffaa00'>DEFENSIVE</span>"; mkt_score += 3
            else: v_stat = "<span style='color:#ff4444'>PANIC</span>"
            health_rows.append({"Indicator": f"VIX Level ({v_cur:.2f})", "Status": v_stat})
            
            # SPY Trend
            if s_cur > s_sma: 
                s_stat = "<span style='color:#00ff00'>PASS</span>"; mkt_score += 1
            else: 
                s_stat = "<span style='color:#ff4444'>FAIL</span>"
            health_rows.append({"Indicator": "SPY Price > SMA18", "Status": s_stat})
            
            # RSP Trend
            r_cur = rsp.iloc[-1]['Close']; r_sma = calc_sma(rsp['Close'], 20).iloc[-1]
            if r_cur > r_sma:
                r_stat = "<span style='color:#00ff00'>PASS</span>"; mkt_score += 1
            else:
                r_stat = "<span style='color:#ff4444'>FAIL</span>"
            health_rows.append({"Indicator": "RSP Price > SMA18", "Status": r_stat})
            
            # TOTAL
            if mkt_score >= 10: msg="AGGRESSIVE (100%)"; cl="#00ff00"
            elif mkt_score >= 8: msg="CAUTIOUS (100%)"; cl="#00ff00"
            elif mkt_score >= 5: msg="DEFENSIVE (50%)"; cl="#ffaa00"
            else: msg="CASH (0%)"; cl="#ff4444"
            
            health_rows.append({"Indicator": "STRATEGY MODE", "Status": f"<span style='color:{cl}; font-weight:bold'>{msg}</span>"})
            
            st.subheader("🏥 Market Health")
            st.markdown(pd.DataFrame(health_rows).style.pipe(style_daily_health).to_html(escape=False), unsafe_allow_html=True)
            st.write("---")

    # 5. SCANNER
    with st.spinner('Running Scanner...'):
        all_tickers = list(set(list(tc.DATA_MAP.keys()) + [x for x in pf_tickers if x != "CASH"]))
        
        for t in all_tickers:
            if t not in cache_d:
                try: cache_d[t] = yf.Ticker("SPY" if t=="MANL" else t).history(period="2y")
                except: pass
        
        results = []
        for t in all_tickers:
            if t not in cache_d or len(cache_d[t]) < 20: continue
            df = cache_d[t]
            
            df['SMA18'] = calc_sma(df['Close'], 18)
            df['SMA50'] = calc_sma(df['Close'], 50)
            df['ATR'] = calc_atr(df['High'], df['Low'], df['Close'])
            df['VolSMA'] = calc_sma(df['Volume'], 18)
            df['RSI5'] = calc_rsi(df['Close'], 5)
            df['RSI20'] = calc_rsi(df['Close'], 20)
            
            curr = df['Close'].iloc[-1]
            sma18 = df['SMA18'].iloc[-1]; sma50 = df['SMA50'].iloc[-1]
            cld = calc_ichimoku(df['High'], df['Low'], df['Close'])[0].iloc[-1]
            struct = calc_structure(df)
            
            score = 0
            if curr > sma18: score += 1
            if sma18 > sma50: score += 1
            if curr > cld: score += 1
            
            action = "AVOID"
            if score == 3: action = "BUY" if struct in ["HH", "HL"] else "SCOUT"
            elif score == 2: action = "WATCH"
            
            smart_stop = round_to_03_07(curr - (2.618 * df['ATR'].iloc[-1]))
            
            r5 = df['RSI5'].iloc[-1]; r20 = df['RSI20'].iloc[-1]
            arrow = "↑" if r5 > df['RSI5'].iloc[-2] else "↓"
            rsi_html = f"{int(r5)}/{int(r20)} {arrow}"
            
            vol_msg = "NORMAL"
            if df['Volume'].iloc[-1] > (df['VolSMA'].iloc[-1] * 1.5): vol_msg = "SPIKE (Live)"
            
            cat = tc.DATA_MAP[t][0] if t in tc.DATA_MAP else "OTHER"
            if "99. DATA" in cat: continue
            
            results.append({
                "Sector": cat, "Ticker": t, "Action": action, 
                "Weekly<br>Score": score, "Structure": struct,
                "Stop Price": f"${smart_stop:.2f}",
                "Dual RSI": rsi_html, "Volume": vol_msg,
                "4W %": f"{((curr/df['Close'].iloc[-20])-1)*100:.1f}%",
                "2W %": f"{((curr/df['Close'].iloc[-10])-1)*100:.1f}%"
            })
            
        if results:
            df_final = pd.DataFrame(results).sort_values(["Sector", "Weekly<br>Score"], ascending=[True, False])
            cols = ["Sector", "Ticker", "4W %", "2W %", "Weekly<br>Score", "Structure", "Volume", "Dual RSI", "Action", "Stop Price"]
            st.markdown(df_final[cols].style.pipe(style_final).to_html(escape=False), unsafe_allow_html=True)
        else:
            st.warning("Scanner returned no results.")
