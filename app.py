import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

# --- IMPORT CONFIG ---
try:
    import titan_config as tc
except ImportError:
    st.error("⚠️ CRITICAL ERROR: `titan_config.py` is missing. Please create it.")
    st.stop()

# --- SAFE IMPORT FOR PLOTLY ---
try:
    import plotly.graph_objects as go
except ImportError:
    st.warning("⚠️ Plotly not found. RRG Charts will not work. (pip install plotly)")
    go = None

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
#  TITAN STRATEGY APP (v61.1 Final Sync)
# ==============================================================================

current_user = st.session_state.user
PORTFOLIO_FILE = f"portfolio_{current_user}.csv"

st.sidebar.write(f"👤 Logged in as: **{current_user.upper()}**")
if st.sidebar.button("Log Out"):
    logout()

st.title(f"🛡️ Titan Strategy v61.1 ({current_user.upper()})")
st.caption("Institutional Protocol: Stylizer Column Sync")

# --- CALCULATIONS ---
def calc_sma(series, length): return series.rolling(window=length).mean()
def calc_ad(high, low, close, volume):
    mfm = ((close - low) - (high - close)) / (high - low); mfm = mfm.fillna(0.0); mfv = mfm * volume
    return mfv.cumsum()
def calc_ichimoku(high, low, close):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26); span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return span_a, span_b
def calc_atr(high, low, close, length=14):
    try:
        tr1 = high - low; tr2 = abs(high - close.shift(1)); tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(com=length-1, adjust=False).mean()
    except: return pd.Series(0, index=close.index)
def calc_rsi(series, length=14):
    try:
        delta = series.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=length-1, adjust=False).mean()
        avg_loss = loss.ewm(com=length-1, adjust=False).mean()
        rs = avg_gain / avg_loss; rs = rs.fillna(0)
        return 100 - (100 / (1 + rs))
    except: return pd.Series(50, index=series.index)

# --- ZIG ZAG ENGINE ---
def calc_structure(df, deviation_pct=0.035):
    if len(df) < 50: return "None"
    pivots = []; trend = 1; last_val = df['Close'].iloc[0]; pivots.append((0, last_val, 1))
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        if trend == 1:
            if price > last_val:
                last_val = price
                if pivots[-1][2] == 1: pivots[-1] = (i, price, 1)
                else: pivots.append((i, price, 1))
            elif price < last_val * (1 - deviation_pct):
                trend = -1; last_val = price; pivots.append((i, price, -1))
        else:
            if price < last_val:
                last_val = price
                if pivots[-1][2] == -1: pivots[-1] = (i, price, -1)
                else: pivots.append((i, price, -1))
            elif price > last_val * (1 + deviation_pct):
                trend = 1; last_val = price; pivots.append((i, price, 1))
    if len(pivots) < 3: return "Range"
    return ("HH" if pivots[-1][1] > pivots[-3][1] else "LH") if pivots[-1][2] == 1 else ("LL" if pivots[-1][1] < pivots[-3][1] else "HL")

def round_to_03_07(price):
    if pd.isna(price): return 0.0
    whole = int(price); candidates = [c for c in [whole + 0.03, whole + 0.07, (whole - 1) + 0.97, (whole - 1) + 0.93] if c > 0]
    return min(candidates, key=lambda x: abs(x - price)) if candidates else price

# --- UNIFIED DATA ENGINE ---
@st.cache_data(ttl=3600) 
def fetch_master_data(ticker_list):
    unique_tickers = sorted(list(set(ticker_list))) 
    data_map = {}
    for t in unique_tickers:
        try:
            fetch_sym = "SPY" if t == "MANL" else t
            tk = yf.Ticker(fetch_sym)
            df = tk.history(period="2y", interval="1d")
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if not df.empty and 'Close' in df.columns:
                data_map[t] = df
        except: continue
    return data_map

def prepare_rrg_inputs(data_map, tickers, benchmark):
    df_wide = pd.DataFrame()
    if benchmark in data_map:
        bench_df = data_map[benchmark].resample('W-FRI').last()
        df_wide[benchmark] = bench_df['Close']
    for t in tickers:
        if t in data_map and t != benchmark:
            w_df = data_map[t].resample('W-FRI').last()
            df_wide[t] = w_df['Close']
    return df_wide.dropna()

# --- RRG LOGIC ---
def calculate_rrg_math(price_data, benchmark_col, window_rs=14, window_mom=5, smooth_factor=3):
    if benchmark_col not in price_data.columns: return pd.DataFrame(), pd.DataFrame()
    df_ratio = pd.DataFrame(); df_mom = pd.DataFrame()
    for col in price_data.columns:
        if col != benchmark_col:
            try:
                rs = price_data[col] / price_data[benchmark_col]
                mean = rs.rolling(window_rs).mean(); std = rs.rolling(window_rs).std()
                ratio = 100 + ((rs - mean) / std) * 1.5
                df_ratio[col] = ratio
            except: continue
    for col in df_ratio.columns:
        try: df_mom[col] = 100 + (df_ratio[col] - df_ratio[col].rolling(window_mom).mean()) * 2
        except: continue
    return df_ratio.rolling(smooth_factor).mean().dropna(), df_mom.rolling(smooth_factor).mean().dropna()

def generate_full_rrg_snapshot(data_map, benchmark="SPY"):
    try:
        all_tickers = list(data_map.keys())
        status_map = {}
        
        # 1. Primary Tickers vs SPY
        wide_df = prepare_rrg_inputs(data_map, all_tickers, benchmark)
        if not wide_df.empty:
            r, m = calculate_rrg_math(wide_df, benchmark)
            if not r.empty:
                l_idx = r.index[-1]
                for t in r.columns:
                    vr, vm = r.at[l_idx, t], m.at[l_idx, t]
                    if vr > 100 and vm > 100: status_map[t] = "LEADING"
                    elif vr > 100 and vm < 100: status_map[t] = "WEAKENING"
                    elif vr < 100 and vm < 100: status_map[t] = "LAGGING"
                    else: status_map[t] = "IMPROVING"

        # 2. SPY vs IEF specifically
        spy_ief_wide = prepare_rrg_inputs(data_map, ["SPY"], "IEF")
        if not spy_ief_wide.empty:
            rs, ms = calculate_rrg_math(spy_ief_wide, "IEF")
            if not rs.empty:
                l_idx = rs.index[-1]
                vrs, vms = rs.at[l_idx, "SPY"], ms.at[l_idx, "SPY"]
                if vrs > 100 and vms > 100: status_map["SPY"] = "LEADING"
                elif vrs > 100 and vms < 100: status_map["SPY"] = "WEAKENING"
                elif vrs < 100 and vms < 100: status_map["SPY"] = "LAGGING"
                else: status_map["SPY"] = "IMPROVING"
        return status_map
    except: return {}

def plot_rrg_chart(ratios, momentums, labels_map, title, is_dark):
    if go is None: return None
    fig = go.Figure()
    text_col = "white" if is_dark else "black"
    template = "plotly_dark" if is_dark else "plotly_white"
    
    for ticker in labels_map.keys():
        if ticker not in ratios.columns: continue
        xt, yt = ratios[ticker].tail(5), momentums[ticker].tail(5)
        if len(xt) < 5: continue
        cx, cy = xt.iloc[-1], yt.iloc[-1]
        
        color = "#00FF00" if cx > 100 and cy > 100 else ("#FFFF00" if cx > 100 else ("#FF4444" if cy < 100 else "#00BFFF"))
        fig.add_trace(go.Scatter(x=xt, y=yt, mode='lines', line=dict(color=color, width=2, shape='spline'), opacity=0.6, showlegend=False))
        fig.add_trace(go.Scatter(x=[cx], y=[cy], mode='markers+text', marker=dict(color=color, size=12), text=[ticker], textposition="top center", textfont=dict(color=text_col)))

    fig.add_hline(y=100, line_dash="dot", line_color="gray"); fig.add_vline(x=100, line_dash="dot", line_color="gray")
    fig.update_layout(title=title, template=template, height=600, xaxis=dict(range=[97, 103], title="Trend"), yaxis=dict(range=[97, 103], title="Momentum"))
    return fig

# --- STYLING (THE KEY FIX) ---
def style_final(styler):
    def color_rotation(val):
        if "LEADING" in val: return 'color: #00FF00; font-weight: bold'
        if "WEAKENING" in val: return 'color: #FFFF00; font-weight: bold'
        if "LAGGING" in val: return 'color: #FF4444; font-weight: bold'
        if "IMPROVING" in val: return 'color: #00BFFF; font-weight: bold'
        return ''
    def color_inst(val):
        if "ACCUMULATION" in val or "BREAKOUT" in val: return 'color: #00FF00; font-weight: bold'
        if "DISTRIBUTION" in val or "LIQUIDATION" in val: return 'color: #FF4444; font-weight: bold'
        return 'color: #888888'
    
    return styler.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#111'), ('color', 'white'), ('font-size', '12px')]}, {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '14px'), ('padding', '8px')]}]).set_properties(**{'background-color': '#222', 'color': 'white', 'border-color': '#444'})\
        .map(lambda v: 'color: #00ff00; font-weight: bold' if v in ["BUY", "STRONG BUY"] else ('color: #ffaa00; font-weight: bold' if "CAUTION" in v else 'color: white'), subset=["Action"])\
        .map(lambda v: 'color: #00ff00; font-weight: bold' if v >= 4 else ('color: #ffaa00' if v == 3 else '#ff4444'), subset=["Weekly<br>Score", "Daily<br>Score"])\
        .map(color_rotation, subset=["Rotation"])\
        .map(color_inst, subset=["Institutional<br>Activity"]).hide(axis='index')

def style_daily_health(styler):
    def color_status(v):
        if "PASS" in v or "NORMAL" in v or "RISING" in v: return 'color: #00ff00; font-weight: bold'
        if "FAIL" in v or "PANIC" in v or "FALLING" in v: return 'color: #ff4444; font-weight: bold'
        return 'color: white'
    return styler.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left'), ('background-color', '#111')]}, {'selector': 'td', 'props': [('text-align', 'left')]}]).map(color_status, subset=['Status']).hide(axis='index')

def style_portfolio(styler):
    return styler.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#111')]}]).hide(axis='index')

# --- PORTFOLIO ENGINE ---
def load_portfolio():
    cols = ["ID", "Ticker", "Date", "Shares", "Cost_Basis", "Status", "Exit_Date", "Exit_Price", "Return", "Realized_PL", "SPY_Return", "Type", "Shadow_SPY"]
    if not os.path.exists(PORTFOLIO_FILE): pd.DataFrame(columns=cols).to_csv(PORTFOLIO_FILE, index=False)
    df = pd.read_csv(PORTFOLIO_FILE)
    if "ID" not in df.columns: df["ID"] = range(1, len(df) + 1)
    return df

def save_portfolio(df): df.to_csv(PORTFOLIO_FILE, index=False)

# --- SIDEBAR ---
st.sidebar.header("💼 Portfolio Manager")
pf_df = load_portfolio()
current_cash = pf_df[(pf_df['Ticker'] == 'CASH') & (pf_df['Status'] == 'OPEN')]['Shares'].sum()
st.sidebar.metric("Cash Available", f"${current_cash:,.2f}")
st.sidebar.tabs(["🟢 Buy", "🔴 Sell", "💵 Cash", "🧮 Calc", "🛠️ Fix"])

# --- MAIN SCANNER CACHING ---
@st.cache_data
def generate_scanner_html(results_df):
    if results_df.empty: return ""
    return results_df.style.pipe(style_final).to_html(escape=False)

# --- MAIN EXECUTION ---
if "run_analysis" not in st.session_state: st.session_state.run_analysis = False
if st.button("RUN ANALYSIS", type="primary"): st.session_state.run_analysis = True; st.rerun()

if st.session_state.run_analysis:
    if st.button("⬅️ Back"): st.session_state.run_analysis = False; st.rerun()
    
    pf_tickers = [x for x in pf_df['Ticker'].unique() if x != "CASH"]
    all_tickers = list(tc.DATA_MAP.keys()) + pf_tickers + list(tc.RRG_SECTORS.keys()) + ["CAD=X", "IEF", "RSP"]
    
    with st.spinner('Downloading Market Data...'):
        master_data = fetch_master_data(all_tickers)
        rrg_snapshot = generate_full_rrg_snapshot(master_data, "SPY")

    mode = st.radio("Navigation", ["Scanner", "Sector Rotation"], horizontal=True)
    
    if mode == "Scanner":
        # 1. Holdings Logic
        open_pos = pf_df[(pf_df['Status'] == 'OPEN') & (pf_df['Ticker'] != 'CASH')]
        eq_val = 0.0; pf_rows = []
        for idx, row in open_pos.iterrows():
            t = row['Ticker']; cp = master_data[t]['Close'].iloc[-1] if t in master_data else row['Cost_Basis']
            val = row['Shares'] * cp; eq_val += val
            pf_rows.append({"Ticker": t, "Shares": int(row['Shares']), "Current": f"${cp:.2f}", "P/L": f"${(val-(row['Shares']*row['Cost_Basis'])):+.2f}"})
        
        c1, c2 = st.columns(2)
        c1.metric("Net Worth (USD)", f"${current_cash + eq_val:,.2f}")
        if pf_rows: st.markdown(pd.DataFrame(pf_rows).style.pipe(style_portfolio).to_html(), unsafe_allow_html=True)
        st.write("---")

        # 2. Market Health
        spy = master_data.get("SPY"); vix = master_data.get("^VIX"); rsp = master_data.get("RSP")
        mkt_score = 0; h_rows = []
        if spy is not None:
            if vix is not None:
                v = vix.iloc[-1]['Close']; s = "NORMAL" if v < 17 else ("CAUTIOUS" if v < 20 else "PANIC")
                mkt_score += 9 if v < 17 else (6 if v < 20 else (3 if v < 25 else 0))
                h_rows.append({"Indicator": f"VIX ({v:.2f})", "Status": s})
            
            s_c = spy['Close'].iloc[-1]; s18 = calc_sma(spy['Close'], 18); s8 = calc_sma(spy['Close'], 8)
            checks = {"SPY > 18": s_c > s18.iloc[-1], "SPY 18 Rising": s18.iloc[-1] >= s18.iloc[-2], "SPY 8 Rising": s8.iloc[-1] > s8.iloc[-2]}
            for k, v in checks.items():
                mkt_score += 1 if v else 0
                h_rows.append({"Indicator": k, "Status": "PASS" if v else "FAIL"})

            if rsp is not None:
                r_c = rsp['Close'].iloc[-1]; r18 = calc_sma(rsp['Close'], 18); r8 = calc_sma(rsp['Close'], 8)
                rchecks = {"RSP > 18": r_c > r18.iloc[-1], "RSP 18 Rising": r18.iloc[-1] >= r18.iloc[-2], "RSP 8 Rising": r8.iloc[-1] > r8.iloc[-2]}
                for k, v in rchecks.items():
                    mkt_score += 1 if v else 0
                    h_rows.append({"Indicator": k, "Status": "PASS" if v else "FAIL"})
            
            col = "#00ff00" if mkt_score >= 8 else ("#ffaa00" if mkt_score >= 5 else "#ff4444")
            h_rows.append({"Indicator": "TOTAL SCORE", "Status": f"SCORE: {mkt_score}/11"})
            st.markdown(pd.DataFrame(h_rows).style.pipe(style_daily_health).to_html(escape=False), unsafe_allow_html=True)
            st.write("---")

        # 3. Scanner Loop
        results = []
        scan_list = list(set(list(tc.DATA_MAP.keys()) + pf_tickers))
        for t in scan_list:
            if t not in master_data or len(master_data[t]) < 50: continue
            df = master_data[t].copy()
            df['SMA18'] = calc_sma(df['Close'], 18); df['SMA40'] = calc_sma(df['Close'], 40); df['AD'] = calc_ad(df['High'], df['Low'], df['Close'], df['Volume'])
            ad18 = calc_sma(df['AD'], 18); ad40 = calc_sma(df['AD'], 40)
            
            # Pine Parity Score
            ad_ok = not ((df['AD'].iloc[-1] < ad18.iloc[-1] and ad18.iloc[-1] <= ad18.iloc[-2]) or (ad18.iloc[-1] < ad40.iloc[-1] and ad18.iloc[-1] < ad18.iloc[-2]))
            d_score = sum([ad_ok, df['Close'].iloc[-1] > df['SMA18'].iloc[-1], df['SMA18'].iloc[-1] >= df['SMA18'].iloc[-2], df['SMA18'].iloc[-1] > df['SMA40'].iloc[-1]])
            
            w_df = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            w_score = 0
            if not w_df.empty:
                ws18 = calc_sma(w_df['Close'], 18)
                if w_df['Close'].iloc[-1] > ws18.iloc[-1]: w_score = 4 # Simplified for speed
            
            rrg_status = rrg_snapshot.get(t, "—")
            act = "BUY" if (w_score >= 4 and d_score >= 4) else "WATCH"
            if "WEAKENING" in rrg_status and act == "BUY": act = "CAUTION"

            r5, r20 = calc_rsi(df['Close'], 5).iloc[-1], calc_rsi(df['Close'], 20).iloc[-1]
            arrow = "↑" if calc_rsi(df['Close'], 5).iloc[-1] > calc_rsi(df['Close'], 5).iloc[-2] else "↓"
            
            # Final Row Construction
            cat = tc.DATA_MAP.get(t, ["OTHER"])[0]
            if "99. DATA" in cat: continue
            results.append({"Sector": cat, "Ticker": t, "Rotation": rrg_status, "Weekly<br>Score": w_score, "Daily<br>Score": d_score, "Institutional<br>Activity": calc_structure(df), "Dual RSI": f"{int(r5)}/{int(r20)} {arrow}", "Action": act})

        if results:
            df_final = pd.DataFrame(results).sort_values(["Sector", "Ticker"])
            # Remove rotation coloring from generate_scanner_html if it was buggy
            st.markdown(generate_scanner_html(df_final), unsafe_allow_html=True)

    if mode == "Sector Rotation":
        rrg_nav = st.radio("View", ["Indices", "Sectors"], horizontal=True)
        if rrg_nav == "Indices":
            bench = st.selectbox("Benchmark", ["SPY", "IEF"])
            idx_list = list(tc.RRG_INDICES.keys())
            if bench == "IEF" and "SPY" not in idx_list: idx_list.append("SPY")
            if st.button("Run"):
                wide = prepare_rrg_inputs(master_data, idx_list, bench)
                r, m = calculate_rrg_math(wide, bench)
                st.plotly_chart(plot_rrg_chart(r, m, tc.RRG_INDICES, f"Indices vs {bench}", True), use_container_width=True)
