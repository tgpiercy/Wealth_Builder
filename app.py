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
#  TITAN STRATEGY APP (v58.4 Unified Engine)
# ==============================================================================

current_user = st.session_state.user
PORTFOLIO_FILE = f"portfolio_{current_user}.csv"

st.sidebar.write(f"👤 Logged in as: **{current_user.upper()}**")
if st.sidebar.button("Log Out"):
    logout()

st.title(f"🛡️ Titan Strategy v58.4 ({current_user.upper()})")
st.caption("Institutional Protocol: Unified Data Engine")

# --- CALCULATIONS ---
def calc_sma(series, length): return series.rolling(window=length).mean()
def calc_ad(high, low, close, volume):
    mfm = ((close - low) - (high - close)) / (high - low); mfm = mfm.fillna(0); mfv = mfm * volume
    return mfv.cumsum()
def calc_ichimoku(high, low, close):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26); span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return span_a, span_b
def calc_atr(high, low, close, length=14):
    tr1 = high - low; tr2 = abs(high - close.shift(1)); tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean()
def calc_rsi(series, length):
    delta = series.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = np.full_like(gain, np.nan); avg_loss = np.full_like(loss, np.nan)
    avg_gain[length] = gain[1:length+1].mean(); avg_loss[length] = loss[1:length+1].mean()
    for i in range(length + 1, len(series)):
        avg_gain[i] = (avg_gain[i-1] * (length - 1) + gain.iloc[i]) / length
        avg_loss[i] = (avg_loss[i-1] * (length - 1) + loss.iloc[i]) / length
    rs = avg_gain / avg_loss; rs = np.where(avg_loss == 0, 100, rs)
    return pd.Series(100 - (100 / (1 + rs)), index=series.index)

# --- ZIG ZAG ---
def calc_structure(df, deviation_pct=0.035):
    if len(df) < 50: return "None"
    pivots = []; trend = 1; last_val = df['Close'].iloc[0]; pivots.append((0, last_val, 1))
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        if trend == 1:
            if price > last_val: last_val = price; (pivots[-1] = (i, price, 1)) if pivots[-1][2] == 1 else pivots.append((i, price, 1))
            elif price < last_val * (1 - deviation_pct): trend = -1; last_val = price; pivots.append((i, price, -1))
        else:
            if price < last_val: last_val = price; (pivots[-1] = (i, price, -1)) if pivots[-1][2] == -1 else pivots.append((i, price, -1))
            elif price > last_val * (1 + deviation_pct): trend = 1; last_val = price; pivots.append((i, price, 1))
    if len(pivots) < 3: return "Range"
    return ("HH" if pivots[-1][1] > pivots[-3][1] else "LH") if pivots[-1][2] == 1 else ("LL" if pivots[-1][1] < pivots[-3][1] else "HL")

def round_to_03_07(price):
    if pd.isna(price): return 0.0
    whole = int(price); candidates = [c for c in [whole + 0.03, whole + 0.07, (whole - 1) + 0.97, (whole - 1) + 0.93] if c > 0]
    return min(candidates, key=lambda x: abs(x - price)) if candidates else price

# --- UNIFIED DATA ENGINE (THE FIX) ---
@st.cache_data(ttl=1800) # Cache for 30 mins
def fetch_master_data(ticker_list):
    """Downloads daily data for ALL tickers once."""
    # Deduplicate
    unique_tickers = list(set(ticker_list))
    data_map = {}
    
    # Download in bulk if possible, or loop for reliability
    # For robust handling, we loop but cache the result object
    for t in unique_tickers:
        try:
            fetch_sym = "SPY" if t == "MANL" else t
            tk = yf.Ticker(fetch_sym)
            # We fetch 2y daily. This is enough for Scanner AND RRG (resampled)
            df = tk.history(period="2y", interval="1d")
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if not df.empty and 'Close' in df.columns:
                data_map[t] = df
        except: continue
    return data_map

def prepare_rrg_inputs(data_map, tickers, benchmark):
    """Converts Daily Dict -> Weekly Wide DataFrame for RRG"""
    df_wide = pd.DataFrame()
    
    # Process Benchmark
    if benchmark in data_map:
        bench_df = data_map[benchmark].resample('W-FRI').last()
        df_wide[benchmark] = bench_df['Close']
    
    # Process Tickers
    for t in tickers:
        if t in data_map and t != benchmark:
            # Resample Daily to Weekly
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
                df_ratio[col] = 100 + ((rs - mean) / std) * 1.5
            except: continue
    for col in df_ratio.columns:
        try: df_mom[col] = 100 + (df_ratio[col] - df_ratio[col].rolling(window_mom).mean()) * 2
        except: continue
    return df_ratio.rolling(smooth_factor).mean().dropna(), df_mom.rolling(smooth_factor).mean().dropna()

def plot_rrg_chart(ratios, momentums, labels_map, title, is_dark):
    if go is None: return None
    fig = go.Figure()
    if is_dark:
        bg_col, text_col = "black", "white"; c_lead, c_weak, c_lag, c_imp = "#00FF00", "#FFFF00", "#FF4444", "#00BFFF"; template = "plotly_dark"
    else:
        bg_col, text_col = "white", "black"; c_lead, c_weak, c_lag, c_imp = "#008000", "#FF8C00", "#CC0000", "#0000FF"; template = "plotly_white"

    has_data = False
    for ticker in labels_map.keys():
        if ticker not in ratios.columns: continue
        xt = ratios[ticker].tail(5); yt = momentums[ticker].tail(5)
        if len(xt) < 5: continue
        has_data = True
        cx, cy = xt.iloc[-1], yt.iloc[-1]
        if cx > 100 and cy > 100: color = c_lead
        elif cx > 100 and cy < 100: color = c_weak
        elif cx < 100 and cy < 100: color = c_lag
        else: color = c_imp
        
        fig.add_trace(go.Scatter(x=xt, y=yt, mode='lines', line=dict(color=color, width=2, shape='spline'), opacity=0.6, showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[cx], y=[cy], mode='markers+text', marker=dict(color=color, size=12, line=dict(color=text_col, width=1)), text=[ticker], textposition="top center", textfont=dict(color=text_col), hovertemplate=f"<b>{labels_map.get(ticker, ticker)}</b><br>T: %{{x:.2f}}<br>M: %{{y:.2f}}"))

    if not has_data: return None
    op = 0.1 if is_dark else 0.05
    fig.add_hline(y=100, line_dash="dot", line_color="gray"); fig.add_vline(x=100, line_dash="dot", line_color="gray")
    fig.add_shape(type="rect", x0=100, y0=100, x1=200, y1=200, fillcolor=f"rgba(0,255,0,{op})", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=100, y0=0, x1=200, y1=100, fillcolor=f"rgba(255,255,0,{op})", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, fillcolor=f"rgba(255,0,0,{op})", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=100, x1=100, y1=200, fillcolor=f"rgba(0,0,255,{op})", layer="below", line_width=0)
    fig.update_layout(title=title, template=template, height=650, showlegend=False, xaxis=dict(range=[96, 104], showgrid=False, title="RS-Ratio (Trend)"), yaxis=dict(range=[96, 104], showgrid=False, title="RS-Momentum (Velocity)"))
    fig.add_annotation(x=104, y=104, text="LEADING", showarrow=False, font=dict(size=16, color=c_lead), xanchor="right", yanchor="top")
    fig.add_annotation(x=104, y=96, text="WEAKENING", showarrow=False, font=dict(size=16, color=c_weak), xanchor="right", yanchor="bottom")
    fig.add_annotation(x=96, y=96, text="LAGGING", showarrow=False, font=dict(size=16, color=c_lag), xanchor="left", yanchor="bottom")
    fig.add_annotation(x=96, y=104, text="IMPROVING", showarrow=False, font=dict(size=16, color=c_imp), xanchor="left", yanchor="top")
    return fig

# --- STYLING ---
def style_final(styler):
    def color_pct(val):
        try: return 'color: #00ff00; font-weight: bold' if float(val.strip('%')) >= 0 else 'color: #ff0000; font-weight: bold'
        except: return ''
    def color_rsi(val):
        try:
            parts = val.split(); r5 = float(parts[0].split('/')[0]); r20 = float(parts[0].split('/')[1]); arrow = parts[1]
            if r5 >= r20: return 'color: #00BFFF; font-weight: bold' if (r20 > 50 and arrow=="↑") else ('color: #00FF00; font-weight: bold' if arrow=="↑" else 'color: #FF4444; font-weight: bold')
            return 'color: #FFA500; font-weight: bold' if r20 > 50 else 'color: #FF4444; font-weight: bold'
        except: return ''
    def color_inst(val):
        if "ACCUMULATION" in val or "BREAKOUT" in val: return 'color: #00FF00; font-weight: bold' 
        if "CAPITULATION" in val: return 'color: #00BFFF; font-weight: bold'       
        if "DISTRIBUTION" in val: return 'color: #FF4444; font-weight: bold' 
        return 'color: #CCFFCC' if "HH" in val else ('color: #FFCCCC' if "LL" in val else 'color: #888888')
    def highlight_ticker_row(row):
        styles = ['' for _ in row.index]
        if 'Ticker' not in row.index: return styles
        idx = row.index.get_loc('Ticker'); act = str(row.get('Action', '')).upper(); vol = str(row.get('Volume', '')).upper(); rsi = str(row.get('Dual RSI', ''))
        if "AVOID" in act: pass
        elif "00BFFF" in rsi and "SPIKE" in vol: styles[idx] = 'background-color: #0044CC; color: white; font-weight: bold'
        elif "BUY" in act: styles[idx] = 'background-color: #006600; color: white; font-weight: bold'
        elif "SCOUT" in act: styles[idx] = 'background-color: #005555; color: white; font-weight: bold'
        elif "SOON" in act: styles[idx] = 'background-color: #CC5500; color: white; font-weight: bold'
        return styles
    return styler.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#111'), ('color', 'white'), ('font-size', '12px')]}, {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '14px'), ('padding', '8px')]}]).set_properties(**{'background-color': '#222', 'color': 'white', 'border-color': '#444'}).apply(highlight_ticker_row, axis=1).map(lambda v: 'color: #00ff00; font-weight: bold' if v in ["BUY", "STRONG BUY"] else ('color: #00ffff; font-weight: bold' if "SCOUT" in v else ('color: #ffaa00; font-weight: bold' if v in ["SOON", "CAUTION"] else 'color: white')), subset=["Action"]).map(lambda v: 'color: #ff00ff; font-weight: bold' if "SPIKE" in v else ('color: #00ff00' if "HIGH" in v else 'color: #ccc'), subset=["Volume"]).map(lambda v: 'color: #00ff00; font-weight: bold' if "STRONG" in v else ('color: #ff0000' if "WEAK" in v else 'color: #ffaa00'), subset=["A/D Breadth"]).map(lambda v: 'color: #ff0000; font-weight: bold' if "FAIL" in v or "NO" in v else 'color: #00ff00', subset=["Ichimoku<br>Cloud", "Weekly<br>SMA8"]).map(lambda v: 'color: #00ff00; font-weight: bold' if "GOOD" in v else ('color: #ffaa00; font-weight: bold' if "WEAK" in v else 'color: #ff0000; font-weight: bold'), subset=["Weekly<br>Impulse"]).map(lambda v: 'color: #00ff00; font-weight: bold' if v >= 4 else ('color: #ffaa00; font-weight: bold' if v == 3 else 'color: #ff0000; font-weight: bold'), subset=["Weekly<br>Score", "Daily<br>Score"]).map(lambda v: 'color: #ff0000; font-weight: bold' if "BELOW 18" in v else 'color: #00ff00', subset=["Structure"]).map(color_pct, subset=["4W %", "2W %"]).map(color_rsi, subset=["Dual RSI"]).map(color_inst, subset=["Institutional<br>Activity"]).hide(axis='index')

def style_daily_health(styler):
    def color_status(v):
        if "PASS" in v or "NORMAL" in v or "RISING" in v or "AGGRESSIVE" in v: return 'color: #00ff00; font-weight: bold'
        if "FAIL" in v or "PANIC" in v or "FALLING" in v or "CASH" in v: return 'color: #ff4444; font-weight: bold'
        return 'color: white; font-weight: bold'
    return styler.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left'), ('background-color', '#111'), ('color', 'white'), ('font-size', '14px')]}, {'selector': 'td', 'props': [('text-align', 'left'), ('font-size', '14px'), ('padding', '8px')]}]).set_properties(**{'background-color': '#222', 'border-color': '#444'}).set_properties(subset=['Indicator'], **{'color': 'white', 'font-weight': 'bold'}).map(color_status, subset=['Status']).hide(axis='index')

def style_portfolio(styler):
    def color_pl(val):
        try: return 'color: #00ff00; font-weight: bold' if float(val.strip('%').replace('+','')) >= 0 else 'color: #ff4444; font-weight: bold'
        except: return ''
    def color_pl_dol(val):
        try: return 'color: #00ff00; font-weight: bold' if float(val.strip('$').replace('+','').replace(',','')) >= 0 else 'color: #ff4444; font-weight: bold'
        except: return ''
    def color_action(val): return 'color: #ff0000; font-weight: bold; background-color: #220000' if "EXIT" in val else ('color: #00ff00; font-weight: bold' if "HOLD" in val else 'color: #ffffff')
    return styler.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#111'), ('color', 'white')]}, {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '14px')]}]).map(color_pl, subset=["% Return"]).map(color_pl_dol, subset=["Gain/Loss ($)"]).map(color_action, subset=["Audit Action"]).hide(axis='index')

def style_history(styler):
    def color_pl(val):
        try: return 'color: #00ff00; font-weight: bold' if float(val.strip('%').replace('+','')) >= 0 else 'color: #ff4444; font-weight: bold'
        except: return ''
    def color_pl_dol(val):
        try: return 'color: #00ff00; font-weight: bold' if float(val.strip('$').replace('+','').replace(',','')) >= 0 else 'color: #ff4444; font-weight: bold'
        except: return ''
    return styler.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#111'), ('color', 'white')]}, {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '14px')]}]).map(color_pl, subset=["% Return"]).map(color_pl_dol, subset=["P/L"]).hide(axis='index')

def fmt_delta(val): return f"-${abs(val):,.2f}" if val < 0 else f"${val:,.2f}"

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

# --- SIDEBAR & TABS ---
st.sidebar.header("💼 Portfolio Manager")
pf_df = load_portfolio()
cash_rows = pf_df[(pf_df['Ticker'] == 'CASH') & (pf_df['Status'] == 'OPEN')]
current_cash = cash_rows['Shares'].sum() if not cash_rows.empty else 0.0
st.sidebar.metric("Cash Available", f"${current_cash:,.2f}")

tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["🟢 Buy", "🔴 Sell", "💵 Cash", "🧮 Calc", "🛠️ Fix"])

with tab1:
    with st.form("buy_trade"):
        b_tick = st.selectbox("Ticker", list(tc.DATA_MAP.keys()))
        b_date = st.date_input("Buy Date"); b_shares = st.number_input("Shares", min_value=1, value=100); b_price = st.number_input("Price", min_value=0.01, value=100.00)
        if st.form_submit_button("Execute Buy"):
            new_id = 1 if pf_df.empty else pf_df["ID"].max() + 1
            pf_df = pd.concat([pf_df, pd.DataFrame([{"ID": new_id, "Ticker": b_tick, "Date": b_date, "Shares": b_shares, "Cost_Basis": b_price, "Status": "OPEN", "Type": "STOCK", "Shadow_SPY": 0.0}])], ignore_index=True)
            if current_cash >= (b_shares * b_price):
                 pf_df = pd.concat([pf_df, pd.DataFrame([{"ID": pf_df["ID"].max()+1, "Ticker": "CASH", "Date": b_date, "Shares": -(b_shares * b_price), "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRADE_CASH", "Shadow_SPY": 0.0}])], ignore_index=True)
            save_portfolio(pf_df); st.success(f"Bought {b_tick}"); st.rerun()

with tab2:
    open_trades = pf_df[(pf_df['Status'] == 'OPEN') & (pf_df['Ticker'] != 'CASH')]
    if not open_trades.empty:
        opts = [f"ID:{r['ID']} | {r['Ticker']} | {int(r['Shares'])}" for i, r in open_trades.iterrows()]
        sel = st.selectbox("Position", opts)
        if sel:
            sel_id = int(sel.split("ID:")[1].split("|")[0].strip())
            row = pf_df[pf_df['ID'] == sel_id].iloc[0]; max_qty = int(row['Shares'])
            with st.form("sell_trade"):
                s_shares = st.number_input("Shares", 1, max_qty, max_qty); s_date = st.date_input("Date"); s_price = st.number_input("Price", 0.01, value=100.00)
                if st.form_submit_button("Execute Sell"):
                    idx = pf_df[pf_df['ID'] == sel_id].index[0]
                    pl = (s_price - row['Cost_Basis']) * s_shares
                    pf_df = pd.concat([pf_df, pd.DataFrame([{"ID": pf_df["ID"].max()+1, "Ticker": "CASH", "Date": s_date, "Shares": (s_price * s_shares), "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRADE_CASH", "Shadow_SPY": 0.0}])], ignore_index=True)
                    if s_shares < max_qty:
                        pf_df.at[idx, 'Shares'] -= s_shares
                        pf_df = pd.concat([pf_df, pd.DataFrame([{"ID": pf_df["ID"].max()+1, "Ticker": row['Ticker'], "Date": row['Date'], "Shares": s_shares, "Cost_Basis": row['Cost_Basis'], "Status": "CLOSED", "Exit_Date": s_date, "Exit_Price": s_price, "Realized_PL": pl, "Type": "STOCK"}])], ignore_index=True)
                    else:
                        pf_df.at[idx, 'Status'] = 'CLOSED'; pf_df.at[idx, 'Exit_Date'] = s_date; pf_df.at[idx, 'Exit_Price'] = s_price; pf_df.at[idx, 'Realized_PL'] = pl
                    save_portfolio(pf_df); st.success("Sold"); st.rerun()

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
            pf_df = pd.concat([pf_df, pd.DataFrame([{"ID": pf_df["ID"].max()+1 if not pf_df.empty else 1, "Ticker": "CASH", "Date": dt, "Shares": final, "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRANSFER", "Shadow_SPY": s_shares}])], ignore_index=True)
            save_portfolio(pf_df); st.success("Done"); st.rerun()

with tab4:
    st.subheader("Calculator"); risk = st.number_input("Risk Unit", 100, value=2300); tk = st.text_input("Ticker").upper()
    if tk:
        try:
            d = yf.Ticker(tk).history("1mo"); c = d['Close'].iloc[-1]; atr = calc_atr(d['High'], d['Low'], d['Close']).iloc[-1]
            stop = round_to_03_07(c - 2.618*atr)
            if c > stop:
                sh = int(risk / (c - stop))
                st.info(f"Entry: ${c:.2f} | Stop: ${stop:.2f} | Shares: {sh} | Cap: ${sh*c:,.0f}")
        except: st.error("Error")

with tab5:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "rb") as f: st.download_button("Download CSV", f, PORTFOLIO_FILE)
    up = st.file_uploader("Restore", type=["csv"])
    if up and st.button("Restore"): pd.read_csv(up).to_csv(PORTFOLIO_FILE, index=False); st.rerun()
    if st.button("Factory Reset"): os.remove(PORTFOLIO_FILE); st.rerun()

# --- MAIN EXECUTION ---
if "run_analysis" not in st.session_state: st.session_state.run_analysis = False
if st.button("RUN ANALYSIS", type="primary"): st.session_state.run_analysis = True; st.rerun()

if st.session_state.run_analysis:
    if st.button("⬅️ Back"): st.session_state.run_analysis = False; st.rerun()
    
    # --- UNIFIED LIST GENERATION ---
    SECTORS = {"XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care", "XLY": "Cons. Discret", "XLP": "Cons. Staples", "XLI": "Industrials", "XLC": "Comm. Services", "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate"}
    INDICES = {"QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWM": "Russell 2000", "IWC": "Micro-Cap", "RSP": "S&P Equal Wgt", "^VIX": "Volatility (VIX)", "HXT.TO": "TSX 60 (Canada)", "EFA": "Foreign Dev", "EEM": "Emerging Mkts"}
    THEMES = {"BOTZ": "Robotics/AI", "AIQ": "Artificial Intel", "SMH": "Semiconductors", "IGV": "Software", "CIBR": "CyberSec", "ARKG": "Genomics", "ICLN": "Clean Energy", "TAN": "Solar", "URA": "Uranium", "PAVE": "Infrastructure", "GLD": "Gold", "SLV": "Silver", "GDX": "Gold Miners", "COPX": "Copper", "MOO": "Agribusiness", "SLX": "Steel", "REMX": "Rare Earths"}
    INDUSTRY_MAP = {
        "XLK": {"SMH": "Semis", "NVDA": "Nvidia", "IGV": "Software", "MSFT": "Microsoft", "CIBR": "CyberSec", "AAPL": "Apple", "SMCI": "Servers (AI)", "DELL": "Dell", "ANET": "Networking", "WDC": "Storage"},
        "XLF": {"KBE": "Banks", "KRE": "Reg. Banks", "IAI": "Brokers", "IAK": "Insurance", "XP": "Fintech"},
        "XLE": {"XOP": "Exploration", "OIH": "Oil Svcs", "CRAK": "Refiners", "XOM": "Exxon", "CVX": "Chevron"},
        "XLV": {"IBB": "Biotech", "IHI": "Med Devices", "PPH": "Pharma", "UNH": "UnitedHealth"},
        "XLY": {"XRT": "Retail", "ITB": "Homebuild", "PEJ": "Leisure", "AMZN": "Amazon", "TSLA": "Tesla"},
        "XLP": {"PBJ": "Food/Bev", "KXI": "Global Stapl", "COST": "Costco", "PG": "Procter", "WMT": "Walmart"},
        "XLI": {"ITA": "Aerospace", "IYT": "Transport", "JETS": "Airlines", "PAVE": "Infrastruct", "CAT": "Caterpillar"},
        "XLC": {"SOCL": "Social", "PBS": "Media", "GOOGL": "Google", "META": "Meta", "NFLX": "Netflix"},
        "XLB": {"GDX": "Gold Miners", "SIL": "Silver", "LIT": "Lithium", "REMX": "Rare Earth", "COPX": "Copper", "MOO": "Agricul", "SLX": "Steel", "AA": "Alcoa", "DD": "DuPont"},
        "XLU": {"IDU": "US Util", "VPU": "Vanguard Util", "NEE": "NextEra", "DUK": "Duke Energy"},
        "XLRE": {"REZ": "Resid. RE", "BBRE": "BetaBuilders", "PLD": "Prologis", "AMT": "Am. Tower"},
        "Canada (TSX)": {"RY.TO": "Royal Bank", "BN.TO": "Brookfield", "CNQ.TO": "Cdn Natural", "CP.TO": "CP Rail", "WSP.TO": "WSP Global", "SHOP.TO": "Shopify", "CSU.TO": "Constell", "NTR.TO": "Nutrien", "TECK-B.TO": "Teck Res"}
    }

    pf_tickers = pf_df['Ticker'].unique().tolist() if not pf_df.empty else []
    pf_tickers = [x for x in pf_tickers if x != "CASH"]
    
    # MASTER LIST (Duplicates removed automatically by set)
    all_tickers = list(tc.DATA_MAP.keys()) + pf_tickers + list(SECTORS.keys()) + list(INDICES.keys()) + list(THEMES.keys())
    for v in INDUSTRY_MAP.values(): all_tickers.extend(list(v.keys()))
    
    # --- MASTER DATA FETCH ---
    with st.spinner('Downloading Unified Market Data...'):
        master_data = fetch_master_data(all_tickers)

    tab_scan, tab_rrg = st.tabs(["🔍 Market Scanner", "🔄 Sector Rotation"])
    
    with tab_scan:
        # 1. HOLDINGS
        open_pos = pf_df[(pf_df['Status'] == 'OPEN') & (pf_df['Ticker'] != 'CASH')]
        eq_val = 0.0; cost_basis = 0.0; rows = []
        if not open_pos.empty:
            for i, r in open_pos.iterrows():
                p = master_data[r['Ticker']]['Close'].iloc[-1] if r['Ticker'] in master_data else r['Cost_Basis']
                val = r['Shares'] * p; eq_val += val; cost_basis += (r['Shares'] * r['Cost_Basis'])
                rows.append({"Ticker": r['Ticker'], "Shares": int(r['Shares']), "Avg Cost": f"${r['Cost_Basis']:.2f}", "Current": f"${p:.2f}", "Gain/Loss ($)": f"${val-(r['Shares']*r['Cost_Basis']):+.2f}", "% Return": f"{(p-r['Cost_Basis'])/r['Cost_Basis']*100:+.2f}%", "Audit Action": "HOLD"})
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Net Worth", f"${current_cash + eq_val:,.2f}"); c2.metric("Cash", f"${current_cash:,.2f}"); c3.metric("Equity", f"${eq_val:,.2f}")
        if rows: st.markdown(pd.DataFrame(rows).style.pipe(style_portfolio).to_html(), unsafe_allow_html=True)
        st.write("---")

        # 2. MARKET HEALTH
        spy = master_data.get("SPY"); vix = master_data.get("^VIX"); rsp = master_data.get("RSP")
        mkt_score = 0; h_rows = []
        if spy is not None:
            if vix is not None:
                v = vix.iloc[-1]['Close']; s = "<span style='color:#00ff00'>NORMAL</span>" if v < 17 else "<span style='color:#ff4444'>PANIC</span>"
                mkt_score += 9 if v < 17 else 0; h_rows.append({"Indicator": f"VIX ({v:.2f})", "Status": s})
            
            sc = spy.iloc[-1]['Close']; s18 = calc_sma(spy['Close'], 18).iloc[-1]; s8 = calc_sma(spy['Close'], 8).iloc[-1]
            if sc > s18: mkt_score += 1
            h_rows.append({"Indicator": "SPY > SMA18", "Status": "<span style='color:#00ff00'>PASS</span>" if sc > s18 else "<span style='color:#ff4444'>FAIL</span>"})
            
            col = "#00ff00" if mkt_score >= 8 else "#ff4444"
            h_rows.append({"Indicator": "TOTAL SCORE", "Status": f"<span style='color:{col}'><b>{mkt_score}/11</b></span>"})
            st.subheader("🏥 Market Health")
            st.markdown(pd.DataFrame(h_rows).style.pipe(style_daily_health).to_html(escape=False), unsafe_allow_html=True)
            st.write("---")

        # 3. SCANNER LOOP
        results = []
        # Filter for only scanner-relevant tickers
        scan_list = list(set(list(tc.DATA_MAP.keys()) + pf_tickers))
        
        for t in scan_list:
            if t not in master_data or len(master_data[t]) < 50: continue
            df = master_data[t].copy()
            # ... (Calculations reused) ...
            df['SMA18'] = calc_sma(df['Close'], 18); df['SMA40'] = calc_sma(df['Close'], 40); df['AD'] = calc_ad(df['High'], df['Low'], df['Close'], df['Volume'])
            df['AD_SMA18'] = calc_sma(df['AD'], 18); df['VolSMA'] = calc_sma(df['Volume'], 18); df['RSI5'] = calc_rsi(df['Close'], 5); df['RSI20'] = calc_rsi(df['Close'], 20)
            
            # Weekly
            df_w = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
            df_w.dropna(inplace=True)
            if len(df_w) < 5: continue
            df_w['SMA8'] = calc_sma(df_w['Close'], 8); df_w['SMA18'] = calc_sma(df_w['Close'], 18); df_w['SMA40'] = calc_sma(df_w['Close'], 40)
            
            # Logic
            dc = df.iloc[-1]; wc = df_w.iloc[-1]
            struc = calc_structure(df)
            w_score = 0
            if wc['Close'] > wc['SMA18']: w_score += 1
            if wc['SMA18'] > df_w.iloc[-2]['SMA18']: w_score += 1
            if wc['SMA18'] > wc['SMA40']: w_score += 1
            if wc['Close'] > wc['SMA8']: w_score += 1
            
            act = "AVOID"; rsn = "Weak"
            if w_score >= 3: act = "WATCH"; rsn = "Improving"
            if w_score == 4: act = "BUY"; rsn = "Strong"
            
            row = {"Sector": tc.DATA_MAP.get(t, ["OTHER"])[0], "Ticker": t, "Weekly<br>Score": w_score, "Structure": struc, "Action": act, "Reasoning": rsn, "Price": f"${dc['Close']:.2f}"}
            results.append(row)

        if results:
            df_final = pd.DataFrame(results).sort_values("Ticker")
            cols = ["Sector", "Ticker", "Price", "Weekly<br>Score", "Structure", "Action", "Reasoning"]
            st.markdown(df_final[cols].style.pipe(style_final).to_html(escape=False), unsafe_allow_html=True)

    with tab_rrg:
        is_dark = st.toggle("🌙 Dark Mode", value=True)
        sub1, sub2, sub3, sub4 = st.tabs(["Indices", "Sectors", "Drill-Down", "Themes"])
        
        def render_rrg(ticker_subset, bench, title):
            if st.button(f"Run {title}", key=f"btn_{title}"):
                # Use master_data directly - NO DOWNLOAD
                wide_df = prepare_rrg_inputs(master_data, ticker_subset, bench)
                if not wide_df.empty:
                    r, m = calculate_rrg_math(wide_df, bench)
                    labels = {t:t for t in ticker_subset} # Simplify labels for now
                    fig = plot_rrg_chart(r, m, labels, title, is_dark)
                    st.plotly_chart(fig, use_container_width=True)
                else: st.error("Data insufficient")

        with sub1:
            render_rrg(list(INDICES.keys()), "SPY", "Indices")
        with sub2:
            render_rrg(list(SECTORS.keys()), "SPY", "Sectors")
        with sub3:
            s_key = st.selectbox("Sector", list(SECTORS.keys()))
            render_rrg(list(INDUSTRY_MAP.get(s_key, {}).keys()), s_key, f"{SECTORS[s_key]} Industries")
        with sub4:
            render_rrg(list(THEMES.keys()), "SPY", "Themes")
