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
#  TITAN STRATEGY APP (v59.4 Variable Patch)
# ==============================================================================

current_user = st.session_state.user
PORTFOLIO_FILE = f"portfolio_{current_user}.csv"

st.sidebar.write(f"👤 Logged in as: **{current_user.upper()}**")
if st.sidebar.button("Log Out"):
    logout()

st.title(f"🛡️ Titan Strategy v59.4 ({current_user.upper()})")
st.caption("Institutional Protocol: Variable Scope Fixed")

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
    """Downloads daily data for ALL tickers once."""
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
        if "DISTRIBUTION" in val or "LIQUIDATION" in val: return 'color: #FF4444; font-weight: bold' 
        if "SELLING" in val: return 'color: #FFA500; font-weight: bold'      
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
        if "PASS" in v or "NORMAL" in v or "CAUTIOUS" in v or "RISING" in v or "AGGRESSIVE" in v: return 'color: #00ff00; font-weight: bold'
        if "FAIL" in v or "PANIC" in v or "DEFENSIVE" in v or "FALLING" in v or "CASH" in v: return 'color: #ff4444; font-weight: bold'
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

# --- SIDEBAR: MANAGER ---
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
        b_price = st.number_input("Price", min_value=0.01, value=100.00)
        if st.form_submit_button("Execute Buy"):
            new_id = 1 if pf_df.empty else pf_df["ID"].max() + 1
            pf_df = pd.concat([pf_df, pd.DataFrame([{"ID": new_id, "Ticker": b_tick, "Date": b_date, "Shares": b_shares, "Cost_Basis": b_price, "Status": "OPEN", "Type": "STOCK", "Shadow_SPY": 0.0}])], ignore_index=True)
            if current_cash >= (b_shares * b_price):
                 pf_df = pd.concat([pf_df, pd.DataFrame([{"ID": pf_df["ID"].max()+1, "Ticker": "CASH", "Date": b_date, "Shares": -(b_shares * b_price), "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRADE_CASH", "Shadow_SPY": 0.0}])], ignore_index=True)
            save_portfolio(pf_df); st.success(f"Bought {b_tick}"); st.rerun()

with tab2:
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
            sel_data = trade_map[selected_trade_str]; sel_id = sel_data['id']; max_qty = sel_data['max_shares']
            with st.form("sell_trade"):
                s_shares = st.number_input("Shares to Sell", min_value=1, max_value=max_qty, value=max_qty, step=1)
                s_date = st.date_input("Date"); s_price = st.number_input("Price", 0.01, value=100.00)
                if st.form_submit_button("Execute Sell"):
                    row_idx = sel_data['idx']; buy_price = float(pf_df.at[row_idx, 'Cost_Basis']); buy_date_str = pf_df.at[row_idx, 'Date']
                    ret_pct = ((s_price - buy_price) / buy_price) * 100; pl_dollars = (s_price - buy_price) * s_shares
                    cash_id = pf_df["ID"].max() + 1
                    cash_row = pd.DataFrame([{"ID": cash_id, "Ticker": "CASH", "Date": s_date, "Shares": (s_price * s_shares), "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRADE_CASH", "Shadow_SPY": 0.0}])
                    pf_df = pd.concat([pf_df, cash_row], ignore_index=True)
                    if s_shares < max_qty:
                        pf_df.at[row_idx, 'Shares'] -= s_shares
                        new_id = pf_df["ID"].max() + 1
                        new_closed_row = pd.DataFrame([{"ID": new_id, "Ticker": pf_df.at[row_idx, 'Ticker'], "Date": buy_date_str, "Shares": s_shares, "Cost_Basis": buy_price, "Status": "CLOSED", "Exit_Date": s_date, "Exit_Price": s_price, "Return": ret_pct, "Realized_PL": pl_dollars, "SPY_Return": 0.0, "Type": "STOCK", "Shadow_SPY": 0.0}])
                        pf_df = pd.concat([pf_df, new_closed_row], ignore_index=True)
                    else:
                        pf_df.at[row_idx, 'Status'] = 'CLOSED'; pf_df.at[row_idx, 'Exit_Date'] = s_date; pf_df.at[row_idx, 'Exit_Price'] = s_price; pf_df.at[row_idx, 'Return'] = ret_pct; pf_df.at[row_idx, 'Realized_PL'] = pl_dollars
                    save_portfolio(pf_df); st.success(f"Sold {s_shares} shares. P&L: ${pl_dollars:+.2f}"); st.rerun()
    else: st.info("No Open Positions")

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
            pf_df = pd.concat([pf_df, pd.DataFrame([{"ID": pf_df["ID"].max()+1, "Ticker": "CASH", "Date": dt, "Shares": final, "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRANSFER", "Shadow_SPY": s_shares}])], ignore_index=True)
            save_portfolio(pf_df); st.success("Done"); st.rerun()

with tab4:
    # --- FIXED VARIABLE NAME HERE ---
    st.subheader("Calculator"); RISK_UNIT_BASE = st.number_input("Risk Unit", 100, value=2300); tk = st.text_input("Ticker").upper()
    if tk:
        try:
            d = yf.Ticker(tk).history("1mo"); c = d['Close'].iloc[-1]; atr = calc_atr(d['High'], d['Low'], d['Close']).iloc[-1]
            stop = round_to_03_07(c - 2.618*atr)
            if c > stop:
                sh = int(RISK_UNIT_BASE / (c - stop)) # Fixed usage
                st.info(f"Entry: ${c:.2f} | Stop: ${stop:.2f} | Shares: {sh} | Cap: ${sh*c:,.0f}")
        except: st.error("Error")

with tab5:
    st.write("### 🛠️ Data Management")
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "rb") as file:
            st.download_button("Download Portfolio CSV", file, PORTFOLIO_FILE, "text/csv")
    else: st.warning("No portfolio file found.")
    st.write("---")
    uploaded_file = st.file_uploader("Restore .csv", type=["csv"])
    if uploaded_file is not None and st.button("CONFIRM RESTORE"):
        try:
            pd.read_csv(uploaded_file).to_csv(PORTFOLIO_FILE, index=False)
            st.success("Data Restored!"); st.rerun()
        except: st.error("Error")
    st.write("---")
    
    action_type = st.radio("Advanced Tools", ["Delete Trade", "Edit Trade", "⚠️ FACTORY RESET", "Rebuild Benchmark History"])
    if action_type == "⚠️ FACTORY RESET" and st.button("CONFIRM RESET"):
        if os.path.exists(PORTFOLIO_FILE): os.remove(PORTFOLIO_FILE)
        st.success("Reset!"); st.rerun()
    elif action_type == "Rebuild Benchmark History" and st.button("RUN REBUILD"):
         with st.spinner("Rebuilding..."):
             try:
                 spy_hist = yf.Ticker("SPY").history(period="10y")
                 for idx, row in pf_df.iterrows():
                     if row['Type'] == 'TRANSFER' and row['Ticker'] == 'CASH':
                         t_date = pd.to_datetime(row['Date'])
                         idx_loc = spy_hist.index.searchsorted(t_date)
                         price = spy_hist.iloc[idx_loc]['Close'] if idx_loc < len(spy_hist) else spy_hist.iloc[-1]['Close']
                         if price > 0: pf_df.at[idx, 'Shadow_SPY'] = float(row['Shares']) / price
                 save_portfolio(pf_df); st.success("Done!"); st.rerun()
             except: st.error("Error")

# --- HTML CACHING ---
@st.cache_data
def generate_scanner_html(results_df):
    if results_df.empty: return ""
    return results_df.style.pipe(style_final).to_html(escape=False)

# --- MAIN EXECUTION ---
if "run_analysis" not in st.session_state: st.session_state.run_analysis = False
if st.button("RUN ANALYSIS", type="primary"): st.session_state.run_analysis = True; st.rerun()

if st.session_state.run_analysis:
    if st.button("⬅️ Back to Menu"): st.session_state.run_analysis = False; st.rerun()
    
    # --- UNIFIED LIST GENERATION ---
    pf_tickers = pf_df['Ticker'].unique().tolist() if not pf_df.empty else []
    pf_tickers = [x for x in pf_tickers if x != "CASH"]
    all_tickers = list(tc.DATA_MAP.keys()) + pf_tickers + list(tc.RRG_SECTORS.keys()) + list(tc.RRG_INDICES.keys()) + list(tc.RRG_THEMES.keys()) + ["CAD=X"] 
    for v in tc.RRG_INDUSTRY_MAP.values(): all_tickers.extend(list(v.keys()))
    
    # --- MASTER DATA FETCH ---
    with st.spinner('Downloading Unified Market Data...'):
        master_data = fetch_master_data(all_tickers)

    # --- STATE-BASED NAVIGATION ---
    mode = st.radio("Navigation", ["Scanner", "Sector Rotation"], horizontal=True, key="main_nav")
    
    if mode == "Scanner":
        # 1. HOLDINGS
        open_pos = pf_df[(pf_df['Status'] == 'OPEN') & (pf_df['Ticker'] != 'CASH')]
        eq_val = 0.0; total_cost_basis = 0.0; pf_rows = []
        
        if not open_pos.empty:
            for idx, row in open_pos.iterrows():
                t = row['Ticker']; shares = row['Shares']; cost = row['Cost_Basis']
                curr_price = cost 
                if t in master_data and not master_data[t].empty: curr_price = master_data[t]['Close'].iloc[-1]
                pos_val = shares * curr_price; eq_val += pos_val
                total_cost_basis += (shares * cost)
                pf_rows.append({"Ticker": t, "Shares": int(shares), "Avg Cost": f"${cost:.2f}", "Current": f"${curr_price:.2f}", "Gain/Loss ($)": f"${(pos_val - (shares * cost)):+.2f}", "% Return": f"{((curr_price - cost) / cost) * 100:+.2f}%", "Audit Action": "HOLD"})
        
        total_net_worth = current_cash + eq_val
        
        # --- CAD CORRECTION ---
        cad_data = master_data.get("CAD=X")
        if cad_data is not None and not cad_data.empty:
            rate = cad_data['Close'].iloc[-1]
            if rate < 1.0: rate = 1.0 / rate 
            cad_rate = rate
        else:
            cad_rate = 1.40 
            
        total_nw_cad = total_net_worth * cad_rate
        open_pl_val = eq_val - total_cost_basis
        open_pl_cad = open_pl_val * cad_rate
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Net Worth (CAD @ {cad_rate:.2f})", f"${total_nw_cad:,.2f}", fmt_delta(open_pl_cad))
        c2.metric("Net Worth (USD)", f"${total_net_worth:,.2f}", fmt_delta(open_pl_val))
        c3.metric("Cash", f"${current_cash:,.2f}"); c4.metric("Equity", f"${eq_val:,.2f}")
        
        if pf_rows: st.markdown(pd.DataFrame(pf_rows).style.pipe(style_portfolio).to_html(), unsafe_allow_html=True)
        else: st.info("No active trades.")
        st.write("---")

        # 2. CLOSED PERFORMANCE
        closed_trades = pf_df[(pf_df['Status'] == 'CLOSED') & (pf_df['Ticker'] != 'CASH')]
        if not closed_trades.empty:
            st.subheader("📜 Closed Performance")
            wins = closed_trades[closed_trades['Return'] > 0]
            win_rate = (len(wins) / len(closed_trades)) * 100
            total_pl = closed_trades['Realized_PL'].sum()
            c1, c2 = st.columns(2)
            c1.metric("Win Rate", f"{win_rate:.0f}%"); c2.metric("Total P&L", f"${total_pl:,.2f}")
            hist_view = closed_trades[["Ticker", "Cost_Basis", "Exit_Price", "Realized_PL", "Return"]].copy()
            hist_view["Open Position"] = hist_view["Cost_Basis"].apply(lambda x: f"${x:,.2f}")
            hist_view["Close Position"] = hist_view["Exit_Price"].apply(lambda x: f"${x:,.2f}")
            hist_view["P/L"] = hist_view["Realized_PL"].apply(lambda x: f"${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}")
            hist_view["% Return"] = hist_view["Return"].apply(lambda x: f"{x:+.2f}%")
            st.dataframe(hist_view[["Ticker", "Open Position", "Close Position", "P/L", "% Return"]].style.pipe(style_history))
            st.write("---")

        # 3. BENCHMARK
        shadow_shares_total = pf_df['Shadow_SPY'].sum()
        spy_data = master_data.get("SPY")
        if spy_data is not None:
            curr_spy = spy_data['Close'].iloc[-1]
            bench_val = shadow_shares_total * curr_spy
            alpha = total_net_worth - bench_val
            alpha_pct = ((total_net_worth - bench_val) / bench_val * 100) if bench_val > 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Titan Net Worth", f"${total_net_worth:,.2f}")
            c2.metric("SPY Benchmark", f"${bench_val:,.2f}")
            c3.metric("Alpha (Edge)", f"${alpha:,.2f}", f"{alpha_pct:+.2f}%")
            st.write("---")

        # 4. MARKET HEALTH (RESTORED FULL)
        spy = master_data.get("SPY"); vix = master_data.get("^VIX"); rsp = master_data.get("RSP")
        mkt_score = 0; h_rows = []
        if spy is not None:
            if vix is not None:
                v = vix.iloc[-1]['Close']
                s = "<span style='color:#00ff00'>NORMAL</span>" if v < 17 else ("<span style='color:#ffaa00'>CAUTIOUS</span>" if v < 20 else "<span style='color:#ff4444'>PANIC</span>")
                mkt_score += 9 if v < 17 else (6 if v < 20 else (3 if v < 25 else 0))
                h_rows.append({"Indicator": f"VIX Level ({v:.2f})", "Status": s})
            
            sc = spy.iloc[-1]['Close']; s18 = calc_sma(spy['Close'], 18).iloc[-1]; s8 = calc_sma(spy['Close'], 8).iloc[-1]
            if sc > s18: mkt_score += 1
            h_rows.append({"Indicator": "SPY Price > SMA18", "Status": "<span style='color:#00ff00'>PASS</span>" if sc > s18 else "<span style='color:#ff4444'>FAIL</span>"})
            
            if s18 >= calc_sma(spy['Close'], 18).iloc[-2]: mkt_score += 1
            h_rows.append({"Indicator": "SPY SMA18 Rising", "Status": "<span style='color:#00ff00'>RISING</span>" if s18 >= calc_sma(spy['Close'], 18).iloc[-2] else "<span style='color:#ff4444'>FALLING</span>"})
            
            if s8 > calc_sma(spy['Close'], 8).iloc[-2]: mkt_score += 1
            h_rows.append({"Indicator": "SPY SMA8 Rising", "Status": "<span style='color:#00ff00'>RISING</span>" if s8 > calc_sma(spy['Close'], 8).iloc[-2] else "<span style='color:#ff4444'>FALLING</span>"})

            if rsp is not None:
                rc = rsp.iloc[-1]['Close']; r18 = calc_sma(rsp['Close'], 18).iloc[-1]
                if rc > r18: mkt_score += 1
                h_rows.append({"Indicator": "RSP Price > SMA18", "Status": "<span style='color:#00ff00'>PASS</span>" if rc > r18 else "<span style='color:#ff4444'>FAIL</span>"})
            
            col = "#00ff00" if mkt_score >= 8 else ("#ffaa00" if mkt_score >= 5 else "#ff4444")
            msg = "AGGRESSIVE" if mkt_score >= 10 else ("CAUTIOUS" if mkt_score >= 8 else "DEFENSIVE")
            risk_per_trade = RISK_UNIT_BASE if mkt_score >= 8 else (RISK_UNIT_BASE * 0.5 if mkt_score >= 5 else 0)
            
            h_rows.append({"Indicator": "TOTAL SCORE", "Status": f"<span style='color:{col}'><b>{mkt_score}/11</b></span>"})
            h_rows.append({"Indicator": "STRATEGY MODE", "Status": f"<span style='color:{col}'><b>{msg}</b></span>"})
            st.subheader("🏥 Daily Market Health")
            st.markdown(pd.DataFrame(h_rows).style.pipe(style_daily_health).to_html(escape=False), unsafe_allow_html=True)
            st.write("---")

        # 5. SCANNER LOOP (CACHED)
        results = []
        scan_list = list(set(list(tc.DATA_MAP.keys()) + pf_tickers))
        analysis_db = {}
        
        for t in scan_list:
            if t not in master_data or len(master_data[t]) < 50: continue
            df = master_data[t].copy()
            df['SMA18'] = calc_sma(df['Close'], 18); df['SMA40'] = calc_sma(df['Close'], 40); df['AD'] = calc_ad(df['High'], df['Low'], df['Close'], df['Volume'])
            df['AD_SMA18'] = calc_sma(df['AD'], 18); df['VolSMA'] = calc_sma(df['Volume'], 18); df['RSI5'] = calc_rsi(df['Close'], 5); df['RSI20'] = calc_rsi(df['Close'], 20)
            df_w = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
            df_w.dropna(inplace=True)
            if len(df_w) < 5: continue
            df_w['SMA8'] = calc_sma(df_w['Close'], 8); df_w['SMA18'] = calc_sma(df_w['Close'], 18); df_w['SMA40'] = calc_sma(df_w['Close'], 40)
            span_a, span_b = calc_ichimoku(df_w['High'], df_w['Low'], df_w['Close']); df_w['Cloud_Top'] = pd.concat([span_a, span_b], axis=1).max(axis=1)

            mom_4w = ""; mom_2w = ""
            if len(df_w) >= 5:
                curr = df_w.iloc[-1]['Close']; prev2 = df_w.iloc[-3]['Close']; prev4 = df_w.iloc[-5]['Close']
                mom_2w = f"{((curr/prev2)-1)*100:.1f}%"; mom_4w = f"{((curr/prev4)-1)*100:.1f}%"

            dc = df.iloc[-1]; wc = df_w.iloc[-1]
            inst_activity = calc_structure(df)
            
            ad_pass = False
            if len(df) > 2 and not pd.isna(df['AD_SMA18'].iloc[-1]):
                 ad_pass = (df['AD'].iloc[-1] >= df['AD_SMA18'].iloc[-1] * 0.995) and (df['AD_SMA18'].iloc[-1] >= df['AD_SMA18'].iloc[-2])
            
            vol_msg = "NORMAL"
            if df['Volume'].iloc[-1] > (df['VolSMA'].iloc[-1] * 1.5): vol_msg = "SPIKE (Live)"
            elif df['Volume'].iloc[-2] > (df['VolSMA'].iloc[-2] * 1.5): vol_msg = "SPIKE (Prev)"
            elif df['Volume'].iloc[-1] > df['VolSMA'].iloc[-1]: vol_msg = "HIGH (Live)"
            
            r5 = df['RSI5'].iloc[-1]; r20 = df['RSI20'].iloc[-1] if not pd.isna(df['RSI20'].iloc[-1]) else 50
            r5_prev = df['RSI5'].iloc[-2]; is_rising = r5 > r5_prev
            
            final_inst_msg = inst_activity
            if "SPIKE" in vol_msg:
                if inst_activity == "HL": final_inst_msg = "ACCUMULATION (HL)" if is_rising else final_inst_msg
                if inst_activity == "HH": final_inst_msg = "BREAKOUT (HH)" if is_rising else "DISTRIBUTION (HH)"
                if inst_activity == "LL": final_inst_msg = "CAPITULATION (LL)" if is_rising else "LIQUIDATION (LL)"
                if inst_activity == "LH": final_inst_msg = "SELLING (LH)"

            w_score = 0
            if wc['Close'] > wc['SMA18']: w_score += 1
            if wc['SMA18'] > df_w.iloc[-2]['SMA18']: w_score += 1
            if wc['SMA18'] > wc['SMA40']: w_score += 1
            if wc['Close'] > wc['Cloud_Top']: w_score += 1
            if wc['Close'] > wc['SMA8']: w_score += 1 
            
            d_chk = 0
            if dc['Close'] > df['SMA18'].iloc[-1]: d_chk += 1
            if df['SMA18'].iloc[-1] >= df['SMA18'].iloc[-2]: d_chk += 1
            if df['SMA18'].iloc[-1] > df['SMA40'].iloc[-1]: d_chk += 1
            if ad_pass: d_chk += 1
            
            w_pulse = "GOOD" if (wc['Close'] > wc['SMA18']) and (dc['Close'] > df['SMA18'].iloc[-1]) else "NO"

            decision = "AVOID"; reason = "Low Score"
            if w_score >= 4:
                if d_chk == 4: decision = "BUY"; reason = "Score 5/5"
                elif d_chk == 3: decision = "SCOUT"; reason = "D-Score 4"
                elif d_chk == 2: decision = "SCOUT"; reason = "Dip Buy"
                else: decision = "WATCH"; reason = "Daily Weak"
            else: decision = "AVOID"; reason = "Weekly Weak"

            if not (wc['Close'] > wc['SMA8']): decision = "AVOID"; reason = "BELOW W-SMA8"
            elif "NO" in w_pulse: decision = "AVOID"; reason = "Impulse NO"
            elif risk_per_trade == 0 and "BUY" in decision: decision = "CAUTION"; reason = "VIX Lock"

            atr = calc_atr(df['High'], df['Low'], df['Close']).iloc[-1]
            raw_stop = dc['Close'] - (2.618 * atr); smart_stop_val = round_to_03_07(raw_stop)
            stop_dist = dc['Close'] - smart_stop_val; stop_pct = (stop_dist / dc['Close']) * 100 if dc['Close'] else 0
            
            num_col = "#FF4444"
            if r5 >= r20: num_col = "#00BFFF" if (r20 > 50 and is_rising) else "#00FF00"
            arrow_col = "#00FF00" if is_rising else "#FF4444"; arrow = "↑" if is_rising else "↓"
            rsi_msg = f"<span style='color:{num_col}'><b>{int(r5)}/{int(r20)}</b></span> <span style='color:{arrow_col}'><b>{arrow}</b></span>"
            
            analysis_db[t] = {"Decision": decision, "Reason": reason, "Price": dc['Close'], "Stop": smart_stop_val, "StopPct": stop_pct, "Mom4W": mom_4w, "Mom2W": mom_2w, "W_SMA8_Pass": (wc['Close']>wc['SMA8']), "W_Pulse": w_pulse, "W_Score": w_score, "D_Score": d_chk, "D_Chk_Price": (dc['Close'] > df['SMA18'].iloc[-1]), "W_Cloud": (wc['Close']>wc['Cloud_Top']), "AD_Pass": ad_pass, "Vol_Msg": vol_msg, "RSI_Msg": rsi_msg, "Inst_Act": final_inst_msg}

        for t in scan_list:
            cat_name = tc.DATA_MAP.get(t, ["OTHER"])[0]
            if "99. DATA" in cat_name: continue
            if t not in analysis_db: continue
            is_scanner = t in tc.DATA_MAP and (tc.DATA_MAP[t][0] != "BENCH" or t in ["DIA", "QQQ", "IWM", "IWC", "HXT.TO"])
            if not is_scanner: continue
            
            db = analysis_db[t]
            final_decision = db['Decision']; final_reason = db['Reason']
            if cat_name in tc.SECTOR_PARENTS:
                parent = tc.SECTOR_PARENTS[cat_name]
                if parent in analysis_db and "AVOID" in analysis_db[parent]['Decision']:
                    if t != parent: final_decision = "AVOID"; final_reason = "Sector Lock"

            rsi_html = db['RSI_Msg']; vol_str = db['Vol_Msg']
            is_blue_spike = ("00BFFF" in rsi_html) and ("SPIKE" in vol_str)
            final_risk = risk_per_trade / 3 if "SCOUT" in final_decision else risk_per_trade
            if is_blue_spike: final_risk = risk_per_trade
            
            if "AVOID" in final_decision and not is_blue_spike: disp_stop = ""; disp_shares = ""
            else:
                shares = int(final_risk / (db['Price'] - db['Stop'])) if (db['Price'] - db['Stop']) > 0 else 0
                disp_stop = f"${db['Stop']:.2f} (-{db['StopPct']:.1f}%)"; disp_shares = f"{shares} shares"

            row = {
                "Sector": cat_name, "Ticker": t, "Rank": (0 if "00." in cat_name else 1), "4W %": db['Mom4W'], "2W %": db['Mom2W'],
                "Weekly<br>SMA8": "PASS" if db['W_SMA8_Pass'] else "FAIL", "Weekly<br>Impulse": db['W_Pulse'], 
                "Weekly<br>Score": db['W_Score'], "Daily<br>Score": db['D_Score'],
                "Structure": "ABOVE 18" if db['D_Chk_Price'] else "BELOW 18",
                "Ichimoku<br>Cloud": "PASS" if db['W_Cloud'] else "FAIL", "A/D Breadth": "STRONG" if db['AD_Pass'] else "WEAK",
                "Volume": db['Vol_Msg'], "Dual RSI": db['RSI_Msg'], "Institutional<br>Activity": db['Inst_Act'],
                "Action": final_decision, "Reasoning": final_reason, "Stop Price": disp_stop, "Position Size": disp_shares
            }
            results.append(row)
            if t == "HXT.TO": row_cad = row.copy(); row_cad["Sector"] = "15. CANADA (HXT)"; row_cad["Rank"] = 0; results.append(row_cad)
            if t in tc.SECTOR_ETFS: row_sec = row.copy(); row_sec["Sector"] = "02. SECTORS (SUMMARY)"; row_sec["Rank"] = 0; results.append(row_sec)

        if results:
            df_final = pd.DataFrame(results).sort_values(["Sector", "Rank", "Ticker"], ascending=[True, True, True])
            df_final["Sector"] = df_final["Sector"].apply(lambda x: x.split(". ", 1)[1].replace("(SUMMARY)", "").strip() if ". " in x else x)
            cols = ["Sector", "Ticker", "4W %", "2W %", "Weekly<br>SMA8", "Weekly<br>Impulse", "Weekly<br>Score", "Daily<br>Score", "Structure", "Ichimoku<br>Cloud", "A/D Breadth", "Volume", "Dual RSI", "Institutional<br>Activity", "Action", "Reasoning", "Stop Price", "Position Size"]
            st.markdown(generate_scanner_html(df_final[cols]), unsafe_allow_html=True)
        else:
            st.warning("Scanner returned no results.")

    if mode == "Sector Rotation":
        st.subheader("🔄 Relative Rotation Graphs (RRG)")
        is_dark = st.toggle("🌙 Dark Mode", value=True)
        rrg_mode = st.radio("View:", ["Indices", "Sectors", "Drill-Down", "Themes"], horizontal=True, key="rrg_nav")
        
        if rrg_mode == "Indices":
            c1, c2 = st.columns([1,3])
            with c1: bench_sel = st.selectbox("Benchmark", ["SPY", "IEF"], key="bench_idx")
            tgt = "IEF" if bench_sel == "IEF" else "SPY"
            idx_list = list(tc.RRG_INDICES.keys())
            if tgt == "IEF": idx_list.append("SPY")
            elif "SPY" in idx_list: idx_list.remove("SPY")
            
            if st.button("Run Indices"):
                wide_df = prepare_rrg_inputs(master_data, idx_list, tgt)
                r, m = calculate_rrg_math(wide_df, tgt)
                st.session_state['fig_idx'] = plot_rrg_chart(r, m, tc.RRG_INDICES, f"Indices vs {tgt}", is_dark)
            if 'fig_idx' in st.session_state: st.plotly_chart(st.session_state['fig_idx'], use_container_width=True)

        elif rrg_mode == "Sectors":
            if st.button("Run Sectors"):
                wide_df = prepare_rrg_inputs(master_data, list(tc.RRG_SECTORS.keys()), "SPY")
                r, m = calculate_rrg_math(wide_df, "SPY")
                st.session_state['fig_sec'] = plot_rrg_chart(r, m, tc.RRG_SECTORS, "Sectors vs SPY", is_dark)
            if 'fig_sec' in st.session_state: st.plotly_chart(st.session_state['fig_sec'], use_container_width=True)

        elif rrg_mode == "Drill-Down":
            c1, c2 = st.columns([1,3])
            with c1:
                def fmt(x): return f"{x} - {tc.RRG_SECTORS[x]}" if x in tc.RRG_SECTORS else x
                opts = list(tc.RRG_SECTORS.keys()) + ["Canada (TSX)"]
                sec_key = st.selectbox("Select Sector", opts, format_func=fmt, key="dd_sel")
            if sec_key == "Canada (TSX)": bench_dd = "HXT.TO"; name_dd = "Canadian Titans"
            else: bench_dd = sec_key; name_dd = tc.RRG_SECTORS[sec_key]
            
            if st.button(f"Run {name_dd}"):
                comp_list = list(tc.RRG_INDUSTRY_MAP.get(sec_key, {}).keys())
                wide_df = prepare_rrg_inputs(master_data, comp_list, bench_dd)
                r, m = calculate_rrg_math(wide_df, bench_dd)
                all_labels = {**tc.RRG_INDUSTRY_MAP.get(sec_key, {}), **tc.RRG_SECTORS}
                st.session_state['fig_dd'] = plot_rrg_chart(r, m, all_labels, f"{name_dd} vs {bench_dd}", is_dark)
            if 'fig_dd' in st.session_state: st.plotly_chart(st.session_state['fig_dd'], use_container_width=True)

        elif rrg_mode == "Themes":
            if st.button("Run Themes"):
                wide_df = prepare_rrg_inputs(master_data, list(tc.RRG_THEMES.keys()), "SPY")
                r, m = calculate_rrg_math(wide_df, "SPY")
                st.session_state['fig_thm'] = plot_rrg_chart(r, m, tc.RRG_THEMES, "Themes vs SPY", is_dark)
            if 'fig_thm' in st.session_state: st.plotly_chart(st.session_state['fig_thm'], use_container_width=True)
