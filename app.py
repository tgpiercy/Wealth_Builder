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
    st.error("=E2=9A=A0=EF=B8=8F CRITICAL ERROR: `titan_config.py` is missi=
ng. Please
create it.")
    st.stop()

# --- SAFE IMPORT FOR PLOTLY ---
try:
    import plotly.graph_objects as go
except ImportError:
    st.warning("=E2=9A=A0=EF=B8=8F Plotly not found. RRG Charts will not wo=
rk. (pip install
plotly)")
    go =3D None

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title=3D"Titan Strategy", layout=3D"wide")

# --- AUTHENTICATION ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated =3D False
    st.session_state.user =3D None

def check_login():
    username =3D st.session_state.username_input
    password =3D st.session_state.password_input
    if username in tc.CREDENTIALS and tc.CREDENTIALS[username] =3D=3D passw=
ord:
        st.session_state.authenticated =3D True
        st.session_state.user =3D username
    else:
        st.error("Incorrect Username or Password")

def logout():
    st.session_state.authenticated =3D False
    st.session_state.user =3D None
    st.rerun()

if not st.session_state.authenticated:
    st.title("=F0=9F=9B=A1=EF=B8=8F Titan Strategy Login")
    with st.form("login_form"):
        st.text_input("Username", key=3D"username_input")
        st.text_input("Password", type=3D"password", key=3D"password_input"=
)
        st.form_submit_button("Login", on_click=3Dcheck_login)
    st.stop()

#
=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=
=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=
=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=
=3D=3D=3D
# TITAN STRATEGY APP (v60.5 Variable Name Fix)
#
=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=
=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=
=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=
=3D=3D=3D

current_user =3D st.session_state.user
PORTFOLIO_FILE =3D f"portfolio_{current_user}.csv"

st.sidebar.write(f"=F0=9F=91=A4 Logged in as: **{current_user.upper()}**")
if st.sidebar.button("Log Out"):
    logout()

st.title(f"=F0=9F=9B=A1=EF=B8=8F Titan Strategy v60.5 ({current_user.upper(=
)})")
st.caption("Institutional Protocol: Pine Parity & Stability")

# --- CALCULATIONS ---
def calc_sma(series, length): return series.rolling(window=3Dlength).mean()
def calc_ad(high, low, close, volume):
    mfm =3D ((close - low) - (high - close)) / (high - low); mfm =3D
mfm.fillna(0.0); mfv =3D mfm * volume
    return mfv.cumsum()
def calc_ichimoku(high, low, close):
    tenkan =3D (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun =3D (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a =3D ((tenkan + kijun) / 2).shift(26); span_b =3D
((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return span_a, span_b
def calc_atr(high, low, close, length=3D14):
    try:
        tr1 =3D high - low; tr2 =3D abs(high - close.shift(1)); tr3 =3D abs=
(low -
close.shift(1))
        tr =3D pd.concat([tr1, tr2, tr3], axis=3D1).max(axis=3D1)
        return tr.ewm(com=3Dlength-1, adjust=3DFalse).mean()
    except: return pd.Series(0, index=3Dclose.index)
def calc_rsi(series, length=3D14):
    try:
        delta =3D series.diff(); gain =3D delta.clip(lower=3D0); loss =3D
-delta.clip(upper=3D0)
        avg_gain =3D gain.ewm(com=3Dlength-1, adjust=3DFalse).mean()
        avg_loss =3D loss.ewm(com=3Dlength-1, adjust=3DFalse).mean()
        rs =3D avg_gain / avg_loss; rs =3D rs.fillna(0)
        return 100 - (100 / (1 + rs))
    except: return pd.Series(50, index=3Dseries.index)

# --- ZIG ZAG ENGINE ---
def calc_structure(df, deviation_pct=3D0.035):
    if len(df) < 50: return "None"
    pivots =3D []; trend =3D 1; last_val =3D df['Close'].iloc[0];
pivots.append((0, last_val, 1))
    for i in range(1, len(df)):
        price =3D df['Close'].iloc[i]
        if trend =3D=3D 1:
            if price > last_val:
                last_val =3D price
                if pivots[-1][2] =3D=3D 1: pivots[-1] =3D (i, price, 1)
                else: pivots.append((i, price, 1))
            elif price < last_val * (1 - deviation_pct):
                trend =3D -1; last_val =3D price; pivots.append((i, price, =
-1))
        else:
            if price < last_val:
                last_val =3D price
                if pivots[-1][2] =3D=3D -1: pivots[-1] =3D (i, price, -1)
                else: pivots.append((i, price, -1))
            elif price > last_val * (1 + deviation_pct):
                trend =3D 1; last_val =3D price; pivots.append((i, price, 1=
))
    if len(pivots) < 3: return "Range"
    return ("HH" if pivots[-1][1] > pivots[-3][1] else "LH") if
pivots[-1][2] =3D=3D 1 else ("LL" if pivots[-1][1] < pivots[-3][1] else "HL=
")

def round_to_03_07(price):
    if pd.isna(price): return 0.0
    whole =3D int(price); candidates =3D [c for c in [whole + 0.03, whole +
0.07, (whole - 1) + 0.97, (whole - 1) + 0.93] if c > 0]
    return min(candidates, key=3Dlambda x: abs(x - price)) if candidates el=
se
price

# --- UNIFIED DATA ENGINE ---
@st.cache_data(ttl=3D3600)
def fetch_master_data(ticker_list):
    """Downloads daily data for ALL tickers once."""
    unique_tickers =3D sorted(list(set(ticker_list)))
    data_map =3D {}
    for t in unique_tickers:
        try:
            fetch_sym =3D "SPY" if t =3D=3D "MANL" else t
            tk =3D yf.Ticker(fetch_sym)
            df =3D tk.history(period=3D"2y", interval=3D"1d")
            df.index =3D pd.to_datetime(df.index).tz_localize(None)
            if not df.empty and 'Close' in df.columns:
                data_map[t] =3D df
        except: continue
    return data_map

def prepare_rrg_inputs(data_map, tickers, benchmark):
    df_wide =3D pd.DataFrame()
    if benchmark in data_map:
        bench_df =3D data_map[benchmark].resample('W-FRI').last()
        df_wide[benchmark] =3D bench_df['Close']
    for t in tickers:
        if t in data_map and t !=3D benchmark:
            w_df =3D data_map[t].resample('W-FRI').last()
            df_wide[t] =3D w_df['Close']
    return df_wide.dropna()

# --- RRG LOGIC ---
def calculate_rrg_math(price_data, benchmark_col, window_rs=3D14,
window_mom=3D5, smooth_factor=3D3):
    if benchmark_col not in price_data.columns: return pd.DataFrame(),
pd.DataFrame()
    df_ratio =3D pd.DataFrame(); df_mom =3D pd.DataFrame()
    for col in price_data.columns:
        if col !=3D benchmark_col:
            try:
                rs =3D price_data[col] / price_data[benchmark_col]
                mean =3D rs.rolling(window_rs).mean(); std =3D
rs.rolling(window_rs).std()
                ratio =3D 100 + ((rs - mean) / std) * 1.5
                df_ratio[col] =3D ratio
            except: continue
    for col in df_ratio.columns:
        try: df_mom[col] =3D 100 + (df_ratio[col] -
df_ratio[col].rolling(window_mom).mean()) * 2
        except: continue
    return df_ratio.rolling(smooth_factor).mean().dropna(),
df_mom.rolling(smooth_factor).mean().dropna()

def generate_full_rrg_snapshot(data_map, benchmark=3D"SPY"):
    try:
        all_tickers =3D list(data_map.keys())
        if benchmark not in all_tickers: return {}
        wide_df =3D prepare_rrg_inputs(data_map, all_tickers, benchmark)
        if wide_df.empty: return {}
        r, m =3D calculate_rrg_math(wide_df, benchmark)
        if r.empty or m.empty: return {}
        status_map =3D {}
        last_idx =3D r.index[-1]
        for t in r.columns:
            try:
                val_r =3D r.at[last_idx, t]; val_m =3D m.at[last_idx, t]
                if val_r > 100 and val_m > 100: status =3D "LEADING"
                elif val_r > 100 and val_m < 100: status =3D "WEAKENING"
                elif val_r < 100 and val_m < 100: status =3D "LAGGING"
                else: status =3D "IMPROVING"
                status_map[t] =3D status
            except: continue
        return status_map
    except: return {}

def plot_rrg_chart(ratios, momentums, labels_map, title, is_dark):
    if go is None: return None
    fig =3D go.Figure()
    if is_dark:
        bg_col, text_col =3D "black", "white"; c_lead, c_weak, c_lag, c_imp=
 =3D
"#00FF00", "#FFFF00", "#FF4444", "#00BFFF"; template =3D "plotly_dark"
    else:
        bg_col, text_col =3D "white", "black"; c_lead, c_weak, c_lag, c_imp=
 =3D
"#008000", "#FF8C00", "#CC0000", "#0000FF"; template =3D "plotly_white"

    has_data =3D False
    for ticker in labels_map.keys():
        if ticker not in ratios.columns: continue
        xt =3D ratios[ticker].tail(5); yt =3D momentums[ticker].tail(5)
        if len(xt) < 5: continue
        has_data =3D True
        cx, cy =3D xt.iloc[-1], yt.iloc[-1]
        if cx > 100 and cy > 100: color =3D c_lead
        elif cx > 100 and cy < 100: color =3D c_weak
        elif cx < 100 and cy < 100: color =3D c_lag
        else: color =3D c_imp

        fig.add_trace(go.Scatter(x=3Dxt, y=3Dyt, mode=3D'lines',
line=3Ddict(color=3Dcolor, width=3D2, shape=3D'spline'), opacity=3D0.6,
showlegend=3DFalse, hoverinfo=3D'skip'))
        fig.add_trace(go.Scatter(x=3D[cx], y=3D[cy], mode=3D'markers+text',
marker=3Ddict(color=3Dcolor, size=3D12, line=3Ddict(color=3Dtext_col, width=
=3D1)),
text=3D[ticker], textposition=3D"top center", textfont=3Ddict(color=3Dtext_=
col),
hovertemplate=3Df"<b>{labels_map.get(ticker, ticker)}</b><br>T:
%{{x:.2f}}<br>M: %{{y:.2f}}"))

    if not has_data: return None
    op =3D 0.1 if is_dark else 0.05
    fig.add_hline(y=3D100, line_dash=3D"dot", line_color=3D"gray");
fig.add_vline(x=3D100, line_dash=3D"dot", line_color=3D"gray")
    fig.add_shape(type=3D"rect", x0=3D100, y0=3D100, x1=3D200, y1=3D200,
fillcolor=3Df"rgba(0,255,0,{op})", layer=3D"below", line_width=3D0)
    fig.add_shape(type=3D"rect", x0=3D100, y0=3D0, x1=3D200, y1=3D100,
fillcolor=3Df"rgba(255,255,0,{op})", layer=3D"below", line_width=3D0)
    fig.add_shape(type=3D"rect", x0=3D0, y0=3D0, x1=3D100, y1=3D100,
fillcolor=3Df"rgba(255,0,0,{op})", layer=3D"below", line_width=3D0)
    fig.add_shape(type=3D"rect", x0=3D0, y0=3D100, x1=3D100, y1=3D200,
fillcolor=3Df"rgba(0,0,255,{op})", layer=3D"below", line_width=3D0)
    fig.update_layout(title=3Dtitle, template=3Dtemplate, height=3D650,
showlegend=3DFalse, xaxis=3Ddict(range=3D[96, 104], showgrid=3DFalse,
title=3D"RS-Ratio (Trend)"), yaxis=3Ddict(range=3D[96, 104], showgrid=3DFal=
se,
title=3D"RS-Momentum (Velocity)"))
    fig.add_annotation(x=3D104, y=3D104, text=3D"LEADING", showarrow=3DFals=
e,
font=3Ddict(size=3D16, color=3Dc_lead), xanchor=3D"right", yanchor=3D"top")
    fig.add_annotation(x=3D104, y=3D96, text=3D"WEAKENING", showarrow=3DFal=
se,
font=3Ddict(size=3D16, color=3Dc_weak), xanchor=3D"right", yanchor=3D"botto=
m")
    fig.add_annotation(x=3D96, y=3D96, text=3D"LAGGING", showarrow=3DFalse,
font=3Ddict(size=3D16, color=3Dc_lag), xanchor=3D"left", yanchor=3D"bottom"=
)
    fig.add_annotation(x=3D96, y=3D104, text=3D"IMPROVING", showarrow=3DFal=
se,
font=3Ddict(size=3D16, color=3Dc_imp), xanchor=3D"left", yanchor=3D"top")
    return fig

# --- STYLING ---
def style_final(styler):
    def color_rotation(val):
        if "LEADING" in val: return 'color: #00FF00; font-weight: bold'
        if "WEAKENING" in val: return 'color: #FFFF00; font-weight: bold'
        if "LAGGING" in val: return 'color: #FF4444; font-weight: bold'
        if "IMPROVING" in val: return 'color: #00BFFF; font-weight: bold'
        return ''
    def color_rsi(val):
        try:
            parts =3D val.split(); r5 =3D float(parts[0].split('/')[0]); r2=
0 =3D
float(parts[0].split('/')[1]); arrow =3D parts[1]
            if r5 >=3D r20: return 'color: #00BFFF; font-weight: bold' if
(r20 > 50 and arrow=3D=3D"=E2=86=91") else ('color: #00FF00; font-weight: b=
old' if
arrow=3D=3D"=E2=86=91" else 'color: #FF4444; font-weight: bold')
            return 'color: #FFA500; font-weight: bold' if r20 > 50 else
'color: #FF4444; font-weight: bold'
        except: return ''
    def color_inst(val):
        if "ACCUMULATION" in val or "BREAKOUT" in val: return 'color:
#00FF00; font-weight: bold'
        if "CAPITULATION" in val: return 'color: #00BFFF; font-weight:
bold'
        if "DISTRIBUTION" in val or "LIQUIDATION" in val: return 'color:
#FF4444; font-weight: bold'
        if "SELLING" in val: return 'color: #FFA500; font-weight: bold'

        return 'color: #CCFFCC' if "HH" in val else ('color: #FFCCCC' if
"LL" in val else 'color: #888888')
    def highlight_ticker_row(row):
        styles =3D ['' for _ in row.index]
        if 'Ticker' not in row.index: return styles
        idx =3D row.index.get_loc('Ticker'); act =3D str(row.get('Action',
'')).upper(); vol =3D str(row.get('Volume', '')).upper(); rsi =3D
str(row.get('Dual RSI', ''))
        if "AVOID" in act: pass
        elif "BUY" in act: styles[idx] =3D 'background-color: #006600; colo=
r:
white; font-weight: bold'
        elif "SCOUT" in act: styles[idx] =3D 'background-color: #005555;
color: white; font-weight: bold'
        elif "SOON" in act: styles[idx] =3D 'background-color: #CC5500;
color: white; font-weight: bold'
        elif "CAUTION" in act: styles[idx] =3D 'background-color: #AA4400;
color: white; font-weight: bold'
        return styles
    return styler.set_table_styles([{'selector': 'th', 'props':
[('text-align', 'center'), ('background-color', '#111'), ('color',
'white'), ('font-size', '12px')]}, {'selector': 'td', 'props':
[('text-align', 'center'), ('font-size', '14px'), ('padding',
'8px')]}]).set_properties(**{'background-color': '#222', 'color': 'white',
'border-color': '#444'}).apply(highlight_ticker_row, axis=3D1).map(lambda v=
:
'color: #00ff00; font-weight: bold' if v in ["BUY", "STRONG BUY"] else
('color: #00ffff; font-weight: bold' if "SCOUT" in v else ('color: #ffaa00;
font-weight: bold' if v in ["SOON", "CAUTION"] else 'color: white')),
subset=3D["Action"]).map(lambda v: 'color: #ff00ff; font-weight: bold' if
"SPIKE" in v else ('color: #00ff00' if "HIGH" in v else 'color: #ccc'),
subset=3D["Volume"]).map(lambda v: 'color: #00ff00; font-weight: bold' if
"STRONG" in v else ('color: #ff0000' if "WEAK" in v else 'color: #ffaa00'),
subset=3D["A/D Breadth"]).map(lambda v: 'color: #ff0000; font-weight: bold'
if "FAIL" in v or "NO" in v else 'color: #00ff00',
subset=3D["Ichimoku<br>Cloud", "Weekly<br>SMA8"]).map(lambda v: 'color:
#00ff00; font-weight: bold' if "GOOD" in v else ('color: #ffaa00;
font-weight: bold' if "WEAK" in v else 'color: #ff0000; font-weight:
bold'), subset=3D["Weekly<br>Impulse"]).map(lambda v: 'color: #00ff00;
font-weight: bold' if v >=3D 4 else ('color: #ffaa00; font-weight: bold' if=
 v
=3D=3D 3 else 'color: #ff0000; font-weight: bold'), subset=3D["Weekly<br>Sc=
ore",
"Daily<br>Score"]).map(lambda v: 'color: #ff0000; font-weight: bold' if
"BELOW 18" in v else 'color: #00ff00',
subset=3D["Structure"]).map(color_rotation,
subset=3D["Rotation"]).map(color_rsi, subset=3D["Dual RSI"]).map(color_inst=
,
subset=3D["Institutional<br>Activity"]).hide(axis=3D'index')

def style_daily_health(styler):
    def color_status(v):
        if "PASS" in v or "NORMAL" in v or "CAUTIOUS" in v or "RISING" in v
or "AGGRESSIVE" in v: return 'color: #00ff00; font-weight: bold'
        if "FAIL" in v or "PANIC" in v or "DEFENSIVE" in v or "FALLING" in
v or "CASH" in v: return 'color: #ff4444; font-weight: bold'
        return 'color: white; font-weight: bold'
    return styler.set_table_styles([{'selector': 'th', 'props':
[('text-align', 'left'), ('background-color', '#111'), ('color', 'white'),
('font-size', '14px')]}, {'selector': 'td', 'props': [('text-align',
'left'), ('font-size', '14px'), ('padding',
'8px')]}]).set_properties(**{'background-color': '#222', 'border-color':
'#444'}).set_properties(subset=3D['Indicator'], **{'color': 'white',
'font-weight': 'bold'}).map(color_status,
subset=3D['Status']).hide(axis=3D'index')

def style_portfolio(styler):
    def color_pl(val):
        try: return 'color: #00ff00; font-weight: bold' if
float(val.strip('%').replace('+','')) >=3D 0 else 'color: #ff4444;
font-weight: bold'
        except: return ''
    def color_pl_dol(val):
        try: return 'color: #00ff00; font-weight: bold' if
float(val.strip('$').replace('+','').replace(',','')) >=3D 0 else 'color:
#ff4444; font-weight: bold'
        except: return ''
    def color_action(val): return 'color: #ff0000; font-weight: bold;
background-color: #220000' if "EXIT" in val else ('color: #00ff00;
font-weight: bold' if "HOLD" in val else 'color: #ffffff')
    return styler.set_table_styles([{'selector': 'th', 'props':
[('text-align', 'center'), ('background-color', '#111'), ('color',
'white')]}, {'selector': 'td', 'props': [('text-align', 'center'),
('font-size', '14px')]}]).map(color_pl, subset=3D["%
Return"]).map(color_pl_dol, subset=3D["Gain/Loss ($)"]).map(color_action,
subset=3D["Audit Action"]).hide(axis=3D'index')

def style_history(styler):
    def color_pl(val):
        try: return 'color: #00ff00; font-weight: bold' if
float(val.strip('%').replace('+','')) >=3D 0 else 'color: #ff4444;
font-weight: bold'
        except: return ''
    def color_pl_dol(val):
        try: return 'color: #00ff00; font-weight: bold' if
float(val.strip('$').replace('+','').replace(',','')) >=3D 0 else 'color:
#ff4444; font-weight: bold'
        except: return ''
    return styler.set_table_styles([{'selector': 'th', 'props':
[('text-align', 'center'), ('background-color', '#111'), ('color',
'white')]}, {'selector': 'td', 'props': [('text-align', 'center'),
('font-size', '14px')]}]).map(color_pl, subset=3D["%
Return"]).map(color_pl_dol, subset=3D["P/L"]).hide(axis=3D'index')

def fmt_delta(val): return f"-${abs(val):,.2f}" if val < 0 else
f"${val:,.2f}"

# --- PORTFOLIO ENGINE ---
def load_portfolio():
    cols =3D ["ID", "Ticker", "Date", "Shares", "Cost_Basis", "Status",
"Exit_Date", "Exit_Price", "Return", "Realized_PL", "SPY_Return", "Type",
"Shadow_SPY"]
    if not os.path.exists(PORTFOLIO_FILE):
pd.DataFrame(columns=3Dcols).to_csv(PORTFOLIO_FILE, index=3DFalse)
    df =3D pd.read_csv(PORTFOLIO_FILE)
    if 'Cost' in df.columns: df.rename(columns=3D{'Cost': 'Cost_Basis'},
inplace=3DTrue)
    if 'Cost_Basis' not in df.columns: df['Cost_Basis'] =3D 0.0
    if "ID" not in df.columns: df["ID"] =3D range(1, len(df) + 1)
    if 'Shadow_SPY' not in df.columns: df['Shadow_SPY'] =3D 0.0
    df['Shadow_SPY'] =3D pd.to_numeric(df['Shadow_SPY'],
errors=3D'coerce').fillna(0.0)
    return df

def save_portfolio(df):
    dollar_cols =3D ['Cost_Basis', 'Exit_Price', 'Realized_PL', 'Return',
'SPY_Return']
    for col in dollar_cols:
        if col in df.columns: df[col] =3D pd.to_numeric(df[col],
errors=3D'coerce').round(2)
    def clean_shares(row): return round(row['Shares'], 2) if row['Ticker']
=3D=3D 'CASH' else int(row['Shares'])
    if not df.empty: df['Shares'] =3D df.apply(clean_shares, axis=3D1)
    df.to_csv(PORTFOLIO_FILE, index=3DFalse)

# --- SIDEBAR: MANAGER ---
st.sidebar.header("=F0=9F=92=BC Portfolio Manager")
pf_df =3D load_portfolio()
cash_rows =3D pf_df[(pf_df['Ticker'] =3D=3D 'CASH') & (pf_df['Status'] =3D=
=3D 'OPEN')]
current_cash =3D cash_rows['Shares'].sum() if not cash_rows.empty else 0.0
st.sidebar.metric("Cash Available", f"${current_cash:,.2f}")

tab1, tab2, tab3, tab4, tab5 =3D st.sidebar.tabs(["=F0=9F=9F=A2 Buy", "=F0=
=9F=94=B4 Sell", "=F0=9F=92=B5
Cash", "=F0=9F=A7=AE Calc", "=F0=9F=9B=A0=EF=B8=8F Fix"])

with tab1:
    with st.form("buy_trade"):
        b_tick =3D st.selectbox("Ticker", list(tc.DATA_MAP.keys()))
        b_date =3D st.date_input("Buy Date")
        b_shares =3D st.number_input("Shares", min_value=3D1, value=3D100)
        b_price =3D st.number_input("Price", min_value=3D0.01, value=3D100.=
00)
        if st.form_submit_button("Execute Buy"):
            new_id =3D 1 if pf_df.empty else pf_df["ID"].max() + 1
            pf_df =3D pd.concat([pf_df, pd.DataFrame([{"ID": new_id,
"Ticker": b_tick, "Date": b_date, "Shares": b_shares, "Cost_Basis":
b_price, "Status": "OPEN", "Type": "STOCK", "Shadow_SPY": 0.0}])],
ignore_index=3DTrue)
            if current_cash >=3D (b_shares * b_price):
                 pf_df =3D pd.concat([pf_df, pd.DataFrame([{"ID":
pf_df["ID"].max()+1, "Ticker": "CASH", "Date": b_date, "Shares": -(b_shares
* b_price), "Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRADE_CASH",
"Shadow_SPY": 0.0}])], ignore_index=3DTrue)
            save_portfolio(pf_df); st.success(f"Bought {b_tick}");
st.rerun()

with tab2:
    open_trades =3D pf_df[(pf_df['Status'] =3D=3D 'OPEN') & (pf_df['Ticker'=
] !=3D
'CASH')]
    if not open_trades.empty:
        trade_map =3D {}
        opts =3D []
        for idx, row in open_trades.iterrows():
            label =3D f"ID:{row['ID']} | {row['Ticker']} |
{int(row['Shares'])} shares | {row['Date']}"
            trade_map[label] =3D {'id': row['ID'], 'max_shares':
int(row['Shares']), 'idx': idx}
            opts.append(label)
        selected_trade_str =3D st.selectbox("Select Position", opts)
        if selected_trade_str:
            sel_data =3D trade_map[selected_trade_str]; sel_id =3D
sel_data['id']; max_qty =3D sel_data['max_shares']
            with st.form("sell_trade"):
                s_shares =3D st.number_input("Shares to Sell", min_value=3D=
1,
max_value=3Dmax_qty, value=3Dmax_qty, step=3D1)
                s_date =3D st.date_input("Date"); s_price =3D
st.number_input("Price", 0.01, value=3D100.00)
                if st.form_submit_button("Execute Sell"):
                    row_idx =3D sel_data['idx']; buy_price =3D
float(pf_df.at[row_idx,
'Cost_Basis']); buy_date_str =3D pf_df.at[row_idx, 'Date']
                    ret_pct =3D ((s_price - buy_price) / buy_price) * 100;
pl_dollars =3D (s_price - buy_price) * s_shares
                    cash_id =3D pf_df["ID"].max() + 1
                    cash_row =3D pd.DataFrame([{"ID": cash_id, "Ticker":
"CASH", "Date": s_date, "Shares": (s_price * s_shares), "Cost_Basis": 1.0,
"Status": "OPEN", "Type": "TRADE_CASH", "Shadow_SPY": 0.0}])
                    pf_df =3D pd.concat([pf_df, cash_row], ignore_index=3DT=
rue)
                    if s_shares < max_qty:
                        pf_df.at[row_idx, 'Shares'] -=3D s_shares
                        new_id =3D pf_df["ID"].max() + 1
                        new_closed_row =3D pd.DataFrame([{"ID": new_id,
"Ticker": pf_df.at[row_idx, 'Ticker'], "Date": buy_date_str, "Shares":
s_shares, "Cost_Basis": buy_price, "Status": "CLOSED", "Exit_Date": s_date,
"Exit_Price": s_price, "Return": ret_pct, "Realized_PL": pl_dollars,
"SPY_Return": 0.0, "Type": "STOCK", "Shadow_SPY": 0.0}])
                        pf_df =3D pd.concat([pf_df, new_closed_row],
ignore_index=3DTrue)
                    else:
                        pf_df.at[row_idx, 'Status'] =3D 'CLOSED';
pf_df.at[row_idx,
'Exit_Date'] =3D s_date; pf_df.at[row_idx, 'Exit_Price'] =3D s_price;
pf_df.at[row_idx,
'Return'] =3D ret_pct; pf_df.at[row_idx, 'Realized_PL'] =3D pl_dollars
                    save_portfolio(pf_df); st.success(f"Sold {s_shares}
shares. P&L: ${pl_dollars:+.2f}"); st.rerun()
    else: st.info("No Open Positions")

with tab3:
    with st.form("cash"):
        op =3D st.radio("Op", ["Deposit", "Withdraw"]); amt =3D
st.number_input("Amt", 100.00); dt =3D st.date_input("Date")
        if st.form_submit_button("Execute"):
            shares =3D 0.0
            try:
                spy =3D yf.Ticker("SPY").history(start=3Ddt,
end=3Ddt+timedelta(days=3D5))
                if not spy.empty: shares =3D amt / spy['Close'].iloc[0]
            except: pass
            final =3D amt if op =3D=3D "Deposit" else -amt; s_shares =3D sh=
ares if
op =3D=3D "Deposit" else -shares
            pf_df =3D pd.concat([pf_df, pd.DataFrame([{"ID":
pf_df["ID"].max()+1, "Ticker": "CASH", "Date": dt, "Shares": final,
"Cost_Basis": 1.0, "Status": "OPEN", "Type": "TRANSFER", "Shadow_SPY":
s_shares}])], ignore_index=3DTrue)
            save_portfolio(pf_df); st.success("Done"); st.rerun()

with tab4:
    st.subheader("Calculator"); RISK_UNIT_BASE =3D st.number_input("Risk
Unit", 100, value=3D2300); tk =3D st.text_input("Ticker").upper()
    if tk:
        try:
            d =3D yf.Ticker(tk).history("1mo"); c =3D d['Close'].iloc[-1]; =
atr
=3D calc_atr(d['High'], d['Low'], d['Close']).iloc[-1]
            stop =3D round_to_03_07(c - 2.618*atr)
            if c > stop:
                sh =3D int(RISK_UNIT_BASE / (c - stop))
                st.info(f"Entry: ${c:.2f} | Stop: ${stop:.2f} | Shares:
{sh} | Cap: ${sh*c:,.0f}")
        except: st.error("Error")

with tab5:
    st.write("### =F0=9F=9B=A0=EF=B8=8F Data Management")
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "rb") as file:
            st.download_button("Download Portfolio CSV", file,
PORTFOLIO_FILE, "text/csv")
    else: st.warning("No portfolio file found.")
    st.write("---")
    uploaded_file =3D st.file_uploader("Restore .csv", type=3D["csv"])
    if uploaded_file is not None and st.button("CONFIRM RESTORE"):
        try:
            pd.read_csv(uploaded_file).to_csv(PORTFOLIO_FILE, index=3DFalse=
)
            st.success("Data Restored!"); st.rerun()
        except: st.error("Error")
    st.write("---")

    action_type =3D st.radio("Advanced Tools", ["Delete Trade", "Edit Trade=
",
"=E2=9A=A0=EF=B8=8F FACTORY RESET", "Rebuild Benchmark History"])
    if action_type =3D=3D "=E2=9A=A0=EF=B8=8F FACTORY RESET" and st.button(=
"CONFIRM RESET"):
        if os.path.exists(PORTFOLIO_FILE): os.remove(PORTFOLIO_FILE)
        st.success("Reset!"); st.rerun()
    elif action_type =3D=3D "Rebuild Benchmark History" and st.button("RUN
REBUILD"):
         with st.spinner("Rebuilding..."):
             try:
                 spy_hist =3D yf.Ticker("SPY").history(period=3D"10y")
                 for idx, row in pf_df.iterrows():
                     if row['Type'] =3D=3D 'TRANSFER' and row['Ticker'] =3D=
=3D
'CASH':
                         t_date =3D pd.to_datetime(row['Date'])
                         idx_loc =3D spy_hist.index.searchsorted(t_date)
                         price =3D spy_hist.iloc[idx_loc]['Close'] if idx_l=
oc
< len(spy_hist) else spy_hist.iloc[-1]['Close']
                         if price > 0: pf_df.at[idx, 'Shadow_SPY'] =3D
float(row['Shares']) / price
                 save_portfolio(pf_df); st.success("Done!"); st.rerun()
             except: st.error("Error")

# --- HTML CACHING ---
@st.cache_data
def generate_scanner_html(results_df):
    if results_df.empty: return ""
    return results_df.style.pipe(style_final).to_html(escape=3DFalse)

# --- MAIN EXECUTION ---
if "run_analysis" not in st.session_state: st.session_state.run_analysis =
=3D
False
if st.button("RUN ANALYSIS", type=3D"primary"): st.session_state.run_analys=
is
=3D True; st.rerun()

if st.session_state.run_analysis:
    if st.button("=E2=AC=85=EF=B8=8F Back to Menu"): st.session_state.run_a=
nalysis =3D False;
st.rerun()

    # --- UNIFIED LIST GENERATION ---
    pf_tickers =3D pf_df['Ticker'].unique().tolist() if not pf_df.empty els=
e
[]
    pf_tickers =3D [x for x in pf_tickers if x !=3D "CASH"]
    all_tickers =3D list(tc.DATA_MAP.keys()) + pf_tickers +
list(tc.RRG_SECTORS.keys()) + list(tc.RRG_INDICES.keys()) +
list(tc.RRG_THEMES.keys()) + ["CAD=3DX"]
    for v in tc.RRG_INDUSTRY_MAP.values():
all_tickers.extend(list(v.keys()))

    # --- MASTER DATA FETCH & RRG CALC ---
    with st.spinner('Downloading Unified Market Data...'):
        master_data =3D fetch_master_data(all_tickers)
        rrg_snapshot =3D generate_full_rrg_snapshot(master_data, "SPY")

    # --- STATE-BASED NAVIGATION ---
    mode =3D st.radio("Navigation", ["Scanner", "Sector Rotation"],
horizontal=3DTrue, key=3D"main_nav")

    if mode =3D=3D "Scanner":
        # 1. HOLDINGS
        open_pos =3D pf_df[(pf_df['Status'] =3D=3D 'OPEN') & (pf_df['Ticker=
'] !=3D
'CASH')]
        eq_val =3D 0.0; total_cost_basis =3D 0.0; pf_rows =3D []

        if not open_pos.empty:
            for idx, row in open_pos.iterrows():
                t =3D row['Ticker']; shares =3D row['Shares']; cost =3D
row['Cost_Basis']
                curr_price =3D cost
                if t in master_data and not master_data[t].empty:
curr_price =3D master_data[t]['Close'].iloc[-1]
                pos_val =3D shares * curr_price; eq_val +=3D pos_val
                total_cost_basis +=3D (shares * cost)
                pf_rows.append({"Ticker": t, "Shares": int(shares), "Avg
Cost": f"${cost:.2f}", "Current": f"${curr_price:.2f}", "Gain/Loss ($)":
f"${(pos_val - (shares * cost)):+.2f}", "% Return": f"{((curr_price - cost)
/ cost) * 100:+.2f}%", "Audit Action": "HOLD"})

        total_net_worth =3D current_cash + eq_val
        cad_data =3D master_data.get("CAD=3DX")
        if cad_data is not None and not cad_data.empty:
            rate =3D cad_data['Close'].iloc[-1]
            if rate < 1.0: rate =3D 1.0 / rate
            cad_rate =3D rate
        else:
            cad_rate =3D 1.40

        total_nw_cad =3D total_net_worth * cad_rate
        open_pl_val =3D eq_val - total_cost_basis
        open_pl_cad =3D open_pl_val * cad_rate

        c1, c2, c3, c4 =3D st.columns(4)
        c1.metric(f"Net Worth (CAD @ {cad_rate:.2f})",
f"${total_nw_cad:,.2f}", fmt_delta(open_pl_cad))
        c2.metric("Net Worth (USD)", f"${total_net_worth:,.2f}",
fmt_delta(open_pl_val))
        c3.metric("Cash", f"${current_cash:,.2f}"); c4.metric("Equity",
f"${eq_val:,.2f}")

        if pf_rows:
st.markdown(pd.DataFrame(pf_rows).style.pipe(style_portfolio).to_html(),
unsafe_allow_html=3DTrue)
        else: st.info("No active trades.")
        st.write("---")

        # 3. BENCHMARK
        shadow_shares_total =3D pf_df['Shadow_SPY'].sum()
        spy_data =3D master_data.get("SPY")
        if spy_data is not None:
            curr_spy =3D spy_data['Close'].iloc[-1]
            bench_val =3D shadow_shares_total * curr_spy
            alpha =3D total_net_worth - bench_val
            alpha_pct =3D ((total_net_worth - bench_val) / bench_val * 100)
if bench_val > 0 else 0
            c1, c2, c3 =3D st.columns(3)
            c1.metric("Titan Net Worth", f"${total_net_worth:,.2f}")
            c2.metric("SPY Benchmark", f"${bench_val:,.2f}")
            c3.metric("Alpha (Edge)", f"${alpha:,.2f}",
f"{alpha_pct:+.2f}%")
            st.write("---")

        # 4. MARKET HEALTH (RESTORED FULL)
        spy =3D master_data.get("SPY"); vix =3D master_data.get("^VIX"); rs=
p =3D
master_data.get("RSP")
        mkt_score =3D 0; h_rows =3D []
        if spy is not None:
            if vix is not None:
                v =3D vix.iloc[-1]['Close']
                s =3D "<span style=3D'color:#00ff00'>NORMAL</span>" if v < =
17
else ("<span style=3D'color:#ffaa00'>CAUTIOUS</span>" if v < 20 else "<span
style=3D'color:#ff4444'>PANIC</span>")
                mkt_score +=3D 9 if v < 17 else (6 if v < 20 else (3 if v <
25 else 0))
                h_rows.append({"Indicator": f"VIX Level ({v:.2f})",
"Status": s})

            sc =3D spy.iloc[-1]['Close']; s18 =3D calc_sma(spy['Close'],
18).iloc[-1]; s8 =3D calc_sma(spy['Close'], 8).iloc[-1]
            if sc > s18: mkt_score +=3D 1
            h_rows.append({"Indicator": "SPY Price > SMA18", "Status":
"<span style=3D'color:#00ff00'>PASS</span>" if sc > s18 else "<span
style=3D'color:#ff4444'>FAIL</span>"})

            if s18 >=3D calc_sma(spy['Close'], 18).iloc[-2]: mkt_score +=3D=
 1
            h_rows.append({"Indicator": "SPY SMA18 Rising", "Status":
"<span style=3D'color:#00ff00'>RISING</span>" if s18 >=3D
calc_sma(spy['Close'], 18).iloc[-2] else "<span
style=3D'color:#ff4444'>FALLING</span>"})

            if s8 > calc_sma(spy['Close'], 8).iloc[-2]: mkt_score +=3D 1
            h_rows.append({"Indicator": "SPY SMA8 Rising", "Status": "<span
style=3D'color:#00ff00'>RISING</span>" if s8 > calc_sma(spy['Close'],
8).iloc[-2] else "<span style=3D'color:#ff4444'>FALLING</span>"})

            if rsp is not None:
                rc =3D rsp.iloc[-1]['Close']; r18 =3D calc_sma(rsp['Close']=
,
18).iloc[-1]
                if rc > r18: mkt_score +=3D 1
                h_rows.append({"Indicator": "RSP Price > SMA18", "Status":
"<span style=3D'color:#00ff00'>PASS</span>" if rc > r18 else "<span
style=3D'color:#ff4444'>FAIL</span>"})

            col =3D "#00ff00" if mkt_score >=3D 8 else ("#ffaa00" if mkt_sc=
ore
>=3D 5 else "#ff4444")
            msg =3D "AGGRESSIVE" if mkt_score >=3D 10 else ("CAUTIOUS" if
mkt_score >=3D 8 else "DEFENSIVE")
            risk_per_trade =3D RISK_UNIT_BASE if mkt_score >=3D 8 else
(RISK_UNIT_BASE * 0.5 if mkt_score >=3D 5 else 0)

            h_rows.append({"Indicator": "TOTAL SCORE", "Status": f"<span
style=3D'color:{col}'><b>{mkt_score}/11</b></span>"})
            h_rows.append({"Indicator": "STRATEGY MODE", "Status": f"<span
style=3D'color:{col}'><b>{msg}</b></span>"})
            st.subheader("=F0=9F=8F=A5 Daily Market Health")

st.markdown(pd.DataFrame(h_rows).style.pipe(style_daily_health).to_html(esc=
ape=3DFalse),
unsafe_allow_html=3DTrue)
            st.write("---")

        # 5. SCANNER LOOP (CACHED)
        results =3D []
        scan_list =3D list(set(list(tc.DATA_MAP.keys()) + pf_tickers))
        analysis_db =3D {}

        for t in scan_list:
            if t not in master_data or len(master_data[t]) < 50: continue
            df =3D master_data[t].copy()
            df['SMA18'] =3D calc_sma(df['Close'], 18); df['SMA40'] =3D
calc_sma(df['Close'], 40); df['AD'] =3D calc_ad(df['High'], df['Low'],
df['Close'], df['Volume'])
            # Pine Parity: Soft Distribution Check
            ad_sma18 =3D calc_sma(df['AD'], 18); ad_sma40 =3D
calc_sma(df['AD'], 40)
            df['VolSMA'] =3D calc_sma(df['Volume'], 18); df['RSI5'] =3D
calc_rsi(df['Close'], 5); df['RSI20'] =3D calc_rsi(df['Close'], 20)

            # --- RS CALC (Stability Band 0.5%) ---
            bench_ticker =3D "SPY"
            if t in tc.DATA_MAP and tc.DATA_MAP[t][1]: bench_ticker =3D
tc.DATA_MAP[t][1]

            rs_score_ok =3D False
            if bench_ticker in master_data:
                bench_series =3D master_data[bench_ticker]['Close']
                common_idx =3D df.index.intersection(bench_series.index)
                rs_series =3D df.loc[common_idx, 'Close'] /
bench_series.loc[common_idx]
                rs_sma18 =3D calc_sma(rs_series, 18)

                if len(rs_series) > 2 and len(rs_sma18) > 2:
                    curr_rs =3D rs_series.iloc[-1]; curr_rs_sma =3D
rs_sma18.iloc[-1]
                    prev_rs_sma =3D rs_sma18.iloc[-2]

                    upper_band =3D curr_rs_sma * 1.005
                    lower_band =3D curr_rs_sma * 0.995

                    rs_strong =3D curr_rs > upper_band
                    rs_stable =3D (curr_rs <=3D upper_band) and (curr_rs >=
=3D
lower_band)
                    rs_not_down =3D curr_rs_sma >=3D prev_rs_sma

                    if rs_strong: rs_score_ok =3D True
                    elif rs_stable and rs_not_down: rs_score_ok =3D True
            else:
                rs_score_ok =3D True

            # Weekly
            df_w =3D
df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'=
last','Volume':'sum'})
            df_w.dropna(inplace=3DTrue)
            if len(df_w) < 5: continue
            df_w['SMA8'] =3D calc_sma(df_w['Close'], 8); df_w['SMA18'] =3D
calc_sma(df_w['Close'], 18); df_w['SMA40'] =3D calc_sma(df_w['Close'], 40)
            span_a, span_b =3D calc_ichimoku(df_w['High'], df_w['Low'],
df_w['Close']); df_w['Cloud_Top'] =3D pd.concat([span_a, span_b],
axis=3D1).max(axis=3D1)

            dc =3D df.iloc[-1]; wc =3D df_w.iloc[-1]
            inst_activity =3D calc_structure(df)

            # --- A/D WEAK DISTRIBUTION CHECK (Pine Match) ---
            ad_score_ok =3D False
            if len(ad_sma18) > 2:
                ad_val =3D df['AD'].iloc[-1]
                ad18 =3D ad_sma18.iloc[-1]; ad18_prev =3D ad_sma18.iloc[-2]
                ad40 =3D ad_sma40.iloc[-1]
                ad_weak_distrib =3D (ad_val < ad18 and ad18 <=3D ad18_prev)=
 or
(ad18 < ad40 and ad18 < ad18_prev)
                ad_score_ok =3D not ad_weak_distrib

            vol_msg =3D "NORMAL"
            if df['Volume'].iloc[-1] > (df['VolSMA'].iloc[-1] * 1.5):
vol_msg =3D "SPIKE (Live)"
            elif df['Volume'].iloc[-2] > (df['VolSMA'].iloc[-2] * 1.5):
vol_msg =3D "SPIKE (Prev)"
            elif df['Volume'].iloc[-1] > df['VolSMA'].iloc[-1]: vol_msg =3D
"HIGH (Live)"

            r5 =3D df['RSI5'].iloc[-1]; r20 =3D df['RSI20'].iloc[-1] if not
pd.isna(df['RSI20'].iloc[-1]) else 50
            r5_prev =3D df['RSI5'].iloc[-2]; is_rising =3D r5 > r5_prev

            final_inst_msg =3D inst_activity
            if "SPIKE" in vol_msg:
                if inst_activity =3D=3D "HL": final_inst_msg =3D "ACCUMULAT=
ION
(HL)" if is_rising else final_inst_msg
                if inst_activity =3D=3D "HH": final_inst_msg =3D "BREAKOUT =
(HH)"
if is_rising else "DISTRIBUTION (HH)"
                if inst_activity =3D=3D "LL": final_inst_msg =3D "CAPITULAT=
ION
(LL)" if is_rising else "LIQUIDATION (LL)"
                if inst_activity =3D=3D "LH": final_inst_msg =3D "SELLING (=
LH)"

            w_score =3D 0
            if wc['Close'] > wc['SMA18']: w_score +=3D 1
            if wc['SMA18'] > df_w.iloc[-2]['SMA18']: w_score +=3D 1
            if wc['SMA18'] > wc['SMA40']: w_score +=3D 1
            if wc['Close'] > wc['Cloud_Top']: w_score +=3D 1
            if wc['Close'] > wc['SMA8']: w_score +=3D 1

            # --- DAILY SCORE (5 Pts - Pine Parity) ---
            d_chk =3D 0
            if ad_score_ok: d_chk +=3D 1
            if rs_score_ok: d_chk +=3D 1
            if dc['Close'] > df['SMA18'].iloc[-1]: d_chk +=3D 1
            if df['SMA18'].iloc[-1] >=3D df['SMA18'].iloc[-2]: d_chk +=3D 1=
 #
18 Rising
            if df['SMA18'].iloc[-1] > df['SMA40'].iloc[-1]: d_chk +=3D 1 #
Structure

            w_pulse =3D "GOOD" if (wc['Close'] > wc['SMA18']) and
(dc['Close'] > df['SMA18'].iloc[-1]) else "NO"

            decision =3D "AVOID"; reason =3D "Low Score"
            if w_score >=3D 4:
                if d_chk =3D=3D 5: decision =3D "BUY"; reason =3D "Score 5/=
5"
                elif d_chk =3D=3D 4: decision =3D "SCOUT"; reason =3D "D-Sc=
ore 4"
                elif d_chk =3D=3D 3: decision =3D "SCOUT"; reason =3D "Dip =
Buy"
                else: decision =3D "WATCH"; reason =3D "Daily Weak"
            else: decision =3D "AVOID"; reason =3D "Weekly Weak"

            if not (wc['Close'] > wc['SMA8']): decision =3D "AVOID"; reason=
 =3D
"BELOW W-SMA8"
            elif "NO" in w_pulse: decision =3D "AVOID"; reason =3D "Impulse=
 NO"
            elif risk_per_trade =3D=3D 0 and "BUY" in decision: decision =
=3D
"CAUTION"; reason =3D "VIX Lock"

            atr =3D calc_atr(df['High'], df['Low'], df['Close']).iloc[-1]
            raw_stop =3D dc['Close'] - (2.618 * atr); smart_stop_val =3D
round_to_03_07(raw_stop)
            stop_dist =3D dc['Close'] - smart_stop_val; stop_pct =3D (stop_=
dist
/ dc['Close']) * 100 if dc['Close'] else 0

            num_col =3D "#FF4444"
            if r5 >=3D r20: num_col =3D "#00BFFF" if (r20 > 50 and is_risin=
g)
else "#00FF00"
            arrow_col =3D "#00FF00" if is_rising else "#FF4444"; arrow =3D =
"=E2=86=91"
if is_rising else "=E2=86=93"

            # --- SPLIT COLORING (FIX) ---
            # Num Color
            if r5 >=3D r20:
                if r20 > 50 and is_rising: n_c =3D "#00BFFF" # Blue Spike
                elif is_rising: n_c =3D "#00FF00" # Green Build
                else: n_c =3D "#FF4444" # Rollover
            else:
                if r20 > 50: n_c =3D "#FFA500" # Weakening
                else: n_c =3D "#FF4444" # Bearish

            # Arrow Color
            a_c =3D "#00FF00" if is_rising else "#FF4444"

            rsi_msg =3D f"<span
style=3D'color:{n_c}'><b>{int(r5)}/{int(r20)}</b></span> <span
style=3D'color:{a_c}'><b>{arrow}</b></span>"

            # --- PHASE INJECTION ---
            rrg_phase =3D rrg_snapshot.get(t, "unknown").upper()

            # --- ROTATION SAFETY LOCK (v60.4) ---
            if "WEAKENING" in rrg_phase and "BUY" in decision:
                decision =3D "CAUTION"
                reason =3D "Rotation Weak"

            analysis_db[t] =3D {"Decision": decision, "Reason": reason,
"Price": dc['Close'], "Stop": smart_stop_val, "StopPct": stop_pct, "RRG":
rrg_phase, "W_SMA8_Pass": (wc['Close']>wc['SMA8']), "W_Pulse": w_pulse,
"W_Score": w_score, "D_Score": d_chk, "D_Chk_Price": (dc['Close'] >
df['SMA18'].iloc[-1]), "W_Cloud": (wc['Close']>wc['Cloud_Top']), "AD_Pass":
ad_score_ok, "Vol_Msg": vol_msg, "RSI_Msg": rsi_msg, "Inst_Act":
final_inst_msg}

        for t in scan_list:
            cat_name =3D tc.DATA_MAP.get(t, ["OTHER"])[0]
            if "99. DATA" in cat_name: continue
            if t not in analysis_db: continue
            is_scanner =3D t in tc.DATA_MAP and (tc.DATA_MAP[t][0] !=3D "BE=
NCH"
or t in ["DIA", "QQQ", "IWM", "IWC", "HXT.TO"])
            if not is_scanner: continue

            db =3D analysis_db[t]
            final_decision =3D db['Decision']; final_reason =3D db['Reason'=
]
            if cat_name in tc.SECTOR_PARENTS:
                parent =3D tc.SECTOR_PARENTS[cat_name]
                if parent in analysis_db and "AVOID" in
analysis_db[parent]['Decision']:
                    if t !=3D parent: final_decision =3D "AVOID"; final_rea=
son
=3D "Sector Lock"

            # Re-Check Logic for Blue Spike using clean data
            # Now we have HTML in RSI_Msg, so we can't parse easily. Use
decision logic.
            is_blue_spike =3D ("#00BFFF" in db['RSI_Msg']) and ("SPIKE" in
db['Vol_Msg'])

            final_risk =3D risk_per_trade / 3 if "SCOUT" in final_decision
else risk_per_trade
            if is_blue_spike: final_risk =3D risk_per_trade

            if "AVOID" in final_decision and not is_blue_spike: disp_stop =
=3D
""; disp_shares =3D ""
            else:
                shares =3D int(final_risk / (db['Price'] - db['Stop'])) if
(db['Price'] - db['Stop']) > 0 else 0
                disp_stop =3D f"${db['Stop']:.2f} (-{db['StopPct']:.1f}%)";
disp_shares =3D f"{shares} shares"

            row =3D {
                "Sector": cat_name, "Ticker": t, "Rank": (0 if "00." in
cat_name else 1), "Rotation": db['RRG'],
                "Weekly<br>SMA8": "PASS" if db['W_SMA8_Pass'] else "FAIL",
"Weekly<br>Impulse": db['W_Pulse'],
                "Weekly<br>Score": db['W_Score'], "Daily<br>Score":
db['D_Score'],
                "Structure": "ABOVE 18" if db['D_Chk_Price'] else "BELOW
18",
                "Ichimoku<br>Cloud": "PASS" if db['W_Cloud'] else "FAIL",
"A/D Breadth": "STRONG" if db['AD_Pass'] else "WEAK",
                "Volume": db['Vol_Msg'], "Dual RSI": db['RSI_Msg'],
"Institutional<br>Activity": db['Inst_Act'],
                "Action": final_decision, "Reasoning": final_reason, "Stop
Price": disp_stop, "Position Size": disp_shares
            }
            results.append(row)
            if t =3D=3D "HXT.TO": row_cad =3D row.copy(); row_cad["Sector"]=
 =3D
"15. CANADA (HXT)"; row_cad["Rank"] =3D 0; results.append(row_cad)
            if t in tc.SECTOR_ETFS: row_sec =3D row.copy(); row_sec["Sector=
"]
=3D "02. SECTORS (SUMMARY)"; row_sec["Rank"] =3D 0; results.append(row_sec)

        if results:
            df_final =3D pd.DataFrame(results).sort_values(["Sector", "Rank=
",
"Ticker"], ascending=3D[True, True, True])
            df_final["Sector"] =3D df_final["Sector"].apply(lambda x:
x.split(". ", 1)[1].replace("(SUMMARY)", "").strip() if ". " in x else x)
            cols =3D ["Sector", "Ticker", "Rotation", "Weekly<br>SMA8",
"Weekly<br>Impulse", "Weekly<br>Score", "Daily<br>Score", "Structure",
"Ichimoku<br>Cloud", "A/D Breadth", "Volume", "Dual RSI",
"Institutional<br>Activity", "Action", "Reasoning", "Stop Price", "Position
Size"]
            st.markdown(generate_scanner_html(df_final[cols]),
unsafe_allow_html=3DTrue)
        else:
            st.warning("Scanner returned no results.")

    if mode =3D=3D "Sector Rotation":
        st.subheader("=F0=9F=94=84 Relative Rotation Graphs (RRG)")
        is_dark =3D st.toggle("=F0=9F=8C=99 Dark Mode", value=3DTrue)
        rrg_mode =3D st.radio("View:", ["Indices", "Sectors", "Drill-Down",
"Themes"], horizontal=3DTrue, key=3D"rrg_nav")

        if rrg_mode =3D=3D "Indices":
            c1, c2 =3D st.columns([1,3])
            with c1: bench_sel =3D st.selectbox("Benchmark", ["SPY", "IEF"]=
,
key=3D"bench_idx")
            tgt =3D "IEF" if bench_sel =3D=3D "IEF" else "SPY"
            idx_list =3D list(tc.RRG_INDICES.keys())
            if tgt =3D=3D "IEF": idx_list.append("SPY")
            elif "SPY" in idx_list: idx_list.remove("SPY")

            if st.button("Run Indices"):
                wide_df =3D prepare_rrg_inputs(master_data, idx_list, tgt)
                r, m =3D calculate_rrg_math(wide_df, tgt)
                st.session_state['fig_idx'] =3D plot_rrg_chart(r, m,
tc.RRG_INDICES, f"Indices vs {tgt}", is_dark)
            if 'fig_idx' in st.session_state:
st.plotly_chart(st.session_state['fig_idx'], use_container_width=3DTrue)

        elif rrg_mode =3D=3D "Sectors":
            if st.button("Run Sectors"):
                wide_df =3D prepare_rrg_inputs(master_data,
list(tc.RRG_SECTORS.keys()), "SPY")
                r, m =3D calculate_rrg_math(wide_df, "SPY")
                st.session_state['fig_sec'] =3D plot_rrg_chart(r, m,
tc.RRG_SECTORS, "Sectors vs SPY", is_dark)
            if 'fig_sec' in st.session_state:
st.plotly_chart(st.session_state['fig_sec'], use_container_width=3DTrue)

        elif rrg_mode =3D=3D "Drill-Down":
            c1, c2 =3D st.columns([1,3])
            with c1:
                def fmt(x): return f"{x} - {tc.RRG_SECTORS[x]}" if x in
tc.RRG_SECTORS else x
                opts =3D list(tc.RRG_SECTORS.keys()) + ["Canada (TSX)"]
                sec_key =3D st.selectbox("Select Sector", opts,
format_func=3Dfmt, key=3D"dd_sel")
            if sec_key =3D=3D "Canada (TSX)": bench_dd =3D "HXT.TO"; name_d=
d =3D
"Canadian Titans"
            else: bench_dd =3D sec_key; name_dd =3D tc.RRG_SECTORS[sec_key]

            if st.button(f"Run {name_dd}"):
                comp_list =3D list(tc.RRG_INDUSTRY_MAP.get(sec_key,
{}).keys())
                wide_df =3D prepare_rrg_inputs(master_data, comp_list,
bench_dd)
                r, m =3D calculate_rrg_math(wide_df, bench_dd)
                all_labels =3D {**tc.RRG_INDUSTRY_MAP.get(sec_key, {}),
**tc.RRG_SECTORS}
                st.session_state['fig_dd'] =3D plot_rrg_chart(r, m,
all_labels, f"{name_dd} vs {bench_dd}", is_dark)
            if 'fig_dd' in st.session_state:
st.plotly_chart(st.session_state['fig_dd'], use_container_width=3DTrue)

        elif rrg_mode =3D=3D "Themes":
            if st.button("Run Themes"):
                wide_df =3D prepare_rrg_inputs(master_data,
list(tc.RRG_THEMES.keys()), "SPY")
                r, m =3D calculate_rrg_math(wide_df, "SPY")
                st.session_state['fig_thm'] =3D plot_rrg_chart(r, m,
tc.RRG_THEMES, "Themes vs SPY", is_dark)
            if 'fig_thm' in st.session_state:
st.plotly_chart(st.session_state['fig_thm'], use_container_width=3DTrue)

--000000000000d622c9064a4c3cb0
Content-Type: text/html; charset="UTF-8"
Content-Transfer-Encoding: quoted-printable

<div dir=3D"auto">import streamlit as st<div dir=3D"auto">import yfinance a=
s yf</div><div dir=3D"auto">import pandas as pd</div><div dir=3D"auto">impo=
rt numpy as np</div><div dir=3D"auto">import os</div><div dir=3D"auto">impo=
rt time</div><div dir=3D"auto">from datetime import datetime, timedelta</di=
v><div dir=3D"auto"><br></div><div dir=3D"auto"># --- IMPORT CONFIG ---</di=
v><div dir=3D"auto">try:</div><div dir=3D"auto">=C2=A0 =C2=A0 import titan_=
config as tc</div><div dir=3D"auto">except ImportError:</div><div dir=3D"au=
to">=C2=A0 =C2=A0 st.error(&quot;=E2=9A=A0=EF=B8=8F CRITICAL ERROR: `titan_=
config.py` is missing. Please create it.&quot;)</div><div dir=3D"auto">=C2=
=A0 =C2=A0 st.stop()</div><div dir=3D"auto"><br></div><div dir=3D"auto"># -=
-- SAFE IMPORT FOR PLOTLY ---</div><div dir=3D"auto">try:</div><div dir=3D"=
auto">=C2=A0 =C2=A0 import plotly.graph_objects as go</div><div dir=3D"auto=
">except ImportError:</div><div dir=3D"auto">=C2=A0 =C2=A0 st.warning(&quot=
;=E2=9A=A0=EF=B8=8F Plotly not found. RRG Charts will not work. (pip instal=
l plotly)&quot;)</div><div dir=3D"auto">=C2=A0 =C2=A0 go =3D None</div><div=
 dir=3D"auto"><br></div><div dir=3D"auto"># --- PAGE CONFIGURATION ---</div=
><div dir=3D"auto">st.set_page_config(page_title=3D&quot;Titan Strategy&quo=
t;, layout=3D&quot;wide&quot;)</div><div dir=3D"auto"><br></div><div dir=3D=
"auto"># --- AUTHENTICATION ---</div><div dir=3D"auto">if &#39;authenticate=
d&#39; not in st.session_state:</div><div dir=3D"auto">=C2=A0 =C2=A0 st.ses=
sion_state.authenticated =3D False</div><div dir=3D"auto">=C2=A0 =C2=A0 st.=
session_state.user =3D None</div><div dir=3D"auto"><br></div><div dir=3D"au=
to">def check_login():</div><div dir=3D"auto">=C2=A0 =C2=A0 username =3D st=
.session_state.username_input</div><div dir=3D"auto">=C2=A0 =C2=A0 password=
 =3D st.session_state.password_input</div><div dir=3D"auto">=C2=A0 =C2=A0 i=
f username in tc.CREDENTIALS and tc.CREDENTIALS[username] =3D=3D password:<=
/div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 st.session_state.authent=
icated =3D True</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 st.sessi=
on_state.user =3D username</div><div dir=3D"auto">=C2=A0 =C2=A0 else:</div>=
<div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 st.error(&quot;Incorrect User=
name or Password&quot;)</div><div dir=3D"auto"><br></div><div dir=3D"auto">=
def logout():</div><div dir=3D"auto">=C2=A0 =C2=A0 st.session_state.authent=
icated =3D False</div><div dir=3D"auto">=C2=A0 =C2=A0 st.session_state.user=
 =3D None</div><div dir=3D"auto">=C2=A0 =C2=A0 st.rerun()</div><div dir=3D"=
auto"><br></div><div dir=3D"auto">if not st.session_state.authenticated:</d=
iv><div dir=3D"auto">=C2=A0 =C2=A0 st.title(&quot;=F0=9F=9B=A1=EF=B8=8F Tit=
an Strategy Login&quot;)</div><div dir=3D"auto">=C2=A0 =C2=A0 with st.form(=
&quot;login_form&quot;):</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0=
 st.text_input(&quot;Username&quot;, key=3D&quot;username_input&quot;)</div=
><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 st.text_input(&quot;Password=
&quot;, type=3D&quot;password&quot;, key=3D&quot;password_input&quot;)</div=
><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 st.form_submit_button(&quot;=
Login&quot;, on_click=3Dcheck_login)</div><div dir=3D"auto">=C2=A0 =C2=A0 s=
t.stop()=C2=A0</div><div dir=3D"auto"><br></div><div dir=3D"auto"># =3D=3D=
=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=
=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=
=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=
=3D</div><div dir=3D"auto">#  TITAN STRATEGY APP (v60.5 Variable Name Fix)<=
/div><div dir=3D"auto"># =3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=
=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=
=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=
=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D=3D</div><div dir=3D"auto"><br></div><div =
dir=3D"auto">current_user =3D st.session_state.user</div><div dir=3D"auto">=
PORTFOLIO_FILE =3D f&quot;portfolio_{current_user}.csv&quot;</div><div dir=
=3D"auto"><br></div><div dir=3D"auto">st.sidebar.write(f&quot;=F0=9F=91=A4 =
Logged in as: **{current_user.upper()}**&quot;)</div><div dir=3D"auto">if s=
t.sidebar.button(&quot;Log Out&quot;):</div><div dir=3D"auto">=C2=A0 =C2=A0=
 logout()</div><div dir=3D"auto"><br></div><div dir=3D"auto">st.title(f&quo=
t;=F0=9F=9B=A1=EF=B8=8F Titan Strategy v60.5 ({current_user.upper()})&quot;=
)</div><div dir=3D"auto">st.caption(&quot;Institutional Protocol: Pine Pari=
ty &amp; Stability&quot;)</div><div dir=3D"auto"><br></div><div dir=3D"auto=
"># --- CALCULATIONS ---</div><div dir=3D"auto">def calc_sma(series, length=
): return series.rolling(window=3Dlength).mean()</div><div dir=3D"auto">def=
 calc_ad(high, low, close, volume):</div><div dir=3D"auto">=C2=A0 =C2=A0 mf=
m =3D ((close - low) - (high - close)) / (high - low); mfm =3D mfm.fillna(0=
.0); mfv =3D mfm * volume</div><div dir=3D"auto">=C2=A0 =C2=A0 return mfv.c=
umsum()</div><div dir=3D"auto">def calc_ichimoku(high, low, close):</div><d=
iv dir=3D"auto">=C2=A0 =C2=A0 tenkan =3D (high.rolling(9).max() + low.rolli=
ng(9).min()) / 2</div><div dir=3D"auto">=C2=A0 =C2=A0 kijun =3D (high.rolli=
ng(26).max() + low.rolling(26).min()) / 2</div><div dir=3D"auto">=C2=A0 =C2=
=A0 span_a =3D ((tenkan + kijun) / 2).shift(26); span_b =3D ((high.rolling(=
52).max() + low.rolling(52).min()) / 2).shift(26)</div><div dir=3D"auto">=
=C2=A0 =C2=A0 return span_a, span_b</div><div dir=3D"auto">def calc_atr(hig=
h, low, close, length=3D14):</div><div dir=3D"auto">=C2=A0 =C2=A0 try:</div=
><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 tr1 =3D high - low; tr2 =3D =
abs(high - close.shift(1)); tr3 =3D abs(low - close.shift(1))</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 tr =3D pd.concat([tr1, tr2, tr3], axi=
s=3D1).max(axis=3D1)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 ret=
urn tr.ewm(com=3Dlength-1, adjust=3DFalse).mean()</div><div dir=3D"auto">=
=C2=A0 =C2=A0 except: return pd.Series(0, index=3Dclose.index)</div><div di=
r=3D"auto">def calc_rsi(series, length=3D14):</div><div dir=3D"auto">=C2=A0=
 =C2=A0 try:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 delta =3D s=
eries.diff(); gain =3D delta.clip(lower=3D0); loss =3D -delta.clip(upper=3D=
0)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 avg_gain =3D gain.ewm=
(com=3Dlength-1, adjust=3DFalse).mean()</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 avg_loss =3D loss.ewm(com=3Dlength-1, adjust=3DFalse).mea=
n()</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 rs =3D avg_gain / av=
g_loss; rs =3D rs.fillna(0)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 return 100 - (100 / (1 + rs))</div><div dir=3D"auto">=C2=A0 =C2=A0 exce=
pt: return pd.Series(50, index=3Dseries.index)</div><div dir=3D"auto"><br><=
/div><div dir=3D"auto"># --- ZIG ZAG ENGINE ---</div><div dir=3D"auto">def =
calc_structure(df, deviation_pct=3D0.035):</div><div dir=3D"auto">=C2=A0 =
=C2=A0 if len(df) &lt; 50: return &quot;None&quot;</div><div dir=3D"auto">=
=C2=A0 =C2=A0 pivots =3D []; trend =3D 1; last_val =3D df[&#39;Close&#39;].=
iloc[0]; pivots.append((0, last_val, 1))</div><div dir=3D"auto">=C2=A0 =C2=
=A0 for i in range(1, len(df)):</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0=
 =C2=A0 price =3D df[&#39;Close&#39;].iloc[i]</div><div dir=3D"auto">=C2=A0=
 =C2=A0 =C2=A0 =C2=A0 if trend =3D=3D 1:</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if price &gt; last_val:</div><div dir=3D"au=
to">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 last_val =3D pr=
ice</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 if pivots[-1][2] =3D=3D 1: pivots[-1] =3D (i, price, 1)</div><div d=
ir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 else: p=
ivots.append((i, price, 1))</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 elif price &lt; last_val * (1 - deviation_pct):</div><div=
 dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 trend=
 =3D -1; last_val =3D price; pivots.append((i, price, -1))</div><div dir=3D=
"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 else:</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if price &lt; last_val:</div><div dir=3D"au=
to">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 last_val =3D pr=
ice</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 if pivots[-1][2] =3D=3D -1: pivots[-1] =3D (i, price, -1)</div><div=
 dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 else:=
 pivots.append((i, price, -1))</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 elif price &gt; last_val * (1 + deviation_pct):</div><=
div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 tr=
end =3D 1; last_val =3D price; pivots.append((i, price, 1))</div><div dir=
=3D"auto">=C2=A0 =C2=A0 if len(pivots) &lt; 3: return &quot;Range&quot;</di=
v><div dir=3D"auto">=C2=A0 =C2=A0 return (&quot;HH&quot; if pivots[-1][1] &=
gt; pivots[-3][1] else &quot;LH&quot;) if pivots[-1][2] =3D=3D 1 else (&quo=
t;LL&quot; if pivots[-1][1] &lt; pivots[-3][1] else &quot;HL&quot;)</div><d=
iv dir=3D"auto"><br></div><div dir=3D"auto">def round_to_03_07(price):</div=
><div dir=3D"auto">=C2=A0 =C2=A0 if pd.isna(price): return 0.0</div><div di=
r=3D"auto">=C2=A0 =C2=A0 whole =3D int(price); candidates =3D [c for c in [=
whole + 0.03, whole + 0.07, (whole - 1) + 0.97, (whole - 1) + 0.93] if c &g=
t; 0]</div><div dir=3D"auto">=C2=A0 =C2=A0 return min(candidates, key=3Dlam=
bda x: abs(x - price)) if candidates else price</div><div dir=3D"auto"><br>=
</div><div dir=3D"auto"># --- UNIFIED DATA ENGINE ---</div><div dir=3D"auto=
">@st.cache_data(ttl=3D3600)=C2=A0</div><div dir=3D"auto">def fetch_master_=
data(ticker_list):</div><div dir=3D"auto">=C2=A0 =C2=A0 &quot;&quot;&quot;D=
ownloads daily data for ALL tickers once.&quot;&quot;&quot;</div><div dir=
=3D"auto">=C2=A0 =C2=A0 unique_tickers =3D sorted(list(set(ticker_list)))=
=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 data_map =3D {}</div><div dir=
=3D"auto">=C2=A0 =C2=A0 for t in unique_tickers:</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 try:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 fetch_sym =3D &quot;SPY&quot; if t =3D=3D &quot;MANL&q=
uot; else t</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 tk =3D yf.Ticker(fetch_sym)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 df =3D tk.history(period=3D&quot;2y&quot;, interval=
=3D&quot;1d&quot;)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 df.index =3D pd.to_datetime(df.index).tz_localize(None)</div><di=
v dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if not df.empty an=
d &#39;Close&#39; in df.columns:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 data_map[t] =3D df</div><div dir=3D"=
auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 except: continue</div><div dir=3D"auto">=
=C2=A0 =C2=A0 return data_map</div><div dir=3D"auto"><br></div><div dir=3D"=
auto">def prepare_rrg_inputs(data_map, tickers, benchmark):</div><div dir=
=3D"auto">=C2=A0 =C2=A0 df_wide =3D pd.DataFrame()</div><div dir=3D"auto">=
=C2=A0 =C2=A0 if benchmark in data_map:</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 bench_df =3D data_map[benchmark].resample(&#39;W-FRI&#39;=
).last()</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 df_wide[benchma=
rk] =3D bench_df[&#39;Close&#39;]</div><div dir=3D"auto">=C2=A0 =C2=A0 for =
t in tickers:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if t in da=
ta_map and t !=3D benchmark:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 w_df =3D data_map[t].resample(&#39;W-FRI&#39;).last()<=
/div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 df_wide[t]=
 =3D w_df[&#39;Close&#39;]</div><div dir=3D"auto">=C2=A0 =C2=A0 return df_w=
ide.dropna()</div><div dir=3D"auto"><br></div><div dir=3D"auto"># --- RRG L=
OGIC ---</div><div dir=3D"auto">def calculate_rrg_math(price_data, benchmar=
k_col, window_rs=3D14, window_mom=3D5, smooth_factor=3D3):</div><div dir=3D=
"auto">=C2=A0 =C2=A0 if benchmark_col not in price_data.columns: return pd.=
DataFrame(), pd.DataFrame()</div><div dir=3D"auto">=C2=A0 =C2=A0 df_ratio =
=3D pd.DataFrame(); df_mom =3D pd.DataFrame()</div><div dir=3D"auto">=C2=A0=
 =C2=A0 for col in price_data.columns:</div><div dir=3D"auto">=C2=A0 =C2=A0=
 =C2=A0 =C2=A0 if col !=3D benchmark_col:</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 try:</div><div dir=3D"auto">=C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 rs =3D price_data[col] / price_da=
ta[benchmark_col]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 mean =3D rs.rolling(window_rs).mean(); std =3D rs.rol=
ling(window_rs).std()</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 ratio =3D 100 + ((rs - mean) / std) * 1.5</div>=
<div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 d=
f_ratio[col] =3D ratio</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 except: continue</div><div dir=3D"auto">=C2=A0 =C2=A0 for col=
 in df_ratio.columns:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 tr=
y: df_mom[col] =3D 100 + (df_ratio[col] - df_ratio[col].rolling(window_mom)=
.mean()) * 2</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 except: con=
tinue</div><div dir=3D"auto">=C2=A0 =C2=A0 return df_ratio.rolling(smooth_f=
actor).mean().dropna(), df_mom.rolling(smooth_factor).mean().dropna()</div>=
<div dir=3D"auto"><br></div><div dir=3D"auto">def generate_full_rrg_snapsho=
t(data_map, benchmark=3D&quot;SPY&quot;):</div><div dir=3D"auto">=C2=A0 =C2=
=A0 try:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 all_tickers =3D=
 list(data_map.keys())</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 i=
f benchmark not in all_tickers: return {}</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 wide_df =3D prepare_rrg_inputs(data_map, all_tickers, ben=
chmark)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if wide_df.empty=
: return {}</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 r, m =3D cal=
culate_rrg_math(wide_df, benchmark)</div><div dir=3D"auto">=C2=A0 =C2=A0 =
=C2=A0 =C2=A0 if r.empty or m.empty: return {}</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 status_map =3D {}</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 last_idx =3D r.index[-1]</div><div dir=3D"auto">=C2=A0=
 =C2=A0 =C2=A0 =C2=A0 for t in r.columns:</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 try:</div><div dir=3D"auto">=C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 val_r =3D <a href=3D"http://r.at"=
>r.at</a>[last_idx, t]; val_m =3D <a href=3D"http://m.at">m.at</a>[last_idx=
, t]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 if val_r &gt; 100 and val_m &gt; 100: status =3D &quot;LEADING&q=
uot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 elif val_r &gt; 100 and val_m &lt; 100: status =3D &quot;WEAKENI=
NG&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 elif val_r &lt; 100 and val_m &lt; 100: status =3D &quot;LAGG=
ING&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 else: status =3D &quot;IMPROVING&quot;</div><div dir=3D"auto"=
>=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 status_map[t] =3D =
status</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 exc=
ept: continue</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 return sta=
tus_map</div><div dir=3D"auto">=C2=A0 =C2=A0 except: return {}</div><div di=
r=3D"auto"><br></div><div dir=3D"auto">def plot_rrg_chart(ratios, momentums=
, labels_map, title, is_dark):</div><div dir=3D"auto">=C2=A0 =C2=A0 if go i=
s None: return None</div><div dir=3D"auto">=C2=A0 =C2=A0 fig =3D go.Figure(=
)</div><div dir=3D"auto">=C2=A0 =C2=A0 if is_dark:</div><div dir=3D"auto">=
=C2=A0 =C2=A0 =C2=A0 =C2=A0 bg_col, text_col =3D &quot;black&quot;, &quot;w=
hite&quot;; c_lead, c_weak, c_lag, c_imp =3D &quot;#00FF00&quot;, &quot;#FF=
FF00&quot;, &quot;#FF4444&quot;, &quot;#00BFFF&quot;; template =3D &quot;pl=
otly_dark&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 else:</div><div dir=3D=
"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 bg_col, text_col =3D &quot;white&quot;, =
&quot;black&quot;; c_lead, c_weak, c_lag, c_imp =3D &quot;#008000&quot;, &q=
uot;#FF8C00&quot;, &quot;#CC0000&quot;, &quot;#0000FF&quot;; template =3D &=
quot;plotly_white&quot;</div><div dir=3D"auto"><br></div><div dir=3D"auto">=
=C2=A0 =C2=A0 has_data =3D False</div><div dir=3D"auto">=C2=A0 =C2=A0 for t=
icker in labels_map.keys():</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 if ticker not in ratios.columns: continue</div><div dir=3D"auto">=C2=A0=
 =C2=A0 =C2=A0 =C2=A0 xt =3D ratios[ticker].tail(5); yt =3D momentums[ticke=
r].tail(5)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if len(xt) &l=
t; 5: continue</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 has_data =
=3D True</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 cx, cy =3D xt.i=
loc[-1], yt.iloc[-1]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if =
cx &gt; 100 and cy &gt; 100: color =3D c_lead</div><div dir=3D"auto">=C2=A0=
 =C2=A0 =C2=A0 =C2=A0 elif cx &gt; 100 and cy &lt; 100: color =3D c_weak</d=
iv><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 elif cx &lt; 100 and cy &l=
t; 100: color =3D c_lag</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
else: color =3D c_imp</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=
=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 fig.add_trace(go.Sca=
tter(x=3Dxt, y=3Dyt, mode=3D&#39;lines&#39;, line=3Ddict(color=3Dcolor, wid=
th=3D2, shape=3D&#39;spline&#39;), opacity=3D0.6, showlegend=3DFalse, hover=
info=3D&#39;skip&#39;))</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
fig.add_trace(go.Scatter(x=3D[cx], y=3D[cy], mode=3D&#39;markers+text&#39;,=
 marker=3Ddict(color=3Dcolor, size=3D12, line=3Ddict(color=3Dtext_col, widt=
h=3D1)), text=3D[ticker], textposition=3D&quot;top center&quot;, textfont=
=3Ddict(color=3Dtext_col), hovertemplate=3Df&quot;&lt;b&gt;{labels_map.get(=
ticker, ticker)}&lt;/b&gt;&lt;br&gt;T: %{{x:.2f}}&lt;br&gt;M: %{{y:.2f}}&qu=
ot;))</div><div dir=3D"auto"><br></div><div dir=3D"auto">=C2=A0 =C2=A0 if n=
ot has_data: return None</div><div dir=3D"auto">=C2=A0 =C2=A0 op =3D 0.1 if=
 is_dark else 0.05</div><div dir=3D"auto">=C2=A0 =C2=A0 fig.add_hline(y=3D1=
00, line_dash=3D&quot;dot&quot;, line_color=3D&quot;gray&quot;); fig.add_vl=
ine(x=3D100, line_dash=3D&quot;dot&quot;, line_color=3D&quot;gray&quot;)</d=
iv><div dir=3D"auto">=C2=A0 =C2=A0 fig.add_shape(type=3D&quot;rect&quot;, x=
0=3D100, y0=3D100, x1=3D200, y1=3D200, fillcolor=3Df&quot;rgba(0,255,0,{op}=
)&quot;, layer=3D&quot;below&quot;, line_width=3D0)</div><div dir=3D"auto">=
=C2=A0 =C2=A0 fig.add_shape(type=3D&quot;rect&quot;, x0=3D100, y0=3D0, x1=
=3D200, y1=3D100, fillcolor=3Df&quot;rgba(255,255,0,{op})&quot;, layer=3D&q=
uot;below&quot;, line_width=3D0)</div><div dir=3D"auto">=C2=A0 =C2=A0 fig.a=
dd_shape(type=3D&quot;rect&quot;, x0=3D0, y0=3D0, x1=3D100, y1=3D100, fillc=
olor=3Df&quot;rgba(255,0,0,{op})&quot;, layer=3D&quot;below&quot;, line_wid=
th=3D0)</div><div dir=3D"auto">=C2=A0 =C2=A0 fig.add_shape(type=3D&quot;rec=
t&quot;, x0=3D0, y0=3D100, x1=3D100, y1=3D200, fillcolor=3Df&quot;rgba(0,0,=
255,{op})&quot;, layer=3D&quot;below&quot;, line_width=3D0)</div><div dir=
=3D"auto">=C2=A0 =C2=A0 fig.update_layout(title=3Dtitle, template=3Dtemplat=
e, height=3D650, showlegend=3DFalse, xaxis=3Ddict(range=3D[96, 104], showgr=
id=3DFalse, title=3D&quot;RS-Ratio (Trend)&quot;), yaxis=3Ddict(range=3D[96=
, 104], showgrid=3DFalse, title=3D&quot;RS-Momentum (Velocity)&quot;))</div=
><div dir=3D"auto">=C2=A0 =C2=A0 fig.add_annotation(x=3D104, y=3D104, text=
=3D&quot;LEADING&quot;, showarrow=3DFalse, font=3Ddict(size=3D16, color=3Dc=
_lead), xanchor=3D&quot;right&quot;, yanchor=3D&quot;top&quot;)</div><div d=
ir=3D"auto">=C2=A0 =C2=A0 fig.add_annotation(x=3D104, y=3D96, text=3D&quot;=
WEAKENING&quot;, showarrow=3DFalse, font=3Ddict(size=3D16, color=3Dc_weak),=
 xanchor=3D&quot;right&quot;, yanchor=3D&quot;bottom&quot;)</div><div dir=
=3D"auto">=C2=A0 =C2=A0 fig.add_annotation(x=3D96, y=3D96, text=3D&quot;LAG=
GING&quot;, showarrow=3DFalse, font=3Ddict(size=3D16, color=3Dc_lag), xanch=
or=3D&quot;left&quot;, yanchor=3D&quot;bottom&quot;)</div><div dir=3D"auto"=
>=C2=A0 =C2=A0 fig.add_annotation(x=3D96, y=3D104, text=3D&quot;IMPROVING&q=
uot;, showarrow=3DFalse, font=3Ddict(size=3D16, color=3Dc_imp), xanchor=3D&=
quot;left&quot;, yanchor=3D&quot;top&quot;)</div><div dir=3D"auto">=C2=A0 =
=C2=A0 return fig</div><div dir=3D"auto"><br></div><div dir=3D"auto"># --- =
STYLING ---</div><div dir=3D"auto">def style_final(styler):</div><div dir=
=3D"auto">=C2=A0 =C2=A0 def color_rotation(val):</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 if &quot;LEADING&quot; in val: return &#39;color: =
#00FF00; font-weight: bold&#39;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0=
 =C2=A0 if &quot;WEAKENING&quot; in val: return &#39;color: #FFFF00; font-w=
eight: bold&#39;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if &quo=
t;LAGGING&quot; in val: return &#39;color: #FF4444; font-weight: bold&#39;<=
/div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if &quot;IMPROVING&quot;=
 in val: return &#39;color: #00BFFF; font-weight: bold&#39;</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 return &#39;&#39;</div><div dir=3D"au=
to">=C2=A0 =C2=A0 def color_rsi(val):</div><div dir=3D"auto">=C2=A0 =C2=A0 =
=C2=A0 =C2=A0 try:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 parts =3D val.split(); r5 =3D float(parts[0].split(&#39;/&#39;)[=
0]); r20 =3D float(parts[0].split(&#39;/&#39;)[1]); arrow =3D parts[1]</div=
><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if r5 &gt;=3D =
r20: return &#39;color: #00BFFF; font-weight: bold&#39; if (r20 &gt; 50 and=
 arrow=3D=3D&quot;=E2=86=91&quot;) else (&#39;color: #00FF00; font-weight: =
bold&#39; if arrow=3D=3D&quot;=E2=86=91&quot; else &#39;color: #FF4444; fon=
t-weight: bold&#39;)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 return &#39;color: #FFA500; font-weight: bold&#39; if r20 &gt; 5=
0 else &#39;color: #FF4444; font-weight: bold&#39;</div><div dir=3D"auto">=
=C2=A0 =C2=A0 =C2=A0 =C2=A0 except: return &#39;&#39;</div><div dir=3D"auto=
">=C2=A0 =C2=A0 def color_inst(val):</div><div dir=3D"auto">=C2=A0 =C2=A0 =
=C2=A0 =C2=A0 if &quot;ACCUMULATION&quot; in val or &quot;BREAKOUT&quot; in=
 val: return &#39;color: #00FF00; font-weight: bold&#39;=C2=A0</div><div di=
r=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if &quot;CAPITULATION&quot; in val: =
return &#39;color: #00BFFF; font-weight: bold&#39;=C2=A0 =C2=A0 =C2=A0 =C2=
=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if &quot;DISTRIBUTIO=
N&quot; in val or &quot;LIQUIDATION&quot; in val: return &#39;color: #FF444=
4; font-weight: bold&#39;=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0=
 =C2=A0 if &quot;SELLING&quot; in val: return &#39;color: #FFA500; font-wei=
ght: bold&#39;=C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 return &#39;color: #CCFFCC&#39; if &quot;HH&quot; in val =
else (&#39;color: #FFCCCC&#39; if &quot;LL&quot; in val else &#39;color: #8=
88888&#39;)</div><div dir=3D"auto">=C2=A0 =C2=A0 def highlight_ticker_row(r=
ow):</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 styles =3D [&#39;&#=
39; for _ in row.index]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
if &#39;Ticker&#39; not in row.index: return styles</div><div dir=3D"auto">=
=C2=A0 =C2=A0 =C2=A0 =C2=A0 idx =3D row.index.get_loc(&#39;Ticker&#39;); ac=
t =3D str(row.get(&#39;Action&#39;, &#39;&#39;)).upper(); vol =3D str(row.g=
et(&#39;Volume&#39;, &#39;&#39;)).upper(); rsi =3D str(row.get(&#39;Dual RS=
I&#39;, &#39;&#39;))</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if =
&quot;AVOID&quot; in act: pass</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 elif &quot;BUY&quot; in act: styles[idx] =3D &#39;background-color: =
#006600; color: white; font-weight: bold&#39;</div><div dir=3D"auto">=C2=A0=
 =C2=A0 =C2=A0 =C2=A0 elif &quot;SCOUT&quot; in act: styles[idx] =3D &#39;b=
ackground-color: #005555; color: white; font-weight: bold&#39;</div><div di=
r=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 elif &quot;SOON&quot; in act: styles=
[idx] =3D &#39;background-color: #CC5500; color: white; font-weight: bold&#=
39;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 elif &quot;CAUTION&q=
uot; in act: styles[idx] =3D &#39;background-color: #AA4400; color: white; =
font-weight: bold&#39;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 r=
eturn styles</div><div dir=3D"auto">=C2=A0 =C2=A0 return styler.set_table_s=
tyles([{&#39;selector&#39;: &#39;th&#39;, &#39;props&#39;: [(&#39;text-alig=
n&#39;, &#39;center&#39;), (&#39;background-color&#39;, &#39;#111&#39;), (&=
#39;color&#39;, &#39;white&#39;), (&#39;font-size&#39;, &#39;12px&#39;)]}, =
{&#39;selector&#39;: &#39;td&#39;, &#39;props&#39;: [(&#39;text-align&#39;,=
 &#39;center&#39;), (&#39;font-size&#39;, &#39;14px&#39;), (&#39;padding&#3=
9;, &#39;8px&#39;)]}]).set_properties(**{&#39;background-color&#39;: &#39;#=
222&#39;, &#39;color&#39;: &#39;white&#39;, &#39;border-color&#39;: &#39;#4=
44&#39;}).apply(highlight_ticker_row, axis=3D1).map(lambda v: &#39;color: #=
00ff00; font-weight: bold&#39; if v in [&quot;BUY&quot;, &quot;STRONG BUY&q=
uot;] else (&#39;color: #00ffff; font-weight: bold&#39; if &quot;SCOUT&quot=
; in v else (&#39;color: #ffaa00; font-weight: bold&#39; if v in [&quot;SOO=
N&quot;, &quot;CAUTION&quot;] else &#39;color: white&#39;)), subset=3D[&quo=
t;Action&quot;]).map(lambda v: &#39;color: #ff00ff; font-weight: bold&#39; =
if &quot;SPIKE&quot; in v else (&#39;color: #00ff00&#39; if &quot;HIGH&quot=
; in v else &#39;color: #ccc&#39;), subset=3D[&quot;Volume&quot;]).map(lamb=
da v: &#39;color: #00ff00; font-weight: bold&#39; if &quot;STRONG&quot; in =
v else (&#39;color: #ff0000&#39; if &quot;WEAK&quot; in v else &#39;color: =
#ffaa00&#39;), subset=3D[&quot;A/D Breadth&quot;]).map(lambda v: &#39;color=
: #ff0000; font-weight: bold&#39; if &quot;FAIL&quot; in v or &quot;NO&quot=
; in v else &#39;color: #00ff00&#39;, subset=3D[&quot;Ichimoku&lt;br&gt;Clo=
ud&quot;, &quot;Weekly&lt;br&gt;SMA8&quot;]).map(lambda v: &#39;color: #00f=
f00; font-weight: bold&#39; if &quot;GOOD&quot; in v else (&#39;color: #ffa=
a00; font-weight: bold&#39; if &quot;WEAK&quot; in v else &#39;color: #ff00=
00; font-weight: bold&#39;), subset=3D[&quot;Weekly&lt;br&gt;Impulse&quot;]=
).map(lambda v: &#39;color: #00ff00; font-weight: bold&#39; if v &gt;=3D 4 =
else (&#39;color: #ffaa00; font-weight: bold&#39; if v =3D=3D 3 else &#39;c=
olor: #ff0000; font-weight: bold&#39;), subset=3D[&quot;Weekly&lt;br&gt;Sco=
re&quot;, &quot;Daily&lt;br&gt;Score&quot;]).map(lambda v: &#39;color: #ff0=
000; font-weight: bold&#39; if &quot;BELOW 18&quot; in v else &#39;color: #=
00ff00&#39;, subset=3D[&quot;Structure&quot;]).map(color_rotation, subset=
=3D[&quot;Rotation&quot;]).map(color_rsi, subset=3D[&quot;Dual RSI&quot;]).=
map(color_inst, subset=3D[&quot;Institutional&lt;br&gt;Activity&quot;]).hid=
e(axis=3D&#39;index&#39;)</div><div dir=3D"auto"><br></div><div dir=3D"auto=
">def style_daily_health(styler):</div><div dir=3D"auto">=C2=A0 =C2=A0 def =
color_status(v):</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if &quo=
t;PASS&quot; in v or &quot;NORMAL&quot; in v or &quot;CAUTIOUS&quot; in v o=
r &quot;RISING&quot; in v or &quot;AGGRESSIVE&quot; in v: return &#39;color=
: #00ff00; font-weight: bold&#39;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 if &quot;FAIL&quot; in v or &quot;PANIC&quot; in v or &quot;DEFE=
NSIVE&quot; in v or &quot;FALLING&quot; in v or &quot;CASH&quot; in v: retu=
rn &#39;color: #ff4444; font-weight: bold&#39;</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 return &#39;color: white; font-weight: bold&#39;</=
div><div dir=3D"auto">=C2=A0 =C2=A0 return styler.set_table_styles([{&#39;s=
elector&#39;: &#39;th&#39;, &#39;props&#39;: [(&#39;text-align&#39;, &#39;l=
eft&#39;), (&#39;background-color&#39;, &#39;#111&#39;), (&#39;color&#39;, =
&#39;white&#39;), (&#39;font-size&#39;, &#39;14px&#39;)]}, {&#39;selector&#=
39;: &#39;td&#39;, &#39;props&#39;: [(&#39;text-align&#39;, &#39;left&#39;)=
, (&#39;font-size&#39;, &#39;14px&#39;), (&#39;padding&#39;, &#39;8px&#39;)=
]}]).set_properties(**{&#39;background-color&#39;: &#39;#222&#39;, &#39;bor=
der-color&#39;: &#39;#444&#39;}).set_properties(subset=3D[&#39;Indicator&#3=
9;], **{&#39;color&#39;: &#39;white&#39;, &#39;font-weight&#39;: &#39;bold&=
#39;}).map(color_status, subset=3D[&#39;Status&#39;]).hide(axis=3D&#39;inde=
x&#39;)</div><div dir=3D"auto"><br></div><div dir=3D"auto">def style_portfo=
lio(styler):</div><div dir=3D"auto">=C2=A0 =C2=A0 def color_pl(val):</div><=
div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 try: return &#39;color: #00ff0=
0; font-weight: bold&#39; if float(val.strip(&#39;%&#39;).replace(&#39;+&#3=
9;,&#39;&#39;)) &gt;=3D 0 else &#39;color: #ff4444; font-weight: bold&#39;<=
/div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 except: return &#39;&#39=
;</div><div dir=3D"auto">=C2=A0 =C2=A0 def color_pl_dol(val):</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 try: return &#39;color: #00ff00; font=
-weight: bold&#39; if float(val.strip(&#39;$&#39;).replace(&#39;+&#39;,&#39=
;&#39;).replace(&#39;,&#39;,&#39;&#39;)) &gt;=3D 0 else &#39;color: #ff4444=
; font-weight: bold&#39;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0=
 except: return &#39;&#39;</div><div dir=3D"auto">=C2=A0 =C2=A0 def color_a=
ction(val): return &#39;color: #ff0000; font-weight: bold; background-color=
: #220000&#39; if &quot;EXIT&quot; in val else (&#39;color: #00ff00; font-w=
eight: bold&#39; if &quot;HOLD&quot; in val else &#39;color: #ffffff&#39;)<=
/div><div dir=3D"auto">=C2=A0 =C2=A0 return styler.set_table_styles([{&#39;=
selector&#39;: &#39;th&#39;, &#39;props&#39;: [(&#39;text-align&#39;, &#39;=
center&#39;), (&#39;background-color&#39;, &#39;#111&#39;), (&#39;color&#39=
;, &#39;white&#39;)]}, {&#39;selector&#39;: &#39;td&#39;, &#39;props&#39;: =
[(&#39;text-align&#39;, &#39;center&#39;), (&#39;font-size&#39;, &#39;14px&=
#39;)]}]).map(color_pl, subset=3D[&quot;% Return&quot;]).map(color_pl_dol, =
subset=3D[&quot;Gain/Loss ($)&quot;]).map(color_action, subset=3D[&quot;Aud=
it Action&quot;]).hide(axis=3D&#39;index&#39;)</div><div dir=3D"auto"><br><=
/div><div dir=3D"auto">def style_history(styler):</div><div dir=3D"auto">=
=C2=A0 =C2=A0 def color_pl(val):</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 try: return &#39;color: #00ff00; font-weight: bold&#39; if float=
(val.strip(&#39;%&#39;).replace(&#39;+&#39;,&#39;&#39;)) &gt;=3D 0 else &#3=
9;color: #ff4444; font-weight: bold&#39;</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 except: return &#39;&#39;</div><div dir=3D"auto">=C2=A0 =
=C2=A0 def color_pl_dol(val):</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 try: return &#39;color: #00ff00; font-weight: bold&#39; if float(val=
.strip(&#39;$&#39;).replace(&#39;+&#39;,&#39;&#39;).replace(&#39;,&#39;,&#3=
9;&#39;)) &gt;=3D 0 else &#39;color: #ff4444; font-weight: bold&#39;</div><=
div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 except: return &#39;&#39;</div=
><div dir=3D"auto">=C2=A0 =C2=A0 return styler.set_table_styles([{&#39;sele=
ctor&#39;: &#39;th&#39;, &#39;props&#39;: [(&#39;text-align&#39;, &#39;cent=
er&#39;), (&#39;background-color&#39;, &#39;#111&#39;), (&#39;color&#39;, &=
#39;white&#39;)]}, {&#39;selector&#39;: &#39;td&#39;, &#39;props&#39;: [(&#=
39;text-align&#39;, &#39;center&#39;), (&#39;font-size&#39;, &#39;14px&#39;=
)]}]).map(color_pl, subset=3D[&quot;% Return&quot;]).map(color_pl_dol, subs=
et=3D[&quot;P/L&quot;]).hide(axis=3D&#39;index&#39;)</div><div dir=3D"auto"=
><br></div><div dir=3D"auto">def fmt_delta(val): return f&quot;-${abs(val):=
,.2f}&quot; if val &lt; 0 else f&quot;${val:,.2f}&quot;</div><div dir=3D"au=
to"><br></div><div dir=3D"auto"># --- PORTFOLIO ENGINE ---</div><div dir=3D=
"auto">def load_portfolio():</div><div dir=3D"auto">=C2=A0 =C2=A0 cols =3D =
[&quot;ID&quot;, &quot;Ticker&quot;, &quot;Date&quot;, &quot;Shares&quot;, =
&quot;Cost_Basis&quot;, &quot;Status&quot;, &quot;Exit_Date&quot;, &quot;Ex=
it_Price&quot;, &quot;Return&quot;, &quot;Realized_PL&quot;, &quot;SPY_Retu=
rn&quot;, &quot;Type&quot;, &quot;Shadow_SPY&quot;]</div><div dir=3D"auto">=
=C2=A0 =C2=A0 if not os.path.exists(PORTFOLIO_FILE): pd.DataFrame(columns=
=3Dcols).to_csv(PORTFOLIO_FILE, index=3DFalse)</div><div dir=3D"auto">=C2=
=A0 =C2=A0 df =3D pd.read_csv(PORTFOLIO_FILE)</div><div dir=3D"auto">=C2=A0=
 =C2=A0 if &#39;Cost&#39; in df.columns: df.rename(columns=3D{&#39;Cost&#39=
;: &#39;Cost_Basis&#39;}, inplace=3DTrue)</div><div dir=3D"auto">=C2=A0 =C2=
=A0 if &#39;Cost_Basis&#39; not in df.columns: df[&#39;Cost_Basis&#39;] =3D=
 0.0</div><div dir=3D"auto">=C2=A0 =C2=A0 if &quot;ID&quot; not in df.colum=
ns: df[&quot;ID&quot;] =3D range(1, len(df) + 1)</div><div dir=3D"auto">=C2=
=A0 =C2=A0 if &#39;Shadow_SPY&#39; not in df.columns: df[&#39;Shadow_SPY&#3=
9;] =3D 0.0</div><div dir=3D"auto">=C2=A0 =C2=A0 df[&#39;Shadow_SPY&#39;] =
=3D pd.to_numeric(df[&#39;Shadow_SPY&#39;], errors=3D&#39;coerce&#39;).fill=
na(0.0)</div><div dir=3D"auto">=C2=A0 =C2=A0 return df</div><div dir=3D"aut=
o"><br></div><div dir=3D"auto">def save_portfolio(df):</div><div dir=3D"aut=
o">=C2=A0 =C2=A0 dollar_cols =3D [&#39;Cost_Basis&#39;, &#39;Exit_Price&#39=
;, &#39;Realized_PL&#39;, &#39;Return&#39;, &#39;SPY_Return&#39;]</div><div=
 dir=3D"auto">=C2=A0 =C2=A0 for col in dollar_cols:</div><div dir=3D"auto">=
=C2=A0 =C2=A0 =C2=A0 =C2=A0 if col in df.columns: df[col] =3D pd.to_numeric=
(df[col], errors=3D&#39;coerce&#39;).round(2)</div><div dir=3D"auto">=C2=A0=
 =C2=A0 def clean_shares(row): return round(row[&#39;Shares&#39;], 2) if ro=
w[&#39;Ticker&#39;] =3D=3D &#39;CASH&#39; else int(row[&#39;Shares&#39;])</=
div><div dir=3D"auto">=C2=A0 =C2=A0 if not df.empty: df[&#39;Shares&#39;] =
=3D df.apply(clean_shares, axis=3D1)</div><div dir=3D"auto">=C2=A0 =C2=A0 d=
f.to_csv(PORTFOLIO_FILE, index=3DFalse)</div><div dir=3D"auto"><br></div><d=
iv dir=3D"auto"># --- SIDEBAR: MANAGER ---</div><div dir=3D"auto">st.sideba=
r.header(&quot;=F0=9F=92=BC Portfolio Manager&quot;)</div><div dir=3D"auto"=
>pf_df =3D load_portfolio()</div><div dir=3D"auto">cash_rows =3D pf_df[(pf_=
df[&#39;Ticker&#39;] =3D=3D &#39;CASH&#39;) &amp; (pf_df[&#39;Status&#39;] =
=3D=3D &#39;OPEN&#39;)]</div><div dir=3D"auto">current_cash =3D cash_rows[&=
#39;Shares&#39;].sum() if not cash_rows.empty else 0.0</div><div dir=3D"aut=
o">st.sidebar.metric(&quot;Cash Available&quot;, f&quot;${current_cash:,.2f=
}&quot;)</div><div dir=3D"auto"><br></div><div dir=3D"auto">tab1, tab2, tab=
3, tab4, tab5 =3D st.sidebar.tabs([&quot;=F0=9F=9F=A2 Buy&quot;, &quot;=F0=
=9F=94=B4 Sell&quot;, &quot;=F0=9F=92=B5 Cash&quot;, &quot;=F0=9F=A7=AE Cal=
c&quot;, &quot;=F0=9F=9B=A0=EF=B8=8F Fix&quot;])</div><div dir=3D"auto"><br=
></div><div dir=3D"auto">with tab1:</div><div dir=3D"auto">=C2=A0 =C2=A0 wi=
th st.form(&quot;buy_trade&quot;):</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 b_tick =3D st.selectbox(&quot;Ticker&quot;, list(tc.DATA_MAP.key=
s()))</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 b_date =3D st.date=
_input(&quot;Buy Date&quot;)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 b_shares =3D st.number_input(&quot;Shares&quot;, min_value=3D1, valu=
e=3D100)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 b_price =3D st.=
number_input(&quot;Price&quot;, min_value=3D0.01, value=3D100.00)</div><div=
 dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if st.form_submit_button(&quot;Ex=
ecute Buy&quot;):</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 new_id =3D 1 if pf_df.empty else pf_df[&quot;ID&quot;].max() + 1</d=
iv><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 pf_df =3D pd=
.concat([pf_df, pd.DataFrame([{&quot;ID&quot;: new_id, &quot;Ticker&quot;: =
b_tick, &quot;Date&quot;: b_date, &quot;Shares&quot;: b_shares, &quot;Cost_=
Basis&quot;: b_price, &quot;Status&quot;: &quot;OPEN&quot;, &quot;Type&quot=
;: &quot;STOCK&quot;, &quot;Shadow_SPY&quot;: 0.0}])], ignore_index=3DTrue)=
</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if curren=
t_cash &gt;=3D (b_shares * b_price):</div><div dir=3D"auto">=C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0pf_df =3D pd.concat([pf_df,=
 pd.DataFrame([{&quot;ID&quot;: pf_df[&quot;ID&quot;].max()+1, &quot;Ticker=
&quot;: &quot;CASH&quot;, &quot;Date&quot;: b_date, &quot;Shares&quot;: -(b=
_shares * b_price), &quot;Cost_Basis&quot;: 1.0, &quot;Status&quot;: &quot;=
OPEN&quot;, &quot;Type&quot;: &quot;TRADE_CASH&quot;, &quot;Shadow_SPY&quot=
;: 0.0}])], ignore_index=3DTrue)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 save_portfolio(pf_df); st.success(f&quot;Bought {b=
_tick}&quot;); st.rerun()</div><div dir=3D"auto"><br></div><div dir=3D"auto=
">with tab2:</div><div dir=3D"auto">=C2=A0 =C2=A0 open_trades =3D pf_df[(pf=
_df[&#39;Status&#39;] =3D=3D &#39;OPEN&#39;) &amp; (pf_df[&#39;Ticker&#39;]=
 !=3D &#39;CASH&#39;)]</div><div dir=3D"auto">=C2=A0 =C2=A0 if not open_tra=
des.empty:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 trade_map =3D=
 {}</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 opts =3D []</div><di=
v dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 for idx, row in open_trades.iter=
rows():</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 la=
bel =3D f&quot;ID:{row[&#39;ID&#39;]} | {row[&#39;Ticker&#39;]} | {int(row[=
&#39;Shares&#39;])} shares | {row[&#39;Date&#39;]}&quot;</div><div dir=3D"a=
uto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 trade_map[label] =3D {&#39;i=
d&#39;: row[&#39;ID&#39;], &#39;max_shares&#39;: int(row[&#39;Shares&#39;])=
, &#39;idx&#39;: idx}</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 opts.append(label)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 selected_trade_str =3D st.selectbox(&quot;Select Position&quot;,=
 opts)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if selected_trade=
_str:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 sel_=
data =3D trade_map[selected_trade_str]; sel_id =3D sel_data[&#39;id&#39;]; =
max_qty =3D sel_data[&#39;max_shares&#39;]</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 with st.form(&quot;sell_trade&quot;):</d=
iv><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 s_shares =3D st.number_input(&quot;Shares to Sell&quot;, min_value=3D1,=
 max_value=3Dmax_qty, value=3Dmax_qty, step=3D1)</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 s_date =3D st.date_inp=
ut(&quot;Date&quot;); s_price =3D st.number_input(&quot;Price&quot;, 0.01, =
value=3D100.00)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 if st.form_submit_button(&quot;Execute Sell&quot;):</d=
iv><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 row_idx =3D sel_data[&#39;idx&#39;]; buy_price =3D float(=
<a href=3D"http://pf_df.at">pf_df.at</a>[row_idx, &#39;Cost_Basis&#39;]); b=
uy_date_str =3D <a href=3D"http://pf_df.at">pf_df.at</a>[row_idx, &#39;Date=
&#39;]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 ret_pct =3D ((s_price - buy_price) / buy_price) * =
100; pl_dollars =3D (s_price - buy_price) * s_shares</div><div dir=3D"auto"=
>=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 cash=
_id =3D pf_df[&quot;ID&quot;].max() + 1</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 cash_row =3D pd=
.DataFrame([{&quot;ID&quot;: cash_id, &quot;Ticker&quot;: &quot;CASH&quot;,=
 &quot;Date&quot;: s_date, &quot;Shares&quot;: (s_price * s_shares), &quot;=
Cost_Basis&quot;: 1.0, &quot;Status&quot;: &quot;OPEN&quot;, &quot;Type&quo=
t;: &quot;TRADE_CASH&quot;, &quot;Shadow_SPY&quot;: 0.0}])</div><div dir=3D=
"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 pf_df =3D pd.concat([pf_df, cash_row], ignore_index=3DTrue)</div><div d=
ir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 if s_shares &lt; max_qty:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 <a href=
=3D"http://pf_df.at">pf_df.at</a>[row_idx, &#39;Shares&#39;] -=3D s_shares<=
/div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 new_id =3D pf_df[&quot;ID&quot;].max() + 1<=
/div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 new_closed_row =3D pd.DataFrame([{&quot;ID&=
quot;: new_id, &quot;Ticker&quot;: <a href=3D"http://pf_df.at">pf_df.at</a>=
[row_idx, &#39;Ticker&#39;], &quot;Date&quot;: buy_date_str, &quot;Shares&q=
uot;: s_shares, &quot;Cost_Basis&quot;: buy_price, &quot;Status&quot;: &quo=
t;CLOSED&quot;, &quot;Exit_Date&quot;: s_date, &quot;Exit_Price&quot;: s_pr=
ice, &quot;Return&quot;: ret_pct, &quot;Realized_PL&quot;: pl_dollars, &quo=
t;SPY_Return&quot;: 0.0, &quot;Type&quot;: &quot;STOCK&quot;, &quot;Shadow_=
SPY&quot;: 0.0}])</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 pf_df =3D pd.concat([pf_d=
f, new_closed_row], ignore_index=3DTrue)</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 else:</div><div=
 dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 <a href=3D"http://pf_df.at">pf_df.at</a>[row_idx, =
&#39;Status&#39;] =3D &#39;CLOSED&#39;; <a href=3D"http://pf_df.at">pf_df.a=
t</a>[row_idx, &#39;Exit_Date&#39;] =3D s_date; <a href=3D"http://pf_df.at"=
>pf_df.at</a>[row_idx, &#39;Exit_Price&#39;] =3D s_price; <a href=3D"http:/=
/pf_df.at">pf_df.at</a>[row_idx, &#39;Return&#39;] =3D ret_pct; <a href=3D"=
http://pf_df.at">pf_df.at</a>[row_idx, &#39;Realized_PL&#39;] =3D pl_dollar=
s</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 save_portfolio(pf_df); st.success(f&quot;Sold {s_share=
s} shares. P&amp;L: ${pl_dollars:+.2f}&quot;); st.rerun()</div><div dir=3D"=
auto">=C2=A0 =C2=A0 else: <a href=3D"http://st.info">st.info</a>(&quot;No O=
pen Positions&quot;)</div><div dir=3D"auto"><br></div><div dir=3D"auto">wit=
h tab3:</div><div dir=3D"auto">=C2=A0 =C2=A0 with st.form(&quot;cash&quot;)=
:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 op =3D st.radio(&quot;=
Op&quot;, [&quot;Deposit&quot;, &quot;Withdraw&quot;]); amt =3D st.number_i=
nput(&quot;Amt&quot;, 100.00); dt =3D st.date_input(&quot;Date&quot;)</div>=
<div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if st.form_submit_button(&quo=
t;Execute&quot;):</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 shares =3D 0.0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 try:=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 =C2=A0 spy =3D yf.Ticker(&quot;SPY&quot;).history(sta=
rt=3Ddt, end=3Ddt+timedelta(days=3D5))</div><div dir=3D"auto">=C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if not spy.empty: shares =3D amt=
 / spy[&#39;Close&#39;].iloc[0]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 except: pass</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 final =3D amt if op =3D=3D &quot;Deposit&quot; els=
e -amt; s_shares =3D shares if op =3D=3D &quot;Deposit&quot; else -shares</=
div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 pf_df =3D p=
d.concat([pf_df, pd.DataFrame([{&quot;ID&quot;: pf_df[&quot;ID&quot;].max()=
+1, &quot;Ticker&quot;: &quot;CASH&quot;, &quot;Date&quot;: dt, &quot;Share=
s&quot;: final, &quot;Cost_Basis&quot;: 1.0, &quot;Status&quot;: &quot;OPEN=
&quot;, &quot;Type&quot;: &quot;TRANSFER&quot;, &quot;Shadow_SPY&quot;: s_s=
hares}])], ignore_index=3DTrue)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 save_portfolio(pf_df); st.success(&quot;Done&quot;); =
st.rerun()</div><div dir=3D"auto"><br></div><div dir=3D"auto">with tab4:</d=
iv><div dir=3D"auto">=C2=A0 =C2=A0 st.subheader(&quot;Calculator&quot;); RI=
SK_UNIT_BASE =3D st.number_input(&quot;Risk Unit&quot;, 100, value=3D2300);=
 tk =3D st.text_input(&quot;Ticker&quot;).upper()</div><div dir=3D"auto">=
=C2=A0 =C2=A0 if tk:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 try=
:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 d =3D yf=
.Ticker(tk).history(&quot;1mo&quot;); c =3D d[&#39;Close&#39;].iloc[-1]; at=
r =3D calc_atr(d[&#39;High&#39;], d[&#39;Low&#39;], d[&#39;Close&#39;]).ilo=
c[-1]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 stop=
 =3D round_to_03_07(c - 2.618*atr)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 if c &gt; stop:</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 sh =3D int(RISK_UNIT_BASE / (=
c - stop))=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 <a href=3D"http://st.info">st.info</a>(f&quot;Entry: $=
{c:.2f} | Stop: ${stop:.2f} | Shares: {sh} | Cap: ${sh*c:,.0f}&quot;)</div>=
<div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 except: st.error(&quot;Error&=
quot;)</div><div dir=3D"auto"><br></div><div dir=3D"auto">with tab5:</div><=
div dir=3D"auto">=C2=A0 =C2=A0 st.write(&quot;### =F0=9F=9B=A0=EF=B8=8F Dat=
a Management&quot;)</div><div dir=3D"auto">=C2=A0 =C2=A0 if os.path.exists(=
PORTFOLIO_FILE):</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 with op=
en(PORTFOLIO_FILE, &quot;rb&quot;) as file:</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 st.download_button(&quot;Download Portfo=
lio CSV&quot;, file, PORTFOLIO_FILE, &quot;text/csv&quot;)</div><div dir=3D=
"auto">=C2=A0 =C2=A0 else: st.warning(&quot;No portfolio file found.&quot;)=
</div><div dir=3D"auto">=C2=A0 =C2=A0 st.write(&quot;---&quot;)</div><div d=
ir=3D"auto">=C2=A0 =C2=A0 uploaded_file =3D st.file_uploader(&quot;Restore =
.csv&quot;, type=3D[&quot;csv&quot;])</div><div dir=3D"auto">=C2=A0 =C2=A0 =
if uploaded_file is not None and st.button(&quot;CONFIRM RESTORE&quot;):</d=
iv><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 try:</div><div dir=3D"auto=
">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 pd.read_csv(uploaded_file).to_c=
sv(PORTFOLIO_FILE, index=3DFalse)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 st.success(&quot;Data Restored!&quot;); st.rerun()=
</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 except: st.error(&quot;=
Error&quot;)</div><div dir=3D"auto">=C2=A0 =C2=A0 st.write(&quot;---&quot;)=
</div><div dir=3D"auto">=C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =
=C2=A0 action_type =3D st.radio(&quot;Advanced Tools&quot;, [&quot;Delete T=
rade&quot;, &quot;Edit Trade&quot;, &quot;=E2=9A=A0=EF=B8=8F FACTORY RESET&=
quot;, &quot;Rebuild Benchmark History&quot;])</div><div dir=3D"auto">=C2=
=A0 =C2=A0 if action_type =3D=3D &quot;=E2=9A=A0=EF=B8=8F FACTORY RESET&quo=
t; and st.button(&quot;CONFIRM RESET&quot;):</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 if os.path.exists(PORTFOLIO_FILE): os.remove(PORTFOLIO=
_FILE)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 st.success(&quot;=
Reset!&quot;); st.rerun()</div><div dir=3D"auto">=C2=A0 =C2=A0 elif action_=
type =3D=3D &quot;Rebuild Benchmark History&quot; and st.button(&quot;RUN R=
EBUILD&quot;):</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0wit=
h st.spinner(&quot;Rebuilding...&quot;):</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0try:</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0spy_hist =3D yf.Tick=
er(&quot;SPY&quot;).history(period=3D&quot;10y&quot;)</div><div dir=3D"auto=
">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0for idx, ro=
w in pf_df.iterrows():</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0if row[&#39;Type&#39;] =3D=
=3D &#39;TRANSFER&#39; and row[&#39;Ticker&#39;] =3D=3D &#39;CASH&#39;:</di=
v><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0t_date =3D pd.to_datetime(row[&#39;Date&=
#39;])</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0idx_loc =3D spy_hist.index.sea=
rchsorted(t_date)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0price =3D spy_hist.=
iloc[idx_loc][&#39;Close&#39;] if idx_loc &lt; len(spy_hist) else spy_hist.=
iloc[-1][&#39;Close&#39;]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0if price =
&gt; 0: <a href=3D"http://pf_df.at">pf_df.at</a>[idx, &#39;Shadow_SPY&#39;]=
 =3D float(row[&#39;Shares&#39;]) / price</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0save_portfolio(pf_df); =
st.success(&quot;Done!&quot;); st.rerun()</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0except: st.error(&quot;Error&quot;)</=
div><div dir=3D"auto"><br></div><div dir=3D"auto"># --- HTML CACHING ---</d=
iv><div dir=3D"auto">@st.cache_data</div><div dir=3D"auto">def generate_sca=
nner_html(results_df):</div><div dir=3D"auto">=C2=A0 =C2=A0 if results_df.e=
mpty: return &quot;&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 return resul=
ts_df.style.pipe(style_final).to_html(escape=3DFalse)</div><div dir=3D"auto=
"><br></div><div dir=3D"auto"># --- MAIN EXECUTION ---</div><div dir=3D"aut=
o">if &quot;run_analysis&quot; not in st.session_state: st.session_state.ru=
n_analysis =3D False</div><div dir=3D"auto">if st.button(&quot;RUN ANALYSIS=
&quot;, type=3D&quot;primary&quot;): st.session_state.run_analysis =3D True=
; st.rerun()</div><div dir=3D"auto"><br></div><div dir=3D"auto">if st.sessi=
on_state.run_analysis:</div><div dir=3D"auto">=C2=A0 =C2=A0 if st.button(&q=
uot;=E2=AC=85=EF=B8=8F Back to Menu&quot;): st.session_state.run_analysis =
=3D False; st.rerun()</div><div dir=3D"auto">=C2=A0 =C2=A0=C2=A0</div><div =
dir=3D"auto">=C2=A0 =C2=A0 # --- UNIFIED LIST GENERATION ---</div><div dir=
=3D"auto">=C2=A0 =C2=A0 pf_tickers =3D pf_df[&#39;Ticker&#39;].unique().tol=
ist() if not pf_df.empty else []</div><div dir=3D"auto">=C2=A0 =C2=A0 pf_ti=
ckers =3D [x for x in pf_tickers if x !=3D &quot;CASH&quot;]</div><div dir=
=3D"auto">=C2=A0 =C2=A0 all_tickers =3D list(tc.DATA_MAP.keys()) + pf_ticke=
rs + list(tc.RRG_SECTORS.keys()) + list(tc.RRG_INDICES.keys()) + list(tc.RR=
G_THEMES.keys()) + [&quot;CAD=3DX&quot;]=C2=A0</div><div dir=3D"auto">=C2=
=A0 =C2=A0 for v in tc.RRG_INDUSTRY_MAP.values(): all_tickers.extend(list(v=
.keys()))</div><div dir=3D"auto">=C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto"=
>=C2=A0 =C2=A0 # --- MASTER DATA FETCH &amp; RRG CALC ---</div><div dir=3D"=
auto">=C2=A0 =C2=A0 with st.spinner(&#39;Downloading Unified Market Data...=
&#39;):</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 master_data =3D =
fetch_master_data(all_tickers)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 rrg_snapshot =3D generate_full_rrg_snapshot(master_data, &quot;SPY&q=
uot;)</div><div dir=3D"auto"><br></div><div dir=3D"auto">=C2=A0 =C2=A0 # --=
- STATE-BASED NAVIGATION ---</div><div dir=3D"auto">=C2=A0 =C2=A0 mode =3D =
st.radio(&quot;Navigation&quot;, [&quot;Scanner&quot;, &quot;Sector Rotatio=
n&quot;], horizontal=3DTrue, key=3D&quot;main_nav&quot;)</div><div dir=3D"a=
uto">=C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 if mode =3D=
=3D &quot;Scanner&quot;:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0=
 # 1. HOLDINGS</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 open_pos =
=3D pf_df[(pf_df[&#39;Status&#39;] =3D=3D &#39;OPEN&#39;) &amp; (pf_df[&#39=
;Ticker&#39;] !=3D &#39;CASH&#39;)]</div><div dir=3D"auto">=C2=A0 =C2=A0 =
=C2=A0 =C2=A0 eq_val =3D 0.0; total_cost_basis =3D 0.0; pf_rows =3D []</div=
><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto"=
>=C2=A0 =C2=A0 =C2=A0 =C2=A0 if not open_pos.empty:</div><div dir=3D"auto">=
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 for idx, row in open_pos.iterrows=
():</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 t =3D row[&#39;Ticker&#39;]; shares =3D row[&#39;Shares&#39;]; cost=
 =3D row[&#39;Cost_Basis&#39;]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 curr_price =3D cost=C2=A0</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if t in m=
aster_data and not master_data[t].empty: curr_price =3D master_data[t][&#39=
;Close&#39;].iloc[-1]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 pos_val =3D shares * curr_price; eq_val +=3D po=
s_val</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 total_cost_basis +=3D (shares * cost)</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 pf_rows.append({&quot;=
Ticker&quot;: t, &quot;Shares&quot;: int(shares), &quot;Avg Cost&quot;: f&q=
uot;${cost:.2f}&quot;, &quot;Current&quot;: f&quot;${curr_price:.2f}&quot;,=
 &quot;Gain/Loss ($)&quot;: f&quot;${(pos_val - (shares * cost)):+.2f}&quot=
;, &quot;% Return&quot;: f&quot;{((curr_price - cost) / cost) * 100:+.2f}%&=
quot;, &quot;Audit Action&quot;: &quot;HOLD&quot;})</div><div dir=3D"auto">=
=C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 total_net_worth =3D current_cash + eq_val</div><div dir=3D"auto"=
>=C2=A0 =C2=A0 =C2=A0 =C2=A0 cad_data =3D master_data.get(&quot;CAD=3DX&quo=
t;)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if cad_data is not N=
one and not cad_data.empty:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 rate =3D cad_data[&#39;Close&#39;].iloc[-1]</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if rate &lt; 1.0: rate =
=3D 1.0 / rate=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 cad_rate =3D rate</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 else:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 cad_rate =3D 1.40=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
total_nw_cad =3D total_net_worth * cad_rate</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 open_pl_val =3D eq_val - total_cost_basis</div><div di=
r=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 open_pl_cad =3D open_pl_val * cad_ra=
te</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 c1, c2, c3, c4 =3D st.columns(4)</div=
><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 c1.metric(f&quot;Net Worth (=
CAD @ {cad_rate:.2f})&quot;, f&quot;${total_nw_cad:,.2f}&quot;, fmt_delta(o=
pen_pl_cad))</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 c2.metric(&=
quot;Net Worth (USD)&quot;, f&quot;${total_net_worth:,.2f}&quot;, fmt_delta=
(open_pl_val))</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 c3.metric=
(&quot;Cash&quot;, f&quot;${current_cash:,.2f}&quot;); c4.metric(&quot;Equi=
ty&quot;, f&quot;${eq_val:,.2f}&quot;)</div><div dir=3D"auto">=C2=A0 =C2=A0=
 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if =
pf_rows: st.markdown(pd.DataFrame(pf_rows).style.pipe(style_portfolio).to_h=
tml(), unsafe_allow_html=3DTrue)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 else: <a href=3D"http://st.info">st.info</a>(&quot;No active tra=
des.&quot;)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 st.write(&qu=
ot;---&quot;)</div><div dir=3D"auto"><br></div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 # 3. BENCHMARK</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 shadow_shares_total =3D pf_df[&#39;Shadow_SPY&#39;].sum()</div><=
div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 spy_data =3D master_data.get(&=
quot;SPY&quot;)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if spy_d=
ata is not None:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 curr_spy =3D spy_data[&#39;Close&#39;].iloc[-1]</div><div dir=3D"aut=
o">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 bench_val =3D shadow_shares_to=
tal * curr_spy</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 alpha =3D total_net_worth - bench_val</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 alpha_pct =3D ((total_net_worth - bench_=
val) / bench_val * 100) if bench_val &gt; 0 else 0</div><div dir=3D"auto">=
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 c1, c2, c3 =3D st.columns(3)</div=
><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 c1.metric(&quo=
t;Titan Net Worth&quot;, f&quot;${total_net_worth:,.2f}&quot;)</div><div di=
r=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 c2.metric(&quot;SPY Be=
nchmark&quot;, f&quot;${bench_val:,.2f}&quot;)</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 c3.metric(&quot;Alpha (Edge)&quot;, =
f&quot;${alpha:,.2f}&quot;, f&quot;{alpha_pct:+.2f}%&quot;)</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 st.write(&quot;---&quot=
;)</div><div dir=3D"auto"><br></div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 # 4. MARKET HEALTH (RESTORED FULL)</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 spy =3D master_data.get(&quot;SPY&quot;); vix =3D master_=
data.get(&quot;^VIX&quot;); rsp =3D master_data.get(&quot;RSP&quot;)</div><=
div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 mkt_score =3D 0; h_rows =3D []=
</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if spy is not None:</di=
v><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if vix is not=
 None:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 v =3D vix.iloc[-1][&#39;Close&#39;]</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 s =3D &quot;&lt;span s=
tyle=3D&#39;color:#00ff00&#39;&gt;NORMAL&lt;/span&gt;&quot; if v &lt; 17 el=
se (&quot;&lt;span style=3D&#39;color:#ffaa00&#39;&gt;CAUTIOUS&lt;/span&gt;=
&quot; if v &lt; 20 else &quot;&lt;span style=3D&#39;color:#ff4444&#39;&gt;=
PANIC&lt;/span&gt;&quot;)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 mkt_score +=3D 9 if v &lt; 17 else (6 if v =
&lt; 20 else (3 if v &lt; 25 else 0))</div><div dir=3D"auto">=C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 h_rows.append({&quot;Indicator&qu=
ot;: f&quot;VIX Level ({v:.2f})&quot;, &quot;Status&quot;: s})</div><div di=
r=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D=
"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 sc =3D spy.iloc[-1][&#39;C=
lose&#39;]; s18 =3D calc_sma(spy[&#39;Close&#39;], 18).iloc[-1]; s8 =3D cal=
c_sma(spy[&#39;Close&#39;], 8).iloc[-1]</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if sc &gt; s18: mkt_score +=3D 1</div><div =
dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 h_rows.append({&quot=
;Indicator&quot;: &quot;SPY Price &gt; SMA18&quot;, &quot;Status&quot;: &qu=
ot;&lt;span style=3D&#39;color:#00ff00&#39;&gt;PASS&lt;/span&gt;&quot; if s=
c &gt; s18 else &quot;&lt;span style=3D&#39;color:#ff4444&#39;&gt;FAIL&lt;/=
span&gt;&quot;})</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 if s18 &gt;=3D calc_sma(spy[&#39;Close&#39;], 18).iloc[-2]: mkt_score +=
=3D 1</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 h_ro=
ws.append({&quot;Indicator&quot;: &quot;SPY SMA18 Rising&quot;, &quot;Statu=
s&quot;: &quot;&lt;span style=3D&#39;color:#00ff00&#39;&gt;RISING&lt;/span&=
gt;&quot; if s18 &gt;=3D calc_sma(spy[&#39;Close&#39;], 18).iloc[-2] else &=
quot;&lt;span style=3D&#39;color:#ff4444&#39;&gt;FALLING&lt;/span&gt;&quot;=
})</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</=
div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if s8 &gt; =
calc_sma(spy[&#39;Close&#39;], 8).iloc[-2]: mkt_score +=3D 1</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 h_rows.append({&quot;In=
dicator&quot;: &quot;SPY SMA8 Rising&quot;, &quot;Status&quot;: &quot;&lt;s=
pan style=3D&#39;color:#00ff00&#39;&gt;RISING&lt;/span&gt;&quot; if s8 &gt;=
 calc_sma(spy[&#39;Close&#39;], 8).iloc[-2] else &quot;&lt;span style=3D&#3=
9;color:#ff4444&#39;&gt;FALLING&lt;/span&gt;&quot;})</div><div dir=3D"auto"=
><br></div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if r=
sp is not None:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 rc =3D rsp.iloc[-1][&#39;Close&#39;]; r18 =3D calc_sma=
(rsp[&#39;Close&#39;], 18).iloc[-1]</div><div dir=3D"auto">=C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if rc &gt; r18: mkt_score +=3D 1<=
/div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 h_rows.append({&quot;Indicator&quot;: &quot;RSP Price &gt; SMA18&quot;,=
 &quot;Status&quot;: &quot;&lt;span style=3D&#39;color:#00ff00&#39;&gt;PASS=
&lt;/span&gt;&quot; if rc &gt; r18 else &quot;&lt;span style=3D&#39;color:#=
ff4444&#39;&gt;FAIL&lt;/span&gt;&quot;})</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 col =3D &quot;#00ff00&quot; if mkt_score &gt;=
=3D 8 else (&quot;#ffaa00&quot; if mkt_score &gt;=3D 5 else &quot;#ff4444&q=
uot;)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 msg =
=3D &quot;AGGRESSIVE&quot; if mkt_score &gt;=3D 10 else (&quot;CAUTIOUS&quo=
t; if mkt_score &gt;=3D 8 else &quot;DEFENSIVE&quot;)</div><div dir=3D"auto=
">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 risk_per_trade =3D RISK_UNIT_BA=
SE if mkt_score &gt;=3D 8 else (RISK_UNIT_BASE * 0.5 if mkt_score &gt;=3D 5=
 else 0)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 h_r=
ows.append({&quot;Indicator&quot;: &quot;TOTAL SCORE&quot;, &quot;Status&qu=
ot;: f&quot;&lt;span style=3D&#39;color:{col}&#39;&gt;&lt;b&gt;{mkt_score}/=
11&lt;/b&gt;&lt;/span&gt;&quot;})</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 h_rows.append({&quot;Indicator&quot;: &quot;STRATE=
GY MODE&quot;, &quot;Status&quot;: f&quot;&lt;span style=3D&#39;color:{col}=
&#39;&gt;&lt;b&gt;{msg}&lt;/b&gt;&lt;/span&gt;&quot;})</div><div dir=3D"aut=
o">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 st.subheader(&quot;=F0=9F=8F=
=A5 Daily Market Health&quot;)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 st.markdown(pd.DataFrame(h_rows).style.pipe(style_dail=
y_health).to_html(escape=3DFalse), unsafe_allow_html=3DTrue)</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 st.write(&quot;---&quot=
;)</div><div dir=3D"auto"><br></div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 # 5. SCANNER LOOP (CACHED)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 results =3D []</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 scan_list =3D list(set(list(tc.DATA_MAP.keys()) + pf_tickers))</div><di=
v dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 analysis_db =3D {}</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 for t in scan_list:</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if t not in master_data or len(master_data[=
t]) &lt; 50: continue</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 df =3D master_data[t].copy()</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 df[&#39;SMA18&#39;] =3D calc_sma(df[&#39=
;Close&#39;], 18); df[&#39;SMA40&#39;] =3D calc_sma(df[&#39;Close&#39;], 40=
); df[&#39;AD&#39;] =3D calc_ad(df[&#39;High&#39;], df[&#39;Low&#39;], df[&=
#39;Close&#39;], df[&#39;Volume&#39;])</div><div dir=3D"auto">=C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 =C2=A0 # Pine Parity: Soft Distribution Check</div><d=
iv dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 ad_sma18 =3D calc=
_sma(df[&#39;AD&#39;], 18); ad_sma40 =3D calc_sma(df[&#39;AD&#39;], 40)</di=
v><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 df[&#39;VolSM=
A&#39;] =3D calc_sma(df[&#39;Volume&#39;], 18); df[&#39;RSI5&#39;] =3D calc=
_rsi(df[&#39;Close&#39;], 5); df[&#39;RSI20&#39;] =3D calc_rsi(df[&#39;Clos=
e&#39;], 20)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
# --- RS CALC (Stability Band 0.5%) ---</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 bench_ticker =3D &quot;SPY&quot;</div><div =
dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if t in tc.DATA_MAP =
and tc.DATA_MAP[t][1]: bench_ticker =3D tc.DATA_MAP[t][1]</div><div dir=3D"=
auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto=
">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 rs_score_ok =3D False</div><div=
 dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if bench_ticker in =
master_data:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 bench_series =3D master_data[bench_ticker][&#39;Close&#39=
;]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 common_idx =3D df.index.intersection(bench_series.index)</div><div d=
ir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 rs_seri=
es =3D df.loc[common_idx, &#39;Close&#39;] / bench_series.loc[common_idx]</=
div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 rs_sma18 =3D calc_sma(rs_series, 18)</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if len(rs_series) &=
gt; 2 and len(rs_sma18) &gt; 2:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 curr_rs =3D rs_series.ilo=
c[-1]; curr_rs_sma =3D rs_sma18.iloc[-1]</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 prev_rs_sma =3D=
 rs_sma18.iloc[-2]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 upper_band =
=3D curr_rs_sma * 1.005</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 lower_band =3D curr_rs_sma * 0.99=
5</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 rs_strong =3D curr_rs &gt; up=
per_band</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 rs_stable =3D (curr_rs &lt;=3D upper_band) and =
(curr_rs &gt;=3D lower_band)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 rs_not_down =3D curr_rs_sm=
a &gt;=3D prev_rs_sma</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if rs_st=
rong: rs_score_ok =3D True</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 elif rs_stable and rs_not_dow=
n: rs_score_ok =3D True</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 else:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 rs_score_ok =3D True=C2=A0</div><div dir=3D"auto">=
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 # Weekly</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 df_w =3D df.resample(&#39;W-FRI&#39;=
).agg({&#39;Open&#39;:&#39;first&#39;,&#39;High&#39;:&#39;max&#39;,&#39;Low=
&#39;:&#39;min&#39;,&#39;Close&#39;:&#39;last&#39;,&#39;Volume&#39;:&#39;su=
m&#39;})</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 d=
f_w.dropna(inplace=3DTrue)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 if len(df_w) &lt; 5: continue</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 df_w[&#39;SMA8&#39;] =3D calc_sma(df=
_w[&#39;Close&#39;], 8); df_w[&#39;SMA18&#39;] =3D calc_sma(df_w[&#39;Close=
&#39;], 18); df_w[&#39;SMA40&#39;] =3D calc_sma(df_w[&#39;Close&#39;], 40)<=
/div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 span_a, sp=
an_b =3D calc_ichimoku(df_w[&#39;High&#39;], df_w[&#39;Low&#39;], df_w[&#39=
;Close&#39;]); df_w[&#39;Cloud_Top&#39;] =3D pd.concat([span_a, span_b], ax=
is=3D1).max(axis=3D1)</div><div dir=3D"auto"><br></div><div dir=3D"auto">=
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 dc =3D df.iloc[-1]; wc =3D df_w.i=
loc[-1]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 in=
st_activity =3D calc_structure(df)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 # --- A/D WEAK DISTRIBUTION CHECK (Pine Match) ---</di=
v><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 ad_score_ok =
=3D False</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
if len(ad_sma18) &gt; 2:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 =C2=A0 ad_val =3D df[&#39;AD&#39;].iloc[-1]</div><div=
 dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 ad18 =
=3D ad_sma18.iloc[-1]; ad18_prev =3D ad_sma18.iloc[-2]</div><div dir=3D"aut=
o">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 ad40 =3D ad_sma4=
0.iloc[-1]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 =C2=A0 ad_weak_distrib =3D (ad_val &lt; ad18 and ad18 &lt;=3D ad18_=
prev) or (ad18 &lt; ad40 and ad18 &lt; ad18_prev)</div><div dir=3D"auto">=
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 ad_score_ok =3D not=
 ad_weak_distrib</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 vol_msg =3D &quot;NORMAL&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 if df[&#39;Volume&#39;].iloc[-1] &gt; (df[&#39;Vol=
SMA&#39;].iloc[-1] * 1.5): vol_msg =3D &quot;SPIKE (Live)&quot;</div><div d=
ir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 elif df[&#39;Volume&#=
39;].iloc[-2] &gt; (df[&#39;VolSMA&#39;].iloc[-2] * 1.5): vol_msg =3D &quot=
;SPIKE (Prev)&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 elif df[&#39;Volume&#39;].iloc[-1] &gt; df[&#39;VolSMA&#39;].ilo=
c[-1]: vol_msg =3D &quot;HIGH (Live)&quot;</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 r5 =3D df[&#39;RSI5&#39;].iloc[-1]; r20 =3D=
 df[&#39;RSI20&#39;].iloc[-1] if not pd.isna(df[&#39;RSI20&#39;].iloc[-1]) =
else 50</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 r5=
_prev =3D df[&#39;RSI5&#39;].iloc[-2]; is_rising =3D r5 &gt; r5_prev</div><=
div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div =
dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 final_inst_msg =3D i=
nst_activity</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 if &quot;SPIKE&quot; in vol_msg:</div><div dir=3D"auto">=C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if inst_activity =3D=3D &quot;HL&=
quot;: final_inst_msg =3D &quot;ACCUMULATION (HL)&quot; if is_rising else f=
inal_inst_msg</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 if inst_activity =3D=3D &quot;HH&quot;: final_inst_msg =
=3D &quot;BREAKOUT (HH)&quot; if is_rising else &quot;DISTRIBUTION (HH)&quo=
t;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 if inst_activity =3D=3D &quot;LL&quot;: final_inst_msg =3D &quot;CAP=
ITULATION (LL)&quot; if is_rising else &quot;LIQUIDATION (LL)&quot;</div><d=
iv dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if =
inst_activity =3D=3D &quot;LH&quot;: final_inst_msg =3D &quot;SELLING (LH)&=
quot;</div><div dir=3D"auto"><br></div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 w_score =3D 0</div><div dir=3D"auto">=C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if wc[&#39;Close&#39;] &gt; wc[&#39;SMA18&#39;=
]: w_score +=3D 1</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 if wc[&#39;SMA18&#39;] &gt; df_w.iloc[-2][&#39;SMA18&#39;]: w_score=
 +=3D 1</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if=
 wc[&#39;SMA18&#39;] &gt; wc[&#39;SMA40&#39;]: w_score +=3D 1</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if wc[&#39;Close&#39;] =
&gt; wc[&#39;Cloud_Top&#39;]: w_score +=3D 1</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if wc[&#39;Close&#39;] &gt; wc[&#39;SMA8=
&#39;]: w_score +=3D 1=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 # --- DAILY SCORE (5 Pts - Pine Parity) ---</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 d_chk =3D 0</div><div d=
ir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if ad_score_ok: d_chk=
 +=3D 1</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if=
 rs_score_ok: d_chk +=3D 1</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 if dc[&#39;Close&#39;] &gt; df[&#39;SMA18&#39;].iloc[-1]:=
 d_chk +=3D 1</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 if df[&#39;SMA18&#39;].iloc[-1] &gt;=3D df[&#39;SMA18&#39;].iloc[-2]: d=
_chk +=3D 1 # 18 Rising</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 if df[&#39;SMA18&#39;].iloc[-1] &gt; df[&#39;SMA40&#39;].iloc=
[-1]: d_chk +=3D 1 # Structure</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 w_pulse =3D &quot;GOOD&quot; if (wc[&#39;Close&#39;] &gt;=
 wc[&#39;SMA18&#39;]) and (dc[&#39;Close&#39;] &gt; df[&#39;SMA18&#39;].ilo=
c[-1]) else &quot;NO&quot;</div><div dir=3D"auto"><br></div><div dir=3D"aut=
o">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 decision =3D &quot;AVOID&quot;=
; reason =3D &quot;Low Score&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 if w_score &gt;=3D 4:</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if d_chk =3D=3D 5: dec=
ision =3D &quot;BUY&quot;; reason =3D &quot;Score 5/5&quot;=C2=A0</div><div=
 dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 elif =
d_chk =3D=3D 4: decision =3D &quot;SCOUT&quot;; reason =3D &quot;D-Score 4&=
quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 elif d_chk =3D=3D 3: decision =3D &quot;SCOUT&quot;; reason =3D =
&quot;Dip Buy&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 else: decision =3D &quot;WATCH&quot;; reason =3D &=
quot;Daily Weak&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 else: decision =3D &quot;AVOID&quot;; reason =3D &quot;Weekly=
 Weak&quot;</div><div dir=3D"auto"><br></div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if not (wc[&#39;Close&#39;] &gt; wc[&#39;SM=
A8&#39;]): decision =3D &quot;AVOID&quot;; reason =3D &quot;BELOW W-SMA8&qu=
ot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 elif &=
quot;NO&quot; in w_pulse: decision =3D &quot;AVOID&quot;; reason =3D &quot;=
Impulse NO&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 elif risk_per_trade =3D=3D 0 and &quot;BUY&quot; in decision: decisi=
on =3D &quot;CAUTION&quot;; reason =3D &quot;VIX Lock&quot;</div><div dir=
=3D"auto"><br></div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 atr =3D calc_atr(df[&#39;High&#39;], df[&#39;Low&#39;], df[&#39;Clos=
e&#39;]).iloc[-1]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 raw_stop =3D dc[&#39;Close&#39;] - (2.618 * atr); smart_stop_val =
=3D round_to_03_07(raw_stop)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 stop_dist =3D dc[&#39;Close&#39;] - smart_stop_val; st=
op_pct =3D (stop_dist / dc[&#39;Close&#39;]) * 100 if dc[&#39;Close&#39;] e=
lse 0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=
=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 num_co=
l =3D &quot;#FF4444&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 if r5 &gt;=3D r20: num_col =3D &quot;#00BFFF&quot; if (r2=
0 &gt; 50 and is_rising) else &quot;#00FF00&quot;</div><div dir=3D"auto">=
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 arrow_col =3D &quot;#00FF00&quot;=
 if is_rising else &quot;#FF4444&quot;; arrow =3D &quot;=E2=86=91&quot; if =
is_rising else &quot;=E2=86=93&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 # --- SPLIT COLORING (FIX) ---</div><div dir=3D"au=
to">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 # Num Color</div><div dir=3D"=
auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if r5 &gt;=3D r20:</div><di=
v dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if r=
20 &gt; 50 and is_rising: n_c =3D &quot;#00BFFF&quot; # Blue Spike</div><di=
v dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 elif=
 is_rising: n_c =3D &quot;#00FF00&quot; # Green Build</div><div dir=3D"auto=
">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 else: n_c =3D &qu=
ot;#FF4444&quot; # Rollover</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 else:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 if r20 &gt; 50: n_c =3D &quot;#FFA500&quot; # W=
eakening</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 else: n_c =3D &quot;#FF4444&quot; # Bearish</div><div dir=3D"=
auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto=
">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 # Arrow Color</div><div dir=3D"=
auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 a_c =3D &quot;#00FF00&quot;=
 if is_rising else &quot;#FF4444&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 rsi_msg =3D f&quot;&lt;span style=3D&#39;color:{n_=
c}&#39;&gt;&lt;b&gt;{int(r5)}/{int(r20)}&lt;/b&gt;&lt;/span&gt; &lt;span st=
yle=3D&#39;color:{a_c}&#39;&gt;&lt;b&gt;{arrow}&lt;/b&gt;&lt;/span&gt;&quot=
;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</d=
iv><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 # --- PHASE =
INJECTION ---</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 rrg_phase =3D rrg_snapshot.get(t, &quot;unknown&quot;).upper()</div><di=
v dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div di=
r=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 # --- ROTATION SAFETY =
LOCK (v60.4) ---</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 if &quot;WEAKENING&quot; in rrg_phase and &quot;BUY&quot; in decisio=
n:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 decision =3D &quot;CAUTION&quot;</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 reason =3D &quot;Rotation Wea=
k&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=
=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 analys=
is_db[t] =3D {&quot;Decision&quot;: decision, &quot;Reason&quot;: reason, &=
quot;Price&quot;: dc[&#39;Close&#39;], &quot;Stop&quot;: smart_stop_val, &q=
uot;StopPct&quot;: stop_pct, &quot;RRG&quot;: rrg_phase, &quot;W_SMA8_Pass&=
quot;: (wc[&#39;Close&#39;]&gt;wc[&#39;SMA8&#39;]), &quot;W_Pulse&quot;: w_=
pulse, &quot;W_Score&quot;: w_score, &quot;D_Score&quot;: d_chk, &quot;D_Ch=
k_Price&quot;: (dc[&#39;Close&#39;] &gt; df[&#39;SMA18&#39;].iloc[-1]), &qu=
ot;W_Cloud&quot;: (wc[&#39;Close&#39;]&gt;wc[&#39;Cloud_Top&#39;]), &quot;A=
D_Pass&quot;: ad_score_ok, &quot;Vol_Msg&quot;: vol_msg, &quot;RSI_Msg&quot=
;: rsi_msg, &quot;Inst_Act&quot;: final_inst_msg}</div><div dir=3D"auto"><b=
r></div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 for t in scan_list:</=
div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 cat_name =
=3D tc.DATA_MAP.get(t, [&quot;OTHER&quot;])[0]</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if &quot;99. DATA&quot; in cat_name:=
 continue</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
if t not in analysis_db: continue</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 is_scanner =3D t in tc.DATA_MAP and (tc.DATA_MAP[t=
][0] !=3D &quot;BENCH&quot; or t in [&quot;DIA&quot;, &quot;QQQ&quot;, &quo=
t;IWM&quot;, &quot;IWC&quot;, &quot;<a href=3D"http://HXT.TO">HXT.TO</a>&qu=
ot;])</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if n=
ot is_scanner: continue</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 db =3D analysis_db[t]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 final_decision =3D db[&#39;Decision&#39;]; final_r=
eason =3D db[&#39;Reason&#39;]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 if cat_name in tc.SECTOR_PARENTS:</div><div dir=3D"aut=
o">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 parent =3D tc.SE=
CTOR_PARENTS[cat_name]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 if parent in analysis_db and &quot;AVOID&quot; =
in analysis_db[parent][&#39;Decision&#39;]:</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if t !=3D pa=
rent: final_decision =3D &quot;AVOID&quot;; final_reason =3D &quot;Sector L=
ock&quot;</div><div dir=3D"auto"><br></div><div dir=3D"auto">=C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 # Re-Check Logic for Blue Spike using clean dat=
a</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 # Now we=
 have HTML in RSI_Msg, so we can&#39;t parse easily. Use decision logic.</d=
iv><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 is_blue_spik=
e =3D (&quot;#00BFFF&quot; in db[&#39;RSI_Msg&#39;]) and (&quot;SPIKE&quot;=
 in db[&#39;Vol_Msg&#39;])</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 final_risk =3D risk_per_trade / 3 if &quot;SCOUT&quot; in fin=
al_decision else risk_per_trade</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 if is_blue_spike: final_risk =3D risk_per_trade</div>=
<div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div=
 dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if &quot;AVOID&quot=
; in final_decision and not is_blue_spike: disp_stop =3D &quot;&quot;; disp=
_shares =3D &quot;&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 =C2=A0 else:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 shares =3D int(final_risk / (db[&#39;Price&#39;] -=
 db[&#39;Stop&#39;])) if (db[&#39;Price&#39;] - db[&#39;Stop&#39;]) &gt; 0 =
else 0</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 disp_stop =3D f&quot;${db[&#39;Stop&#39;]:.2f} (-{db[&#39;StopPc=
t&#39;]:.1f}%)&quot;; disp_shares =3D f&quot;{shares} shares&quot;</div><di=
v dir=3D"auto"><br></div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 row =3D {</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 &quot;Sector&quot;: cat_name, &quot;Ticker&quot;: =
t, &quot;Rank&quot;: (0 if &quot;00.&quot; in cat_name else 1), &quot;Rotat=
ion&quot;: db[&#39;RRG&#39;],</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 &quot;Weekly&lt;br&gt;SMA8&quot;: &quot;=
PASS&quot; if db[&#39;W_SMA8_Pass&#39;] else &quot;FAIL&quot;, &quot;Weekly=
&lt;br&gt;Impulse&quot;: db[&#39;W_Pulse&#39;],=C2=A0</div><div dir=3D"auto=
">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 &quot;Weekly&lt;b=
r&gt;Score&quot;: db[&#39;W_Score&#39;], &quot;Daily&lt;br&gt;Score&quot;: =
db[&#39;D_Score&#39;],</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 &quot;Structure&quot;: &quot;ABOVE 18&quot; if =
db[&#39;D_Chk_Price&#39;] else &quot;BELOW 18&quot;,</div><div dir=3D"auto"=
>=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 &quot;Ichimoku&lt;=
br&gt;Cloud&quot;: &quot;PASS&quot; if db[&#39;W_Cloud&#39;] else &quot;FAI=
L&quot;, &quot;A/D Breadth&quot;: &quot;STRONG&quot; if db[&#39;AD_Pass&#39=
;] else &quot;WEAK&quot;,</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 &quot;Volume&quot;: db[&#39;Vol_Msg&#39;], =
&quot;Dual RSI&quot;: db[&#39;RSI_Msg&#39;], &quot;Institutional&lt;br&gt;A=
ctivity&quot;: db[&#39;Inst_Act&#39;],</div><div dir=3D"auto">=C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 &quot;Action&quot;: final_decisi=
on, &quot;Reasoning&quot;: final_reason, &quot;Stop Price&quot;: disp_stop,=
 &quot;Position Size&quot;: disp_shares</div><div dir=3D"auto">=C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 }</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 results.append(row)</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if t =3D=3D &quot;<a href=3D"http://HXT.=
TO">HXT.TO</a>&quot;: row_cad =3D row.copy(); row_cad[&quot;Sector&quot;] =
=3D &quot;15. CANADA (HXT)&quot;; row_cad[&quot;Rank&quot;] =3D 0; results.=
append(row_cad)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 if t in tc.SECTOR_ETFS: row_sec =3D row.copy(); row_sec[&quot;Sector=
&quot;] =3D &quot;02. SECTORS (SUMMARY)&quot;; row_sec[&quot;Rank&quot;] =
=3D 0; results.append(row_sec)</div><div dir=3D"auto"><br></div><div dir=3D=
"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if results:</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 df_final =3D pd.DataFrame(results).s=
ort_values([&quot;Sector&quot;, &quot;Rank&quot;, &quot;Ticker&quot;], asce=
nding=3D[True, True, True])</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 df_final[&quot;Sector&quot;] =3D df_final[&quot;Sector&qu=
ot;].apply(lambda x: x.split(&quot;. &quot;, 1)[1].replace(&quot;(SUMMARY)&=
quot;, &quot;&quot;).strip() if &quot;. &quot; in x else x)</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 cols =3D [&quot;Sector&=
quot;, &quot;Ticker&quot;, &quot;Rotation&quot;, &quot;Weekly&lt;br&gt;SMA8=
&quot;, &quot;Weekly&lt;br&gt;Impulse&quot;, &quot;Weekly&lt;br&gt;Score&qu=
ot;, &quot;Daily&lt;br&gt;Score&quot;, &quot;Structure&quot;, &quot;Ichimok=
u&lt;br&gt;Cloud&quot;, &quot;A/D Breadth&quot;, &quot;Volume&quot;, &quot;=
Dual RSI&quot;, &quot;Institutional&lt;br&gt;Activity&quot;, &quot;Action&q=
uot;, &quot;Reasoning&quot;, &quot;Stop Price&quot;, &quot;Position Size&qu=
ot;]</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 st.ma=
rkdown(generate_scanner_html(df_final[cols]), unsafe_allow_html=3DTrue)</di=
v><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 else:</div><div dir=3D"auto=
">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 st.warning(&quot;Scanner return=
ed no results.&quot;)</div><div dir=3D"auto"><br></div><div dir=3D"auto">=
=C2=A0 =C2=A0 if mode =3D=3D &quot;Sector Rotation&quot;:</div><div dir=3D"=
auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 st.subheader(&quot;=F0=9F=94=84 Relative =
Rotation Graphs (RRG)&quot;)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =
=C2=A0 is_dark =3D st.toggle(&quot;=F0=9F=8C=99 Dark Mode&quot;, value=3DTr=
ue)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 rrg_mode =3D st.radi=
o(&quot;View:&quot;, [&quot;Indices&quot;, &quot;Sectors&quot;, &quot;Drill=
-Down&quot;, &quot;Themes&quot;], horizontal=3DTrue, key=3D&quot;rrg_nav&qu=
ot;)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 if rrg_mode =3D=3D &quot;Indices&quot=
;:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 c1, c2 =
=3D st.columns([1,3])</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 with c1: bench_sel =3D st.selectbox(&quot;Benchmark&quot;, [&=
quot;SPY&quot;, &quot;IEF&quot;], key=3D&quot;bench_idx&quot;)</div><div di=
r=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 tgt =3D &quot;IEF&quot=
; if bench_sel =3D=3D &quot;IEF&quot; else &quot;SPY&quot;</div><div dir=3D=
"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 idx_list =3D list(tc.RRG_I=
NDICES.keys())</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 if tgt =3D=3D &quot;IEF&quot;: idx_list.append(&quot;SPY&quot;)</div=
><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 elif &quot;SPY=
&quot; in idx_list: idx_list.remove(&quot;SPY&quot;)</div><div dir=3D"auto"=
>=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if st.button(&quot;Run Indices&quot;=
):</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 wide_df =3D prepare_rrg_inputs(master_data, idx_list, tgt)</div><div=
 dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 r, m =
=3D calculate_rrg_math(wide_df, tgt)</div><div dir=3D"auto">=C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 st.session_state[&#39;fig_idx&#39=
;] =3D plot_rrg_chart(r, m, tc.RRG_INDICES, f&quot;Indices vs {tgt}&quot;, =
is_dark)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 i=
f &#39;fig_idx&#39; in st.session_state: st.plotly_chart(st.session_state[&=
#39;fig_idx&#39;], use_container_width=3DTrue)</div><div dir=3D"auto"><br><=
/div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 elif rrg_mode =3D=3D &qu=
ot;Sectors&quot;:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 if st.button(&quot;Run Sectors&quot;):</div><div dir=3D"auto">=C2=
=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 wide_df =3D prepare_rr=
g_inputs(master_data, list(tc.RRG_SECTORS.keys()), &quot;SPY&quot;)</div><d=
iv dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 r, =
m =3D calculate_rrg_math(wide_df, &quot;SPY&quot;)</div><div dir=3D"auto">=
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 st.session_state[&#=
39;fig_sec&#39;] =3D plot_rrg_chart(r, m, tc.RRG_SECTORS, &quot;Sectors vs =
SPY&quot;, is_dark)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 if &#39;fig_sec&#39; in st.session_state: st.plotly_chart(st.ses=
sion_state[&#39;fig_sec&#39;], use_container_width=3DTrue)</div><div dir=3D=
"auto"><br></div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 elif rrg_mod=
e =3D=3D &quot;Drill-Down&quot;:</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 c1, c2 =3D st.columns([1,3])</div><div dir=3D"auto=
">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 with c1:</div><div dir=3D"auto"=
>=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 def fmt(x): return=
 f&quot;{x} - {tc.RRG_SECTORS[x]}&quot; if x in tc.RRG_SECTORS else x</div>=
<div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 o=
pts =3D list(tc.RRG_SECTORS.keys()) + [&quot;Canada (TSX)&quot;]</div><div =
dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 sec_ke=
y =3D st.selectbox(&quot;Select Sector&quot;, opts, format_func=3Dfmt, key=
=3D&quot;dd_sel&quot;)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 if sec_key =3D=3D &quot;Canada (TSX)&quot;: bench_dd =3D &quo=
t;<a href=3D"http://HXT.TO">HXT.TO</a>&quot;; name_dd =3D &quot;Canadian Ti=
tans&quot;</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
 else: bench_dd =3D sec_key; name_dd =3D tc.RRG_SECTORS[sec_key]</div><div =
dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=C2=A0</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if st.button(f&quot;Run=
 {name_dd}&quot;):</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 =C2=A0 comp_list =3D list(tc.RRG_INDUSTRY_MAP.get(sec_key=
, {}).keys())</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 =C2=A0 =C2=A0 wide_df =3D prepare_rrg_inputs(master_data, comp_list, be=
nch_dd)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 =C2=A0 r, m =3D calculate_rrg_math(wide_df, bench_dd)</div><div dir=
=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 all_label=
s =3D {**tc.RRG_INDUSTRY_MAP.get(sec_key, {}), **tc.RRG_SECTORS}</div><div =
dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 st.ses=
sion_state[&#39;fig_dd&#39;] =3D plot_rrg_chart(r, m, all_labels, f&quot;{n=
ame_dd} vs {bench_dd}&quot;, is_dark)</div><div dir=3D"auto">=C2=A0 =C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 if &#39;fig_dd&#39; in st.session_state: st.plo=
tly_chart(st.session_state[&#39;fig_dd&#39;], use_container_width=3DTrue)</=
div><div dir=3D"auto"><br></div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=
=A0 elif rrg_mode =3D=3D &quot;Themes&quot;:</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if st.button(&quot;Run Themes&quot;):</d=
iv><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=
=A0 wide_df =3D prepare_rrg_inputs(master_data, list(tc.RRG_THEMES.keys()),=
 &quot;SPY&quot;)</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0=
 =C2=A0 =C2=A0 =C2=A0 r, m =3D calculate_rrg_math(wide_df, &quot;SPY&quot;)=
</div><div dir=3D"auto">=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 =
=C2=A0 st.session_state[&#39;fig_thm&#39;] =3D plot_rrg_chart(r, m, tc.RRG_=
THEMES, &quot;Themes vs SPY&quot;, is_dark)</div><div dir=3D"auto">=C2=A0 =
=C2=A0 =C2=A0 =C2=A0 =C2=A0 =C2=A0 if &#39;fig_thm&#39; in st.session_state=
: st.plotly_chart(st.session_state[&#39;fig_thm&#39;], use_container_width=
=3DTrue)</div><div dir=3D"auto"><br></div></div>
