import streamlit as st
import pandas as pd
import numpy as np

# --- 1. SAFE IMPORTS (Prevents immediate crash) ---
try:
    import plotly.graph_objects as go
    import yfinance as yf
except ImportError:
    st.error("⚠️ Libraries missing. Please install: pip install plotly yfinance")
    st.stop()

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Sector RRG")
st.title("🔄 Sector & Industry Rotation (RRG)")
st.caption("Institutional Money Flow Tracker | v57.5 Final Standalone")

# --- DATA UNIVERSE (Updated from your File) ---
BENCHMARK_US = "SPY"
BENCHMARK_CA = "HXT.TO"

SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care", 
    "XLY": "Cons. Discret", "XLP": "Cons. Staples", "XLI": "Industrials", 
    "XLC": "Comm. Services", "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate"
}

INDICES = {
    "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWM": "Russell 2000", 
    "IWC": "Micro-Cap", "RSP": "S&P Equal Wgt", "^VIX": "Volatility (VIX)",
    "HXT.TO": "TSX 60 (Canada)", "EFA": "Foreign Dev", "EEM": "Emerging Mkts"
}

THEMES = {
    "BOTZ": "Robotics/AI", "AIQ": "Artificial Intel", "SMH": "Semiconductors", 
    "IGV": "Software", "CIBR": "CyberSec", "ARKG": "Genomics",
    "ICLN": "Clean Energy", "TAN": "Solar", "URA": "Uranium", "PAVE": "Infrastructure",
    "GLD": "Gold", "SLV": "Silver", "GDX": "Gold Miners", "COPX": "Copper",
    "MOO": "Agribusiness", "SLX": "Steel", "REMX": "Rare Earths"
}

INDUSTRY_MAP = {
    "XLK": {
        "SMH": "Semis", "NVDA": "Nvidia", "IGV": "Software", "MSFT": "Microsoft", 
        "CIBR": "CyberSec", "AAPL": "Apple", "SMCI": "Servers (AI)", 
        "DELL": "Dell", "ANET": "Networking", "WDC": "Storage"
    },
    "XLF": {"KBE": "Banks", "KRE": "Reg. Banks", "IAI": "Brokers", "IAK": "Insurance", "XP": "Fintech"},
    "XLE": {"XOP": "Exploration", "OIH": "Oil Svcs", "CRAK": "Refiners", "XOM": "Exxon", "CVX": "Chevron"},
    "XLV": {"IBB": "Biotech", "IHI": "Med Devices", "PPH": "Pharma", "UNH": "UnitedHealth"},
    "XLY": {"XRT": "Retail", "ITB": "Homebuild", "PEJ": "Leisure", "AMZN": "Amazon", "TSLA": "Tesla"},
    "XLP": {"PBJ": "Food/Bev", "KXI": "Global Stapl", "COST": "Costco", "PG": "Procter", "WMT": "Walmart"},
    "XLI": {"ITA": "Aerospace", "IYT": "Transport", "JETS": "Airlines", "PAVE": "Infrastruct", "CAT": "Caterpillar"},
    "XLC": {"SOCL": "Social", "PBS": "Media", "GOOGL": "Google", "META": "Meta", "NFLX": "Netflix"},
    "XLB": {
        "GDX": "Gold Miners", "SIL": "Silver", "LIT": "Lithium", 
        "REMX": "Rare Earth", "COPX": "Copper", "MOO": "Agricul", 
        "SLX": "Steel", "AA": "Alcoa", "DD": "DuPont"
    },
    "XLU": {"IDU": "US Util", "VPU": "Vanguard Util", "NEE": "NextEra", "DUK": "Duke Energy"},
    "XLRE": {"REZ": "Resid. RE", "BBRE": "BetaBuilders", "PLD": "Prologis", "AMT": "Am. Tower"},
    "Canada (TSX)": {
        "RY.TO": "Royal Bank", "BN.TO": "Brookfield", "CNQ.TO": "Cdn Natural", 
        "CP.TO": "CP Rail", "WSP.TO": "WSP Global", "SHOP.TO": "Shopify", 
        "CSU.TO": "Constell", "NTR.TO": "Nutrien", "TECK-B.TO": "Teck Res"
    }
}

# --- CALCULATION ENGINE ---
def calculate_rrg(price_data, benchmark_col, window_rs=14, window_mom=5, smooth_factor=3):
    if benchmark_col not in price_data.columns:
        return pd.DataFrame(), pd.DataFrame()

    df_ratio = pd.DataFrame()
    df_mom = pd.DataFrame()
    
    for col in price_data.columns:
        if col != benchmark_col:
            try:
                rs = price_data[col] / price_data[benchmark_col]
                mean = rs.rolling(window_rs).mean()
                std = rs.rolling(window_rs).std()
                ratio = 100 + ((rs - mean) / std) * 1.5 
                df_ratio[col] = ratio
            except: continue

    for col in df_ratio.columns:
        try:
            mom = 100 + (df_ratio[col] - df_ratio[col].rolling(window_mom).mean()) * 2
            df_mom[col] = mom
        except: continue
        
    return df_ratio.rolling(smooth_factor).mean().dropna(), df_mom.rolling(smooth_factor).mean().dropna()

# --- PLOTTING ENGINE ---
def plot_rrg_chart(ratios, momentums, labels_map, title, is_dark):
    fig = go.Figure()
    
    # Theme Colors
    if is_dark:
        bg_col, text_col = "black", "white"
        c_lead, c_weak, c_lag, c_imp = "#00FF00", "#FFFF00", "#FF4444", "#00BFFF"
        template = "plotly_dark"
    else:
        bg_col, text_col = "white", "black"
        c_lead, c_weak, c_lag, c_imp = "#008000", "#FF8C00", "#CC0000", "#0000FF"
        template = "plotly_white"

    has_data = False
    
    for ticker in labels_map.keys():
        if ticker not in ratios.columns: continue
        
        xt = ratios[ticker].tail(5)
        yt = momentums[ticker].tail(5)
        if len(xt) < 5: continue
        
        has_data = True
        cx, cy = xt.iloc[-1], yt.iloc[-1]
        
        # Color Phase
        if cx > 100 and cy > 100: color = c_lead
        elif cx > 100 and cy < 100: color = c_weak
        elif cx < 100 and cy < 100: color = c_lag
        else: color = c_imp

        # Trail
        fig.add_trace(go.Scatter(
            x=xt, y=yt, mode='lines',
            line=dict(color=color, width=2, shape='spline'),
            opacity=0.6, showlegend=False, hoverinfo='skip'
        ))
        
        # Head
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode='markers+text',
            marker=dict(color=color, size=12, line=dict(color=text_col, width=1)),
            text=[ticker], textposition="top center",
            textfont=dict(color=text_col),
            hovertemplate=f"<b>{labels_map.get(ticker, ticker)}</b><br>T: %{{x:.2f}}<br>M: %{{y:.2f}}"
        ))

    if not has_data: return None

    # Static Quadrants (Fixed View 96-104)
    op = 0.1 if is_dark else 0.05
    fig.add_hline(y=100, line_dash="dot", line_color="gray")
    fig.add_vline(x=100, line_dash="dot", line_color="gray")
    fig.add_shape(type="rect", x0=100, y0=100, x1=200, y1=200, fillcolor=f"rgba(0,255,0,{op})", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=100, y0=0, x1=200, y1=100, fillcolor=f"rgba(255,255,0,{op})", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, fillcolor=f"rgba(255,0,0,{op})", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=100, x1=100, y1=200, fillcolor=f"rgba(0,0,255,{op})", layer="below", line_width=0)

    fig.update_layout(
        title=title, template=template, height=650, showlegend=False,
        xaxis=dict(range=[96, 104], showgrid=False, title="RS-Ratio (Trend)"),
        yaxis=dict(range=[96, 104], showgrid=False, title="RS-Momentum (Velocity)")
    )
    
    # Labels
    fig.add_annotation(x=104, y=104, text="LEADING", showarrow=False, font=dict(size=16, color=c_lead), xanchor="right", yanchor="top")
    fig.add_annotation(x=104, y=96, text="WEAKENING", showarrow=False, font=dict(size=16, color=c_weak), xanchor="right", yanchor="bottom")
    fig.add_annotation(x=96, y=96, text="LAGGING", showarrow=False, font=dict(size=16, color=c_lag), xanchor="left", yanchor="bottom")
    fig.add_annotation(x=96, y=104, text="IMPROVING", showarrow=False, font=dict(size=16, color=c_imp), xanchor="left", yanchor="top")
    
    return fig

# --- APP LAYOUT ---
is_dark = st.toggle("🌙 Dark Mode", value=True)

tab1, tab2, tab3, tab4 = st.tabs(["📈 Indices", "🌍 Sectors", "🔬 Drill-Down", "💡 Themes"])

def run_rrg_scan(tickers, benchmark, title, session_key):
    if st.button(f"Run {title}", key=f"btn_{session_key}"):
        with st.spinner("Fetching Data..."):
            fetch_list = list(set(tickers + [benchmark]))
            # 6mo is enough for 14-wk MA + 5-wk Momentum
            data = yf.download(fetch_list, period="6mo", interval="1wk", progress=False)['Close']
            
            if not data.empty and benchmark in data.columns:
                r, m = calculate_rrg(data, benchmark)
                labels = {t:t for t in tickers} # Fallback labels
                # Try to use pretty names if available in our dictionaries
                for t in tickers:
                    if t in SECTORS: labels[t] = SECTORS[t]
                    elif t in INDICES: labels[t] = INDICES[t]
                    elif t in THEMES: labels[t] = THEMES[t]
                    else:
                        # Check deep industry map
                        for k, v in INDUSTRY_MAP.items():
                            if t in v: labels[t] = v[t]

                fig = plot_rrg_chart(r, m, labels, f"{title} vs {benchmark}", is_dark)
                if fig:
                    st.session_state[f"fig_{session_key}"] = fig
                else:
                    st.warning("Insufficient history for these tickers.")
            else:
                st.error("Download failed or Benchmark missing.")

    if f"fig_{session_key}" in st.session_state:
        st.plotly_chart(st.session_state[f"fig_{session_key}"], use_container_width=True)


# TAB 1: INDICES
with tab1:
    c1, c2 = st.columns([1,3])
    with c1:
        bench_sel = st.selectbox("Benchmark", ["SPY", "IEF"])
    
    tgt = "IEF" if bench_sel == "IEF" else "SPY"
    idx_list = list(INDICES.keys())
    # If Benchmark is IEF, we want to see SPY on the chart.
    if tgt == "IEF": idx_list.append("SPY")
    # If Benchmark is SPY, remove SPY from the list to avoid flat line.
    elif "SPY" in idx_list: idx_list.remove("SPY")
    
    run_rrg_scan(idx_list, tgt, "Indices", "idx")

# TAB 2: SECTORS
with tab2:
    run_rrg_scan(list(SECTORS.keys()), BENCHMARK_US, "Sectors", "sec")

# TAB 3: DRILL DOWN
with tab3:
    c1, c2 = st.columns([1,3])
    with c1:
        def fmt(x): return f"{x} - {SECTORS[x]}" if x in SECTORS else x
        opts = list(SECTORS.keys()) + ["Canada (TSX)"]
        sec_key = st.selectbox("Select Sector", opts, format_func=fmt)
    
    if sec_key == "Canada (TSX)":
        bench_dd = BENCHMARK_CA
        name_dd = "Canadian Titans"
    else:
        bench_dd = sec_key
        name_dd = SECTORS[sec_key]
        
    comp_list = list(INDUSTRY_MAP.get(sec_key, {}).keys())
    if comp_list:
        run_rrg_scan(comp_list, bench_dd, f"{name_dd} Industries", "dd")
    else:
        st.info("No components defined for this sector.")

# TAB 4: THEMES
with tab4:
    run_rrg_scan(list(THEMES.keys()), BENCHMARK_US, "Themes", "thm")
