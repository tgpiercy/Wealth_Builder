import streamlit as st
import pandas as pd
import numpy as np

# --- 1. SAFE IMPORT (Prevents Crash if Plotly is missing) ---
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("⚠️ CRITICAL ERROR: 'plotly' library is missing.")
    st.info("If you are on mobile/local: Run 'pip install plotly'")
    st.stop()

try:
    import yfinance as yf
except ImportError:
    st.error("⚠️ CRITICAL ERROR: 'yfinance' library is missing.")
    st.stop()

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Sector RRG (Safe Mode)")
st.title("🔄 Sector RRG (Safe Mode)")
st.caption("Robust Institutional Tracker | Debugging Enabled")

# --- DATA UNIVERSE ---
BENCHMARK_US = "SPY"
BENCHMARK_CA = "HXT.TO"

SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care", 
    "XLY": "Cons. Discret", "XLP": "Cons. Staples", "XLI": "Industrials", 
    "XLC": "Comm. Services", "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate"
}

INDICES = {
    "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWM": "Russell 2000", 
    "IWC": "Micro-Cap", "RSP": "S&P Equal Wgt", "^VIX": "Volatility",
    "HXT.TO": "TSX 60 (Canada)", "EFA": "Foreign Dev", "EEM": "Emerging Mkts"
}

THEMES = {
    "BOTZ": "Robotics/AI", "AIQ": "Artificial Intel", "SMH": "Semiconductors", 
    "IGV": "Software", "CIBR": "CyberSec", "ARKG": "Genomics",
    "ICLN": "Clean Energy", "TAN": "Solar", "URA": "Uranium", "PAVE": "Infrastructure",
    "GLD": "Gold", "SLV": "Silver", "GDX": "Gold Miners", "COPX": "Copper",
    "MOO": "Agricul", "SLX": "Steel"
}

INDUSTRY_MAP = {
    "XLK": {"SMH": "Semis", "NVDA": "Nvidia", "IGV": "Software", "MSFT": "Microsoft", "CIBR": "CyberSec", "AAPL": "Apple", "SMCI": "Servers", "ANET": "Networks"},
    "XLF": {"KBE": "Banks", "KRE": "Reg. Banks", "IAI": "Brokers", "IAK": "Insurance", "XP": "Fintech"},
    "XLE": {"XOP": "Exploration", "OIH": "Oil Svcs", "CRAK": "Refiners", "XOM": "Exxon", "CVX": "Chevron"},
    "XLV": {"IBB": "Biotech", "IHI": "Med Devices", "PPH": "Pharma", "UNH": "UnitedHealth"},
    "XLY": {"XRT": "Retail", "ITB": "Homebuild", "PEJ": "Leisure", "AMZN": "Amazon", "TSLA": "Tesla"},
    "XLP": {"PBJ": "Food/Bev", "KXI": "Global Stapl", "COST": "Costco", "PG": "Procter", "WMT": "Walmart"},
    "XLI": {"ITA": "Aerospace", "IYT": "Transport", "JETS": "Airlines", "PAVE": "Infrastruct", "CAT": "Caterpillar"},
    "XLC": {"SOCL": "Social", "PBS": "Media", "GOOGL": "Google", "META": "Meta", "NFLX": "Netflix"},
    "XLB": {"GDX": "Gold Miners", "SIL": "Silver", "LIT": "Lithium", "REMX": "Rare Earth", "COPX": "Copper", "MOO": "Ag", "SLX": "Steel"},
    "XLU": {"IDU": "US Util", "VPU": "Vanguard Util", "NEE": "NextEra", "DUK": "Duke Energy"},
    "XLRE": {"REZ": "Resid. RE", "BBRE": "BetaBuilders", "PLD": "Prologis", "AMT": "Am. Tower"},
    "Canada (TSX)": {"RY.TO": "Royal Bank", "BN.TO": "Brookfield", "CNQ.TO": "Cdn Natural", "CP.TO": "CP Rail", "WSP.TO": "WSP Global", "SHOP.TO": "Shopify", "CSU.TO": "Constell", "NTR.TO": "Nutrien", "TECK-B.TO": "Teck Res"}
}

# --- CALCULATION ENGINE (Robust) ---
def calculate_rrg(price_data, benchmark_col, window_rs=14, window_mom=5, smooth_factor=3):
    if benchmark_col not in price_data.columns:
        st.error(f"❌ Benchmark '{benchmark_col}' failed to download. Check ticker symbol.")
        return pd.DataFrame(), pd.DataFrame()

    df_ratio = pd.DataFrame()
    df_mom = pd.DataFrame()
    
    # Calculate RS-Ratio
    for col in price_data.columns:
        if col != benchmark_col:
            try:
                # Handle potential division by zero or NaN
                rs = price_data[col] / price_data[benchmark_col]
                mean = rs.rolling(window_rs).mean()
                std = rs.rolling(window_rs).std()
                
                # Z-Score Proxy
                ratio = 100 + ((rs - mean) / std) * 1.5 
                df_ratio[col] = ratio
            except:
                continue # Skip bad columns

    # Calculate RS-Momentum
    for col in df_ratio.columns:
        try:
            mom = 100 + (df_ratio[col] - df_ratio[col].rolling(window_mom).mean()) * 2
            df_mom[col] = mom
        except:
            continue

    # Smoothing
    df_ratio = df_ratio.rolling(window=smooth_factor).mean()
    df_mom = df_mom.rolling(window=smooth_factor).mean()
        
    return df_ratio.dropna(), df_mom.dropna()

# --- PLOTTING ENGINE (Robust) ---
def plot_rrg_chart(ratios, momentums, labels_map, title, is_dark=True):
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
    all_vals = []

    for ticker in labels_map.keys():
        if ticker not in ratios.columns: continue
        
        x_trail = ratios[ticker].tail(5)
        y_trail = momentums[ticker].tail(5)
        if len(x_trail) < 5: continue
        
        has_data = True
        curr_x = x_trail.iloc[-1]
        curr_y = y_trail.iloc[-1]
        all_vals.extend([curr_x, curr_y])

        # Color Logic
        if curr_x > 100 and curr_y > 100: color = c_lead
        elif curr_x > 100 and curr_y < 100: color = c_weak
        elif curr_x < 100 and curr_y < 100: color = c_lag
        else: color = c_imp

        # Trail
        fig.add_trace(go.Scatter(
            x=x_trail, y=y_trail, mode='lines',
            line=dict(color=color, width=2, shape='spline'),
            opacity=0.6, showlegend=False, hoverinfo='skip'
        ))
        
        # Dot
        fig.add_trace(go.Scatter(
            x=[curr_x], y=[curr_y], mode='markers+text',
            marker=dict(color=color, size=12, line=dict(color=text_col, width=1)),
            text=[ticker], textposition="top center",
            textfont=dict(color=text_col),
            hovertemplate=f"<b>{labels_map.get(ticker, ticker)}</b><br>Trend: %{{x:.2f}}<br>Mom: %{{y:.2f}}"
        ))

    if not has_data:
        return None

    # Static Quadrants (Fixed Range for Stability)
    # We fix the view to 96-104 to prevent crazy zooming on outliers
    fig.add_hline(y=100, line_dash="dot", line_color="gray")
    fig.add_vline(x=100, line_dash="dot", line_color="gray")
    
    # Backgrounds
    op = 0.1 if is_dark else 0.05
    fig.add_shape(type="rect", x0=100, y0=100, x1=200, y1=200, fillcolor=f"rgba(0,255,0,{op})", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=100, y0=0, x1=200, y1=100, fillcolor=f"rgba(255,255,0,{op})", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, fillcolor=f"rgba(255,0,0,{op})", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=100, x1=100, y1=200, fillcolor=f"rgba(0,0,255,{op})", layer="below", line_width=0)

    fig.update_layout(
        title=title, xaxis_title="RS-Ratio (Trend)", yaxis_title="RS-Momentum (Velocity)",
        template=template, height=600, showlegend=False,
        xaxis=dict(range=[96, 104], showgrid=False),
        yaxis=dict(range=[96, 104], showgrid=False)
    )
    return fig

# --- APP EXECUTION ---
is_dark = st.toggle("🌙 Dark Mode", value=True)

tab1, tab2, tab3, tab4 = st.tabs(["📈 Indices", "🌍 Sectors", "🔬 Drill-Down", "💡 Themes"])

def run_scan(tickers, benchmark, title, key_suffix):
    if st.button(f"Run {title}", key=f"btn_{key_suffix}"):
        with st.spinner("Fetching Data..."):
            try:
                # 1. Deduplicate
                fetch_list = list(set(tickers + [benchmark]))
                
                # 2. Fetch Data (Reduced period to save memory)
                data = yf.download(fetch_list, period="6mo", interval="1wk", progress=False)['Close']
                
                # 3. Validation
                if data.empty:
                    st.error("No data received. YFinance might be blocked.")
                    return
                
                if benchmark not in data.columns:
                    st.error(f"Benchmark {benchmark} missing from data.")
                    return

                # 4. Calculation
                rat, mom = calculate_rrg(data, benchmark)
                
                # 5. Plotting
                labels = {t: t for t in tickers} # Simplified labels for robustness
                fig = plot_rrg_chart(rat, mom, labels, f"{title} vs {benchmark}", is_dark)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough history to generate plot.")

            except Exception as e:
                st.error(f"Crash Prevented: {e}")

# TAB 1: INDICES
with tab1:
    bench = st.selectbox("Benchmark", ["SPY", "IEF"])
    tgt = "IEF" if bench == "IEF" else "SPY"
    # Logic: If IEF is bench, add SPY to list. If SPY is bench, remove it from list.
    idx_list = list(INDICES.keys())
    if tgt == "IEF": idx_list.append("SPY")
    elif "SPY" in idx_list: idx_list.remove("SPY")
        
    run_scan(idx_list, tgt, "Indices", "idx")

# TAB 2: SECTORS
with tab2:
    run_scan(list(SECTORS.keys()), BENCHMARK_US, "Sectors", "sec")

# TAB 3: DRILL DOWN
with tab3:
    sec = st.selectbox("Sector", list(INDUSTRY_MAP.keys()))
    if sec == "Canada (TSX)":
        bench_dd = BENCHMARK_CA
    else:
        bench_dd = sec 
    
    comps = list(INDUSTRY_MAP[sec].keys())
    run_scan(comps, bench_dd, f"{sec} Components", "dd")

# TAB 4: THEMES
with tab4:
    run_scan(list(THEMES.keys()), BENCHMARK_US, "Themes", "thm")
