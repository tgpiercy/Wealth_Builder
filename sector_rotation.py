import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Sector RRG Analysis")
st.title("🔄 Sector & Industry Rotation (RRG)")
st.caption("Institutional Money Flow Tracker | Macro & Micro Analysis")

# --- THEME SELECTOR ---
is_dark = st.toggle("🌙 Dark Mode Chart", value=True)

if is_dark:
    TEMPLATE = "plotly_dark"
    C_TEXT = "white"
    C_LEADING = "#00FF00"   
    C_WEAKENING = "#FFFF00" 
    C_LAGGING = "#FF4444"   
    C_IMPROVING = "#00BFFF" 
else:
    TEMPLATE = "plotly_white"
    C_TEXT = "black"
    C_LEADING = "#008000"   
    C_WEAKENING = "#FF8C00" 
    C_LAGGING = "#CC0000"   
    C_IMPROVING = "#0000FF" 

# --- DATA UNIVERSE ---
BENCHMARK = "SPY"

# 1. Macro Sectors
SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care", 
    "XLY": "Cons. Discret", "XLP": "Cons. Staples", "XLI": "Industrials", 
    "XLC": "Comm. Services", "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate"
}

# 2. Micro Industries (Map Sector -> Representative ETFs/Stocks)
INDUSTRY_MAP = {
    "XLK": {"SMH": "Semis", "IGV": "Software", "CIBR": "CyberSec", "AAPL": "Apple", "MSFT": "Microsoft", "VGT": "Vanguard Tech"},
    "XLF": {"KBE": "Banks", "KRE": "Reg. Banks", "IAI": "Brokers", "IAK": "Insurance", "XP": "Fintech"},
    "XLE": {"XOP": "Exploration", "OIH": "Oil Svcs", "CRAK": "Refiners", "XOM": "Exxon", "CVX": "Chevron"},
    "XLV": {"IBB": "Biotech", "IHI": "Med Devices", "PPH": "Pharma", "XHE": "Equipment", "UNH": "UnitedHealth"},
    "XLY": {"XRT": "Retail", "ITB": "Homebuild", "PEJ": "Leisure", "AMZN": "Amazon", "TSLA": "Tesla"},
    "XLP": {"PBJ": "Food/Bev", "KXI": "Global Stapl", "COST": "Costco", "PG": "Procter", "WMT": "Walmart"},
    "XLI": {"ITA": "Aerospace", "IYT": "Transport", "JETS": "Airlines", "PAVE": "Infrastruct", "CAT": "Caterpillar"},
    "XLC": {"SOCL": "Social", "PBS": "Media", "GOOGL": "Google", "META": "Meta", "NFLX": "Netflix"},
    "XLB": {"GDX": "Gold Miners", "SIL": "Silver", "LIT": "Lithium", "REMX": "Rare Earth", "COPX": "Copper"},
    "XLU": {"IDU": "US Util", "VPU": "Vanguard Util", "NEE": "NextEra", "DUK": "Duke Energy"},
    "XLRE": {"REZ": "Resid. RE", "BBRE": "BetaBuilders", "PLD": "Prologis", "AMT": "Am. Tower"}
}

# --- CALCULATION ENGINE ---
def calculate_rrg(price_data, benchmark_col, window_rs=14, window_mom=5, smooth_factor=3):
    df_ratio = pd.DataFrame()
    df_mom = pd.DataFrame()
    
    for col in price_data.columns:
        if col != benchmark_col:
            rs = price_data[col] / price_data[benchmark_col]
            # RS-Ratio (Normalized Trend vs Benchmark)
            # Centered at 100
            ratio = 100 + ((rs - rs.rolling(window_rs).mean()) / rs.rolling(window_rs).std()) * 1.5 
            df_ratio[col] = ratio

    for col in df_ratio.columns:
        # RS-Momentum (Velocity of the Ratio)
        mom = 100 + (df_ratio[col] - df_ratio[col].rolling(window=window_mom).mean()) * 2
        df_mom[col] = mom
        
    # Smoothing
    df_ratio = df_ratio.rolling(window=smooth_factor).mean()
    df_mom = df_mom.rolling(window=smooth_factor).mean()
        
    return df_ratio.dropna(), df_mom.dropna()

# --- PLOTTING ENGINE ---
def plot_rrg_chart(ratios, momentums, labels_map, title, tail_len=5):
    all_x = []
    all_y = []
    
    fig = go.Figure()

    for ticker in labels_map.keys():
        if ticker not in ratios.columns: continue
        
        # Trail Data
        x_trail = ratios[ticker].tail(tail_len)
        y_trail = momentums[ticker].tail(tail_len)
        
        if len(x_trail) < tail_len: continue

        all_x.extend(x_trail.values)
        all_y.extend(y_trail.values)

        curr_x = x_trail.iloc[-1]
        curr_y = y_trail.iloc[-1]
        
        # Phase Logic
        color = "gray"
        if curr_x > 100 and curr_y > 100: color = C_LEADING
        elif curr_x > 100 and curr_y < 100: color = C_WEAKENING
        elif curr_x < 100 and curr_y < 100: color = C_LAGGING
        elif curr_x < 100 and curr_y > 100: color = C_IMPROVING

        # Draw Trail
        fig.add_trace(go.Scatter(
            x=x_trail, y=y_trail, mode='lines',
            line=dict(color=color, width=2, shape='spline'),
            opacity=0.6, showlegend=False, hoverinfo='skip'
        ))
        
        # Draw Head
        text_col = C_TEXT if not is_dark else "white"
        fig.add_trace(go.Scatter(
            x=[curr_x], y=[curr_y], mode='markers+text',
            marker=dict(color=color, size=12, line=dict(color=text_col, width=1)),
            text=[ticker], textposition="top center",
            textfont=dict(color=text_col),
            name=labels_map[ticker],
            hovertemplate=f"<b>{labels_map[ticker]}</b><br>Trend: %{{x:.2f}}<br>Mom: %{{y:.2f}}"
        ))

    # Dynamic Scaling
    if not all_x: return go.Figure() # Empty return if no data
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    pad = 1.5
    
    # Keep 100 centered
    x_min = min(x_min, 98); x_max = max(x_max, 102)
    y_min = min(y_min, 98); y_max = max(y_max, 102)

    # Background Quadrants
    bg_opacity = 0.1 if is_dark else 0.05
    fig.add_shape(type="rect", x0=100, y0=100, x1=x_max+pad, y1=y_max+pad, fillcolor=f"rgba(0, 255, 0, {bg_opacity})", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=100, y0=y_min-pad, x1=x_max+pad, y1=100, fillcolor=f"rgba(255, 255, 0, {bg_opacity})", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=x_min-pad, y0=y_min-pad, x1=100, y1=100, fillcolor=f"rgba(255, 0, 0, {bg_opacity})", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=x_min-pad, y0=100, x1=100, y1=y_max+pad, fillcolor=f"rgba(0, 0, 255, {bg_opacity})", layer="below", line_width=0)

    # Crosshair
    axis_col = "gray" if is_dark else "black"
    fig.add_hline(y=100, line_dash="dot", line_color=axis_col)
    fig.add_vline(x=100, line_dash="dot", line_color=axis_col)

    fig.update_layout(
        title=title, xaxis_title="RS-Ratio (Trend)", yaxis_title="RS-Momentum (Velocity)",
        width=1100, height=700, template=TEMPLATE,
        xaxis=dict(range=[x_min-pad, x_max+pad], showgrid=False, zeroline=False), 
        yaxis=dict(range=[y_min-pad, y_max+pad], showgrid=False, zeroline=False),
        showlegend=False
    )
    
    # Labels
    fig.add_annotation(x=x_max, y=y_max, text="LEADING", showarrow=False, font=dict(size=16, color=C_LEADING), xanchor="right", yanchor="top")
    fig.add_annotation(x=x_max, y=y_min, text="WEAKENING", showarrow=False, font=dict(size=16, color=C_WEAKENING), xanchor="right", yanchor="bottom")
    fig.add_annotation(x=x_min, y=y_min, text="LAGGING", showarrow=False, font=dict(size=16, color=C_LAGGING), xanchor="left", yanchor="bottom")
    fig.add_annotation(x=x_min, y=y_max, text="IMPROVING", showarrow=False, font=dict(size=16, color=C_IMPROVING), xanchor="left", yanchor="top")

    return fig

# --- APP LAYOUT ---

tab1, tab2 = st.tabs(["🌍 Global Sectors", "🔬 Industry Drill-Down"])

# 1. MACRO VIEW
with tab1:
    st.subheader(f"Macro View: Sectors vs {BENCHMARK}")
    
    # Check session state logic
    run_global = st.button("Run Global Scan", key="btn_global")
    
    if run_global:
        with st.spinner("Analyzing Global Flows..."):
            tickers = list(SECTORS.keys()) + [BENCHMARK]
            data = yf.download(tickers, period="1y", interval="1wk", progress=False)['Close']
            
            if not data.empty:
                rat, mom = calculate_rrg(data, BENCHMARK)
                fig_global = plot_rrg_chart(rat, mom, SECTORS, f"Sector Rotation vs {BENCHMARK}")
                
                # SAVE TO SESSION STATE
                st.session_state['fig_global'] = fig_global
            else:
                st.error("Data fetch failed.")

    # Render from State if available
    if 'fig_global' in st.session_state:
        st.plotly_chart(st.session_state['fig_global'], use_container_width=True)

# 2. MICRO VIEW
with tab2:
    st.subheader("Micro View: Industry vs Sector")
    
    sel_sector_key = st.selectbox("Select Sector to Drill Down:", list(SECTORS.keys()), 
                                  format_func=lambda x: f"{x} - {SECTORS[x]}")
    
    sel_benchmark = sel_sector_key 
    
    run_micro = st.button(f"Analyze {SECTORS[sel_sector_key]} Industries", key="btn_drill")
    
    if run_micro:
        with st.spinner(f"Fetching {SECTORS[sel_sector_key]} Components..."):
            ind_map = INDUSTRY_MAP.get(sel_sector_key, {})
            if not ind_map:
                st.warning("No sub-industries defined.")
                st.stop()
                
            tickers = list(ind_map.keys()) + [sel_benchmark]
            data = yf.download(tickers, period="1y", interval="1wk", progress=False)['Close']
            
            if not data.empty:
                rat, mom = calculate_rrg(data, sel_benchmark)
                fig_micro = plot_rrg_chart(rat, mom, ind_map, f"Industry Rotation vs {sel_benchmark} ({SECTORS[sel_sector_key]})")
                
                # Calc Table
                latest_rat = rat.iloc[-1].to_dict()
                latest_mom = mom.iloc[-1].to_dict()
                tbl = []
                for t, name in ind_map.items():
                    if t in latest_rat:
                        r = latest_rat[t]
                        m = latest_mom[t]
                        status = "LAGGING"
                        if r > 100 and m > 100: status = "LEADING"
                        elif r > 100 and m < 100: status = "WEAKENING"
                        elif r < 100 and m > 100: status = "IMPROVING"
                        tbl.append({"Ticker": t, "Name": name, "Status": status, "Trend": r, "Momentum": m})
                
                df_micro = pd.DataFrame(tbl)
                
                # SAVE TO SESSION STATE
                st.session_state['fig_micro'] = fig_micro
                st.session_state['df_micro'] = df_micro
                
            else:
                st.error(f"Could not fetch data for {sel_sector_key}.")

    # Render from State if available
    if 'fig_micro' in st.session_state:
        st.plotly_chart(st.session_state['fig_micro'], use_container_width=True)
        
        if 'df_micro' in st.session_state:
            st.divider()
            st.caption("Industry Data")
            st.dataframe(st.session_state['df_micro'].style.format({"Trend": "{:.2f}", "Momentum": "{:.2f}"}))
