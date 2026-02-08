import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Sector RRG Analysis")
st.title("🔄 Sector Rotation: Relative Rotation Graph (RRG)")
st.caption("Institutional Money Flow Tracker | Benchmark: SPY")

# --- THEME SELECTOR ---
# This allows switching between Dad's Light Mode and your Dark Mode
is_dark = st.toggle("🌙 Dark Mode Chart", value=True)

# Define Color Palette based on selection
if is_dark:
    TEMPLATE = "plotly_dark"
    C_TEXT = "white"
    C_LEADING = "#00FF00"   # Bright Green
    C_WEAKENING = "#FFFF00" # Bright Yellow
    C_LAGGING = "#FF4444"   # Bright Red
    C_IMPROVING = "#00BFFF" # Deep Sky Blue
else:
    TEMPLATE = "plotly_white"
    C_TEXT = "black"
    C_LEADING = "#008000"   # Darker Green
    C_WEAKENING = "#FF8C00" # Dark Orange (Readable on white)
    C_LAGGING = "#CC0000"   # Darker Red
    C_IMPROVING = "#0000FF" # Solid Blue (Readable on white)

# 1. Define Universe
SECTORS = {
    "XLK": "Technology", 
    "XLF": "Financials", 
    "XLE": "Energy", 
    "XLV": "Health Care", 
    "XLY": "Cons. Discret", 
    "XLP": "Cons. Staples",
    "XLI": "Industrials", 
    "XLC": "Comm. Services", 
    "XLU": "Utilities", 
    "XLB": "Materials", 
    "XLRE": "Real Estate"
}
BENCHMARK = "SPY"

# 2. RRG Calculation Engine
def calculate_rrg_components(price_data, benchmark_col, window_rs=14, window_mom=5, smooth_factor=3):
    """
    Calculates RRG coordinates:
    - RS-Ratio (X-Axis): Normalized Relative Strength (Trend)
    - RS-Momentum (Y-Axis): Rate of Change of the Ratio (Velocity)
    """
    df_ratio = pd.DataFrame()
    df_mom = pd.DataFrame()
    
    for col in price_data.columns:
        if col != benchmark_col:
            # 1. Relative Strength (RS)
            rs = price_data[col] / price_data[benchmark_col]
            
            # 2. RS-Ratio (Normalized Trend)
            ma_rs = rs.rolling(window=window_rs).mean()
            std_rs = rs.rolling(window=window_rs).std()
            
            # This formula centers the values around 100
            ratio = 100 + ((rs - ma_rs) / std_rs) * 1.5 
            df_ratio[col] = ratio

    # 3. RS-Momentum (Rate of Change of Ratio)
    for col in df_ratio.columns:
        mom = 100 + (df_ratio[col] - df_ratio[col].rolling(window=window_mom).mean()) * 2
        df_mom[col] = mom
        
    # 4. Final Smoothing
    df_ratio = df_ratio.rolling(window=smooth_factor).mean()
    df_mom = df_mom.rolling(window=smooth_factor).mean()
        
    return df_ratio.dropna(), df_mom.dropna()

# 3. Main Execution
if st.button("Generate RRG Matrix"):
    with st.spinner("Fetching Data & Calculating Vectors..."):
        # Fetch 1 Year of Weekly Data
        tickers = list(SECTORS.keys()) + [BENCHMARK]
        data = yf.download(tickers, period="1y", interval="1wk", progress=False)['Close']
        
        if not data.empty:
            # Run Math
            ratios, momentums = calculate_rrg_components(data, BENCHMARK)
            
            # Define tail length (history to draw)
            tail_len = 5 
            
            # Variables to track min/max for dynamic scaling
            all_x = []
            all_y = []

            # --- PLOTLY CHART ---
            fig = go.Figure()

            # Plot Sector Trails
            table_data = []
            
            for ticker in SECTORS.keys():
                if ticker not in ratios.columns: continue
                
                # Slice data for the trail
                x_trail = ratios[ticker].tail(tail_len)
                y_trail = momentums[ticker].tail(tail_len)
                
                if len(x_trail) < tail_len: continue

                # Collect points for auto-scaling
                all_x.extend(x_trail.values)
                all_y.extend(y_trail.values)

                curr_x = x_trail.iloc[-1]
                curr_y = y_trail.iloc[-1]
                
                # Determine Color/Phase based on current position
                phase = "LAGGING"
                color = "gray"
                if curr_x > 100 and curr_y > 100: 
                    color = C_LEADING; phase = "LEADING"
                elif curr_x > 100 and curr_y < 100: 
                    color = C_WEAKENING; phase = "WEAKENING"
                elif curr_x < 100 and curr_y < 100: 
                    color = C_LAGGING; phase = "LAGGING"
                elif curr_x < 100 and curr_y > 100: 
                    color = C_IMPROVING; phase = "IMPROVING"

                # Draw Trail (Smoothed Line)
                fig.add_trace(go.Scatter(
                    x=x_trail, y=y_trail, mode='lines',
                    line=dict(color=color, width=2, shape='spline'),
                    opacity=0.6, showlegend=False, hoverinfo='skip'
                ))
                
                # Draw Head (Marker + Text)
                # Text color adapts to theme
                text_col = C_TEXT if not is_dark else "white"
                
                fig.add_trace(go.Scatter(
                    x=[curr_x], y=[curr_y], mode='markers+text',
                    marker=dict(color=color, size=12, line=dict(color=text_col, width=1)),
                    text=[ticker], textposition="top center",
                    textfont=dict(color=text_col), # Adaptive text color
                    name=SECTORS[ticker],
                    hovertemplate=f"<b>{ticker}</b><br>Trend: %{{x:.2f}}<br>Mom: %{{y:.2f}}"
                ))
                
                table_data.append({
                    "Sector": SECTORS[ticker],
                    "Ticker": ticker,
                    "Phase": phase,
                    "RS-Ratio": curr_x,
                    "RS-Momentum": curr_y
                })

            # Calculate Dynamic Range with Padding
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            pad = 2.0 
            
            # Ensure 100 is always visible (center reference)
            x_min = min(x_min, 98); x_max = max(x_max, 102)
            y_min = min(y_min, 98); y_max = max(y_max, 102)

            # Draw Background Quadrants (Dynamic Size with Adaptive Opacity)
            bg_opacity = 0.1 if is_dark else 0.05 # Lighter backgrounds for Light Mode
            
            # Green (Leading)
            fig.add_shape(type="rect", x0=100, y0=100, x1=x_max+pad, y1=y_max+pad, fillcolor=f"rgba(0, 255, 0, {bg_opacity})", layer="below", line_width=0)
            # Yellow (Weakening)
            fig.add_shape(type="rect", x0=100, y0=y_min-pad, x1=x_max+pad, y1=100, fillcolor=f"rgba(255, 255, 0, {bg_opacity})", layer="below", line_width=0)
            # Red (Lagging)
            fig.add_shape(type="rect", x0=x_min-pad, y0=y_min-pad, x1=100, y1=100, fillcolor=f"rgba(255, 0, 0, {bg_opacity})", layer="below", line_width=0)
            # Blue (Improving)
            fig.add_shape(type="rect", x0=x_min-pad, y0=100, x1=100, y1=y_max+pad, fillcolor=f"rgba(0, 0, 255, {bg_opacity})", layer="below", line_width=0)

            # Add Axes (Crosshair at 100,100)
            axis_col = "gray" if is_dark else "black"
            fig.add_hline(y=100, line_dash="dot", line_color=axis_col)
            fig.add_vline(x=100, line_dash="dot", line_color=axis_col)

            # Chart Layout
            fig.update_layout(
                title="Sector Rotation Trails (Smoothed History)",
                xaxis_title="RS-Ratio (Trend)", yaxis_title="RS-Momentum (Velocity)",
                width=1100, height=800, template=TEMPLATE,
                xaxis=dict(range=[x_min-pad, x_max+pad], showgrid=False, zeroline=False), 
                yaxis=dict(range=[y_min-pad, y_max+pad], showgrid=False, zeroline=False),
                showlegend=False
            )
            
            # Dynamic Quadrant Labels (Colors adapted to theme)
            fig.add_annotation(x=x_max, y=y_max, text="LEADING", showarrow=False, font=dict(size=18, color=C_LEADING), xanchor="right", yanchor="top")
            fig.add_annotation(x=x_max, y=y_min, text="WEAKENING", showarrow=False, font=dict(size=18, color=C_WEAKENING), xanchor="right", yanchor="bottom")
            fig.add_annotation(x=x_min, y=y_min, text="LAGGING", showarrow=False, font=dict(size=18, color=C_LAGGING), xanchor="left", yanchor="bottom")
            fig.add_annotation(x=x_min, y=y_max, text="IMPROVING", showarrow=False, font=dict(size=18, color=C_IMPROVING), xanchor="left", yanchor="top")

            st.plotly_chart(fig, use_container_width=True)
            
            # 5. Data Table
            st.divider()
            st.subheader("📊 Sector Data")
            df = pd.DataFrame(table_data).sort_values("RS-Ratio", ascending=False)
            
            def color_phase(val):
                c = "black" # Default safety
                if val == "LEADING": c = C_LEADING
                elif val == "WEAKENING": c = C_WEAKENING
                elif val == "LAGGING": c = C_LAGGING
                elif val == "IMPROVING": c = C_IMPROVING
                return f'color: {c}; font-weight: bold'

            st.dataframe(df.style.map(color_phase, subset=['Phase'])
                                 .format({'RS-Ratio': "{:.2f}", 'RS-Momentum': "{:.2f}"}))
            
        else:
            st.error("Data Fetch Failed.")
