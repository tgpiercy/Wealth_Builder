import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Sector RRG Analysis")
st.title("🔄 Sector Rotation: Relative Rotation Graph (RRG)")
st.caption("Institutional Money Flow Tracker | Benchmark: SPY")

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
            # We use a standard score (Z-Score proxy) centered at 100
            ma_rs = rs.rolling(window=window_rs).mean()
            std_rs = rs.rolling(window=window_rs).std()
            
            # This formula centers the values around 100
            ratio = 100 + ((rs - ma_rs) / std_rs) * 1.5 
            df_ratio[col] = ratio

    # 3. RS-Momentum (Rate of Change of Ratio)
    for col in df_ratio.columns:
        # Momentum is the velocity of the ratio compared to its own moving average
        mom = 100 + (df_ratio[col] - df_ratio[col].rolling(window=window_mom).mean()) * 2
        df_mom[col] = mom
        
    # 4. Final Smoothing (To make curves look organic)
    df_ratio = df_ratio.rolling(window=smooth_factor).mean()
    df_mom = df_mom.rolling(window=smooth_factor).mean()
        
    return df_ratio.dropna(), df_mom.dropna()

# 3. Main Execution
if st.button("Generate RRG Matrix"):
    with st.spinner("Fetching Data & Calculating Vectors..."):
        # Fetch 1 Year of Weekly Data (Standard for RRG Trend Analysis)
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
                    color = "#00FF00"; phase = "LEADING"
                elif curr_x > 100 and curr_y < 100: 
                    color = "#FFFF00"; phase = "WEAKENING"
                elif curr_x < 100 and curr_y < 100: 
                    color = "#FF4444"; phase = "LAGGING"
                elif curr_x < 100 and curr_y > 100: 
                    color = "#00BFFF"; phase = "IMPROVING"

                # Draw Trail (Smoothed Line)
                fig.add_trace(go.Scatter(
                    x=x_trail, y=y_trail, mode='lines',
                    line=dict(color=color, width=2, shape='spline'), # Spline makes it curvy
                    opacity=0.6, showlegend=False, hoverinfo='skip'
                ))
                
                # Draw Head (Marker + Text)
                fig.add_trace(go.Scatter(
                    x=[curr_x], y=[curr_y], mode='markers+text',
                    marker=dict(color=color, size=12, line=dict(color='white', width=1)),
                    text=[ticker], textposition="top center",
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
            pad = 2.0 # Buffer padding
            
            # Ensure 100 is always visible (center reference)
            x_min = min(x_min, 98); x_max = max(x_max, 102)
            y_min = min(y_min, 98); y_max = max(y_max, 102)

            # Draw Background Quadrants (Dynamic Size)
            # Green (Leading) - Top Right
            fig.add_shape(type="rect", x0=100, y0=100, x1=x_max+pad, y1=y_max+pad, fillcolor="rgba(0, 255, 0, 0.1)", layer="below", line_width=0)
            # Yellow (Weakening) - Bottom Right
            fig.add_shape(type="rect", x0=100, y0=y_min-pad, x1=x_max+pad, y1=100, fillcolor="rgba(255, 255, 0, 0.1)", layer="below", line_width=0)
            # Red (Lagging) - Bottom Left
            fig.add_shape(type="rect", x0=x_min-pad, y0=y_min-pad, x1=100, y1=100, fillcolor="rgba(255, 0, 0, 0.1)", layer="below", line_width=0)
            # Blue (Improving) - Top Left
            fig.add_shape(type="rect", x0=x_min-pad, y0=100, x1=100, y1=y_max+pad, fillcolor="rgba(0, 0, 255, 0.1)", layer="below", line_width=0)

            # Add Axes (Crosshair at 100,100)
            fig.add_hline(y=100, line_dash="dot", line_color="gray")
            fig.add_vline(x=100, line_dash="dot", line_color="gray")

            # Chart Layout
            fig.update_layout(
                title="Sector Rotation Trails (Smoothed History)",
                xaxis_title="RS-Ratio (Trend)", yaxis_title="RS-Momentum (Velocity)",
                width=1100, height=800, template="plotly_dark",
                xaxis=dict(range=[x_min-pad, x_max+pad], showgrid=False, zeroline=False), 
                yaxis=dict(range=[y_min-pad, y_max+pad], showgrid=False, zeroline=False),
                showlegend=False
            )
            
            # Dynamic Quadrant Labels (Fixed relative to view)
            fig.add_annotation(x=x_max, y=y_max, text="LEADING", showarrow=False, font=dict(size=18, color="green"), xanchor="right", yanchor="top")
            fig.add_annotation(x=x_max, y=y_min, text="WEAKENING", showarrow=False, font=dict(size=18, color="yellow"), xanchor="right", yanchor="bottom")
            fig.add_annotation(x=x_min, y=y_min, text="LAGGING", showarrow=False, font=dict(size=18, color="red"), xanchor="left", yanchor="bottom")
            fig.add_annotation(x=x_min, y=y_max, text="IMPROVING", showarrow=False, font=dict(size=18, color="cyan"), xanchor="left", yanchor="top")

            st.plotly_chart(fig, use_container_width=True)
            
            # 5. Data Table
            st.divider()
            st.subheader("📊 Sector Data")
            df = pd.DataFrame(table_data).sort_values("RS-Ratio", ascending=False)
            
            def color_phase(val):
                c = "white"
                if val == "LEADING": c = "#00FF00"
                elif val == "WEAKENING": c = "#FFA500"
                elif val == "LAGGING": c = "#FF4444"
                elif val == "IMPROVING": c = "#00BFFF"
                return f'color: {c}; font-weight: bold'

            st.dataframe(df.style.map(color_phase, subset=['Phase'])
                                 .format({'RS-Ratio': "{:.2f}", 'RS-Momentum': "{:.2f}"}))
            
        else:
            st.error("Data Fetch Failed.")
