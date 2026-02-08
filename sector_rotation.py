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
def calculate_rrg_components(price_data, benchmark_col, window_rs=14, window_mom=5):
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
            
            # --- PLOTLY CHART ---
            fig = go.Figure()

            # 1. Draw Background Quadrants
            # Green (Leading) - Top Right
            fig.add_shape(type="rect", x0=100, y0=100, x1=110, y1=110, fillcolor="rgba(0, 255, 0, 0.1)", layer="below", line_width=0)
            # Yellow (Weakening) - Bottom Right
            fig.add_shape(type="rect", x0=100, y0=90, x1=110, y1=100, fillcolor="rgba(255, 255, 0, 0.1)", layer="below", line_width=0)
            # Red (Lagging) - Bottom Left
            fig.add_shape(type="rect", x0=90, y0=90, x1=100, y1=100, fillcolor="rgba(255, 0, 0, 0.1)", layer="below", line_width=0)
            # Blue (Improving) - Top Left
            fig.add_shape(type="rect", x0=90, y0=100, x1=100, y1=110, fillcolor="rgba(0, 0, 255, 0.1)", layer="below", line_width=0)

            # 2. Add Axes (Crosshair at 100,100)
            fig.add_hline(y=100, line_dash="dot", line_color="gray")
            fig.add_vline(x=100, line_dash="dot", line_color="gray")

            # 3. Plot Sector Trails
            table_data = []
            
            for ticker in SECTORS.keys():
                if ticker not in ratios.columns: continue
                
                # Slice data for the trail
                x_trail = ratios[ticker].tail(tail_len)
                y_trail = momentums[ticker].tail(tail_len)
                
                if len(x_trail) < tail_len: continue

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

                # Draw Trail (Line)
                fig.add_trace(go.Scatter(
                    x=x_trail, y=y_trail, mode='lines',
                    line=dict(color=color, width=2), opacity=0.5, showlegend=False, hoverinfo='skip'
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

            # 4. Chart Layout
            fig.update_layout(
                title="Sector Rotation Trails (Last 5 Weeks)",
                xaxis_title="RS-Ratio (Trend)", yaxis_title="RS-Momentum (Velocity)",
                width=1100, height=800, template="plotly_dark",
                xaxis=dict(range=[96, 104], showgrid=False, zeroline=False), 
                yaxis=dict(range=[96, 104], showgrid=False, zeroline=False),
                showlegend=False
            )
            
            # Quadrant Labels
            fig.add_annotation(x=103.5, y=103.5, text="LEADING", showarrow=False, font=dict(size=18, color="green"))
            fig.add_annotation(x=103.5, y=96.5, text="WEAKENING", showarrow=False, font=dict(size=18, color="yellow"))
            fig.add_annotation(x=96.5, y=96.5, text="LAGGING", showarrow=False, font=dict(size=18, color="red"))
            fig.add_annotation(x=96.5, y=103.5, text="IMPROVING", showarrow=False, font=dict(size=18, color="cyan"))

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
