import pandas as pd
import numpy as np
import plotly.graph_objects as go
import titan_math as tm

# --- RRG MATH ENGINE ---
def calculate_rrg_components(price_series, bench_series, len_rs=14, len_mom=14):
    """
    Calculates JdK RS-Ratio and RS-Momentum series.
    """
    # 1. Relative Strength (Price / Bench)
    # Align dates first
    common_idx = price_series.index.intersection(bench_series.index)
    p = price_series.loc[common_idx]
    b = bench_series.loc[common_idx]
    rs_raw = 100 * (p / b)
    
    # 2. RS-Ratio (Trend of RS) -> Normalized around 100
    # Logic: 100 + ((RS - SMA(RS)) / StdDev(RS)) * scaling, 
    # simplified to standard moving average ratio for stability in this implementation:
    # Standard JdK uses complex normalization, we use a robust approximation:
    # Ratio = 100 * (RS / SMA(RS))
    rs_mean = rs_raw.rolling(window=len_rs).mean()
    rs_ratio = 100 * (rs_raw / rs_mean)
    
    # 3. RS-Momentum (Rate of Change of Ratio)
    # Mom = 100 * (Ratio / SMA(Ratio))
    ratio_mean = rs_ratio.rolling(window=len_mom).mean()
    rs_mom = 100 * (rs_ratio / ratio_mean)
    
    return rs_ratio.dropna(), rs_mom.dropna()

def prepare_rrg_inputs(master_data, tickers, benchmark):
    """
    Returns a wide DataFrame of Closes for the RRG calculation.
    """
    data = {}
    if benchmark not in master_data: return pd.DataFrame()
    
    # Add Benchmark
    data[benchmark] = master_data[benchmark]['Close']
    
    # Add Tickers
    for t in tickers:
        if t in master_data and len(master_data[t]) > 50:
            data[t] = master_data[t]['Close']
            
    df = pd.DataFrame(data).ffill().dropna()
    return df

def calculate_rrg_math(wide_df, benchmark_ticker):
    """
    Returns the Ratio and Momentum DataFrames for the last 5 periods (for tails).
    """
    r_dict = {}
    m_dict = {}
    
    if benchmark_ticker not in wide_df.columns: return pd.DataFrame(), pd.DataFrame()
    bench = wide_df[benchmark_ticker]
    
    for col in wide_df.columns:
        if col == benchmark_ticker: continue
        r, m = calculate_rrg_components(wide_df[col], bench)
        r_dict[col] = r
        m_dict[col] = m
        
    r_df = pd.DataFrame(r_dict)
    m_df = pd.DataFrame(m_dict)
    return r_df, m_df

def get_heading(r_series, m_series):
    """
    Determines the directional arrow based on the movement from T-1 to T.
    """
    if len(r_series) < 2 or len(m_series) < 2: return ""
    
    dx = r_series.iloc[-1] - r_series.iloc[-2] # Change in Ratio (X-axis)
    dy = m_series.iloc[-1] - m_series.iloc[-2] # Change in Momentum (Y-axis)
    
    # Logic:
    # NE (Top Right): +Ratio, +Mom (Strongest)
    # NW (Top Left): -Ratio, +Mom (Improving)
    # SE (Bottom Right): +Ratio, -Mom (Weakening)
    # SW (Bottom Left): -Ratio, -Mom (Weakest)
    
    if dx > 0 and dy > 0: return "↗️" # Charging
    if dx > 0 and dy < 0: return "↘️" # Rolling Over
    if dx < 0 and dy < 0: return "↙️" # Dumping
    if dx < 0 and dy > 0: return "↖️" # Bottoming
    return "➡️"

def generate_full_rrg_snapshot(master_data, benchmark="SPY"):
    """
    Runs RRG logic for ALL tickers in master_data vs Benchmark.
    Returns a dict: {Ticker: "PHASE ↗️"}
    """
    if not master_data: return {}
    
    # Create Wide DF
    scan_tickers = list(master_data.keys())
    if benchmark in scan_tickers: scan_tickers.remove(benchmark)
    
    wide_df = prepare_rrg_inputs(master_data, scan_tickers, benchmark)
    if wide_df.empty: return {}
    
    r_df, m_df = calculate_rrg_math(wide_df, benchmark)
    snapshot = {}
    
    for t in r_df.columns:
        try:
            curr_r = r_df[t].iloc[-1]
            curr_m = m_df[t].iloc[-1]
            heading = get_heading(r_df[t], m_df[t])
            
            phase = "UNKNOWN"
            if curr_r > 100 and curr_m > 100: phase = "LEADING"
            elif curr_r < 100 and curr_m > 100: phase = "IMPROVING"
            elif curr_r < 100 and curr_m < 100: phase = "LAGGING"
            elif curr_r > 100 and curr_m < 100: phase = "WEAKENING"
            
            snapshot[t] = f"{phase} {heading}"
        except:
            snapshot[t] = "UNKNOWN"
            
    return snapshot

def plot_rrg_chart(r_df, m_df, label_map, title, is_dark=True):
    """
    Plots the RRG Scatter chart with TAILS (Lines history).
    """
    fig = go.Figure()
    
    # Quadrant Background
    bg_color = "#1e1e1e" if is_dark else "#ffffff"
    grid_color = "#333" if is_dark else "#ddd"
    
    # Add Quadrants
    fig.add_shape(type="rect", x0=100, y0=100, x1=110, y1=110, fillcolor="rgba(0,255,0,0.1)", line_width=0) # Leading
    fig.add_shape(type="rect", x0=90, y0=100, x1=100, y1=110, fillcolor="rgba(0,0,255,0.1)", line_width=0) # Improving
    fig.add_shape(type="rect", x0=90, y0=90, x1=100, y1=100, fillcolor="rgba(255,0,0,0.1)", line_width=0) # Lagging
    fig.add_shape(type="rect", x0=100, y0=90, x1=110, y1=100, fillcolor="rgba(255,255,0,0.1)", line_width=0) # Weakening

    # Plot Tails & Dots
    for col in r_df.columns:
        # Get last 5 points for the tail
        tail_len = 10
        if len(r_df) < tail_len: continue
        
        x_tail = r_df[col].iloc[-tail_len:]
        y_tail = m_df[col].iloc[-tail_len:]
        
        # Color Logic based on current position
        curr_x = x_tail.iloc[-1]
        curr_y = y_tail.iloc[-1]
        
        if curr_x > 100 and curr_y > 100: color = '#00ff00' # Green
        elif curr_x < 100 and curr_y > 100: color = '#00bfff' # Blue
        elif curr_x < 100 and curr_y < 100: color = '#ff4444' # Red
        else: color = '#ffff00' # Yellow
        
        label = label_map.get(col, col)
        
        # Draw Tail (Line)
        fig.add_trace(go.Scatter(
            x=x_tail, y=y_tail,
            mode='lines',
            line=dict(color=color, width=1),
            opacity=0.5,
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Draw Head (Marker)
        fig.add_trace(go.Scatter(
            x=[curr_x], y=[curr_y],
            mode='markers+text',
            marker=dict(color=color, size=10, line=dict(color='white', width=1)),
            text=[label],
            textposition="top center",
            name=label,
            hoverinfo='text+x+y'
        ))

    # Axis Formatting
    fig.update_layout(
        title=title,
        xaxis=dict(title="RS-Ratio (Trend)", showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor='white'),
        yaxis=dict(title="RS-Momentum (Velocity)", showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor='white'),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color="white" if is_dark else "black"),
        width=1000, height=800,
        showlegend=False
    )
    
    return fig
