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
    common_idx = price_series.index.intersection(bench_series.index)
    p = price_series.loc[common_idx]
    b = bench_series.loc[common_idx]
    rs_raw = 100 * (p / b)
    
    # 2. RS-Ratio (Trend of RS)
    rs_mean = rs_raw.rolling(window=len_rs).mean()
    rs_ratio = 100 * (rs_raw / rs_mean)
    
    # 3. RS-Momentum (Rate of Change of Ratio)
    ratio_mean = rs_ratio.rolling(window=len_mom).mean()
    rs_mom = 100 * (rs_ratio / ratio_mean)
    
    return rs_ratio.dropna(), rs_mom.dropna()

def prepare_rrg_inputs(master_data, tickers, benchmark):
    """
    Returns a wide DataFrame of Closes for the RRG calculation.
    """
    data = {}
    if benchmark not in master_data: return pd.DataFrame()
    data[benchmark] = master_data[benchmark]['Close']
    
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
    Returns HTML formatted arrow for larger visibility (22px).
    """
    # HTML Wrapper for size
    style = "style='font-size: 22px; line-height: 1;'"
    
    if len(r_series) < 2 or len(m_series) < 2: 
        return f"<span {style}>➡️</span>"
    
    dx = r_series.iloc[-1] - r_series.iloc[-2] # Change in Ratio (X-axis)
    dy = m_series.iloc[-1] - m_series.iloc[-2] # Change in Momentum (Y-axis)
    
    # Directional Logic
    if dx > 0 and dy > 0: return f"<span {style}>↗️</span>" # NE (Strongest)
    if dx > 0 and dy < 0: return f"<span {style}>↘️</span>" # SE (Weakening)
    if dx < 0 and dy < 0: return f"<span {style}>↙️</span>" # SW (Weakest)
    if dx < 0 and dy > 0: return f"<span {style}>↖️</span>" # NW (Improving)
    
    return f"<span {style}>➡️</span>"

def generate_full_rrg_snapshot(master_data, benchmark="SPY"):
    """
    Runs RRG logic for ALL tickers in master_data vs Benchmark.
    Returns a dict: {Ticker: "PHASE ↗️"}
    """
    if not master_data: return {}
    
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
            
            # Get Heading Arrow (Now Large HTML)
            heading = get_heading(r_df[t], m_df[t])
            
            # Determine Quadrant Phase
            phase = "UNKNOWN"
            if curr_r > 100 and curr_m > 100: phase = "LEADING"
            elif curr_r < 100 and curr_m > 100: phase = "IMPROVING"
            elif curr_r < 100 and curr_m < 100: phase = "LAGGING"
            elif curr_r > 100 and curr_m < 100: phase = "WEAKENING"
            
            # Combine
            snapshot[t] = f"{phase} {heading}"
        except:
            snapshot[t] = "UNKNOWN"
            
    return snapshot

def plot_rrg_chart(r_df, m_df, label_map, title, is_dark=True):
    """
    Plots the RRG Scatter chart with TAILS (Lines history).
    """
    fig = go.Figure()
    
    bg_color = "#1e1e1e" if is_dark else "#ffffff"
    grid_color = "#333" if is_dark else "#ddd"
    
    # Add Quadrants
    fig.add_shape(type="rect", x0=100, y0=100, x1=110, y1=110, fillcolor="rgba(0,255,0,0.1)", line_width=0)
    fig.add_shape(type="rect", x0=90, y0=100, x1=100, y1=110, fillcolor="rgba(0,0,255,0.1)", line_width=0)
    fig.add_shape(type="rect", x0=90, y0=90, x1=100, y1=100, fillcolor="rgba(255,0,0,0.1)", line_width=0)
    fig.add_shape(type="rect", x0=100, y0=90, x1=110, y1=100, fillcolor="rgba(255,255,0,0.1)", line_width=0)

    for col in r_df.columns:
        tail_len = 10
        if len(r_df) < tail_len: continue
        
        x_tail = r_df[col].iloc[-tail_len:]
        y_tail = m_df[col].iloc[-tail_len:]
        curr_x = x_tail.iloc[-1]
        curr_y = y_tail.iloc[-1]
        
        if curr_x > 100 and curr_y > 100: color = '#00ff00' 
        elif curr_x < 100 and curr_y > 100: color = '#00bfff'
        elif curr_x < 100 and curr_y < 100: color = '#ff4444' 
        else: color = '#ffff00'
        
        label = label_map.get(col, col)
        
        # Tail
        fig.add_trace(go.Scatter(
            x=x_tail, y=y_tail, mode='lines', line=dict(color=color, width=1),
            opacity=0.5, hoverinfo='skip', showlegend=False
        ))
        
        # Head
        fig.add_trace(go.Scatter(
            x=[curr_x], y=[curr_y], mode='markers+text',
            marker=dict(color=color, size=10, line=dict(color='white', width=1)),
            text=[label], textposition="top center", name=label, hoverinfo='text+x+y'
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="RS-Ratio (Trend)", showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor='white'),
        yaxis=dict(title="RS-Momentum (Velocity)", showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor='white'),
        paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font=dict(color="white" if is_dark else "black"),
        width=1000, height=800, showlegend=False
    )
    return fig
