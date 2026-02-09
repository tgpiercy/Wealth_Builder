import pandas as pd
import numpy as np
import plotly.graph_objects as go
import titan_math as tm

# --- RRG MATH ENGINE ---
def calculate_rrg_components(price_series, bench_series, len_rs=14, len_mom=14):
    """
    Calculates JdK RS-Ratio and RS-Momentum series.
    Uses index intersection to ensure date alignment per pair.
    """
    # 1. Relative Strength (Price / Bench)
    # Critical: Only calculate on valid overlapping dates
    common_idx = price_series.dropna().index.intersection(bench_series.dropna().index)
    
    if len(common_idx) < 50: return pd.Series(dtype=float), pd.Series(dtype=float)

    p = price_series.loc[common_idx]
    b = bench_series.loc[common_idx]
    
    # Avoid div by zero
    b = b.replace(0, np.nan)
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
    Returns a wide DataFrame of Closes.
    CRITICAL FIX v65.3: Do NOT dropna() globally. 
    Preserve recent data for liquid tickers even if illiquid ones are missing it.
    """
    data = {}
    if benchmark not in master_data: return pd.DataFrame()
    
    # Always include benchmark
    data[benchmark] = master_data[benchmark]['Close']
    
    for t in tickers:
        if t in master_data and len(master_data[t]) > 50:
            data[t] = master_data[t]['Close']
            
    # ffill() handles holidays/halts, but we keep NaNs at the end if data is truly missing
    df = pd.DataFrame(data).ffill()
    return df

def calculate_rrg_math(wide_df, benchmark_ticker):
    """
    Returns the Ratio and Momentum DataFrames.
    """
    r_dict = {}
    m_dict = {}
    
    if benchmark_ticker not in wide_df.columns: return pd.DataFrame(), pd.DataFrame()
    bench = wide_df[benchmark_ticker]
    
    for col in wide_df.columns:
        if col == benchmark_ticker: continue
        
        # Calculate pair-wise to maximize data availability for each ticker
        r, m = calculate_rrg_components(wide_df[col], bench)
        
        # Re-align to the master index so we can combine them later
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
    style = "style='font-size: 22px; line-height: 1;'"
    
    # Ensure we look at the last VALID values (ignoring recent NaNs if any)
    valid_r = r_series.dropna()
    valid_m = m_series.dropna()
    
    if len(valid_r) < 2 or len(valid_m) < 2: 
        return f"<span {style}>➡️</span>"
    
    # Match indices to ensure we compare the same dates
    common_idx = valid_r.index.intersection(valid_m.index)
    if len(common_idx) < 2: return f"<span {style}>➡️</span>"
    
    r_curr = valid_r.loc[common_idx[-1]]; r_prev = valid_r.loc[common_idx[-2]]
    m_curr = valid_m.loc[common_idx[-1]]; m_prev = valid_m.loc[common_idx[-2]]
    
    dx = r_curr - r_prev
    dy = m_curr - m_prev
    
    if dx > 0 and dy > 0: return f"<span {style}>↗️</span>" 
    if dx > 0 and dy < 0: return f"<span {style}>↘️</span>" 
    if dx < 0 and dy < 0: return f"<span {style}>↙️</span>" 
    if dx < 0 and dy > 0: return f"<span {style}>↖️</span>" 
    
    return f"<span {style}>➡️</span>"

def generate_full_rrg_snapshot(master_data, benchmark="SPY"):
    """
    Runs RRG logic for ALL tickers in master_data vs Benchmark.
    """
    if not master_data: return {}
    
    scan_tickers = list(master_data.keys())
    if benchmark in scan_tickers: scan_tickers.remove(benchmark)
    
    # Get wide DF (Sparse, no global dropna)
    wide_df = prepare_rrg_inputs(master_data, scan_tickers, benchmark)
    if wide_df.empty: return {}
    
    r_df, m_df = calculate_rrg_math(wide_df, benchmark)
    snapshot = {}
    
    for t in r_df.columns:
        try:
            # Drop NaNs specific to this ticker to find the true "Last Price"
            ts_r = r_df[t].dropna()
            ts_m = m_df[t].dropna()
            
            if ts_r.empty or ts_m.empty:
                snapshot[t] = "UNKNOWN"
                continue

            curr_r = ts_r.iloc[-1]
            curr_m = ts_m.iloc[-1]
            
            # Pass full series to get_heading for robust T-1 calculation
            heading = get_heading(ts_r, ts_m)
            
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
    Plots the RRG Scatter chart.
    """
    fig = go.Figure()
    
    bg_color = "#1e1e1e" if is_dark else "#ffffff"
    grid_color = "#333" if is_dark else "#ddd"
    
    # Quadrants
    fig.add_shape(type="rect", x0=100, y0=100, x1=115, y1=115, fillcolor="rgba(0,255,0,0.1)", line_width=0)
    fig.add_shape(type="rect", x0=85, y0=100, x1=100, y1=115, fillcolor="rgba(0,0,255,0.1)", line_width=0)
    fig.add_shape(type="rect", x0=85, y0=85, x1=100, y1=100, fillcolor="rgba(255,0,0,0.1)", line_width=0)
    fig.add_shape(type="rect", x0=100, y0=85, x1=115, y1=100, fillcolor="rgba(255,255,0,0.1)", line_width=0)

    for col in r_df.columns:
        # Get valid data for this specific ticker
        valid_idx = r_df[col].dropna().index.intersection(m_df[col].dropna().index)
        if len(valid_idx) < 5: continue
        
        # Slice last 10 points
        x_tail = r_df.loc[valid_idx, col].iloc[-10:]
        y_tail = m_df.loc[valid_idx, col].iloc[-10:]
        
        curr_x = x_tail.iloc[-1]
        curr_y = y_tail.iloc[-1]
        
        if curr_x > 100 and curr_y > 100: color = '#00ff00' 
        elif curr_x < 100 and curr_y > 100: color = '#00bfff'
        elif curr_x < 100 and curr_y < 100: color = '#ff4444' 
        else: color = '#ffff00'
        
        label = label_map.get(col, col)
        
        fig.add_trace(go.Scatter(
            x=x_tail, y=y_tail, mode='lines', line=dict(color=color, width=1),
            opacity=0.5, hoverinfo='skip', showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[curr_x], y=[curr_y], mode='markers+text',
            marker=dict(color=color, size=10, line=dict(color='white', width=1)),
            text=[label], textposition="top center", name=label, hoverinfo='text+x+y'
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="RS-Ratio (Trend)", showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor='white', range=[95, 105]),
        yaxis=dict(title="RS-Momentum (Velocity)", showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor='white', range=[95, 105]),
        paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font=dict(color="white" if is_dark else "black"),
        width=1000, height=800, showlegend=False
    )
    return fig
