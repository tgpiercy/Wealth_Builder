import pandas as pd
import numpy as np
import plotly.graph_objects as go
import titan_math as tm
import math

# --- RRG MATH ENGINE ---
def calculate_rrg_components(price_series, bench_series, len_rs=14, len_mom=14, smooth_period=3):
    """
    Calculates JdK RS-Ratio and RS-Momentum series.
    Includes SMOOTHING to create curvy tails.
    """
    # 1. Align Data
    common_idx = price_series.dropna().index.intersection(bench_series.dropna().index)
    if len(common_idx) < 30: return pd.Series(dtype=float), pd.Series(dtype=float)

    p = price_series.loc[common_idx]
    b = bench_series.loc[common_idx]
    b = b.replace(0, np.nan)
    
    # 2. RS Calculation
    rs_raw = 100 * (p / b)
    
    # 3. RS-Ratio (Trend)
    rs_mean = rs_raw.rolling(window=len_rs).mean()
    rs_ratio = 100 * (rs_raw / rs_mean)
    
    # 4. RS-Momentum (Velocity)
    ratio_mean = rs_ratio.rolling(window=len_mom).mean()
    rs_mom = 100 * (rs_ratio / ratio_mean)
    
    # 5. SMOOTHING (The Fix)
    rs_ratio_smooth = rs_ratio.rolling(window=smooth_period).mean()
    rs_mom_smooth = rs_mom.rolling(window=smooth_period).mean()
    
    return rs_ratio_smooth.dropna(), rs_mom_smooth.dropna()

def prepare_rrg_inputs(master_data, tickers, benchmark):
    """
    Returns a wide DataFrame of Closes, RESAMPLED TO WEEKLY.
    """
    data = {}
    
    def get_weekly(df):
        return df['Close'].resample('W-FRI').last().ffill()

    if benchmark in master_data:
        data[benchmark] = get_weekly(master_data[benchmark])
    
    for t in tickers:
        if t in master_data and len(master_data[t]) > 50:
            data[t] = get_weekly(master_data[t])
            
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
        r, m = calculate_rrg_components(wide_df[col], bench)
        r_dict[col] = r
        m_dict[col] = m
        
    r_df = pd.DataFrame(r_dict)
    m_df = pd.DataFrame(m_dict)
    return r_df, m_df

def get_heading_from_tail(x_series, y_series):
    """
    Determines 8-POINT COMPASS direction using atan2 trigonometry.
    """
    style = "style='font-size: 22px; line-height: 1;'"
    
    if len(x_series) < 2 or len(y_series) < 2: 
        return f"<span {style}>➡️</span>"
    
    dx = x_series.iloc[-1] - x_series.iloc[-2]
    dy = y_series.iloc[-1] - y_series.iloc[-2]
    
    if dx == 0 and dy == 0: return f"<span {style}>➡️</span>"
    
    angle = math.degrees(math.atan2(dy, dx))
    
    if -22.5 <= angle < 22.5: return f"<span {style}>➡️</span>"   # E
    elif 22.5 <= angle < 67.5: return f"<span {style}>↗️</span>"  # NE
    elif 67.5 <= angle < 112.5: return f"<span {style}>⬆️</span>" # N
    elif 112.5 <= angle < 157.5: return f"<span {style}>↖️</span>" # NW
    elif 157.5 <= angle <= 180: return f"<span {style}>⬅️</span>"  # W
    elif -180 <= angle < -157.5: return f"<span {style}>⬅️</span>" # W (Wrap)
    elif -157.5 <= angle < -112.5: return f"<span {style}>↙️</span>" # SW
    elif -112.5 <= angle < -67.5: return f"<span {style}>⬇️</span>" # S
    elif -67.5 <= angle < -22.5: return f"<span {style}>↘️</span>" # SE
    
    return f"<span {style}>➡️</span>"

def generate_full_rrg_snapshot(master_data, benchmark="SPY"):
    """
    Runs Weekly RRG logic for ALL tickers.
    """
    if not master_data: return {}
    
    snapshot = {}
    scan_tickers = list(master_data.keys())
    if benchmark in scan_tickers: scan_tickers.remove(benchmark)
    
    wide_df = prepare_rrg_inputs(master_data, scan_tickers, benchmark)
    
    if not wide_df.empty:
        r_df, m_df = calculate_rrg_math(wide_df, benchmark)
        
        for t in r_df.columns:
            try:
                ts_r = r_df[t].dropna(); ts_m = m_df[t].dropna()
                if ts_r.empty or ts_m.empty: continue
                
                common = ts_r.index.intersection(ts_m.index)
                if len(common) < 2: continue
                
                curr_r = ts_r.loc[common[-1]]
                curr_m = ts_m.loc[common[-1]]
                
                heading = get_heading_from_tail(ts_r.loc[common], ts_m.loc[common])
                
                phase = "UNKNOWN"
                if curr_r > 100 and curr_m > 100: phase = "LEADING"
                elif curr_r < 100 and curr_m > 100: phase = "IMPROVING"
                elif curr_r < 100 and curr_m < 100: phase = "LAGGING"
                elif curr_r > 100 and curr_m < 100: phase = "WEAKENING"
                
                snapshot[t] = f"{phase} {heading}"
            except:
                snapshot[t] = "UNKNOWN"

    # Special Exception: SPY vs IEF
    if "SPY" in master_data and "IEF" in master_data:
        try:
            spy_s = master_data["SPY"]['Close'].resample('W-FRI').last().ffill()
            ief_s = master_data["IEF"]['Close'].resample('W-FRI').last().ffill()
            r_spy, m_spy = calculate_rrg_components(spy_s, ief_s)
            
            if not r_spy.empty and not m_spy.empty:
                c_idx = r_spy.index.intersection(m_spy.index)
                if len(c_idx) > 2:
                    curr_r = r_spy.loc[c_idx[-1]]; curr_m = m_spy.loc[c_idx[-1]]
                    heading = get_heading_from_tail(r_spy.loc[c_idx], m_spy.loc[c_idx])
                    phase = "UNKNOWN"
                    if curr_r > 100 and curr_m > 100: phase = "LEADING"
                    elif curr_r < 100 and curr_m > 100: phase = "IMPROVING"
                    elif curr_r < 100 and curr_m < 100: phase = "LAGGING"
                    elif curr_r > 100 and curr_m < 100: phase = "WEAKENING"
                    snapshot["SPY"] = f"{phase} {heading}"
        except: pass

    return snapshot

def plot_rrg_chart(r_df, m_df, label_map, title, is_dark=True):
    """
    Plots RRG Scatter with Tails (5-week smooth).
    AUTO-SCALING ENABLED (Fixed range removed).
    """
    fig = go.Figure()
    
    bg_color = "#1e1e1e" if is_dark else "#ffffff"
    grid_color = "#333" if is_dark else "#ddd"
    
    # Quadrants - Drawn as infinite shapes or large enough rectangles
    # Since we want auto-scale, we draw shapes relative to center (100,100)
    # We use 'layer="below"' so data is on top.
    
    # Leading (NE)
    fig.add_shape(type="rect", x0=100, y0=100, x1=200, y1=200, fillcolor="rgba(0,255,0,0.1)", line_width=0, layer="below")
    # Improving (NW)
    fig.add_shape(type="rect", x0=0, y0=100, x1=100, y1=200, fillcolor="rgba(0,0,255,0.1)", line_width=0, layer="below")
    # Lagging (SW)
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, fillcolor="rgba(255,0,0,0.1)", line_width=0, layer="below")
    # Weakening (SE)
    fig.add_shape(type="rect", x0=100, y0=0, x1=200, y1=100, fillcolor="rgba(255,255,0,0.1)", line_width=0, layer="below")

    for col in r_df.columns:
        valid_idx = r_df[col].dropna().index.intersection(m_df[col].dropna().index)
        if len(valid_idx) < 5: continue
        
        # 5-Week Tail
        x_tail = r_df.loc[valid_idx, col].iloc[-5:] 
        y_tail = m_df.loc[valid_idx, col].iloc[-5:] 
        
        curr_x = x_tail.iloc[-1]
        curr_y = y_tail.iloc[-1]
        
        if curr_x > 100 and curr_y > 100: color = '#00ff00' 
        elif curr_x < 100 and curr_y > 100: color = '#00bfff'
        elif curr_x < 100 and curr_y < 100: color = '#ff4444' 
        else: color = '#ffff00'
        
        # Label is TICKER only (column name)
        label = col 
        
        fig.add_trace(go.Scatter(
            x=x_tail, y=y_tail, mode='lines', 
            line=dict(color=color, width=1),
            opacity=0.5, hoverinfo='skip', showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[curr_x], y=[curr_y], mode='markers+text',
            marker=dict(color=color, size=10, line=dict(color='white', width=1)),
            text=[label], textposition="top center", 
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        # REMOVED FIXED RANGES for Auto-Scaling
        xaxis=dict(title="RS-Ratio (Trend)", showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor='white'),
        yaxis=dict(title="RS-Momentum (Velocity)", showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor='white'),
        paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font=dict(color="white" if is_dark else "black"),
        width=1000, height=800, showlegend=False,
        dragmode='pan'
    )
    return fig
