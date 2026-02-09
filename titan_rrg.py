import pandas as pd
import plotly.graph_objects as go

# Note: Redefining basic rolling calc here to keep module independent
def calculate_rrg_math(price_data, benchmark_col, window_rs=14, window_mom=5, smooth_factor=3):
    if benchmark_col not in price_data.columns: return pd.DataFrame(), pd.DataFrame()
    df_ratio = pd.DataFrame(); df_mom = pd.DataFrame()
    for col in price_data.columns:
        if col != benchmark_col:
            try:
                rs = price_data[col] / price_data[benchmark_col]
                mean = rs.rolling(window_rs).mean(); std = rs.rolling(window_rs).std()
                ratio = 100 + ((rs - mean) / std) * 1.5
                df_ratio[col] = ratio
            except: continue
    for col in df_ratio.columns:
        try: df_mom[col] = 100 + (df_ratio[col] - df_ratio[col].rolling(window_mom).mean()) * 2
        except: continue
    return df_ratio.rolling(smooth_factor).mean().dropna(), df_mom.rolling(smooth_factor).mean().dropna()

def prepare_rrg_inputs(data_map, tickers, benchmark):
    df_wide = pd.DataFrame()
    if benchmark in data_map:
        bench_df = data_map[benchmark].resample('W-FRI').last()
        df_wide[benchmark] = bench_df['Close']
    for t in tickers:
        if t in data_map and t != benchmark:
            w_df = data_map[t].resample('W-FRI').last()
            df_wide[t] = w_df['Close']
    return df_wide.dropna()

def generate_full_rrg_snapshot(data_map, benchmark="SPY"):
    status_map = {}
    try:
        all_tickers = list(data_map.keys())
        wide_df = prepare_rrg_inputs(data_map, all_tickers, benchmark)
        if not wide_df.empty:
            r, m = calculate_rrg_math(wide_df, benchmark)
            if not r.empty:
                l_idx = r.index[-1]
                for t in r.columns:
                    try:
                        vr, vm = r.at[l_idx, t], m.at[l_idx, t]
                        if vr > 100 and vm > 100: status_map[t] = "LEADING"
                        elif vr > 100 and vm < 100: status_map[t] = "WEAKENING"
                        elif vr < 100 and vm < 100: status_map[t] = "LAGGING"
                        else: status_map[t] = "IMPROVING"
                    except: continue
        # SPY vs IEF Check
        if "SPY" in data_map and "IEF" in data_map:
            spy_ief = prepare_rrg_inputs(data_map, ["SPY"], "IEF")
            rs, ms = calculate_rrg_math(spy_ief, "IEF")
            if not rs.empty:
                l_idx = rs.index[-1]
                vrs, vms = rs.at[l_idx, "SPY"], ms.at[l_idx, "SPY"]
                if vrs > 100 and vms > 100: status_map["SPY"] = "LEADING"
                elif vrs > 100 and vms < 100: status_map["SPY"] = "WEAKENING"
                elif vrs < 100 and vms < 100: status_map["SPY"] = "LAGGING"
                else: status_map["SPY"] = "IMPROVING"
    except: pass
    return status_map

def plot_rrg_chart(ratios, momentums, labels_map, title, is_dark):
    fig = go.Figure()
    if is_dark:
        bg_col, text_col = "black", "white"; c_lead, c_weak, c_lag, c_imp = "#00FF00", "#FFFF00", "#FF4444", "#00BFFF"; template = "plotly_dark"
    else:
        bg_col, text_col = "white", "black"; c_lead, c_weak, c_lag, c_imp = "#008000", "#FF8C00", "#CC0000", "#0000FF"; template = "plotly_white"

    has_data = False; x_vals = []; y_vals = []
    
    # Ensure SPY is in the plot list if available
    plot_tickers = list(labels_map.keys())
    if "SPY" in ratios.columns and "SPY" not in plot_tickers: plot_tickers.append("SPY")

    for ticker in plot_tickers:
        if ticker not in ratios.columns: continue
        xt = ratios[ticker].tail(5); yt = momentums[ticker].tail(5)
        if len(xt) < 5: continue
        has_data = True
        cx, cy = xt.iloc[-1], yt.iloc[-1]
        x_vals.extend(xt.values); y_vals.extend(yt.values)

        if cx > 100 and cy > 100: color = c_lead
        elif cx > 100 and cy < 100: color = c_weak
        elif cx < 100 and cy < 100: color = c_lag
        else: color = c_imp
        
        # FIX: Get full name for tooltip, but use TICKER for chart label
        full_name = labels_map.get(ticker, ticker)
        
        fig.add_trace(go.Scatter(
            x=xt, y=yt, mode='lines', 
            line=dict(color=color, width=2, shape='spline'), 
            opacity=0.6, showlegend=False, hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode='markers+text', 
            marker=dict(color=color, size=12, line=dict(color=text_col, width=1)), 
            text=[ticker],  # <--- CHANGED FROM full_name TO ticker
            textposition="top center", 
            textfont=dict(color=text_col), 
            hovertemplate=f"<b>{full_name} ({ticker})</b><br>T: %{{x:.2f}}<br>M: %{{y:.2f}}"
        ))

    if not has_data: return None
    
    min_x, max_x = min(x_vals + [96]), max(x_vals + [104])
    min_y, max_y = min(y_vals + [96]), max(y_vals + [104])
    buff_x = (max_x - min_x) * 0.05; buff_y = (max_y - min_y) * 0.05
    rx = [min_x - buff_x, max_x + buff_x]; ry = [min_y - buff_y, max_y + buff_y]

    op = 0.1 if is_dark else 0.05
    fig.add_hline(y=100, line_dash="dot", line_color="gray"); fig.add_vline(x=100, line_dash="dot", line_color="gray")
    fig.add_shape(type="rect", x0=100, y0=100, x1=500, y1=500, fillcolor=f"rgba(0,255,0,{op})", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=100, y0=-500, x1=500, y1=100, fillcolor=f"rgba(255,255,0,{op})", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=-500, y0=-500, x1=100, y1=100, fillcolor=f"rgba(255,0,0,{op})", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=-500, y0=100, x1=100, y1=500, fillcolor=f"rgba(0,0,255,{op})", layer="below", line_width=0)
    
    fig.update_layout(title=title, template=template, height=650, showlegend=False, xaxis=dict(range=rx, showgrid=False, title="RS-Ratio (Trend)"), yaxis=dict(range=ry, showgrid=False, title="RS-Momentum (Velocity)"))
    fig.add_annotation(x=rx[1], y=ry[1], text="LEADING", showarrow=False, font=dict(size=16, color=c_lead), xanchor="right", yanchor="top")
    fig.add_annotation(x=rx[1], y=ry[0], text="WEAKENING", showarrow=False, font=dict(size=16, color=c_weak), xanchor="right", yanchor="bottom")
    fig.add_annotation(x=rx[0], y=ry[0], text="LAGGING", showarrow=False, font=dict(size=16, color=c_lag), xanchor="left", yanchor="bottom")
    fig.add_annotation(x=rx[0], y=ry[1], text="IMPROVING", showarrow=False, font=dict(size=16, color=c_imp), xanchor="left", yanchor="top")
    return fig
