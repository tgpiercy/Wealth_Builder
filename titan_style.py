def style_final(styler):
    def color_rotation(val):
        if "LEADING" in val: return 'color: #00FF00; font-weight: bold'
        if "WEAKENING" in val: return 'color: #FFFF00; font-weight: bold'
        if "LAGGING" in val: return 'color: #FF4444; font-weight: bold'
        if "IMPROVING" in val: return 'color: #00BFFF; font-weight: bold'
        return ''
    def color_inst(val):
        if "ACCUMULATION" in val or "BREAKOUT" in val: return 'color: #00FF00; font-weight: bold' 
        if "CAPITULATION" in val: return 'color: #00BFFF; font-weight: bold'       
        if "DISTRIBUTION" in val or "LIQUIDATION" in val: return 'color: #FF4444; font-weight: bold' 
        if "SELLING" in val: return 'color: #FFA500; font-weight: bold'      
        return 'color: #CCFFCC' if "HH" in val else ('color: #FFCCCC' if "LL" in val else 'color: #888888')
    def highlight_ticker_row(row):
        styles = ['' for _ in row.index]
        if 'Ticker' not in row.index: return styles
        idx = row.index.get_loc('Ticker'); act = str(row.get('Action', '')).upper()
        if "AVOID" in act: pass
        elif "BUY" in act: styles[idx] = 'background-color: #006600; color: white; font-weight: bold'
        elif "SCOUT" in act: styles[idx] = 'background-color: #005555; color: white; font-weight: bold'
        elif "SOON" in act: styles[idx] = 'background-color: #CC5500; color: white; font-weight: bold'
        elif "CAUTION" in act: styles[idx] = 'background-color: #AA4400; color: white; font-weight: bold'
        return styles
    
    return styler.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#111'), ('color', 'white'), ('font-size', '12px')]}, {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '14px'), ('padding', '8px')]}]).set_properties(**{'background-color': '#222', 'color': 'white', 'border-color': '#444'}).apply(highlight_ticker_row, axis=1).map(lambda v: 'color: #00ff00; font-weight: bold' if v in ["BUY", "STRONG BUY"] else ('color: #00ffff; font-weight: bold' if "SCOUT" in v else ('color: #ffaa00; font-weight: bold' if v in ["SOON", "CAUTION"] else 'color: white')), subset=["Action"]).map(lambda v: 'color: #ff00ff; font-weight: bold' if "SPIKE" in v else ('color: #00ff00' if "HIGH" in v else 'color: #ccc'), subset=["Volume"]).map(lambda v: 'color: #00ff00; font-weight: bold' if "STRONG" in v else ('color: #ff0000' if "WEAK" in v else 'color: #ffaa00'), subset=["A/D Breadth"]).map(lambda v: 'color: #ff0000; font-weight: bold' if "FAIL" in v or "NO" in v else 'color: #00ff00', subset=["Ichimoku<br>Cloud", "Weekly<br>SMA8"]).map(lambda v: 'color: #00ff00; font-weight: bold' if "GOOD" in v else ('color: #ffaa00; font-weight: bold' if "WEAK" in v else 'color: #ff0000; font-weight: bold'), subset=["Weekly<br>Impulse"]).map(lambda v: 'color: #00ff00; font-weight: bold' if v >= 4 else ('color: #ffaa00; font-weight: bold' if v == 3 else 'color: #ff0000; font-weight: bold'), subset=["Weekly<br>Score", "Daily<br>Score"]).map(lambda v: 'color: #ff0000; font-weight: bold' if "BELOW 18" in v else 'color: #00ff00', subset=["Structure"]).map(color_rotation, subset=["Rotation"]).map(color_inst, subset=["Institutional<br>Activity"]).hide(axis='index')

def style_daily_health(styler):
    def color_status(v):
        if "PASS" in v or "NORMAL" in v or "CAUTIOUS" in v or "RISING" in v or "AGGRESSIVE" in v: return 'color: #00ff00; font-weight: bold'
        if "FAIL" in v or "PANIC" in v or "DEFENSIVE" in v or "FALLING" in v or "CASH" in v: return 'color: #ff4444; font-weight: bold'
        return 'color: white; font-weight: bold'
    return styler.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left'), ('background-color', '#111'), ('color', 'white'), ('font-size', '14px')]}, {'selector': 'td', 'props': [('text-align', 'left'), ('font-size', '14px'), ('padding', '8px')]}]).set_properties(**{'background-color': '#222', 'border-color': '#444'}).set_properties(subset=['Indicator'], **{'color': 'white', 'font-weight': 'bold'}).map(color_status, subset=['Status']).hide(axis='index')

def style_portfolio(styler):
    def color_pl(val):
        try: return 'color: #00ff00; font-weight: bold' if float(val.strip('%').replace('+','')) >= 0 else 'color: #ff4444; font-weight: bold'
        except: return ''
    def color_pl_dol(val):
        try: return 'color: #00ff00; font-weight: bold' if float(val.strip('$').replace('+','').replace(',','')) >= 0 else 'color: #ff4444; font-weight: bold'
        except: return ''
    def color_action(val): return 'color: #ff0000; font-weight: bold; background-color: #220000' if "EXIT" in val else ('color: #00ff00; font-weight: bold' if "HOLD" in val else 'color: #ffffff')
    return styler.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#111'), ('color', 'white')]}, {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '14px')]}]).map(color_pl, subset=["% Return"]).map(color_pl_dol, subset=["Gain/Loss ($)"]).map(color_action, subset=["Audit Action"]).hide(axis='index')

def fmt_delta(val): return f"-${abs(val):,.2f}" if val < 0 else f"${val:,.2f}"

