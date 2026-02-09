import pandas as pd

# --- HELPER FORMATTERS ---
def fmt_delta(val):
    color = "#00ff00" if val >= 0 else "#ff4444"
    return f"<span style='color:{color}'>{val:+.2f}</span>"

# --- COLOR LOGIC FUNCTIONS ---
def color_action(val):
    val = str(val).upper()
    if 'BUY' in val: return 'color: #00ff00; font-weight: bold'
    if 'SCOUT' in val: return 'color: #00bfff; font-weight: bold'
    if 'WATCH' in val: return 'color: #ffff00; font-weight: bold'
    if 'CAUTION' in val: return 'color: #ffaa00; font-weight: bold'
    if 'AVOID' in val: return 'color: #ff4444'
    return 'color: white'

def color_rotation(val):
    val = str(val).upper()
    if 'LEADING' in val: return 'color: #00ff00'
    if 'WEAKENING' in val: return 'color: #ffaa00'
    if 'LAGGING' in val: return 'color: #ff4444'
    if 'IMPROVING' in val: return 'color: #00bfff'
    return 'color: white'

def color_pass_fail(val):
    val = str(val).upper()
    if 'PASS' in val: return 'color: #00ff00'
    if 'FAIL' in val: return 'color: #ff4444'
    return 'color: white'

def color_impulse(val):
    val = str(val).upper()
    if 'GOOD' in val: return 'color: #00ff00'
    if 'NO' in val: return 'color: #ff4444'
    return 'color: white'

def color_structure(val):
    val = str(val).upper()
    if 'BULLISH' in val: return 'color: #00ff00'
    if 'BEARISH' in val: return 'color: #ff4444'
    if 'ABOVE' in val: return 'color: #00ff00'
    if 'BELOW' in val: return 'color: #ff4444'
    return 'color: white'

def color_score(val):
    try:
        v = int(val)
        if v == 5: return 'color: #00ff00; font-weight: bold'
        if v == 4: return 'color: #ccff66'
        if v == 3: return 'color: #ffff00'
        if v <= 2: return 'color: #ff4444'
    except: pass
    return 'color: white'

def color_ad_breadth(val):
    val = str(val).upper()
    if 'ACCUMULATION' in val: return 'color: #00ff00; font-weight: bold'
    if 'NEUTRAL' in val: return 'color: #ffaa00'
    if 'DISTRIBUTION' in val: return 'color: #ff4444'
    if 'STRONG' in val: return 'color: #00ff00'
    if 'WEAK' in val: return 'color: #ff4444'
    return 'color: white'

# --- MAIN STYLER ---
def style_final(styler):
    # 1. Apply Column Specific Coloring
    styler.applymap(color_rotation, subset=['Rotation'])
    styler.applymap(color_pass_fail, subset=['Weekly<br>SMA8'])
    styler.applymap(color_impulse, subset=['Weekly<br>Impulse'])
    styler.applymap(color_score, subset=['Weekly<br>Score', 'Daily<br>Score'])
    styler.applymap(color_structure, subset=['Structure', 'Daily<br>Score']) 
    styler.applymap(color_ad_breadth, subset=['A/D Breadth']) # New Column
    styler.applymap(color_action, subset=['Action'])

    # 2. Global Table Formatting
    styler.hide(axis='index')
    styler.set_table_styles([
        {'selector': 'thead th', 'props': [
            ('background-color', '#1E1E1E'),
            ('color', '#ffffff'),
            ('font-size', '12px'),
            ('text-align', 'center'),
            ('border-bottom', '2px solid #444')
        ]},
        {'selector': 'tbody td', 'props': [
            ('font-size', '13px'),
            ('padding', '8px 12px'),
            ('text-align', 'center'),
            ('border-bottom', '1px solid #333')
        ]},
        {'selector': 'tbody tr:hover', 'props': [('background-color', '#2C2C2C')]}
    ])
    return styler

# --- PORTFOLIO STYLER ---
def style_portfolio(styler):
    styler.hide(axis='index')
    styler.set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', '#333'), ('color', 'white')]},
        {'selector': 'tbody td', 'props': [('text-align', 'center')]}
    ])
    return styler

# --- DAILY HEALTH STYLER ---
def style_daily_health(styler):
    styler.hide(axis='index')
    styler.hide(axis='columns') # Hides headers for cleaner look if desired, or remove this line
    return styler
