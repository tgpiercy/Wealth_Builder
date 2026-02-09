import pandas as pd
import numpy as np

def calc_sma(series, window):
    return series.rolling(window=window).mean()

def calc_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_ad(high, low, close, volume):
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * volume
    return mfv.cumsum()

def calc_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def calc_rising(series, lookback=2):
    if len(series) < lookback + 1: return False
    return series.iloc[-1] > series.iloc[-1 - lookback]

def calc_structure(df):
    # Pivot High/Low Logic (Simplified for Trend)
    if len(df) < 5: return "NEUTRAL"
    c = df['Close'].iloc[-1]; c_prev = df['Close'].iloc[-2]
    h = df['High'].iloc[-1]; h_prev = df['High'].iloc[-2]
    l = df['Low'].iloc[-1]; l_prev = df['Low'].iloc[-2]
    
    if h > h_prev and l > l_prev: return "HH" # Higher High
    if h < h_prev and l < l_prev: return "LL" # Lower Low
    if h < h_prev and l > l_prev: return "ID" # Inside Day
    return "HL" # Mixed/Higher Low

def round_to_03_07(val):
    """Smart Stops: Round to nearest .03 or .07 to avoid round number clusters."""
    int_part = int(val)
    dec_part = val - int_part
    if dec_part < 0.05: final_dec = 0.03
    elif dec_part < 0.50: final_dec = 0.47 
    elif dec_part < 0.95: final_dec = 0.93
    else: final_dec = 0.97
    return int_part + final_dec

def calc_ichimoku(high, low, close):
    # Standard settings: 9, 26, 52
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return span_a, span_b

# --- THE MISSING FUNCTION (VSA LOGIC) ---
def calc_smart_money(df):
    """
    Analyzes Volume + Spread + Close Location to detect Intent.
    """
    if len(df) < 20: return "INSUFFICIENT DATA"
    
    # 1. Get Current Bar Data
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    vol = curr['Volume']
    # Ensure VolSMA exists in DF before accessing
    if 'VolSMA' in df.columns:
        vol_avg = df['VolSMA'].iloc[-1]
    else:
        vol_avg = vol # Fallback
    
    high = curr['High']; low = curr['Low']; close = curr['Close']
    
    # 2. Calculate Spread (Range)
    spread = high - low
    avg_spread = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
    
    # 3. Relative Metrics
    rel_vol = vol / vol_avg if vol_avg > 0 else 0
    rel_spread = spread / avg_spread if avg_spread > 0 else 0
    
    # 4. Close Location (0.0 = Low, 1.0 = High)
    loc = (close - low) / spread if spread > 0 else 0.5
    
    # --- LOGIC TREE ---
    
    # SCENARIO A: High Volume (Institutional Participation)
    if rel_vol >= 1.5:
        # 1. Churning (High Vol + Tiny Range) -> Indecision
        if rel_spread < 0.75:
            return "CHURNING (Neutral)"
            
        # 2. Up Move
        if close > prev['Close']:
            if loc < 0.30: return "UPTHRUST (Trap)" # Sold into rally
            if loc > 0.70: return "IGNITION (Buy)"   # Strong buying
            return "RALLY (Strong)"
            
        # 3. Down Move
        else:
            if loc > 0.70: return "STOPPING (Absorb)" # Bought into dip
            if loc < 0.30: return "DUMPING (Panic)"   # Panic selling
            return "DROP (Heavy)"

    # SCENARIO B: Low Volume (No Interest)
    elif rel_vol < 0.6:
        if close > prev['Close']: return "DRIFT (No Demand)"
        else: return "TEST (No Supply)"
        
    # SCENARIO C: Normal Volume -> Use Structure
    else:
        struct = calc_structure(df)
        if struct == "HH": return "TRENDING UP"
        if struct == "LL": return "TRENDING DOWN"
        return "CONSOLIDATING"
