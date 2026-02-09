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
    """Simple Pivot Check for Fallback"""
    if len(df) < 5: return "NEUTRAL"
    h = df['High'].iloc[-1]; h_prev = df['High'].iloc[-2]
    l = df['Low'].iloc[-1]; l_prev = df['Low'].iloc[-2]
    
    if h > h_prev and l > l_prev: return "HH" # Higher High
    if h < h_prev and l < l_prev: return "LL" # Lower Low
    if h < h_prev and l > l_prev: return "ID" # Inside Day
    return "HL" 

def round_to_03_07(val):
    try:
        int_part = int(val)
        dec_part = val - int_part
        if dec_part < 0.05: final_dec = 0.03
        elif dec_part < 0.50: final_dec = 0.47 
        elif dec_part < 0.95: final_dec = 0.93
        else: final_dec = 0.97
        return int_part + final_dec
    except: return val

def calc_ichimoku(high, low, close):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return span_a, span_b

# --- VSA LOGIC (Optimized & Safe) ---
def calc_smart_money(df):
    """
    Analyzes Volume + Spread + Close Location to detect Intent.
    Includes Safety Checks for Zero Division.
    """
    if len(df) < 20: return "INSUFFICIENT DATA"
    
    try:
        curr = df.iloc[-1]; prev = df.iloc[-2]
        
        # 1. Volume Analysis
        vol = curr.get('Volume', 0)
        # Use pre-calculated VolSMA if available, else roll it
        if 'VolSMA' in df.columns: vol_avg = df['VolSMA'].iloc[-1]
        else: vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
        
        # Safety: Avoid division by zero
        if vol_avg is None or vol_avg == 0: rel_vol = 0
        else: rel_vol = vol / vol_avg
        
        # 2. Spread (Range) Analysis
        high = curr['High']; low = curr['Low']; close = curr['Close']
        spread = high - low
        avg_spread = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        
        if avg_spread is None or avg_spread == 0: rel_spread = 0
        else: rel_spread = spread / avg_spread
        
        # 3. Location of Close (0.0 = Low, 1.0 = High)
        if spread == 0: loc = 0.5
        else: loc = (close - low) / spread
        
        # --- INTERPRETATION ENGINE ---
        
        # A. HIGH VOLUME (Institution Active)
        if rel_vol >= 1.5:
            # Churning: High effort, no result (Tiny Range)
            if rel_spread < 0.75: return "CHURNING (Neutral)"
            
            # Up Moves
            if close > prev['Close']:
                if loc < 0.30: return "UPTHRUST (Trap)" # Sold into strength
                if loc > 0.70: return "IGNITION (Buy)"   # Real buying
                return "RALLY (Strong)"
                
            # Down Moves
            else:
                if loc > 0.70: return "STOPPING (Absorb)" # Institutions buying the dip
                if loc < 0.30: return "DUMPING (Panic)"   # Real selling
                return "DROP (Heavy)"

        # B. LOW VOLUME (No Interest)
        elif rel_vol < 0.6:
            if close > prev['Close']: return "DRIFT (No Demand)"
            return "TEST (No Supply)"
            
        # C. NORMAL VOLUME
        else:
            # Fallback to simple structure
            struct = calc_structure(df)
            if struct == "HH": return "TRENDING UP"
            if struct == "LL": return "TRENDING DOWN"
            return "NORMAL"
            
    except Exception as e:
        return "ERROR"
