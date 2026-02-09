import pandas as pd
import numpy as np

def calc_sma(series, length): 
    return series.rolling(window=length).mean()

def calc_ad(high, low, close, volume):
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0.0)
    mfv = mfm * volume
    return mfv.cumsum()

def calc_ichimoku(high, low, close):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return span_a, span_b

def calc_atr(high, low, close, length=14):
    try:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(com=length-1, adjust=False).mean()
    except:
        return pd.Series(0, index=close.index)

def calc_rsi(series, length=14):
    try:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=length-1, adjust=False).mean()
        avg_loss = loss.ewm(com=length-1, adjust=False).mean()
        rs = avg_gain / avg_loss
        rs = rs.fillna(0)
        return 100 - (100 / (1 + rs))
    except:
        return pd.Series(50, index=series.index)

def calc_rising(series, length=2):
    """
    STRICT PINE SCRIPT PARITY (ta.rising)
    Returns True ONLY if values have increased consecutively for 'length' bars.
    ta.rising(x, 2) --> x[0] > x[1] AND x[1] > x[2]
    """
    if len(series) < length + 1: return False
    
    is_rising = True
    # Iterate backwards from current (-1) to length
    for i in range(1, length + 1):
        curr = series.iloc[-i]
        prev = series.iloc[-(i + 1)]
        if curr <= prev:  # Strict: If any bar is flat or down, fail.
            is_rising = False
            break
    return is_rising

def calc_structure(df, deviation_pct=0.035):
    if len(df) < 50: return "None"
    pivots = []; trend = 1; last_val = df['Close'].iloc[0]; pivots.append((0, last_val, 1))
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        if trend == 1:
            if price > last_val:
                last_val = price
                if pivots[-1][2] == 1: pivots[-1] = (i, price, 1)
                else: pivots.append((i, price, 1))
            elif price < last_val * (1 - deviation_pct):
                trend = -1; last_val = price; pivots.append((i, price, -1))
        else:
            if price < last_val:
                last_val = price
                if pivots[-1][2] == -1: pivots[-1] = (i, price, -1)
                else: pivots.append((i, price, -1))
            elif price > last_val * (1 + deviation_pct):
                trend = 1; last_val = price; pivots.append((i, price, 1))
    if len(pivots) < 3: return "Range"
    return ("HH" if pivots[-1][1] > pivots[-3][1] else "LH") if pivots[-1][2] == 1 else ("LL" if pivots[-1][1] < pivots[-3][1] else "HL")

def round_to_03_07(price):
    if pd.isna(price): return 0.0
    whole = int(price)
    candidates = [c for c in [whole + 0.03, whole + 0.07, (whole - 1) + 0.97, (whole - 1) + 0.93] if c > 0]
    return min(candidates, key=lambda x: abs(x - price)) if candidates else price
