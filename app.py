import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Titan Strategy", layout="wide")
st.title("🛡️ Titan Strategy v46.6")
st.caption("Institutional Protocol: Trend + Volatility + Relative Strength")

RISK_UNIT = 2300  

# --- DATA MAP ---
DATA_MAP = {
    "XLB": ["SECTOR", "SPY", "Materials"],
    "XLC": ["SECTOR", "SPY", "Comm Services"],
    "XLE": ["SECTOR", "SPY", "Energy"],
    "XLF": ["SECTOR", "SPY", "Financials"],
    "XLI": ["SECTOR", "SPY", "Industrials"],
    "XLK": ["SECTOR", "SPY", "Technology"],
    "XLV": ["SECTOR", "SPY", "Health Care"],
    "XLY": ["SECTOR", "SPY", "Cons Discret"],
    "HXT.TO": ["CANADA", "SPY", "TSX 60 Index"],
    "GLD": ["COMMODITY", "SPY", "Gold Bullion"],
    "SLV": ["COMMODITY", "SPY", "Silver Bullion"],
    "BTC-USD": ["COMMODITY", "SPY", "Bitcoin (USD)"],
    "BOTZ": ["THEME", "SPY", "Robotics & AI"],
    "XBI":  ["THEME", "SPY", "Biotechnology"],
    "ICLN": ["THEME", "SPY", "Clean Energy"],
    "REMX": ["THEME", "SPY", "Rare Earth Metals"],
    "GDX":  ["THEME", "SPY", "Gold Miners"],
    "SPY": ["BENCH", "SPY", "S&P 500"],
    "IEF": ["BENCH", "SPY", "7-10 Year Treasuries"],
    "RSP": ["BENCH", "SPY", "S&P 500 Equal Weight"], 
    "^VIX": ["BENCH", "SPY", "VIX Volatility Index"]
}

# --- CALCULATIONS ---
def calc_sma(series, length):
    return series.rolling(window=length).mean()

def calc_ad(high, low, close, volume):
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)
    mfv = mfm * volume
    return mfv.cumsum()

def calc_ichimoku(high, low, close):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return span_a, span_b

def calc_atr(high, low, close, length=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean()

# --- STYLING FUNCTION ---
def style_final(styler):
    return styler.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#111'), ('color', 'white'), ('font-size', '12px')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '14px'), ('padding', '8px')]}
    ]).set_properties(**{'background-color': '#222', 'color': 'white', 'border-color': '#444'})\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if v in ["BUY", "STRONG BUY"] else ('color: #00ffff; font-weight: bold' if "SCOUT" in v else ('color: #ffaa00' if "SOON" in v else 'color: white')), subset=["Action"])\
      .map(lambda v: 'color: #ff00ff; font-weight: bold' if "SPIKE" in v else ('color: #00ff00' if "HIGH" in v else 'color: #ccc'), subset=["Volume"])\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if "STRONG" in v else ('color: #ff0000' if "WEAK" in v else 'color: #ffaa00'), subset=["A/D Breadth"])\
      .map(lambda v: 'color: #ff0000; font-weight: bold' if "FAIL" in v or "NO" in v else 'color: #00ff00', subset=["Ichimoku Cloud", "Weekly SMA8"])\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if "GOOD" in v else ('color: #ffaa00; font-weight: bold' if "WEAK" in v else 'color: #ff0000; font-weight: bold'), subset=["Weekly Impulse"])\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if v >= 4 else ('color: #ffaa00; font-weight: bold' if v == 3 else 'color: #ff0000; font-weight: bold'), subset=["Weekly Score (Max 5)", "Daily Score (Max 5)"])\
      .map(lambda v: 'color: #ff0000; font-weight: bold' if "BELOW 18" in v else 'color: #00ff00', subset=["Structure"])\
      .hide(axis='index')

# --- EXECUTION ---
if st.button("RUN ANALYSIS", type="primary"):
    with st.spinner('Fetching Data & Crunching Numbers...'):
        tickers = list(DATA_MAP.keys())
        data_cache = {}

        # 1. ROBUST DATA FETCHING
        for t in tickers:
            try:
                # Fetch 10y to ensure moving averages are stable
                df = yf.download(t, period="10y", interval="1d", progress=False, auto_adjust=True)
                
                # --- FIX: COLUMN CLEANER ---
                # Check if columns are MultiIndex (e.g., ('AAPL', 'Close'))
                if isinstance(df.columns, pd.MultiIndex):
                    # If the ticker name is in the columns, drop that level
                    if t in df.columns.get_level_values(0):
                         df.columns = df.columns.droplevel(0)
                    # Backup: Try dropping level 1 if level 0 failed
                    elif t in df.columns.get_level_values(1):
                         df.columns = df.columns.droplevel(1)
                
                # --- FIX: TIMEZONE ---
                # Strip timezones to prevent index mismatch errors
                df.index = pd.to_datetime(df.index).tz_localize(None)

                if not df.empty and 'Close' in df.columns:
                     data_cache[t] = df
            except Exception as e:
                print(f"Error fetching {t}: {e}")

        # 2. CALCULATION ENGINE
        def get_titan_score(ticker, timeframe='D'):
            try:
                if ticker not in data_cache: return 0, {}, 0, False, "NORMAL", "NEUTRAL", "NO", False, False, 0, 0, False
                df_raw = data_cache[ticker].copy()
                bench_ticker = DATA_MAP[ticker][1]
                
                # PREPARE DATA
                df_w = pd.DataFrame()
                df_w['Close'] = df_raw['Close'].resample('W').last()
                df_w['SMA18'] = calc_sma(df_w['Close'], 18)
                df_w['SMA40'] = calc_sma(df_w['Close'], 40)
                
                df_d = df_raw.copy()
                df_d['SMA8'] = calc_sma(df_d['Close'], 8)
                df_d['SMA18'] = calc_sma(df_d['Close'], 18)
                df_d['SMA40'] = calc_sma(df_d['Close'], 40)
                df_d['SMA200'] = calc_sma(df_d['Close'], 200)
                df_d['AD'] = calc_ad(df_d['High'], df_d['Low'], df_d['Close'], df_d['Volume'])
                df_d['AD_SMA18'] = calc_sma(df_d['AD'], 18)
                df_d['AD_SMA40'] = calc_sma(df_d['AD'], 40)
                df_d['VolSMA'] = calc_sma(df_d['Volume'], 18)
                
                if timeframe == 'W':
                    df = df_w.copy()
                    df['High'] = df_raw['High'].resample('W').max()
                    df['Low'] = df_raw['Low'].resample('W').min()
                    df['Close'] = df_raw['Close'].resample('W').last()
                    df['Volume'] = df_raw['Volume'].resample('W').sum()
                    df['VolSMA'] = calc_sma(df['Volume'], 18)
                else:
                    df = df_d.copy()

                # RS & BANDS
                if bench_ticker in data_cache:
                    bench_df = data_cache[bench_ticker]
                    # Ensure alignment of dates
                    aligned = pd.concat([df_d['Close'], bench_df['Close']], axis=1, join='inner')
                    rs_series = aligned.iloc[:,0] / aligned.iloc[:,1]
                    rs_sma18 = calc_sma(rs_series, 18)
                    
                    upper_band = rs_sma18 * 1.005
                    lower_band = rs_sma18 * 0.995
                    
                    c_rs = rs_series.iloc[-1]
                    c_rs_sma = rs_sma18.iloc[-1]
                    p_rs_sma = rs_sma18.iloc[-2]
                    pp_rs_sma = rs_sma18.iloc[-3]
                    
                    rs_in_zone = c_rs >= lower_band.iloc[-1]
                    rs_not_down = c_rs_sma >= p_rs_sma
                    rs_score_pass = rs_in_zone and rs_not_down
                    rs_breakdown = (c_rs < c_rs_sma) and (c_rs_sma < p_rs_sma) and (p_rs_sma < pp_rs_sma)
                else:
                    rs_score_pass = False
                    rs_breakdown = False

                # AD LOGIC
                c_ad = df['AD'].iloc[-1]
                c_ad_sma = df['AD_SMA18'].iloc[-1]
                p_ad_sma = df['AD_SMA18'].iloc[-2]
                ad_lower_band = c_ad_sma * 0.995
                ad_score_pass = (c_ad >= ad_lower_band) and (c_ad_sma >= p_ad_sma)

                # IMPULSE (Composite)
                wc = df_w.iloc[-1]; wp = df_w.iloc[-2]
                w_uptrend = (wc['Close'] > wc['SMA18']) and (wc['SMA18'] > wc['SMA40']) and (wc['SMA18'] > wp['SMA18'])
                dc = df_d.iloc[-1]; dp = df_d.iloc[-2]
                d_health_ok = (dc['Close'] > dc['SMA18']) and (dc['SMA18'] >= dp['SMA18']) and ad_score_pass
                
                pulse_status = "NO"
                if w_uptrend:
                    pulse_status = "GOOD" if d_health_ok else "WEAK"
                    
                # SCORE
                c = df.iloc[-1]; p = df.iloc[-2]
                chk = {'Price': c['Close'] > c['SMA18'], 'Trend': c['SMA18'] >= p['SMA18'], 'Align': c['SMA18'] > c['SMA40'], 'A/D': ad_score_pass, 'RS': rs_score_pass}
                score = sum(chk.values())
                
                ad_status = "STRONG" if ad_score_pass else "WEAK"
                span_a, span_b = calc_ichimoku(df['High'], df['Low'], df['Close'])
                cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1).iloc[-1]
                cloud_ok = c['Close'] > cloud_top if timeframe == 'W' else True
                atr_val = calc_atr(df['High'], df['Low'], df['Close']).iloc[-1]
                vol_status = "SPIKE" if c['Volume'] > (c['VolSMA'] * 1.5) else ("HIGH" if c['Volume'] > c['VolSMA'] else "NORMAL")

                return score, chk, atr_val, cloud_ok, vol_status, ad_status, pulse_status, c['Close'] > c['SMA200'], c['Close'] > c['SMA8'], c['Close'], atr_val, rs_breakdown
            except: return 0, {}, 0, False, "NORMAL", "NEUTRAL", "NO", False, False, 0, 0, False

        # MARKET HEALTH
        spy = data_cache.get("SPY"); ief = data_cache.get("IEF"); vix = data_cache.get("^VIX")
        
        if spy is not None and ief is not None and vix is not None:
            aligned = pd.concat([spy['Close'], ief['Close']], axis=1, join='inner')
            ratio = aligned.iloc[:,0] / aligned.iloc[:,1]
            ratio_sma18 = calc_sma(ratio, 18)
            
            mkt_score = 0
            if spy.iloc[-1]['Close'] > calc_sma(spy['Close'], 18).iloc[-1]: mkt_score += 1
            if ratio.iloc[-1] > ratio_sma18.iloc[-1]: mkt_score += 1
            if vix.iloc[-1]['Close'] < 20: mkt_score += 1
            
            vix_val = vix.iloc[-1]['Close']
            if vix_val > 25: risk_per_trade = 0; vix_msg = "⛔ MARKET LOCK (>25)"
            elif vix_val > 20: risk_per_trade = RISK_UNIT / 2
