import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Titan Strategy", layout="wide")
st.title("🛡️ Titan Strategy v47.4")
st.caption("Institutional Protocol: Strict Logic Match")

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

# --- STYLING ---
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
    
    # PHASE 1: MARKET DIAGNOSTIC
    with st.spinner('Checking Market Vitals...'):
        market_tickers = ["SPY", "IEF", "^VIX"]
        market_data = {}
        
        for t in market_tickers:
            try:
                df = yf.download(t, period="2y", interval="1d", progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    if t in df.columns.get_level_values(0): df.columns = df.columns.droplevel(0)
                    elif t in df.columns.get_level_values(1): df.columns = df.columns.droplevel(1)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                if not df.empty and 'Close' in df.columns: market_data[t] = df
            except: pass
        
        spy = market_data.get("SPY"); ief = market_data.get("IEF"); vix = market_data.get("^VIX")
        mkt_score = 0; vix_val = 0; vix_msg = "Error"
        
        if spy is not None and ief is not None and vix is not None:
            if spy.iloc[-1]['Close'] > calc_sma(spy['Close'], 18).iloc[-1]: mkt_score += 1
            
            aligned = pd.concat([spy['Close'], ief['Close']], axis=1, join='inner')
            ratio = aligned.iloc[:,0] / aligned.iloc[:,1]
            if ratio.iloc[-1] > calc_sma(ratio, 18).iloc[-1]: mkt_score += 1

            vix_val = vix.iloc[-1]['Close']
            if vix_val < 20: mkt_score += 1

            if vix_val > 25: risk_per_trade = 0; vix_msg = "⛔ MARKET LOCK (>25)"
            elif vix_val > 20: risk_per_trade = RISK_UNIT / 2; vix_msg = "⚠️ HALF SIZE (>20)"
            else: risk_per_trade = RISK_UNIT; vix_msg = "✅ FULL SIZE"
        else:
            risk_per_trade = 0

    col1, col2 = st.columns(2)
    score_color = "Bullish" if mkt_score==3 else ("Caution" if mkt_score > 0 else "Bearish")
    col1.metric("Market Score", f"{mkt_score}/3", delta=score_color)
    col2.metric("VIX", f"{vix_val:.2f}", delta=vix_msg, delta_color="inverse")

    # PHASE 2: SCANNER
    with st.spinner('Scanning Assets...'):
        tickers = list(DATA_MAP.keys())
        cache_d = {} 
        cache_d.update(market_data)
        
        for t in tickers:
            if t in cache_d: continue
            try:
                tk = yf.Ticker(t)
                df_d = tk.history(period="5y", interval="1d")
                df_d.index = pd.to_datetime(df_d.index).tz_localize(None)
                if not df_d.empty and 'Close' in df_d.columns: cache_d[t] = df_d
            except: pass

        results = []
        for t, meta in DATA_MAP.items():
            if meta[0] in ["BENCH"]: continue
            if t not in cache_d: continue
            
            # DATA
            df_d = cache_d[t].copy()
            df_d['SMA18'] = calc_sma(df_d['Close'], 18)
            df_d['SMA40'] = calc_sma(df_d['Close'], 40)
            df_d['SMA200'] = calc_sma(df_d['Close'], 200)
            df_d['AD'] = calc_ad(df_d['High'], df_d['Low'], df_d['Close'], df_d['Volume'])
            df_d['AD_SMA18'] = calc_sma(df_d['AD'], 18)
            df_d['AD_SMA40'] = calc_sma(df_d['AD'], 40)
            df_d['VolSMA'] = calc_sma(df_d['Volume'], 18)
            
            # WEEKLY RESAMPLE (Fixed)
            logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df_w = df_d.resample('W-FRI').agg(logic)
            df_w.dropna(subset=['Close'], inplace=True)
            df_w['SMA8'] = calc_sma(df_w['Close'], 8)
            df_w['SMA18'] = calc_sma(df_w['Close'], 18)
            df_w['SMA40'] = calc_sma(df_w['Close'], 40)
            span_a, span_b = calc_ichimoku(df_w['High'], df_w['Low'], df_w['Close'])
            df_w['Cloud_Top'] = pd.concat([span_a, span_b], axis=1).max(axis=1)

            # POINTERS
            dc = df_d.iloc[-1]; dp = df_d.iloc[-2]
            wc = df_w.iloc[-1]; wp = df_w.iloc[-2]

            # RS & AD & VOL
            bench_ticker = meta[1]
            rs_score_pass = False; rs_breakdown = False
            if bench_ticker in cache_d:
                bench_df = cache_d[bench_ticker]
                aligned = pd.concat([df_d['Close'], bench_df['Close']], axis=1, join='inner')
                rs_series = aligned.iloc[:,0] / aligned.iloc[:,1]
                rs_sma18 = calc_sma(rs_series, 18)
                c_rs = rs_series.iloc[-1]; c_rs_sma = rs_sma18.iloc[-1]; p_rs_sma = rs_sma18.iloc[-2]
                rs_score_pass = (c_rs >= rs_sma18.iloc[-1] * 0.995) and (c_rs_sma >= p_rs_sma)
                pp_rs_sma = rs_sma18.iloc[-3]
                rs_breakdown = (c_rs < c_rs_sma) and (c_rs_sma < p_rs_sma) and (p_rs_sma < pp_rs_sma)

            ad_pass = False
            if not pd.isna(dc['AD_SMA18']):
                ad_pass = (dc['AD'] >= dc['AD_SMA18'] * 0.995) and (dc['AD_SMA18'] >= dp['AD_SMA18'])
            
            vol_msg = "NORMAL"
            if dc['Volume'] > (dc['VolSMA'] * 1.5): vol_msg = "SPIKE (Live)"
            elif dp['Volume'] > (dp['VolSMA'] * 1.5): vol_msg = "SPIKE (Prev)"
            elif dc['Volume'] > dc['VolSMA']: vol_msg = "HIGH (Live)"
            elif dp['Volume'] > dp['VolSMA']: vol_msg = "HIGH (Prev)"

            # IMPULSE
            w_uptrend = (wc['Close'] > wc['SMA18']) and (wc['SMA18'] > wc['SMA40']) and (wc['SMA18'] > wp['SMA18'])
            d_health_ok = (dc['Close'] > dc['SMA18']) and (dc['SMA18'] >= dp['SMA18']) and ad_pass
            w_pulse = "NO"
            if w_uptrend: w_pulse = "GOOD" if d_health_ok else "WEAK"
            
            # --- SCORES (CORRECTED) ---
            w_score = 0
            if wc['Close'] > wc['SMA18']: w_score += 1
            if wc['SMA18'] > wp['SMA18']: w_score += 1
            if wc['SMA18'] > wc['SMA40']: w_score += 1
            if wc['Close'] > wc['Cloud_Top']: w_score += 1
            # BUG FIX: Was checking SMA8 > 0. Now checks Price > SMA8
            if wc['Close'] > wc['SMA8']: w_score += 1 
            
            d_chk = {'Price': dc['Close'] > dc['SMA18'], 'Trend': dc['SMA18'] >= dp['SMA18'], 'Align': dc['SMA18'] > dc['SMA40'], 'A/D': ad_pass, 'RS': rs_score_pass}
            d_score = sum(d_chk.values())

            # DECISION
            decision = "AVOID"; reason = "Low Score"
            if w_score >= 4:
                if d_score == 5: decision = "BUY"; reason = "Score 5/5" if w_score==5 else "Score 4/5"
                elif d_score == 4: decision = "SOON" if w_score==5 else "SCOUT"; reason = "D-Score 4"
                elif d_score == 3: decision = "SCOUT"; reason = "Dip Buy"
                else: decision = "WATCH"; reason = "Daily Weak"
            else: decision = "AVOID"; reason = "Weekly Weak"

            # Filters
            w_sma8_pass = wc['Close'] > wc['SMA8']
            w_cloud_pass = wc['Close'] > wc['Cloud_Top']
            
            if not w_sma8_pass: decision = "WATCH"; reason = "BELOW\nW-SMA8"
            elif "SCOUT" in decision and "WEAK" in w_pulse: decision = "WATCH"; reason = "Impulse Weak"
            elif "NO" in w_pulse: decision = "AVOID"; reason = "Impulse NO"
            elif "BUY" in decision and not (dc['Close'] > dc['SMA200']): decision = "SCOUT"; reason = "Below 200MA"
            elif not w_cloud_pass and "BUY" in decision: decision = "WATCH"; reason = "Cloud Fail"
            elif "SCOUT" in decision and not d_chk['Price']: decision = "WATCH"; reason = "Price Low"
            elif rs_breakdown: decision = "WATCH"; reason = "RS BREAK"
            elif risk_per_trade == 0 and ("BUY" in decision or "SCOUT" in decision): decision = "WATCH"; reason = "VIX Lock"

            # Stops
            atr = calc_atr(df_d['High'], df_d['Low'], df_d['Close']).iloc[-1]
            stop_dist = 2.618 * atr
            final_risk = risk_per_trade / 3 if "SCOUT" in decision else risk_per_trade
            shares = int(final_risk / stop_dist) if stop_dist > 0 and ("BUY" in decision or "SCOUT" in decision) else 0
            stop_price = dc['Close'] - stop_dist
            stop_pct = (stop_dist / dc['Close']) * 100 if dc['Close'] else 0

            row = {
                "Sector": meta[0], "Industry": meta[2], "Ticker": t,
                "Weekly SMA8": "PASS" if w_sma8_pass else "FAIL", 
                "Weekly Impulse": w_pulse, 
                "Weekly Score (Max 5)": w_score, "Daily Score (Max 5)": d_score,
                "Structure": "Above 18" if d_chk['Price'] else "BELOW 18",
                "Ichimoku Cloud": "PASS" if w_cloud_pass else "FAIL", "A/D Breadth": "STRONG" if ad_pass else "WEAK",
                "Volume": vol_msg, "Action": decision, "Reasoning": reason,
                "Stop Price": f"${stop_price:.2f} (-{stop_pct:.1f}%)", "Position Size": f"{shares} shares"
            }
            results.append(row)

        df_final = pd.DataFrame(results).sort_values(["Sector", "Action"], ascending=[True, True])
        st.markdown(df_final.style.pipe(style_final).to_html(), unsafe_allow_html=True)
