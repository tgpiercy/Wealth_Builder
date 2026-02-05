import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Titan Strategy", layout="wide")
st.title("🛡️ Titan Strategy v47.8")
st.caption("Institutional Protocol: Deep History (10y) + Detailed Health Metrics")

RISK_UNIT = 2300  

# --- DATA MAP ---
DATA_MAP = {
    # INDICES & CORE
    "SPY": ["BENCH", "SPY", "S&P 500"],
    "DIA": ["BENCH", "SPY", "Dow Jones"],
    "QQQ": ["BENCH", "SPY", "Nasdaq 100"],
    "IWM": ["BENCH", "SPY", "Russell 2000"],
    "IWC": ["BENCH", "SPY", "Micro-Cap"],
    "HXT.TO": ["CANADA", "SPY", "TSX 60 Index"],
    "IBB": ["THEME", "SPY", "Biotech Core"],

    # SECTORS
    "XLB": ["SECTOR", "SPY", "Materials"],
    "XLC": ["SECTOR", "SPY", "Comm Services"],
    "XLE": ["SECTOR", "SPY", "Energy"],
    "XLF": ["SECTOR", "SPY", "Financials"],
    "XLI": ["SECTOR", "SPY", "Industrials"],
    "XLK": ["SECTOR", "SPY", "Technology"],
    "XLV": ["SECTOR", "SPY", "Health Care"],
    "XLY": ["SECTOR", "SPY", "Cons Discret"],
    "XLP": ["SECTOR", "SPY", "Cons Staples"],
    "XLRE": ["SECTOR", "SPY", "Real Estate"],
    "XLU": ["SECTOR", "SPY", "Utilities"],

    # COMMODITIES
    "GLD": ["COMMODITY", "SPY", "Gold Bullion"],
    "SLV": ["COMMODITY", "SPY", "Silver Bullion"],
    "BTC-USD": ["COMMODITY", "SPY", "Bitcoin (USD)"],
    
    # THEMES
    "BOTZ": ["THEME", "SPY", "Robotics & AI"],
    "XBI":  ["THEME", "SPY", "Biotechnology"],
    "ICLN": ["THEME", "SPY", "Clean Energy"],
    "REMX": ["THEME", "SPY", "Rare Earth Metals"],
    "GDX":  ["THEME", "SPY", "Gold Miners"],
    
    # MACRO
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
    def color_pct(val):
        if isinstance(val, str) and '%' in val:
            try:
                num = float(val.strip('%'))
                return 'color: #00ff00; font-weight: bold' if num >= 0 else 'color: #ff4444; font-weight: bold'
            except: return ''
        return ''

    return styler.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#111'), ('color', 'white'), ('font-size', '12px'), ('white-space', 'normal')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '14px'), ('padding', '8px')]}
    ]).set_properties(**{'background-color': '#222', 'color': 'white', 'border-color': '#444'})\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if v in ["BUY", "STRONG BUY"] else ('color: #00ffff; font-weight: bold' if "SCOUT" in v else ('color: #ffaa00' if "SOON" in v else 'color: white')), subset=["Action"])\
      .map(lambda v: 'color: #ff00ff; font-weight: bold' if "SPIKE" in v else ('color: #00ff00' if "HIGH" in v else 'color: #ccc'), subset=["Volume"])\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if "STRONG" in v else ('color: #ff0000' if "WEAK" in v else 'color: #ffaa00'), subset=["A/D Breadth"])\
      .map(lambda v: 'color: #ff0000; font-weight: bold' if "FAIL" in v or "NO" in v else 'color: #00ff00', subset=["Ichimoku Cloud", "Weekly SMA8"])\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if "GOOD" in v else ('color: #ffaa00; font-weight: bold' if "WEAK" in v else 'color: #ff0000; font-weight: bold'), subset=["Weekly Impulse"])\
      .map(lambda v: 'color: #00ff00; font-weight: bold' if v >= 4 else ('color: #ffaa00; font-weight: bold' if v == 3 else 'color: #ff0000; font-weight: bold'), subset=["Weekly Score (Max 5)", "Daily Score (Max 5)"])\
      .map(lambda v: 'color: #ff0000; font-weight: bold' if "BELOW 18" in v else 'color: #00ff00', subset=["Structure"])\
      .map(color_pct, subset=["4W %", "2W %"])\
      .hide(axis='index')

def style_market(styler):
    return styler.set_table_styles([
         {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#333'), ('color', 'white')]},
         {'selector': 'td', 'props': [('text-align', 'center'), ('font-weight', 'bold')]}
    ]).map(lambda v: 'color: #00ff00' if v in ["BULLISH", "RISK ON", "CALM"] else ('color: #ffaa00' if v in ["STABLE", "CAUTION"] else 'color: #ff0000'), subset=["Status"])

# --- EXECUTION ---
if st.button("RUN ANALYSIS", type="primary"):
    
    # --- PHASE 1: MARKET HEALTH (Detailed) ---
    with st.spinner('Checking Market Vitals (SPY, IEF, VIX)...'):
        market_tickers = ["SPY", "IEF", "^VIX"]
        market_data = {}
        for t in market_tickers:
            try:
                tk = yf.Ticker(t)
                df = tk.history(period="2y", interval="1d") # 2y is plenty for market health logic
                df.index = pd.to_datetime(df.index).tz_localize(None)
                if not df.empty and 'Close' in df.columns: market_data[t] = df
            except: pass
        
        spy = market_data.get("SPY"); ief = market_data.get("IEF"); vix = market_data.get("^VIX")
        mkt_score = 0; total_exp = 0; exposure_rows = []
        
        if spy is not None and ief is not None and vix is not None:
            # 1. SPY Trend
            spy_c = spy.iloc[-1]['Close']
            spy_sma18 = calc_sma(spy['Close'], 18).iloc[-1]
            if spy_c > spy_sma18:
                status = "BULLISH"; total_exp += 40
            else: status = "BEARISH"
            exposure_rows.append({
                "Metric": "Trend (SPY > SMA18)", 
                "Value": f"${spy_c:.2f}",
                "Status": status
            })

            # 2. Ratio
            aligned = pd.concat([spy['Close'], ief['Close']], axis=1, join='inner')
            ratio = aligned.iloc[:,0] / aligned.iloc[:,1]
            ratio_c = ratio.iloc[-1]; ratio_sma18 = calc_sma(ratio, 18).iloc[-1]
            
            # Distance from SMA
            dist_pct = ((ratio_c - ratio_sma18) / ratio_sma18) * 100
            
            if ratio_c > ratio_sma18:
                status = "RISK ON"; total_exp += 40
            elif ratio_c >= ratio_sma18 * 0.99:
                status = "STABLE"; total_exp += 40
            else: status = "RISK OFF"
            
            exposure_rows.append({
                "Metric": "Power (SPY:IEF)", 
                "Value": f"{ratio_c:.3f} ({dist_pct:+.2f}%)",
                "Status": status
            })

            # 3. VIX
            vix_c = vix.iloc[-1]['Close']
            if vix_c < 20:
                status = "CALM"; total_exp += 20
            elif vix_c < 25: status = "CAUTION"
            else: status = "PANIC"
            exposure_rows.append({
                "Metric": "VIX (<20)", 
                "Value": f"{vix_c:.2f}",
                "Status": status
            })
            
            exposure_rows.append({"Metric": "TOTAL EXPOSURE", "Value": "---", "Status": f"{total_exp}%"})
            
            # Display
            df_mkt = pd.DataFrame(exposure_rows)
            st.subheader(f"📊 Market Health: {total_exp}% Invested")
            st.markdown(df_mkt.style.pipe(style_market).to_html(), unsafe_allow_html=True)
            st.write("---")
            
            risk_per_trade = RISK_UNIT * 0.5 if total_exp <= 40 else RISK_UNIT
            if total_exp == 0: risk_per_trade = 0
            
        else:
            risk_per_trade = 0
            st.error("Market Data Failed to Load")

    # --- PHASE 2: MASTER SCANNER (10-Year Depth) ---
    with st.spinner('Running Titan Protocol (10-Year Deep Scan)...'):
        tickers = list(DATA_MAP.keys())
        cache_d = {}
        # We re-fetch market data with 10y depth for the scanner if needed, or just use separate calls.
        # Ideally, we fetch everything with 10y now to be safe.
        
        for t in tickers:
            if t in cache_d: continue
            try:
                tk = yf.Ticker(t)
                # CHANGE: Back to 10y for maximum precision on 200MA and Volatility
                df = tk.history(period="10y", interval="1d") 
                df.index = pd.to_datetime(df.index).tz_localize(None)
                if not df.empty and 'Close' in df.columns: cache_d[t] = df
            except: pass

        results = []
        for t, meta in DATA_MAP.items():
            if t not in cache_d: continue
            if meta[0] == "BENCH" and t not in ["DIA", "QQQ", "IWM", "IWC", "HXT.TO"]: continue 
            
            # DATA
            df_d = cache_d[t].copy()
            df_d['SMA18'] = calc_sma(df_d['Close'], 18)
            df_d['SMA40'] = calc_sma(df_d['Close'], 40)
            df_d['SMA200'] = calc_sma(df_d['Close'], 200)
            df_d['AD'] = calc_ad(df_d['High'], df_d['Low'], df_d['Close'], df_d['Volume'])
            df_d['AD_SMA18'] = calc_sma(df_d['AD'], 18)
            df_d['AD_SMA40'] = calc_sma(df_d['AD'], 40)
            df_d['VolSMA'] = calc_sma(df_d['Volume'], 18)
            
            # WEEKLY RESAMPLE
            logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df_w = df_d.resample('W-FRI').agg(logic)
            df_w.dropna(subset=['Close'], inplace=True)
            df_w['SMA8'] = calc_sma(df_w['Close'], 8)
            df_w['SMA18'] = calc_sma(df_w['Close'], 18)
            df_w['SMA40'] = calc_sma(df_w['Close'], 40)
            span_a, span_b = calc_ichimoku(df_w['High'], df_w['Low'], df_w['Close'])
            df_w['Cloud_Top'] = pd.concat([span_a, span_b], axis=1).max(axis=1)

            # MOMENTUM (4W & 2W)
            mom_4w = ""; mom_2w = ""
            if len(df_w) >= 5:
                curr = df_w.iloc[-1]['Close']
                prev2 = df_w.iloc[-3]['Close']
                prev4 = df_w.iloc[-5]['Close']
                chg2 = ((curr / prev2) - 1) * 100
                chg4 = ((curr / prev4) - 1) * 100
                mom_2w = f"{chg2:.1f}%"
                mom_4w = f"{chg4:.1f}%"

            # POINTERS
            dc = df_d.iloc[-1]; dp = df_d.iloc[-2]
            wc = df_w.iloc[-1]; wp = df_w.iloc[-2]

            # RS
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

            # AD & VOL
            ad_pass = False
            if not pd.isna(dc['AD_SMA18']):
                ad_pass = (dc['AD'] >= dc['AD_SMA18'] * 0.995) and (dc['AD_SMA18'] >= dp['AD_SMA18'])
            
            vol_msg = "NORMAL"
            if dc['Volume'] > (dc['VolSMA'] * 1.5): vol_msg = "SPIKE (Live)"
            elif dp['Volume'] > (dp['VolSMA'] * 1.5): vol_msg = "SPIKE (Prev)"
            elif dc['Volume'] > dc['VolSMA']: vol_msg = "HIGH (Live)"
            elif dp['Volume'] > dp['VolSMA']: vol_msg = "HIGH (Prev)"

            # SCORES
            w_score = 0
            if wc['Close'] > wc['SMA18']: w_score += 1
            if wc['SMA18'] > wp['SMA18']: w_score += 1
            if wc['SMA18'] > wc['SMA40']: w_score += 1
            if wc['Close'] > wc['Cloud_Top']: w_score += 1
            if wc['Close'] > wc['SMA8']: w_score += 1 
            
            d_chk = {'Price': dc['Close'] > dc['SMA18'], 'Trend': dc['SMA18'] >= dp['SMA18'], 'Align': dc['SMA18'] > dc['SMA40'], 'A/D': ad_pass, 'RS': rs_score_pass}
            d_score = sum(d_chk.values())
            
            # IMPULSE & LOGIC
            w_uptrend = (wc['Close'] > wc['SMA18']) and (wc['SMA18'] > wc['SMA40']) and (wc['SMA18'] > wp['SMA18'])
            d_health_ok = (dc['Close'] > dc['SMA18']) and (dc['SMA18'] >= dp['SMA18']) and ad_pass
            w_pulse = "NO"; w_pulse = "GOOD" if w_uptrend and d_health_ok else ("WEAK" if w_uptrend else "NO")

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
                "4W %": mom_4w, "2W %": mom_2w,
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
        cols = ["Sector", "Ticker", "4W %", "2W %", "Weekly SMA8", "Weekly Impulse", "Weekly Score (Max 5)", "Daily Score (Max 5)", "Structure", "Ichimoku Cloud", "A/D Breadth", "Volume", "Action", "Reasoning", "Stop Price", "Position Size"]
        st.markdown(df_final[cols].style.pipe(style_final).to_html(), unsafe_allow_html=True)
