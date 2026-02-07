import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(layout="wide", page_title="Sector Rotation Matrix")

st.title("🔄 Sector Rotation Matrix (RRG Logic)")
st.caption("Institutional Money Flow Tracker")

# 1. Define Universe
SECTORS = {
    "XLK": "Technology", 
    "XLF": "Financials", 
    "XLE": "Energy",
    "XLV": "Health Care", 
    "XLY": "Cons. Discret", 
    "XLP": "Cons. Staples",
    "XLI": "Industrials", 
    "XLC": "Comm. Services", 
    "XLU": "Utilities",
    "XLB": "Materials", 
    "XLRE": "Real Estate"
}
BENCHMARK = "SPY"

if st.button("🔄 Scan Sectors"):
    with st.spinner("Analyzing Money Flow..."):
        tickers = list(SECTORS.keys()) + [BENCHMARK]
        # Fetch 3mo of data to calculate trends
        data = yf.download(tickers, period="6mo", progress=False)['Close']
        
        if not data.empty:
            # 2. Calculate Returns
            curr = data.iloc[-1]
            prev_1w = data.iloc[-6]  # 1 Week ago
            prev_1m = data.iloc[-22] # 1 Month ago
            prev_3m = data.iloc[-65] # 3 Months ago
            
            # % Change
            chg_1w = ((curr - prev_1w) / prev_1w) * 100
            chg_1m = ((curr - prev_1m) / prev_1m) * 100
            chg_3m = ((curr - prev_3m) / prev_3m) * 100
            
            # Benchmark Performance
            spy_1w = chg_1w[BENCHMARK]
            spy_1m = chg_1m[BENCHMARK]
            spy_3m = chg_3m[BENCHMARK]
            
            results = []
            for t in SECTORS:
                # 3. Calculate Alpha (Relative Strength)
                # Positive Alpha = Outperforming SPY
                alpha_1w = chg_1w[t] - spy_1w
                alpha_1m = chg_1m[t] - spy_1m
                alpha_3m = chg_3m[t] - spy_3m
                
                # 4. RRG Classification Logic
                # Leading: Winning Trend + Winning Momentum
                # Weakening: Winning Trend + Losing Momentum
                # Improving: Losing Trend + Winning Momentum
                # Lagging: Losing Trend + Losing Momentum
                
                status = "LAGGING" # Default
                if alpha_1m > 0 and alpha_1w > 0: status = "LEADING"
                elif alpha_1m > 0 and alpha_1w < 0: status = "WEAKENING"
                elif alpha_1m < 0 and alpha_1w > 0: status = "IMPROVING"
                
                results.append({
                    "Ticker": t,
                    "Sector": SECTORS[t],
                    "Price": curr[t],
                    "Alpha 1W": alpha_1w,
                    "Alpha 1M": alpha_1m,
                    "Alpha 3M": alpha_3m,
                    "Status": status
                })
                
            df = pd.DataFrame(results).sort_values("Alpha 1W", ascending=False)

            # 5. Styling
            def style_status(val):
                color = 'white'
                weight = 'normal'
                if val == "LEADING": color = '#00FF00'; weight='bold'   # Green
                elif val == "WEAKENING": color = '#FFA500'; weight='bold' # Orange
                elif val == "LAGGING": color = '#FF4444'; weight='bold'   # Red
                elif val == "IMPROVING": color = '#00BFFF'; weight='bold' # Blue
                return f'color: {color}; font-weight: {weight}'

            def color_alpha(val):
                color = '#00FF00' if val > 0 else '#FF4444'
                return f'color: {color}'

            # Summary Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("SPY 1-Week", f"{spy_1w:.2f}%")
            c2.metric("SPY 1-Month", f"{spy_1m:.2f}%")
            c3.metric("SPY 3-Month", f"{spy_3m:.2f}%")
            
            st.divider()
            
            # Final Table
            st.dataframe(
                df.style.map(style_status, subset=['Status'])
                        .map(color_alpha, subset=['Alpha 1W', 'Alpha 1M', 'Alpha 3M'])
                        .format({
                            'Price': "${:.2f}", 
                            'Alpha 1W': "{:+.2f}%", 
                            'Alpha 1M': "{:+.2f}%", 
                            'Alpha 3M': "{:+.2f}%"
                        })
            )
        else:
            st.error("Could not fetch data.")
