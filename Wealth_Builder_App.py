import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Niagara Wealth Architecture", layout="wide")
st.title("🏛️ The Niagara Wealth Engine")
st.markdown("A deterministic projection and validation model to track the path to Age 60.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Core Assumptions")
current_age = st.sidebar.number_input("Current Age", value=43)
target_age = st.sidebar.number_input("Retirement Age", value=60)
expected_return = st.sidebar.slider("Expected Nominal Return (%)", 4.0, 12.0, 7.0, 0.1) / 100
inflation = st.sidebar.slider("Expected Inflation (%)", 1.0, 5.0, 2.5, 0.1) / 100
real_return = (1 + expected_return) / (1 + inflation) - 1

st.sidebar.header("2. Starting Balances (Age 43)")
rrsp_start = st.sidebar.number_input("Spousal RRSP ($)", value=825000, step=10000)
tfsa_start = st.sidebar.number_input("TFSA ($)", value=45000, step=1000)
resp_start = st.sidebar.number_input("RESP ($)", value=104000, step=1000)
mortgage_start = st.sidebar.number_input("Mortgage Remaining ($)", value=65000, step=1000)

st.sidebar.header("3. Cash Flow Parameters")
rrsp_contrib = st.sidebar.number_input("Annual RRSP Contrib (Until 50)", value=35000)
tax_refund = st.sidebar.number_input("Est. Tax Refund -> TFSA", value=15000)
mortgage_pmt = st.sidebar.number_input("Annual Mortgage Pmt", value=26000)
post_mortgage_tfsa = st.sidebar.number_input("Post-Mortgage Extra TFSA", value=10000)

st.sidebar.header("4. Live Holdings (Unit Counts)")
rrsp_units = st.sidebar.number_input("XEQT Units (Spousal RRSP)", value=21500, step=100)
tfsa_units = st.sidebar.number_input("XEQT Units (TFSA)", value=1180, step=10)
nonreg_units = st.sidebar.number_input("XEQT Units (Non-Reg)", value=0, step=10)

# --- PROJECTION ENGINE ---
@st.cache_data
def calculate_projection(cagr, rrsp, tfsa, resp, mortgage):
    data = []
    current_year = datetime.now().year
    non_reg = 0
    tfsa_room = 109000  # Starting unused room
    
    for i in range(target_age - current_age + 1):
        age = current_age + i
        year = current_year + i
        
        # 1. Base annual TFSA room increase (~$7k/yr)
        if i > 0: tfsa_room += 7000 
            
        cur_rrsp_in = 0
        cur_tfsa_in = 0
        cur_nonreg_in = 0
        
        # 2. Mortgage Logic
        if mortgage > 0:
            mortgage -= mortgage_pmt
            if mortgage < 0: mortgage = 0
            
        # 3. Phase & Cash Flow Routing Logic
        if age < 46:
            # Phase 1: Mortgage active. Hammer RRSP, route refund to TFSA.
            cur_rrsp_in = rrsp_contrib
            cur_tfsa_in = min(tax_refund, tfsa_room)
            
        elif age >= 46 and age < 50:
            # Phase 2: Mortgage dead. Redirect freed up cash flow to TFSA.
            cur_rrsp_in = rrsp_contrib
            total_to_tfsa = tax_refund + post_mortgage_tfsa
            cur_tfsa_in = min(total_to_tfsa, tfsa_room)
            cur_nonreg_in = max(0, total_to_tfsa - tfsa_room) # Spillover
            
        elif age >= 50:
            # Phase 3: RRSP stops. Redirect total previous cash flow to TFSA/Non-Reg
            cur_rrsp_in = 0
            total_cashflow = rrsp_contrib + post_mortgage_tfsa 
            cur_tfsa_in = min(total_cashflow, tfsa_room)
            cur_nonreg_in = total_cashflow - cur_tfsa_in

        # Update TFSA Room remaining
        tfsa_room -= cur_tfsa_in
        
        # 4. Market Growth (End of Year Compounding)
        if i > 0: # Don't grow Year 0
            rrsp = (rrsp + cur_rrsp_in) * (1 + cagr)
            tfsa = (tfsa + cur_tfsa_in) * (1 + cagr)
            non_reg = (non_reg + cur_nonreg_in) * (1 + cagr)
            
            # RESP Glidepath: 7% until age 47, then 4% safety
            resp_cagr = cagr if age < 47 else 0.04
            resp = resp * (1 + resp_cagr)
            
        liquid_nw = rrsp + tfsa + non_reg
        
        data.append({
            "Year": year,
            "Age": age,
            "RRSP": rrsp,
            "TFSA": tfsa,
            "Non-Reg": non_reg,
            "RESP": resp,
            "Mortgage": mortgage,
            "Liquid NW": liquid_nw
        })
        
    return pd.DataFrame(data)

# Generate Baseline Matrix
df_proj = calculate_projection(real_return, rrsp_start, tfsa_start, resp_start, mortgage_start)

# --- LIVE API INGESTION LAYER ---
@st.cache_data(ttl=3600)  # Cache for 1 hour to prevent API rate limits
def get_live_price(ticker="XEQT.TO"):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return None
    except Exception as e:
        return None

live_price = get_live_price("XEQT.TO")

# --- DASHBOARD UI ---
st.divider()

if live_price:
    # Calculate Actuals
    actual_rrsp = rrsp_units * live_price
    actual_tfsa = tfsa_units * live_price
    actual_nonreg = nonreg_units * live_price
    actual_liquid_nw = actual_rrsp + actual_tfsa + actual_nonreg
    
    # Calculate Fractional Age for exact X-Axis placement
    day_of_year = datetime.now().timetuple().tm_yday
    fractional_age = current_age + (day_of_year / 365.25)
    
    # Calculate Variance against Model
    expected_nw_current = df_proj.loc[df_proj['Age'] == current_age, 'Liquid NW'].values[0]
    variance = actual_liquid_nw - expected_nw_current
    variance_pct = (variance / expected_nw_current) * 100

    st.subheader(f"⚡ Live Market Tracking (XEQT.TO @ ${live_price:.2f})")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Projected Wealth (Age 60)", f"${df_proj['Liquid NW'].iloc[-1]:,.0f}")
    with col2:
        st.metric("Live Liquid NW", f"${actual_liquid_nw:,.0f}")
    with col3:
        st.metric("Variance vs Model", f"${variance:,.0f}", f"{variance_pct:.2f}%")
    with col4:
        mortgage_free_age = df_proj[df_proj['Mortgage'] == 0]['Age'].iloc[0] if not df_proj[df_proj['Mortgage'] == 0].empty else "N/A"
        st.metric("Mortgage Free Age", mortgage_free_age)

else:
    st.warning("Liquidity Warning: yfinance API is unreachable. Displaying base projections only.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Projected Wealth (Age 60)", f"${df_proj['Liquid NW'].iloc[-1]:,.0f}")
    with col2:
        st.metric("Safe Withdrawal (4%)", f"${(df_proj['Liquid NW'].iloc[-1] * 0.04):,.0f} / yr")
    with col3:
        mortgage_free_age = df_proj[df_proj['Mortgage'] == 0]['Age'].iloc[0] if not df_proj[df_proj['Mortgage'] == 0].empty else "N/A"
        st.metric("Mortgage Free Age", mortgage_free_age)

st.divider()

# --- PLOTTING ---
st.subheader("📈 The 17-Year Compound Matrix")

fig = go.Figure()

# Base Projection Area Charts
fig.add_trace(go.Scatter(x=df_proj['Age'], y=df_proj['RRSP'], mode='lines', stackgroup='one', name='Spousal RRSP', line=dict(color='#2E86C1')))
fig.add_trace(go.Scatter(x=df_proj['Age'], y=df_proj['TFSA'], mode='lines', stackgroup='one', name='TFSA', line=dict(color='#28B463')))
fig.add_trace(go.Scatter(x=df_proj['Age'], y=df_proj['Non-Reg'], mode='lines', stackgroup='one', name='Non-Registered', line=dict(color='#F1C40F')))

# Live Data Overlay
if live_price:
    fig.add_trace(go.Scatter(
        x=[fractional_age], 
        y=[actual_liquid_nw],
        mode='markers',
        marker=dict(size=14, color='white', line=dict(width=2, color='black')),
        name='Live Net Worth'
    ))

fig.update_layout(
    xaxis_title="Age",
    yaxis_title="Portfolio Value ($)",
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

# Milestone Lines
fig.add_vline(x=46, line_dash="dash", line_color="red", annotation_text="Mortgage Dead")
fig.add_vline(x=50, line_dash="dash", line_color="white", annotation_text="RRSP Pivot")

st.plotly_chart(fig, use_container_width=True)

# Data Table
with st.expander("🔍 View Raw Projection Data Matrix"):
    st.dataframe(df_proj.set_index("Age").style.format("${:,.0f}"))

