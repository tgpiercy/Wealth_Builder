import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Niagara Wealth Architecture", layout="wide")
st.title("🏛️ The Niagara Wealth Engine")
st.markdown("A deterministic full-lifecycle projection and validation model.")

# --- SIDEBAR: 1. CORE ASSUMPTIONS ---
st.sidebar.header("1. Core Assumptions")
current_age = st.sidebar.number_input("Current Age", value=43)
target_age = st.sidebar.number_input("Retirement Age", value=60)
terminal_age = st.sidebar.number_input("Terminal Age (Longevity)", value=95, min_value=80, max_value=110)
expected_return = st.sidebar.slider("Accumulation Nominal Return (%)", 4.0, 12.0, 7.0, 0.1) / 100
inflation = st.sidebar.slider("Expected Inflation (%)", 1.0, 5.0, 2.5, 0.1) / 100
real_return = (1 + expected_return) / (1 + inflation) - 1

# --- SIDEBAR: 2. STARTING BALANCES ---
st.sidebar.header("2. Starting Balances (Age 43)")
rrsp_start = st.sidebar.number_input("Spousal RRSP ($)", value=825000, step=10000)
tfsa_start = st.sidebar.number_input("TFSA ($)", value=45000, step=1000)
resp_start = st.sidebar.number_input("RESP ($)", value=104000, step=1000)
mortgage_start = st.sidebar.number_input("Mortgage Remaining ($)", value=65000, step=1000)

# --- SIDEBAR: 3. CASH FLOW (ACCUMULATION) ---
st.sidebar.header("3. Cash Flow (Accumulation)")
rrsp_contrib = st.sidebar.number_input("Annual RRSP Contrib (Until 50)", value=35000)
tax_refund = st.sidebar.number_input("Est. Tax Refund -> TFSA", value=15000)
mortgage_pmt = st.sidebar.number_input("Annual Mortgage Pmt", value=26000)
post_mortgage_tfsa = st.sidebar.number_input("Post-Mortgage Extra TFSA", value=10000)

# --- SIDEBAR: 4. DECUMULATION (AGE 60+) ---
st.sidebar.header("4. Decumulation Logic")
post_retire_expected_return = st.sidebar.slider("Post-Retire Nominal Return (%)", 2.0, 10.0, 5.0, 0.1) / 100
post_retire_real_return = (1 + post_retire_expected_return) / (1 + inflation) - 1

target_net_income = st.sidebar.number_input("Target Net Income ($)", value=120000, step=5000)
est_effective_tax = st.sidebar.slider("Est. Effective Tax Rate (%)", 10, 40, 20) / 100

st.sidebar.subheader("Gov Benefits (Household Combined)")
base_cpp = st.sidebar.number_input("Est. CPP at 65", value=24000, step=1000)
base_oas = st.sidebar.number_input("Est. OAS at 65", value=17000, step=1000)
cpp_start = st.sidebar.selectbox("CPP Start Age", [60, 65, 70], index=0) 
oas_start = st.sidebar.selectbox("OAS Start Age", [65, 70], index=0)     

# --- SIDEBAR: 5. STRESS TESTING (SORR) ---
st.sidebar.header("5. Stress Testing (SORR)")
apply_sorr = st.sidebar.checkbox("Enable Market Crash Simulation", value=False)
sorr_age = st.sidebar.number_input("Age of Crash", min_value=target_age, max_value=terminal_age, value=target_age)
sorr_drop = st.sidebar.slider("Crash Depth (%)", -50, -10, -30, 5)

# --- SIDEBAR: 6. LIVE HOLDINGS ---
st.sidebar.header("6. Live Holdings (Unit Counts)")
rrsp_units = st.sidebar.number_input("XEQT Units (Spousal RRSP)", value=21500, step=100)
tfsa_units = st.sidebar.number_input("XEQT Units (TFSA)", value=1180, step=10)
nonreg_units = st.sidebar.number_input("XEQT Units (Non-Reg)", value=0, step=10)

# --- LIVE API INGESTION LAYER ---
@st.cache_data(ttl=3600)  
def get_live_price(ticker="XEQT.TO"):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return None
    except Exception as e:
        return None

live_price = get_live_price("XEQT.TO")

# --- PROJECTION ENGINE (Unified Architecture) ---
data = []
current_year = datetime.now().year

# Initialize Balances
rrsp = rrsp_start
tfsa = tfsa_start
non_reg = 0
resp = resp_start
mortgage = mortgage_start
tfsa_room = 109000  

for i in range(terminal_age - current_age + 1):
    age = current_age + i
    year = current_year + i
    
    cur_rrsp_in = 0
    cur_tfsa_in = 0
    cur_nonreg_in = 0
    
    current_cpp = 0
    current_oas = 0
    rrsp_draw = 0
    nonreg_draw = 0
    tfsa_draw = 0
    tax_paid = 0
    net_income_achieved = 0
    
    if i > 0: tfsa_room += 7000 
    
    # --- PHASE 1 & 2: ACCUMULATION LOGIC ---
    if age < target_age:
        if mortgage > 0:
            mortgage -= mortgage_pmt
            if mortgage < 0: mortgage = 0
            
        if age < 46:
            cur_rrsp_in = rrsp_contrib
            cur_tfsa_in = min(tax_refund, tfsa_room)
        elif age >= 46 and age < 50:
            cur_rrsp_in = rrsp_contrib
            total_to_tfsa = tax_refund + post_mortgage_tfsa
            cur_tfsa_in = min(total_to_tfsa, tfsa_room)
            cur_nonreg_in = max(0, total_to_tfsa - tfsa_room)
        elif age >= 50:
            total_cashflow = rrsp_contrib + post_mortgage_tfsa 
            cur_tfsa_in = min(total_cashflow, tfsa_room)
            cur_nonreg_in = total_cashflow - cur_tfsa_in

        tfsa_room -= cur_tfsa_in
            
    # --- PHASE 3: DECUMULATION LOGIC ---
    else:
        if age >= cpp_start:
            if cpp_start == 60: current_cpp = base_cpp * 0.64
            elif cpp_start == 65: current_cpp = base_cpp
            elif cpp_start == 70: current_cpp = base_cpp * 1.42
            
        if age >= oas_start:
            if oas_start == 65: current_oas = base_oas
            elif oas_start == 70: current_oas = base_oas * 1.36
            
        tax_paid = (current_cpp + current_oas) * est_effective_tax
        net_gov_income = (current_cpp + current_oas) - tax_paid
        net_shortfall = target_net_income - net_gov_income
        
        if net_shortfall > 0:
            gross_needed_taxable = net_shortfall / (1 - est_effective_tax)
            
            # Waterfall: RRSP -> NonReg -> TFSA
            if rrsp >= gross_needed_taxable:
                rrsp_draw = gross_needed_taxable
                rrsp -= rrsp_draw
                tax_paid += rrsp_draw * est_effective_tax
                net_shortfall = 0
            else:
                rrsp_draw = rrsp
                tax_paid += rrsp_draw * est_effective_tax
                net_shortfall -= rrsp_draw * (1 - est_effective_tax)
                rrsp = 0
                
                gross_needed_nonreg = net_shortfall / (1 - est_effective_tax)
                if non_reg >= gross_needed_nonreg:
                    nonreg_draw = gross_needed_nonreg
                    non_reg -= nonreg_draw
                    tax_paid += nonreg_draw * est_effective_tax
                    net_shortfall = 0
                else:
                    nonreg_draw = non_reg
                    tax_paid += nonreg_draw * est_effective_tax
                    net_shortfall -= nonreg_draw * (1 - est_effective_tax)
                    non_reg = 0
                    
                    if tfsa >= net_shortfall:
                        tfsa_draw = net_shortfall
                        tfsa -= tfsa_draw
                        net_shortfall = 0
                    else:
                        tfsa_draw = tfsa
                        net_shortfall -= tfsa_draw
                        tfsa = 0 
                        
        net_income_achieved = target_net_income - net_shortfall

    # --- UNIFIED MARKET COMPOUNDING ENGINE ---
    # 1. Determine baseline CAGR based on phase
    current_year_real_return = real_return if age < target_age else post_retire_real_return
    
    # 2. Inject SORR Crash if active
    if apply_sorr and age == sorr_age:
        # Convert nominal crash to real crash
        current_year_real_return = ((1 + (sorr_drop / 100)) / (1 + inflation)) - 1
        
    # 3. Apply Compounding to remaining balances
    if i > 0:
        rrsp = (rrsp + cur_rrsp_in) * (1 + current_year_real_return)
        tfsa = (tfsa + cur_tfsa_in) * (1 + current_year_real_return)
        non_reg = (non_reg + cur_nonreg_in) * (1 + current_year_real_return)
        
        resp_cagr = real_return if age < 47 else 0.04
        resp = resp * (1 + resp_cagr)
        
    liquid_nw = rrsp + tfsa + non_reg
    
    data.append({
        "Year": year,
        "Age": age,
        "RRSP": max(0, rrsp),
        "TFSA": max(0, tfsa),
        "Non-Reg": max(0, non_reg),
        "RESP": resp,
        "Mortgage": mortgage,
        "Liquid NW": max(0, liquid_nw),
        "Upper Bound (+10%)": liquid_nw * 1.10,
        "Lower Bound (-10%)": liquid_nw * 0.90,
        "CPP": current_cpp,
        "OAS": current_oas,
        "RRSP Draw": rrsp_draw,
        "NonReg Draw": nonreg_draw,
        "TFSA Draw": tfsa_draw,
        "Est. Taxes Paid": tax_paid,
        "Net Income": net_income_achieved
    })
    
df_proj = pd.DataFrame(data)

# --- DASHBOARD UI ---
st.divider()

if live_price:
    actual_rrsp = rrsp_units * live_price
    actual_tfsa = tfsa_units * live_price
    actual_nonreg = nonreg_units * live_price
    actual_liquid_nw = actual_rrsp + actual_tfsa + actual_nonreg
    
    day_of_year = datetime.now().timetuple().tm_yday
    fractional_age = current_age + (day_of_year / 365.25)
    
    expected_nw_current = df_proj.loc[df_proj['Age'] == current_age, 'Liquid NW'].values[0]
    variance = actual_liquid_nw - expected_nw_current
    variance_pct = (variance / expected_nw_current) * 100

    st.subheader(f"⚡ Live Market Tracking (XEQT.TO @ ${live_price:.2f})")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Peak Projected Wealth (Age 60)", f"${df_proj.loc[df_proj['Age'] == target_age, 'Liquid NW'].values[0]:,.0f}")
    with col2:
        st.metric("Live Liquid NW", f"${actual_liquid_nw:,.0f}")
    with col3:
        st.metric("Variance vs Model", f"${variance:,.0f}", f"{variance_pct:.2f}%")
    with col4:
        st.metric("Terminal Wealth", f"${df_proj['Liquid NW'].iloc[-1]:,.0f}")

else:
    st.warning("Liquidity Warning: yfinance API is unreachable. Displaying base projections only.")

st.divider()

# --- TABS ARCHITECTURE ---
tab1, tab2 = st.tabs(["📈 Visual Matrix", "🧮 Raw Ledger (Decumulation)"])

with tab1:
    fig = go.Figure()

    # Base Projection Area Charts
    fig.add_trace(go.Scatter(x=df_proj['Age'], y=df_proj['RRSP'], mode='lines', stackgroup='one', name='Spousal RRSP', line=dict(color='#2E86C1')))
    fig.add_trace(go.Scatter(x=df_proj['Age'], y=df_proj['TFSA'], mode='lines', stackgroup='one', name='TFSA', line=dict(color='#28B463')))
    fig.add_trace(go.Scatter(x=df_proj['Age'], y=df_proj['Non-Reg'], mode='lines', stackgroup='one', name='Non-Registered', line=dict(color='#F1C40F')))

    # Drawdown Tolerance Bands
    fig.add_trace(go.Scatter(x=df_proj['Age'], y=df_proj['Upper Bound (+10%)'], mode='lines', line=dict(color='rgba(255, 255, 255, 0.3)', dash='dot'), name='+10% Variance'))
    fig.add_trace(go.Scatter(x=df_proj['Age'], y=df_proj['Lower Bound (-10%)'], mode='lines', line=dict(color='rgba(255, 255, 255, 0.3)', dash='dot'), name='-10% Variance'))

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

    fig.add_vline(x=46, line_dash="dash", line_color="red", annotation_text="Mortgage Dead")
    fig.add_vline(x=target_age, line_dash="dash", line_color="white", annotation_text="Decumulation Pivot")
    
    # Optional SORR Marker
    if apply_sorr:
        fig.add_vline(x=sorr_age, line_dash="solid", line_color="orange", annotation_text=f"{sorr_drop}% Crash")

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### The Withdrawal Waterfall")
    st.markdown("This ledger mathematically tracks how the target net income is achieved by draining the Spousal RRSP to the tax threshold, then bridging the gap with the TFSA.")
    
    df_decum = df_proj[df_proj['Age'] >= target_age][
        ["Age", "Liquid NW", "Net Income", "CPP", "OAS", "RRSP Draw", "NonReg Draw", "TFSA Draw", "Est. Taxes Paid"]
    ]

    st.dataframe(
        df_decum.set_index("Age").style.format("${:,.0f}"),
        use_container_width=True,
        height=600
    )
