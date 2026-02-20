# FILE: titan_efi_extremes.py
# ROLE: Extreme Volume Anomaly Backtester
# ARCHITECTURE: 100% SPY Core (Testing EFI Panic Exits & Capitulation Entries)

import pandas as pd
import numpy as np
import yfinance as yf
import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) 

print("🦅 TITAN VOLUME LAB: DEEP SCANNING EFI EXTREMES...")
start_time = time.time()

# ==============================================================================
# 1. UNIVERSE INGESTION
# ==============================================================================
print(f"📡 Downloading 5 years of historical data for SPY & IEF...")
data = yf.download(["SPY", "IEF"], period="5y", interval="1d", progress=False)

if isinstance(data.columns, pd.MultiIndex):
    close_px = data['Close'].ffill()
    open_px = data['Open'].ffill()
    high_px = data['High'].ffill()
    low_px = data['Low'].ffill()
    volume = data['Volume']['SPY'].ffill()
else:
    print("❌ Fatal Error: Data fetch failed.")
    exit()

df = pd.DataFrame({
    'Open': open_px['SPY'], 'High': high_px['SPY'], 'Low': low_px['SPY'], 
    'Close': close_px['SPY'], 'Volume': volume
})
ief_close = close_px['IEF']

# ==============================================================================
# 2. VECTORIZED PHYSICS PRE-COMPUTE 
# ==============================================================================
print("⚙️ Pre-computing dynamics (100 SMA, RRG, & EFI Z-Scores)...")

df['SMA_100'] = df['Close'].rolling(100).mean()

# Standard Risk & RRG Math
df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
df['ATR_14'] = df['TR'].rolling(14).mean()

rs = df['Close'] / ief_close
df['RRG_Ratio'] = 100 * (rs / rs.rolling(10).mean())
df['RRG_Mom'] = 100 + df['RRG_Ratio'].pct_change(fill_method=None) * 100

# EFI Calculation & Z-Score (To detect extreme anomalies)
df['EFI'] = (df['Close'] - df['Close'].shift(1)) * df['Volume']
df['EFI_Mean'] = df['EFI'].rolling(50).mean()
df['EFI_Std'] = df['EFI'].rolling(50).std()
df['EFI_Z'] = (df['EFI'] - df['EFI_Mean']) / df['EFI_Std'] # Standard Deviations

# Rolling min to check if a capitulation happened recently (last 10 days)
df['Recent_Capitulation'] = df['EFI_Z'].rolling(10).min() < -2.5

# ==============================================================================
# 3. THE SIMULATION ENGINE
# ==============================================================================
def run_simulation(mode="Baseline"):
    dates = df.dropna().index
    start_idx = len(dates) - (252 * 3) 
    if start_idx < 0: start_idx = 0

    capital = 100000.0
    shares = 0
    entry_price = 0
    stop_loss = 0
    phoenix_active = False
    
    trade_count = 0
    equity_curve = []
    
    for i in range(start_idx, len(dates)-1): 
        today = dates[i]; tomorrow = dates[i+1]
        t_today = df.loc[today]
        t_yest = df.iloc[df.index.get_loc(today) - 1]
        
        curr_eq = capital if shares == 0 else capital + (shares * t_today['Close'])
        equity_curve.append(curr_eq)
            
        # --- PROCESS EXITS ---
        if shares > 0:
            if stop_loss > 0 and t_today['Low'] <= stop_loss:
                capital += shares * stop_loss
                shares = 0; stop_loss = 0; phoenix_active = False
                continue
                
            is_100_break = t_today['Close'] < t_today['SMA_100']
            
            # The EFI Panic Exit Override
            is_efi_panic = False
            if mode == "EFI_Panic_Exit" and not phoenix_active:
                is_efi_panic = t_today['EFI_Z'] < -2.5 # Violent institutional dumping
            
            if phoenix_active:
                if t_today['Close'] > t_today['SMA_100']:
                    phoenix_active = False 
            else:
                if is_100_break or is_efi_panic:
                    exit_px = open_px['SPY'].loc[tomorrow]
                    capital += shares * exit_px
                    shares = 0; stop_loss = 0

        # --- PROCESS ENTRIES ---
        if shares == 0:
            is_bull = t_today['Close'] > t_today['SMA_100']
            
            # The Baseline Entry (1.25 RRG Thrust)
            is_thrust = (t_today['RRG_Mom'] - t_yest['RRG_Mom']) >= 1.25
            
            # The Capitulation Entry Override
            if mode == "EFI_Capitulation_Entry":
                # Only allow the RRG thrust if we saw an extreme volume panic recently
                is_thrust = is_thrust and t_today['Recent_Capitulation']
            
            if is_thrust or is_bull:
                phoenix_active = is_thrust and not is_bull
                stop_loss = t_today['Close'] - (3.5 * t_today['ATR_14']) if phoenix_active else 0
                
                shares = int(capital / t_today['Close'])
                if shares > 0:
                    entry_price = t_today['Close']
                    capital -= (shares * entry_price)
                    trade_count += 1

    if shares > 0:
        capital += shares * df['Close'].iloc[-1]
    
    equity_curve.append(capital)
    s = pd.Series(equity_curve)
    mdd = (s - s.cummax()) / s.cummax()
    return capital - 100000.0, mdd.min() * 100, trade_count

# ==============================================================================
# 4. EXECUTION MATRIX
# ==============================================================================
print("\n" + "="*85)
print("📊 TITAN ARCHITECTURE: EXTREME VOLUME (EFI) DEEP SCAN")
print("="*85)
print(f"{'Strategy Variant':<35} | {'Net PnL':<15} | {'Max DD':<10} | {'Trades':<6}")
print("-" * 80)

modes = [
    ("Baseline", "V11.1 Baseline (100 SMA / 1.25 RRG)"),
    ("EFI_Panic_Exit", "Early Exit on Extreme EFI Drop"),
    ("EFI_Capitulation_Entry", "RRG Entry + Capitulation Filter")
]

for mode, label in modes:
    pnl, dd, tc = run_simulation(mode)
    print(f"{label:<35} | ${pnl:<14,.2f} | {dd:<9.2f}% | {tc:<6}")

print("="*85)
print(f"⏱️ Total Compute Time: {time.time() - start_time:.2f} seconds")
