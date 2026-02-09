# ==============================================================================
#  TITAN CONFIGURATION (v62.6 Hierarchy Map)
# ==============================================================================

# --- CREDENTIALS ---
CREDENTIALS = {
    "admin": "password123",
    "user": "titan2025"
}

# --- MASTER DATA MAP ---
# Format: "TICKER": ["Category", "BENCHMARK_TICKER"]
# The benchmark is used for RS Scoring.

DATA_MAP = {
    # --- 1. INDICES (Benchmarks) ---
    "SPY":    ["01. INDEX", "IEF"],     # Equities vs Bonds (Risk On/Off)
    "QQQ":    ["01. INDEX", "SPY"],     # Tech vs Broad
    "IWM":    ["01. INDEX", "SPY"],     # Small Cap vs Broad
    "DIA":    ["01. INDEX", "SPY"],     # Dow vs Broad
    "HXT.TO": ["01. INDEX", "SPY"],     # Canada vs US
    "EEM":    ["01. INDEX", "SPY"],     # Emerging vs US
    "IEF":    ["01. INDEX", "USD"],     # Bonds vs Cash (proxy)
    "TLT":    ["01. INDEX", "SPY"],

    # --- 2. US SECTORS (vs SPY) ---
    "XLK": ["02. SECTOR", "SPY"], # Technology
    "XLF": ["02. SECTOR", "SPY"], # Financials
    "XLV": ["02. SECTOR", "SPY"], # Healthcare
    "XLY": ["02. SECTOR", "SPY"], # Discretionary
    "XLP": ["02. SECTOR", "SPY"], # Staples
    "XLE": ["02. SECTOR", "SPY"], # Energy
    "XLC": ["02. SECTOR", "SPY"], # Comm Services
    "XLI": ["02. SECTOR", "SPY"], # Industrials
    "XLB": ["02. SECTOR", "SPY"], # Materials
    "XLRE": ["02. SECTOR", "SPY"], # Real Estate
    "XLU": ["02. SECTOR", "SPY"], # Utilities

    # --- 3. COMMODITIES & MINERS ---
    "GLD":  ["03. COMMODITY", "SPY"],
    "SLV":  ["03. COMMODITY", "SPY"],
    "GDX":  ["03. COMMODITY", "SPY"],      # Senior Miners vs Market
    "GDXJ": ["03. COMMODITY", "GDX"],      # Junior vs Senior (Risk Ladder)
    "SIL":  ["03. COMMODITY", "GDX"],      # Silver Miners vs Gold Miners
    "COPX": ["03. COMMODITY", "SPY"],

    # --- 4. THEMES & SUB-SECTORS ---
    "BOTZ": ["04. THEME", "SPY"],          # Robotics vs Market
    "AIQ":  ["04. THEME", "BOTZ"],         # AI vs Robotics (Niche vs Niche)
    
    "ICLN": ["04. THEME", "SPY"],          # Clean Energy vs Market
    "TAN":  ["04. THEME", "ICLN"],         # Solar vs Clean Energy
    "NLR":  ["04. THEME", "ICLN"],         # Nuclear vs Clean Energy
    "URA":  ["04. THEME", "ICLN"],         # Uranium vs Clean Energy
    "REMX": ["04. THEME", "SPY"],          # Rare Earths vs Market
    
    "IAK":  ["04. THEME", "SPY"],          # Insurance vs Market
    "SMH":  ["04. THEME", "XLK"],          # Semis vs Tech
    "IGV":  ["04. THEME", "XLK"],          # Software vs Tech
    "XBI":  ["04. THEME", "XLV"],          # Biotech vs Healthcare
    "ITA":  ["04. THEME", "XLI"],          # Defense vs Industrials

    # --- 5. US STOCKS (Mapped to SECTORS) ---
    # Tech (XLK)
    "AAPL": ["Technology", "XLK"], "MSFT": ["Technology", "XLK"], 
    "NVDA": ["Technology", "XLK"], "AMD":  ["Technology", "XLK"],
    "PLTR": ["Technology", "XLK"], "ORCL": ["Technology", "XLK"],
    
    # Financials (XLF)
    "JPM":  ["Financials", "XLF"], "BAC":  ["Financials", "XLF"],
    "GS":   ["Financials", "XLF"], "V":    ["Financials", "XLF"],
    
    # Comm Services (XLC)
    "GOOGL": ["Comm Svcs", "XLC"], "META": ["Comm Svcs", "XLC"],
    "NFLX":  ["Comm Svcs", "XLC"], "DIS":  ["Comm Svcs", "XLC"],
    
    # Consumer (XLY/XLP)
    "AMZN": ["Discretionary", "XLY"], "TSLA": ["Discretionary", "XLY"],
    "COST": ["Staples", "XLP"],       "PG":   ["Staples", "XLP"],
    
    # Energy (XLE)
    "XOM":  ["Energy", "XLE"], "CVX": ["Energy", "XLE"],
    
    # Healthcare (XLV)
    "LLY":  ["Healthcare", "XLV"], "UNH": ["Healthcare", "XLV"],

    # --- 6. CANADIAN STOCKS (Mapped to HXT) ---
    "SHOP.TO": ["Canada Tech", "HXT.TO"],
    "RY.TO":   ["Canada Fin",  "HXT.TO"],
    "TD.TO":   ["Canada Fin",  "HXT.TO"],
    "CNQ.TO":  ["Canada Energy", "HXT.TO"],
    "SU.TO":   ["Canada Energy", "HXT.TO"],
    "CSU.TO":  ["Canada Tech", "HXT.TO"],
    "DOL.TO":  ["Canada Disc", "HXT.TO"],
    "ATD.TO":  ["Canada Disc", "HXT.TO"],
    "BMO.TO":  ["Canada Fin",  "HXT.TO"],
    "CP.TO":   ["Canada Ind",  "HXT.TO"],
    "CNR.TO":  ["Canada Ind",  "HXT.TO"],
    "BN.TO":   ["Canada Fin",  "HXT.TO"],
    "POW.TO":  ["Canada Fin",  "HXT.TO"],
    "MFC.TO":  ["Canada Fin",  "HXT.TO"],
    "SLF.TO":  ["Canada Fin",  "HXT.TO"],
    "GWO.TO":  ["Canada Fin",  "HXT.TO"],
    "TRI.TO":  ["Canada Ind",  "HXT.TO"],
    "K.TO":    ["Canada Mat",  "HXT.TO"],
    "AEM.TO":  ["Canada Mat",  "HXT.TO"],
    "WPM.TO":  ["Canada Mat",  "HXT.TO"],
    "CCO.TO":  ["Canada Energy", "HXT.TO"],
}

# --- RRG GROUPS (For Charts) ---

RRG_INDICES = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "DIA": "Dow Jones",
    "MDY": "MidCap 400",
    "HXT.TO": "TSX 60"
}

RRG_SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLY": "Discretionary",
    "XLP": "Staples",
    "XLE": "Energy",
    "XLC": "Comm Svcs",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities"
}

RRG_THEMES = {
    "SMH": "Semiconductors",
    "IGV": "Software",
    "XBI": "Biotech",
    "ITA": "Defense",
    "KWEB": "China Internet",
    "GDX": "Gold Miners",
    "XOP": "Oil & Gas Exp",
    "XHB": "Homebuilders",
    "JETS": "Airlines",
    "TAN": "Solar"
}

# --- DRILL DOWN MAPS ---
RRG_INDUSTRY_MAP = {
    "XLK": {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMD": "AMD", "ORCL": "Oracle", "CRM": "Salesforce"},
    "XLF": {"JPM": "JPMorgan", "BAC": "Bank of America", "V": "Visa", "MA": "Mastercard", "GS": "Goldman", "BLK": "Blackrock"},
    "XLC": {"GOOGL": "Google", "META": "Meta", "NFLX": "Netflix", "DIS": "Disney", "TMUS": "T-Mobile"},
    "XLE": {"XOM": "Exxon", "CVX": "Chevron", "EOG": "EOG", "SLB": "Schlumberger", "COP": "Conoco"},
    "XLV": {"LLY": "Lilly", "UNH": "UnitedHealth", "JNJ": "J&J", "ABBV": "AbbVie", "MRK": "Merck"},
    "XLY": {"AMZN": "Amazon", "TSLA": "Tesla", "HD": "Home Depot", "MCD": "McDonalds", "NKE": "Nike"},
    "XLI": {"CAT": "Caterpillar", "DE": "Deere", "HON": "Honeywell", "GE": "General Electric", "UNP": "Union Pacific"},
    "XLB": {"LIN": "Linde", "SHW": "Sherwin", "FCX": "Freeport", "NEM": "Newmont", "APD": "Air Products"},
    "Canada (TSX)": {
        "SHOP.TO": "Shopify", "RY.TO": "RBC", "TD.TO": "TD Bank", "CNQ.TO": "Canadian Natural", 
        "CP.TO": "CP Rail", "CNR.TO": "CN Rail", "BN.TO": "Brookfield", "CSU.TO": "Constellation",
        "ATD.TO": "Couche-Tard", "WPM.TO": "Wheaton"
    }
}

# --- PARENT SECTOR LOOKUP (For Scanner Filtering) ---
SECTOR_PARENTS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Discretionary": "XLY",
    "Staples": "XLP",
    "Energy": "XLE",
    "Comm Svcs": "XLC",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Canada Tech": "HXT.TO",
    "Canada Fin": "HXT.TO",
    "Canada Energy": "HXT.TO",
    "Canada Ind": "HXT.TO",
    "Canada Mat": "HXT.TO",
    "Canada Disc": "HXT.TO"
}

# --- ETF LISTS ---
SECTOR_ETFS = list(RRG_SECTORS.keys())
