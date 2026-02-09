# ==============================================================================
#  TITAN STRATEGY CONFIGURATION (v66.8 - Consolidated Headers)
# ==============================================================================

# --- AUTHENTICATION ---
CREDENTIALS = {
    "son": "",  # No password
    "dad": ""   # No password
}

# --- MASTER DATA MAP ---
# Format: { Ticker : (Display_Category, Benchmark_Ticker) }

DATA_MAP = {
    # --- 01. MARKETS ---
    "SPY": ("01. MARKETS", "SPY"),
    "RSP": ("01. MARKETS", "SPY"),
    "DIA": ("01. MARKETS", "SPY"),
    "IWM": ("01. MARKETS", "SPY"),
    "IWC": ("01. MARKETS", "SPY"),
    "QQQ": ("01. MARKETS", "SPY"),
    "HXT.TO": ("01. MARKETS", "SPY"),
    "IEF": ("01. MARKETS", "SPY"),
    "^VIX": ("01. MARKETS", "SPY"),

    # --- 02. SECTORS ---
    # Includes all Tier 1, Tier 2, Defensive, and XBI
    "XLK": ("02. SECTORS", "SPY"), # Tech
    "XLC": ("02. SECTORS", "SPY"), # Comms
    "XLF": ("02. SECTORS", "SPY"), # Financials
    "XLI": ("02. SECTORS", "SPY"), # Industrials
    "XLE": ("02. SECTORS", "SPY"), # Energy
    "XLB": ("02. SECTORS", "SPY"), # Materials
    "XLV": ("02. SECTORS", "SPY"), # Healthcare
    "XLY": ("02. SECTORS", "SPY"), # Discretionary
    "XLP": ("02. SECTORS", "SPY"), # Staples
    "XLRE": ("02. SECTORS", "SPY"), # Real Estate
    "XLU": ("02. SECTORS", "SPY"), # Utilities
    "XBI": ("02. SECTORS", "SPY"), # Biotech (Moved here per request)

    # --- 03. THEMES ---
    "BOTZ": ("03. THEMES", "SPY"),
    "REMX": ("03. THEMES", "SPY"),
    "ICLN": ("03. THEMES", "SPY"),
    "GDX": ("03. THEMES", "SPY"),

    # --- 04. METALS ---
    "GLD": ("04. METALS", "SPY"),
    "SLV": ("04. METALS", "SPY"),

    # --- 05. INDUSTRIES ---
    # Grouped by header, sorted by sector logic
    "IGV": ("05. INDUSTRIES", "XLK"),  # Software (Tech)
    "SMH": ("05. INDUSTRIES", "XLK"),  # Semis (Tech)
    "XOP": ("05. INDUSTRIES", "XLE"),  # E&P (Energy)
    "KRE": ("05. INDUSTRIES", "XLF"),  # Reg Banks (Fin)
    "ITA": ("05. INDUSTRIES", "XLI"),  # Aerospace (Ind)
    "ITB": ("05. INDUSTRIES", "XLY"),  # Homebuilders (Disc)

    # --- 06. CANADA ---
    # Individual TSX Stocks
    "RY.TO": ("06. CANADA", "HXT.TO"),
    "BN.TO": ("06. CANADA", "HXT.TO"),
    "CNQ.TO": ("06. CANADA", "HXT.TO"),
    "CP.TO": ("06. CANADA", "HXT.TO"),
    "WSP.TO": ("06. CANADA", "HXT.TO"),
    "SHOP.TO": ("06. CANADA", "HXT.TO"),
    "CSU.TO": ("06. CANADA", "HXT.TO"),
    "NTR.TO": ("06. CANADA", "HXT.TO"),
    "TECK-B.TO": ("06. CANADA", "HXT.TO"),
    
    # --- DRILL DOWN DATA ONLY (HIDDEN FROM SCANNER) ---
    "AIQ": ("99. DATA", "BOTZ"),
    "IBB": ("99. DATA", "XBI"),
    "ARKG": ("99. DATA", "XBI"),
    "TAN": ("99. DATA", "ICLN"),
    "NLR": ("99. DATA", "ICLN"),
    "URA": ("99. DATA", "ICLN"),
    "GDXJ": ("99. DATA", "GDX"),
    "SIL": ("99. DATA", "GDX"),
    "COPX": ("99. DATA", "XLB"),
    "AA": ("99. DATA", "XLB"),
    "SLX": ("99. DATA", "XLB"),
    "DD": ("99. DATA", "XLB"),
    "MOO": ("99. DATA", "XLB"),
    "META": ("99. DATA", "XLC"),
    "GOOGL": ("99. DATA", "XLC"),
    "OIH": ("99. DATA", "XLE"),
    "MLPX": ("99. DATA", "XLE"),
    "KBE": ("99. DATA", "XLF"),
    "IAK": ("99. DATA", "XLF"),
    "IYT": ("99. DATA", "XLI"),
    "PAVE": ("99. DATA", "XLI"),
    "SMCI": ("99. DATA", "XLK"),
    "DELL": ("99. DATA", "XLK"),
    "WDC": ("99. DATA", "XLK"),
    "PSTG": ("99. DATA", "XLK"),
    "ANET": ("99. DATA", "XLK"),
    "MSFT": ("99. DATA", "XLK"),
    "NVDA": ("99. DATA", "XLK"),
    "PPH": ("99. DATA", "XLV"),
    "IHI": ("99. DATA", "XLV"),
    "AMZN": ("99. DATA", "XLY"),
}

# --- RRG GROUPS (Unchanged) ---
RRG_INDICES = {
    "SPY": "S&P 500", "RSP": "S&P 500 Eq", "DIA": "Dow Jones",
    "IWM": "Russell 2000", "IWC": "Micro-Cap", "QQQ": "Nasdaq 100",
    "HXT.TO": "TSX 60 (CAD)", "IEF": "Bonds (7-10Y)"
}

RRG_SECTORS = {
    "XLK": "Technology", "XLC": "Comms", "XLY": "Discretionary",
    "XLF": "Financials", "XLV": "Healthcare", "XLI": "Industrials",
    "XLE": "Energy", "XLB": "Materials", "XLRE": "Real Estate",
    "XLU": "Utilities", "XLP": "Staples"
}

RRG_THEMES = {
    "BOTZ": "Robotics/AI", "XBI": "Biotech", "ICLN": "Clean Energy",
    "GDX": "Gold Miners", "SMH": "Semiconductors", "IGV": "Software",
    "XOP": "Oil & Gas E&P", "KRE": "Regional Banks",
    "ITA": "Aerospace", "ITB": "Homebuilders"
}

RRG_INDUSTRY_MAP = {
    "XLK": {"MSFT": "Microsoft", "NVDA": "Nvidia", "SMH": "Semis", "IGV": "Software", "ANET": "Arista", "DELL": "Dell"},
    "XLC": {"META": "Meta", "GOOGL": "Google"},
    "XLF": {"KBE": "Banks", "KRE": "Reg Banks", "IAK": "Insurance"},
    "XLE": {"XOP": "Exploration", "OIH": "Oil Services", "MLPX": "Midstream"},
    "XLV": {"PPH": "Pharma", "IHI": "Med Devices", "XBI": "Biotech"},
    "XLI": {"ITA": "Aerospace", "IYT": "Transport", "PAVE": "Infrastructure"},
    "XLB": {"COPX": "Copper", "SLX": "Steel", "MOO": "Agri", "AA": "Alcoa", "DD": "DuPont"},
    "XLY": {"AMZN": "Amazon", "ITB": "Homebuilders"},
    "GDX": {"GDXJ": "Junior Miners", "SIL": "Silver Miners"},
    "BOTZ": {"AIQ": "Artificial Intel"},
    "ICLN": {"TAN": "Solar", "NLR": "Nuclear", "URA": "Uranium"}
}

SECTOR_PARENTS = {
    "02. SECTORS": "SPY",
    "03. THEMES": "SPY",
    "05. INDUSTRIES": "SPY",
    "06. CANADA": "HXT.TO"
}

SECTOR_ETFS = list(RRG_SECTORS.keys()) + list(RRG_THEMES.keys())
