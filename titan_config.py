# ==============================================================================
#  TITAN STRATEGY CONFIGURATION (v66.9 - Full List Restored)
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
    "XBI": ("02. SECTORS", "SPY"), # Biotech

    # --- 03. THEMES ---
    "BOTZ": ("03. THEMES", "SPY"),
    "REMX": ("03. THEMES", "SPY"),
    "ICLN": ("03. THEMES", "SPY"),
    "GDX": ("03. THEMES", "SPY"),

    # --- 04. METALS ---
    "GLD": ("04. METALS", "SPY"),
    "SLV": ("04. METALS", "SPY"),

    # --- 05. INDUSTRIES (Specialized ETFs & Top Stocks) ---
    
    # Tech & AI
    "IGV": ("05. INDUSTRIES", "XLK"),   # Software
    "SMH": ("05. INDUSTRIES", "XLK"),   # Semis
    "AIQ": ("05. INDUSTRIES", "BOTZ"),  # AI
    "MSFT": ("05. INDUSTRIES", "XLK"),  # Microsoft
    "NVDA": ("05. INDUSTRIES", "SMH"),  # Nvidia
    "SMCI": ("05. INDUSTRIES", "XLK"),  # Super Micro
    "DELL": ("05. INDUSTRIES", "XLK"),  # Dell
    "ANET": ("05. INDUSTRIES", "XLK"),  # Arista
    "WDC": ("05. INDUSTRIES", "XLK"),   # Western Digital
    "PSTG": ("05. INDUSTRIES", "XLK"),  # Pure Storage

    # Comms
    "META": ("05. INDUSTRIES", "XLC"),  # Meta
    "GOOGL": ("05. INDUSTRIES", "XLC"), # Google

    # Energy & Clean Energy
    "XOP": ("05. INDUSTRIES", "XLE"),   # E&P
    "OIH": ("05. INDUSTRIES", "XLE"),   # Oil Services
    "MLPX": ("05. INDUSTRIES", "XLE"),  # Midstream
    "TAN": ("05. INDUSTRIES", "ICLN"),  # Solar
    "NLR": ("05. INDUSTRIES", "ICLN"),  # Nuclear
    "URA": ("05. INDUSTRIES", "ICLN"),  # Uranium

    # Financials
    "KRE": ("05. INDUSTRIES", "XLF"),   # Reg Banks
    "KBE": ("05. INDUSTRIES", "XLF"),   # Banks
    "IAK": ("05. INDUSTRIES", "XLF"),   # Insurance

    # Industrials
    "ITA": ("05. INDUSTRIES", "XLI"),   # Aerospace
    "IYT": ("05. INDUSTRIES", "XLI"),   # Transport
    "PAVE": ("05. INDUSTRIES", "XLI"),  # Infrastructure

    # Materials & Miners
    "GDXJ": ("05. INDUSTRIES", "GDX"),  # Jr Miners
    "SIL": ("05. INDUSTRIES", "GDX"),   # Silver Miners
    "COPX": ("05. INDUSTRIES", "XLB"),  # Copper
    "SLX": ("05. INDUSTRIES", "XLB"),   # Steel
    "MOO": ("05. INDUSTRIES", "XLB"),   # Agribusiness
    "AA": ("05. INDUSTRIES", "XLB"),    # Alcoa
    "DD": ("05. INDUSTRIES", "XLB"),    # DuPont

    # Healthcare & Biotech
    "IBB": ("05. INDUSTRIES", "XBI"),   # Biotech
    "ARKG": ("05. INDUSTRIES", "XBI"),  # Genomics
    "PPH": ("05. INDUSTRIES", "XLV"),   # Pharma
    "IHI": ("05. INDUSTRIES", "XLV"),   # Med Devices

    # Discretionary
    "ITB": ("05. INDUSTRIES", "XLY"),   # Homebuilders
    "AMZN": ("05. INDUSTRIES", "XLY"),  # Amazon

    # --- 06. CANADA ---
    "RY.TO": ("06. CANADA", "HXT.TO"),
    "BN.TO": ("06. CANADA", "HXT.TO"),
    "CNQ.TO": ("06. CANADA", "HXT.TO"),
    "CP.TO": ("06. CANADA", "HXT.TO"),
    "WSP.TO": ("06. CANADA", "HXT.TO"),
    "SHOP.TO": ("06. CANADA", "HXT.TO"),
    "CSU.TO": ("06. CANADA", "HXT.TO"),
    "NTR.TO": ("06. CANADA", "HXT.TO"),
    "TECK-B.TO": ("06. CANADA", "HXT.TO"),
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
