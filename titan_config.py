# ==============================================================================
#  TITAN STRATEGY CONFIGURATION (v66.5 - Headers Restructured)
# ==============================================================================

# --- AUTHENTICATION ---
CREDENTIALS = {
    "son": "",  # No password
    "dad": ""   # No password
}

# --- MASTER DATA MAP ---
# Format: { Ticker : (Display_Category, Benchmark_Ticker) }
# The 'Display_Category' determines the HEADER in the scanner.
# The ORDER of lines determines the SORT ORDER in the scanner.

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

    # --- 02. TECHNOLOGY ---
    "XLK": ("02. TECHNOLOGY", "SPY"),
    "IGV": ("02. TECHNOLOGY", "XLK"),  # Software
    "SMH": ("02. TECHNOLOGY", "XLK"),  # Semis

    # --- 03. ENERGY ---
    "XLE": ("03. ENERGY", "SPY"),
    "XOP": ("03. ENERGY", "XLE"),      # E&P

    # --- 04. FINANCIALS ---
    "XLF": ("04. FINANCIALS", "SPY"),
    "KRE": ("04. FINANCIALS", "XLF"),  # Regional Banks

    # --- 05. INDUSTRIALS ---
    "XLI": ("05. INDUSTRIALS", "SPY"),
    "ITA": ("05. INDUSTRIALS", "XLI"), # Aerospace & Defense

    # --- 06. DISCRETIONARY ---
    "XLY": ("06. DISCRETIONARY", "SPY"),
    "ITB": ("06. DISCRETIONARY", "XLY"), # Homebuilders

    # --- 07. SECTORS (GENERAL) ---
    "XLC": ("07. SECTORS", "SPY"),
    "XLB": ("07. SECTORS", "SPY"),
    "XLV": ("07. SECTORS", "SPY"),
    "XLP": ("07. SECTORS", "SPY"),
    "XLRE": ("07. SECTORS", "SPY"),
    "XLU": ("07. SECTORS", "SPY"),
    "XBI": ("07. SECTORS", "SPY"),     # Biotech (Moved here per request)

    # --- 08. THEMES ---
    "BOTZ": ("08. THEMES", "SPY"),
    "REMX": ("08. THEMES", "SPY"),
    "ICLN": ("08. THEMES", "SPY"),
    "GDX": ("08. THEMES", "SPY"),

    # --- 09. METALS ---
    "GLD": ("09. METALS", "SPY"),
    "SLV": ("09. METALS", "SPY"),

    # --- 10. CANADA (HXT) ---
    "RY.TO": ("10. CANADA (HXT)", "HXT.TO"),
    "BN.TO": ("10. CANADA (HXT)", "HXT.TO"),
    "CNQ.TO": ("10. CANADA (HXT)", "HXT.TO"),
    "CP.TO": ("10. CANADA (HXT)", "HXT.TO"),
    "WSP.TO": ("10. CANADA (HXT)", "HXT.TO"),
    "SHOP.TO": ("10. CANADA (HXT)", "HXT.TO"),
    "CSU.TO": ("10. CANADA (HXT)", "HXT.TO"),
    "NTR.TO": ("10. CANADA (HXT)", "HXT.TO"),
    "TECK-B.TO": ("10. CANADA (HXT)", "HXT.TO"),
    
    # --- DRILL DOWN DATA ONLY (HIDDEN FROM SCANNER) ---
    # These are used for the 'Drill Down' tab but not the main list
    # We assign them a '99.' category so the scanner logic skips them.
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
    "02. TECHNOLOGY": "XLK",
    "03. ENERGY": "XLE",
    "04. FINANCIALS": "XLF",
    "05. INDUSTRIALS": "XLI",
    "06. DISCRETIONARY": "XLY",
    "07. SECTORS": "SPY", 
    "08. THEMES": "SPY",
    "10. CANADA (HXT)": "HXT.TO"
}

SECTOR_ETFS = list(RRG_SECTORS.keys()) + list(RRG_THEMES.keys())
