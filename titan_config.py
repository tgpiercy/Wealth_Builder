# ==============================================================================
#  TITAN STRATEGY CONFIGURATION (v65.9 - Clean List)
# ==============================================================================

# --- AUTHENTICATION ---
CREDENTIALS = {
    "admin": "password",  # Change this for security
    "user": "titan"
}

# --- MASTER DATA MAP ---
# Format: { Ticker : (Display_Category, Benchmark_Ticker) }
DATA_MAP = {
    # --- 01. MARKETS ---
    "SPY": ("01. MARKETS", "SPY"),
    "RSP": ("01. MARKETS", "SPY"),
    "DIA": ("01. MARKETS", "SPY"),
    "QQQ": ("01. MARKETS", "SPY"),
    "IWM": ("01. MARKETS", "SPY"),
    "IWC": ("01. MARKETS", "SPY"),
    "HXT.TO": ("01. MARKETS", "SPY"), # TSX 60
    "IEF": ("01. MARKETS", "SPY"),    # Bonds
    "^VIX": ("01. MARKETS", "SPY"),   # Volatility

    # --- 02. SECTORS (TIER 1 - PRIMARY) ---
    "XLC": ("02. SECTORS", "SPY"),
    "XLF": ("02. SECTORS", "SPY"),
    "XLI": ("02. SECTORS", "SPY"),
    "XLK": ("02. SECTORS", "SPY"),

    # --- 02. SECTORS (TIER 2 - TACTICAL) ---
    "XLB": ("02. SECTORS", "SPY"),
    "XLE": ("02. SECTORS", "SPY"),
    "XLV": ("02. SECTORS", "SPY"),
    "XLY": ("02. SECTORS", "SPY"),

    # --- 02. SECTORS (DEFENSIVE) ---
    "XLP": ("02. SECTORS", "SPY"),
    "XLRE": ("02. SECTORS", "SPY"),
    "XLU": ("02. SECTORS", "SPY"),

    # --- 03. SPECIALIZED (TREND) ---
    "BOTZ": ("03. THEMES", "SPY"),
    "REMX": ("03. THEMES", "SPY"),
    "ICLN": ("03. THEMES", "SPY"),

    # --- 03. SPECIALIZED (VOLATILE) ---
    "XBI": ("03. THEMES", "SPY"),
    "GDX": ("03. THEMES", "SPY"),

    # --- 04. PRECIOUS METALS ---
    "GLD": ("04. METALS", "SPY"),
    "SLV": ("04. METALS", "SPY"),

    # --- INDUSTRIES / DRILL DOWN ---
    
    # AI & ROBOTICS (BOTZ)
    "AIQ": ("05. AI & ROBOTICS", "BOTZ"),

    # BIOTECH (XBI)
    "IBB": ("06. BIOTECH (XBI)", "XBI"),
    "ARKG": ("06. BIOTECH (XBI)", "XBI"),

    # CLEAN ENERGY (ICLN)
    "TAN": ("07. CLEAN ENERGY (ICLN)", "ICLN"),
    "NLR": ("07. CLEAN ENERGY (ICLN)", "ICLN"),
    "URA": ("07. CLEAN ENERGY (ICLN)", "ICLN"),

    # MINERS (GDX)
    "GDXJ": ("08. MINERS (GDX)", "GDX"),
    "SIL": ("08. MINERS (GDX)", "GDX"),

    # MATERIALS (XLB)
    "COPX": ("09. MATERIALS (XLB)", "XLB"),
    "AA": ("09. MATERIALS (XLB)", "XLB"),
    "SLX": ("09. MATERIALS (XLB)", "XLB"),
    "DD": ("09. MATERIALS (XLB)", "XLB"),
    "MOO": ("09. MATERIALS (XLB)", "XLB"),

    # COMMS (XLC)
    "META": ("10. COMMS (XLC)", "XLC"),
    "GOOGL": ("10. COMMS (XLC)", "XLC"),

    # ENERGY (XLE)
    "XOP": ("11. ENERGY (XLE)", "XLE"),
    "OIH": ("11. ENERGY (XLE)", "XLE"),
    "MLPX": ("11. ENERGY (XLE)", "XLE"),

    # FINANCIALS (XLF)
    "KBE": ("12. FINANCIALS (XLF)", "XLF"),
    "KRE": ("12. FINANCIALS (XLF)", "XLF"),
    "IAK": ("12. FINANCIALS (XLF)", "XLF"),

    # INDUSTRIALS (XLI)
    "ITA": ("13. INDUSTRIALS (XLI)", "XLI"),
    "IYT": ("13. INDUSTRIALS (XLI)", "XLI"),
    "PAVE": ("13. INDUSTRIALS (XLI)", "XLI"),

    # TECHNOLOGY (XLK)
    "SMCI": ("14. TECHNOLOGY (XLK)", "XLK"),
    "DELL": ("14. TECHNOLOGY (XLK)", "XLK"),
    "WDC": ("14. TECHNOLOGY (XLK)", "XLK"),
    "PSTG": ("14. TECHNOLOGY (XLK)", "XLK"),
    "ANET": ("14. TECHNOLOGY (XLK)", "XLK"),
    "IGV": ("14. TECHNOLOGY (XLK)", "XLK"),
    "MSFT": ("14. TECHNOLOGY (XLK)", "XLK"),
    "SMH": ("14. TECHNOLOGY (XLK)", "XLK"),
    "NVDA": ("14. TECHNOLOGY (XLK)", "XLK"),

    # HEALTHCARE (XLV)
    "PPH": ("15. HEALTHCARE (XLV)", "XLV"),
    "IHI": ("15. HEALTHCARE (XLV)", "XLV"),

    # CONSUMER DISCRETIONARY (XLY)
    "ITB": ("16. DISCRETIONARY (XLY)", "XLY"),
    "AMZN": ("16. DISCRETIONARY (XLY)", "XLY"),

    # CANADA (HXT)
    "RY.TO": ("17. CANADA (HXT)", "HXT.TO"),
    "BN.TO": ("17. CANADA (HXT)", "HXT.TO"),
    "CNQ.TO": ("17. CANADA (HXT)", "HXT.TO"),
    "CP.TO": ("17. CANADA (HXT)", "HXT.TO"),
    "WSP.TO": ("17. CANADA (HXT)", "HXT.TO"),
    "SHOP.TO": ("17. CANADA (HXT)", "HXT.TO"),
    "CSU.TO": ("17. CANADA (HXT)", "HXT.TO"),
    "NTR.TO": ("17. CANADA (HXT)", "HXT.TO"),
    "TECK-B.TO": ("17. CANADA (HXT)", "HXT.TO"),
}

# --- RRG GROUPS ---

# 1. INDICES (Markets)
RRG_INDICES = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "DIA": "Dow Jones",
    "IWM": "Russell 2000",
    "IWC": "Micro-Cap",
    "HXT.TO": "TSX 60 (CAD)",
    "IEF": "Bonds (7-10Y)"
}

# 2. SECTORS (All US Sectors)
RRG_SECTORS = {
    "XLK": "Technology",
    "XLC": "Comms",
    "XLY": "Discretionary",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLE": "Energy",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLP": "Staples"
}

# 3. THEMES (Specialized)
RRG_THEMES = {
    "BOTZ": "Robotics/AI",
    "XBI": "Biotech",
    "ICLN": "Clean Energy",
    "GDX": "Gold Miners",
    "SMH": "Semiconductors",
    "IGV": "Software",
    "XOP": "Oil & Gas E&P",
    "KRE": "Regional Banks",
    "ITA": "Aerospace",
    "ITB": "Homebuilders"
}

# 4. DRILL-DOWN MAP (For 'Drill-Down' Tab)
# Maps a Parent Sector (Key) to a Dictionary of Children (Values)
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

# 5. PARENT MAPPING (For Logic Locks)
# If Parent is AVOID, Child is Caution
SECTOR_PARENTS = {
    "05. AI & ROBOTICS": "BOTZ",
    "06. BIOTECH (XBI)": "XBI",
    "07. CLEAN ENERGY (ICLN)": "ICLN",
    "08. MINERS (GDX)": "GDX",
    "09. MATERIALS (XLB)": "XLB",
    "10. COMMS (XLC)": "XLC",
    "11. ENERGY (XLE)": "XLE",
    "12. FINANCIALS (XLF)": "XLF",
    "13. INDUSTRIALS (XLI)": "XLI",
    "14. TECHNOLOGY (XLK)": "XLK",
    "15. HEALTHCARE (XLV)": "XLV",
    "16. DISCRETIONARY (XLY)": "XLY",
    "17. CANADA (HXT)": "HXT.TO"
}

# 6. SECTOR ETFS LIST (For Deduplication in Scanner)
SECTOR_ETFS = list(RRG_SECTORS.keys()) + list(RRG_THEMES.keys())
