# ==============================================================================
#  TITAN STRATEGY CONFIGURATION (v67.0 - Specific Industry Headers)
# ==============================================================================

# --- AUTHENTICATION ---
CREDENTIALS = {
    "son": "",  # No password
    "dad": ""   # No password
    "map": "" # No password
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

    # --- 02. SECTORS (Major ETFs) ---
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
    "XBI": ("02. SECTORS", "SPY"), # Biotech (Major)

    # --- 03. THEMES (Major ETFs) ---
    "BOTZ": ("03. THEMES", "SPY"),
    "REMX": ("03. THEMES", "SPY"),
    "ICLN": ("03. THEMES", "SPY"),
    "GDX": ("03. THEMES", "SPY"),

    # --- 04. METALS ---
    "GLD": ("04. METALS", "SPY"),
    "SLV": ("04. METALS", "SPY"),

    # === INDUSTRIES & CONSTITUENTS ===

    # --- 05. TECHNOLOGY (XLK) ---
    "IGV": ("05. TECHNOLOGY (XLK)", "XLK"),   # Software
    "SMH": ("05. TECHNOLOGY (XLK)", "XLK"),   # Semis
    "MSFT": ("05. TECHNOLOGY (XLK)", "XLK"),
    "NVDA": ("05. TECHNOLOGY (XLK)", "SMH"),
    "SMCI": ("05. TECHNOLOGY (XLK)", "XLK"),
    "DELL": ("05. TECHNOLOGY (XLK)", "XLK"),
    "ANET": ("05. TECHNOLOGY (XLK)", "XLK"),
    "WDC": ("05. TECHNOLOGY (XLK)", "XLK"),
    "PSTG": ("05. TECHNOLOGY (XLK)", "XLK"),

    # --- 06. COMMUNICATIONS (XLC) ---
    "META": ("06. COMMUNICATIONS (XLC)", "XLC"),
    "GOOGL": ("06. COMMUNICATIONS (XLC)", "XLC"),

    # --- 07. ENERGY (XLE) ---
    "XOP": ("07. ENERGY (XLE)", "XLE"),   # E&P
    "OIH": ("07. ENERGY (XLE)", "XLE"),   # Services
    "MLPX": ("07. ENERGY (XLE)", "XLE"),  # Midstream

    # --- 08. FINANCIALS (XLF) ---
    "KRE": ("08. FINANCIALS (XLF)", "XLF"), # Reg Banks
    "KBE": ("08. FINANCIALS (XLF)", "XLF"), # Banks
    "IAK": ("08. FINANCIALS (XLF)", "XLF"), # Insurance

    # --- 09. INDUSTRIALS (XLI) ---
    "ITA": ("09. INDUSTRIALS (XLI)", "XLI"),  # Aerospace
    "IYT": ("09. INDUSTRIALS (XLI)", "XLI"),  # Transport
    "PAVE": ("09. INDUSTRIALS (XLI)", "XLI"), # Infrastructure
     "GRID": ("09. INDUSTRIALS (XLI)", "XLI"), # Infrastructure

    # --- 10. MATERIALS (XLB) ---
    "COPX": ("10. MATERIALS (XLB)", "XLB"), # Copper
    "SLX": ("10. MATERIALS (XLB)", "XLB"),  # Steel
    "MOO": ("10. MATERIALS (XLB)", "XLB"),  # Agribusiness
    "AA": ("10. MATERIALS (XLB)", "XLB"),   # Alcoa
    "DD": ("10. MATERIALS (XLB)", "XLB"),   # DuPont

    # --- 11. GOLD MINERS (GDX) ---
    "GDXJ": ("11. GOLD MINERS (GDX)", "GDX"), # Jr Miners
    "SIL": ("11. GOLD MINERS (GDX)", "GDX"),  # Silver Miners

    # --- 12. HEALTHCARE (XLV) ---
    "PPH": ("12. HEALTHCARE (XLV)", "XLV"),  # Pharma
    "IHI": ("12. HEALTHCARE (XLV)", "XLV"),  # Med Devices

    # --- 13. BIOTECH (XBI) ---
    "IBB": ("13. BIOTECH (XBI)", "XBI"),
    "ARKG": ("13. BIOTECH (XBI)", "XBI"),

    # --- 14. DISCRETIONARY (XLY) ---
    "ITB": ("14. DISCRETIONARY (XLY)", "XLY"), # Homebuilders
    "AMZN": ("14. DISCRETIONARY (XLY)", "XLY"),

    # --- 15. AI & ROBOTICS (BOTZ) ---
    "AIQ": ("15. AI & ROBOTICS (BOTZ)", "BOTZ"),

    # --- 16. CLEAN ENERGY (ICLN) ---
    "TAN": ("16. CLEAN ENERGY (ICLN)", "ICLN"), # Solar
    "NLR": ("16. CLEAN ENERGY (ICLN)", "ICLN"), # Nuclear
    "URA": ("16. CLEAN ENERGY (ICLN)", "ICLN"), # Uranium

    # --- 17. CANADA (HXT) ---
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
  #  "XLK": {"MSFT": "Microsoft", "NVDA": "Nvidia", "SMH": "Semis", "IGV": "Software", "ANET": "Arista", "DELL": "Dell"},
 #   "XLC": {"META": "Meta", "GOOGL": "Google"},
  #  "XLF": {"KBE": "Banks", "KRE": "Reg Banks", "IAK": "Insurance"},
    "XLE": {"XOP": "Exploration", "OIH": "Oil Services", "MLPX": "Midstream"},
  #  "XLV": {"PPH": "Pharma", "IHI": "Med Devices", "XBI": "Biotech"},
    "XLI": {"ITA": "Aerospace", "IYT": "Transport", "PAVE": "Infrastructure", "GRID": "Elec. Infra"},
    "XLB": {"COPX": "Copper", "SLX": "Steel", "MOO": "Agri", "AA": "Alcoa", "DD": "DuPont"},
  #  "XLY": {"AMZN": "Amazon", "ITB": "Homebuilders"},
   # "GDX": {"GDXJ": "Junior Miners", "SIL": "Silver Miners"},
 #   "BOTZ": {"AIQ": "Artificial Intel"},
  #  "ICLN": {"TAN": "Solar", "NLR": "Nuclear", "URA": "Uranium"}
}

# --- PARENT MAPPING (UPDATED FOR NEW HEADERS) ---
# Ensures "Caution/Avoid" logic flows from Parent ETF to Child Header
SECTOR_PARENTS = {
    "02. SECTORS": "SPY",
    "03. THEMES": "SPY",
    "05. TECHNOLOGY (XLK)": "XLK",
    "06. COMMUNICATIONS (XLC)": "XLC",
    "07. ENERGY (XLE)": "XLE",
    "08. FINANCIALS (XLF)": "XLF",
    "09. INDUSTRIALS (XLI)": "XLI",
    "10. MATERIALS (XLB)": "XLB",
    "11. GOLD MINERS (GDX)": "GDX",
    "12. HEALTHCARE (XLV)": "XLV",
    "13. BIOTECH (XBI)": "XBI",
    "14. DISCRETIONARY (XLY)": "XLY",
    "15. AI & ROBOTICS (BOTZ)": "BOTZ",
    "16. CLEAN ENERGY (ICLN)": "ICLN",
    "17. CANADA (HXT)": "HXT.TO"
}

SECTOR_ETFS = list(RRG_SECTORS.keys()) + list(RRG_THEMES.keys())
