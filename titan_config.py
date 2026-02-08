# titan_config.py

BENCHMARK_US = "SPY"
BENCHMARK_CA = "HXT.TO"

# 1. Macro Sectors (US)
SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care", 
    "XLY": "Cons. Discret", "XLP": "Cons. Staples", "XLI": "Industrials", 
    "XLC": "Comm. Services", "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate"
}

# 2. Major Indices
INDICES = {
    "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWM": "Russell 2000", 
    "IWC": "Micro-Cap", "RSP": "S&P Equal Wgt", "^VIX": "Volatility",
    "HXT.TO": "TSX 60 (Canada)", "EFA": "Foreign Dev (EAFE)", "EEM": "Emerging Mkts"
}

# 3. Structural Themes
THEMES = {
    "BOTZ": "Robotics/AI", "AIQ": "Artificial Intel", "SMH": "Semiconductors", 
    "IGV": "Software", "CIBR": "CyberSec", "ARKG": "Genomics",
    "ICLN": "Clean Energy", "TAN": "Solar", "URA": "Uranium", "PAVE": "Infrastructure",
    "GLD": "Gold", "SLV": "Silver", "GDX": "Gold Miners", "COPX": "Copper",
    "MOO": "Agricul", "SLX": "Steel"
}

# 4. Micro Industries
INDUSTRY_MAP = {
    "XLK": {"SMH": "Semis", "NVDA": "Nvidia", "IGV": "Software", "MSFT": "Microsoft", "CIBR": "CyberSec", "AAPL": "Apple", "SMCI": "Servers", "ANET": "Networks"},
    "XLF": {"KBE": "Banks", "KRE": "Reg. Banks", "IAI": "Brokers", "IAK": "Insurance", "XP": "Fintech"},
    "XLE": {"XOP": "Exploration", "OIH": "Oil Svcs", "CRAK": "Refiners", "XOM": "Exxon", "CVX": "Chevron"},
    "XLV": {"IBB": "Biotech", "IHI": "Med Devices", "PPH": "Pharma", "UNH": "UnitedHealth"},
    "XLY": {"XRT": "Retail", "ITB": "Homebuild", "PEJ": "Leisure", "AMZN": "Amazon", "TSLA": "Tesla"},
    "XLP": {"PBJ": "Food/Bev", "KXI": "Global Stapl", "COST": "Costco", "PG": "Procter", "WMT": "Walmart"},
    "XLI": {"ITA": "Aerospace", "IYT": "Transport", "JETS": "Airlines", "PAVE": "Infrastruct", "CAT": "Caterpillar"},
    "XLC": {"SOCL": "Social", "PBS": "Media", "GOOGL": "Google", "META": "Meta", "NFLX": "Netflix"},
    "XLB": {"GDX": "Gold Miners", "SIL": "Silver", "LIT": "Lithium", "REMX": "Rare Earth", "COPX": "Copper", "MOO": "Ag", "SLX": "Steel"},
    "XLU": {"IDU": "US Util", "VPU": "Vanguard Util", "NEE": "NextEra", "DUK": "Duke Energy"},
    "XLRE": {"REZ": "Resid. RE", "BBRE": "BetaBuilders", "PLD": "Prologis", "AMT": "Am. Tower"},
    "Canada (TSX)": {"RY.TO": "Royal Bank", "BN.TO": "Brookfield", "CNQ.TO": "Cdn Natural", "CP.TO": "CP Rail", "WSP.TO": "WSP Global", "SHOP.TO": "Shopify", "CSU.TO": "Constell", "NTR.TO": "Nutrien", "TECK-B.TO": "Teck Res"}
}

