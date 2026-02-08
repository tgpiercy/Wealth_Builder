# titan_config.py

# --- SECURITY ---
CREDENTIALS = {
    "dad": "1234",
    "son": "1234"
}

# --- SECTOR PARENT MAP (Scanner Logic) ---
SECTOR_PARENTS = {
    "04. MATERIALS": "XLB", "05. ENERGY": "XLE", "06. FINANCIALS": "XLF",
    "07. INDUSTRIALS": "XLI", "08. TECHNOLOGY": "XLK", "09. COMM SERVICES": "XLC",
    "10. HEALTH CARE": "XLV", "11. CONS DISCRET": "XLY", "12. CONS STAPLES": "XLP",
    "13. UTILITIES / RE": "XLU", "15. CANADA (HXT)": "HXT.TO", "03. THEMES": "SPY"
}

SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLC", "XLV", "XLY", "XLP", "XLU", "XLRE"]

# --- DATA MAP (Scanner Universe) ---
DATA_MAP = {
    "DIA": ["00. INDICES", "SPY", "Dow Jones"], "QQQ": ["00. INDICES", "SPY", "Nasdaq 100"],
    "IWM": ["00. INDICES", "SPY", "Russell 2000"], "IWC": ["00. INDICES", "SPY", "Micro-Cap"],
    "SPY": ["00. INDICES", "SPY", "S&P 500 Base"], "HXT.TO": ["00. INDICES", "SPY", "TSX 60 Index"], 
    "^VIX": ["99. DATA", "SPY", "VIX Volatility"], "RSP": ["99. DATA", "SPY", "S&P Equal Weight"],
    "VOO": ["99. DATA", "SPY", "Vanguard S&P 500"], "IEF": ["01. BONDS/FX", "SPY", "7-10 Year Treasuries"],
    "DLR.TO": ["01. BONDS/FX", None, "USD/CAD Currency"],
    
    # THEMES
    "BOTZ": ["03. THEMES", "SPY", "Robotics & AI"], "AIQ": ["03. THEMES", "SPY", "Artificial Intel"],
    "ARKG": ["03. THEMES", "SPY", "Genomics"], "ICLN": ["03. THEMES", "SPY", "Clean Energy"],
    "TAN": ["03. THEMES", "SPY", "Solar Energy"], "NLR": ["03. THEMES", "SPY", "Nuclear"],
    "URA": ["03. THEMES", "SPY", "Uranium"], "GDX": ["03. THEMES", "SPY", "Gold Miners"],
    "SILJ": ["03. THEMES", "SPY", "Junior Silver"], "COPX": ["03. THEMES", "SPY", "Copper Miners"],
    "REMX": ["03. THEMES", "SPY", "Rare Earths"], "PAVE": ["03. THEMES", "SPY", "Infrastructure"],

    # SECTORS & STOCKS
    "XLB": ["04. MATERIALS", "SPY", "Materials Sector"], "GLD": ["04. MATERIALS", "SPY", "Gold Bullion"],
    "SLV": ["04. MATERIALS", "SPY", "Silver Bullion"], "AA": ["04. MATERIALS", "XLB", "Alcoa"],
    "XLE": ["05. ENERGY", "SPY", "Energy Sector"], "XOP": ["05. ENERGY", "SPY", "Oil & Gas Exp"],
    "OIH": ["05. ENERGY", "SPY", "Oil Services"], "MLPX": ["05. ENERGY", "SPY", "MLP Infra"],
    "XLF": ["06. FINANCIALS", "SPY", "Financials Sector"], "KBE": ["06. FINANCIALS", "SPY", "Bank ETF"],
    "KRE": ["06. FINANCIALS", "SPY", "Regional Banks"], "IAK": ["06. FINANCIALS", "SPY", "Insurance"],
    "XLI": ["07. INDUSTRIALS", "SPY", "Industrials Sector"], "ITA": ["07. INDUSTRIALS", "SPY", "Aerospace"],
    "IYT": ["07. INDUSTRIALS", "SPY", "Transport"],
    "XLK": ["08. TECHNOLOGY", "SPY", "Technology Sector"], "AAPL": ["08. TECHNOLOGY", "QQQ", "Apple Inc"], 
    "MSFT": ["08. TECHNOLOGY", "QQQ", "Microsoft"], "NVDA": ["08. TECHNOLOGY", "QQQ", "Nvidia"],
    "SMH": ["08. TECHNOLOGY", "SPY", "Semiconductors"], "IGV": ["08. TECHNOLOGY", "SPY", "Tech Software"],
    "SMCI": ["08. TECHNOLOGY", "QQQ", "Super Micro"], "DELL": ["08. TECHNOLOGY", "QQQ", "Dell Tech"],
    "WDC": ["08. TECHNOLOGY", "QQQ", "Western Digital"], "ANET": ["08. TECHNOLOGY", "QQQ", "Arista"],
    "CIBR": ["08. TECHNOLOGY", "QQQ", "CyberSecurity"], "PSTG": ["08. TECHNOLOGY", "QQQ", "Pure Storage"],
    "XLC": ["09. COMM SERVICES", "SPY", "Comm Services"], "META": ["09. COMM SERVICES", "QQQ", "Meta"],
    "GOOGL": ["09. COMM SERVICES", "QQQ", "Alphabet"],
    "XLV": ["10. HEALTH CARE", "SPY", "Health Care Sector"], "IBB": ["10. HEALTH CARE", "SPY", "Biotech"],
    "XBI": ["10. HEALTH CARE", "SPY", "Biotech SPDR"], "PPH": ["10. HEALTH CARE", "SPY", "Pharma"],
    "IHI": ["10. HEALTH CARE", "SPY", "Med Devices"],
    "XLY": ["11. CONS DISCRET", "SPY", "Cons Discret"], "AMZN": ["11. CONS DISCRET", "QQQ", "Amazon"],
    "ITB": ["11. CONS DISCRET", "SPY", "Home Construction"],
    "XLP": ["12. CONS STAPLES", "SPY", "Cons Staples"], "MOO": ["12. CONS STAPLES", "SPY", "Agribusiness"],
    "XLU": ["13. UTIL / RE", "SPY", "Utilities"], "XLRE": ["13. UTIL / RE", "SPY", "Real Estate"],
    "CNQ.TO": ["15. CANADA (HXT)", "HXT.TO", "Cdn Natural"], "CP.TO": ["15. CANADA (HXT)", "HXT.TO", "CP Rail"],
    "WSP.TO": ["15. CANADA (HXT)", "HXT.TO", "WSP Global"], "SHOP.TO": ["15. CANADA (HXT)", "HXT.TO", "Shopify"],
    "CSU.TO": ["15. CANADA (HXT)", "HXT.TO", "Constellation"], "NTR.TO": ["15. CANADA (HXT)", "HXT.TO", "Nutrien"],
    "TECK-B.TO": ["15. CANADA (HXT)", "HXT.TO", "Teck Res"], "RY.TO": ["15. CANADA (HXT)", "HXT.TO", "Royal Bank"],
    "BN.TO": ["15. CANADA (HXT)", "HXT.TO", "Brookfield"]
}

# --- RRG DEFINITIONS (New Section) ---
RRG_SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
    "XLY": "Cons. Discret", "XLP": "Cons. Staples", "XLI": "Industrials",
    "XLC": "Comm. Services", "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate"
}

RRG_INDICES = {
    "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWM": "Russell 2000",
    "IWC": "Micro-Cap", "RSP": "S&P Equal Wgt", "^VIX": "Volatility (VIX)",
    "HXT.TO": "TSX 60 (Canada)", "EFA": "Foreign Dev", "EEM": "Emerging Mkts"
}

RRG_THEMES = {
    "BOTZ": "Robotics/AI", "AIQ": "Artificial Intel", "SMH": "Semiconductors",
    "IGV": "Software", "CIBR": "CyberSec", "ARKG": "Genomics",
    "ICLN": "Clean Energy", "TAN": "Solar", "URA": "Uranium", "PAVE": "Infrastructure",
    "GLD": "Gold", "SLV": "Silver", "GDX": "Gold Miners", "COPX": "Copper",
    "MOO": "Agribusiness", "SLX": "Steel", "REMX": "Rare Earths"
}

RRG_INDUSTRY_MAP = {
    "XLK": {"SMH": "Semis", "NVDA": "Nvidia", "IGV": "Software", "MSFT": "Microsoft", "CIBR": "CyberSec", "AAPL": "Apple", "SMCI": "Servers (AI)", "DELL": "Dell", "ANET": "Networking", "WDC": "Storage"},
    "XLF": {"KBE": "Banks", "KRE": "Reg. Banks", "IAI": "Brokers", "IAK": "Insurance", "XP": "Fintech"},
    "XLE": {"XOP": "Exploration", "OIH": "Oil Svcs", "CRAK": "Refiners", "XOM": "Exxon", "CVX": "Chevron"},
    "XLV": {"IBB": "Biotech", "IHI": "Med Devices", "PPH": "Pharma", "UNH": "UnitedHealth"},
    "XLY": {"XRT": "Retail", "ITB": "Homebuild", "PEJ": "Leisure", "AMZN": "Amazon", "TSLA": "Tesla"},
    "XLP": {"PBJ": "Food/Bev", "KXI": "Global Stapl", "COST": "Costco", "PG": "Procter", "WMT": "Walmart"},
    "XLI": {"ITA": "Aerospace", "IYT": "Transport", "JETS": "Airlines", "PAVE": "Infrastruct", "CAT": "Caterpillar"},
    "XLC": {"SOCL": "Social", "PBS": "Media", "GOOGL": "Google", "META": "Meta", "NFLX": "Netflix"},
    "XLB": {"GDX": "Gold Miners", "SIL": "Silver", "LIT": "Lithium", "REMX": "Rare Earth", "COPX": "Copper", "MOO": "Agricul", "SLX": "Steel", "AA": "Alcoa", "DD": "DuPont"},
    "XLU": {"IDU": "US Util", "VPU": "Vanguard Util", "NEE": "NextEra", "DUK": "Duke Energy"},
    "XLRE": {"REZ": "Resid. RE", "BBRE": "BetaBuilders", "PLD": "Prologis", "AMT": "Am. Tower"},
    "Canada (TSX)": {"RY.TO": "Royal Bank", "BN.TO": "Brookfield", "CNQ.TO": "Cdn Natural", "CP.TO": "CP Rail", "WSP.TO": "WSP Global", "SHOP.TO": "Shopify", "CSU.TO": "Constell", "NTR.TO": "Nutrien", "TECK-B.TO": "Teck Res"}
}
