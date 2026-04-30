"""
NSE Stock Universe — NIFTY 50 + NIFTY Next 50 + select high-liquidity mid-caps.
yfinance requires the '.NS' suffix for NSE-listed stocks.
"""

NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "LT.NS", "BAJFINANCE.NS", "WIPRO.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "TITAN.NS", "NESTLEIND.NS", "SUNPHARMA.NS", "POWERGRID.NS",
    "NTPC.NS", "ULTRACEMCO.NS", "BAJAJFINSV.NS", "HCLTECH.NS", "BAJAJ-AUTO.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "INDUSINDBK.NS",
    "JSWSTEEL.NS", "M&M.NS", "ONGC.NS", "SBILIFE.NS", "SHREECEM.NS",
    "TATACONSUM.NS", "TATASTEEL.NS", "TECHM.NS", "TATAMOTORS.NS", "BPCL.NS",
    "CIPLA.NS", "HDFCLIFE.NS", "BRITANNIA.NS", "UPL.NS", "APOLLOHOSP.NS",
]

NIFTY_NEXT_50 = [
    "ADANIGREEN.NS", "ADANIENSOL.NS", "AMBUJACEM.NS", "AUROPHARMA.NS",
    "BANKBARODA.NS", "BERGEPAINT.NS", "BIOCON.NS", "BOSCHLTD.NS",
    "COLPAL.NS", "CONCOR.NS", "DABUR.NS", "DLF.NS", "GAIL.NS",
    "GODREJCP.NS", "GODREJPROP.NS", "HAVELLS.NS", "HINDZINC.NS",
    "ICICIPRULI.NS", "ICICIGI.NS", "IOC.NS", "INDUSTOWER.NS",
    "LUPIN.NS", "MARICO.NS", "UNITDSPR.NS", "MUTHOOTFIN.NS",
    "NAUKRI.NS", "NMDC.NS", "PAGEIND.NS", "PETRONET.NS", "PIDILITIND.NS",
    "PNB.NS", "RECLTD.NS", "SAIL.NS", "SIEMENS.NS", "SRF.NS",
    "TORNTPHARM.NS", "TRENT.NS", "VEDL.NS", "VOLTAS.NS", "ZYDUSLIFE.NS",
]

HIGH_MOMENTUM_MIDCAP = [
    "TATAPOWER.NS", "NHPC.NS", "IRFC.NS", "RVNL.NS", "PFC.NS",
    "CANBK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "ABCAPITAL.NS",
    "ASHOKLEY.NS", "BALKRISIND.NS", "CUMMINSIND.NS", "DIXON.NS",
    "EXIDEIND.NS", "GMRAIRPORT.NS", "INDIANB.NS", "IRCTC.NS",
    "JUBLFOOD.NS", "KPITTECH.NS", "LAURUSLABS.NS", "LICHSGFIN.NS",
    "MFSL.NS", "MOTHERSON.NS", "MPHASIS.NS", "OBEROIRLTY.NS",
    "PERSISTENT.NS", "POLYCAB.NS", "RAMCOCEM.NS", "SBICARD.NS",
    "SUPREMEIND.NS", "TVSMOTOR.NS", "ETERNAL.NS",
]

ALL_STOCKS = NIFTY_50 + NIFTY_NEXT_50 + HIGH_MOMENTUM_MIDCAP

SECTOR_MAP = {
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "BPCL.NS": "Energy",
    "GAIL.NS": "Energy", "PETRONET.NS": "Energy", "IOC.NS": "Energy",
    "TATAPOWER.NS": "Energy", "ADANIGREEN.NS": "Energy", "NHPC.NS": "Energy",
    "NTPC.NS": "Energy", "POWERGRID.NS": "Energy", "PFC.NS": "Energy",
    "RECLTD.NS": "Energy", "ADANIENSOL.NS": "Energy",

    "TCS.NS": "IT", "INFY.NS": "IT", "WIPRO.NS": "IT", "HCLTECH.NS": "IT",
    "TECHM.NS": "IT", "MPHASIS.NS": "IT", "PERSISTENT.NS": "IT",
    "KPITTECH.NS": "IT", "NAUKRI.NS": "IT", "DIXON.NS": "IT",

    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking",
    "AXISBANK.NS": "Banking", "KOTAKBANK.NS": "Banking", "INDUSINDBK.NS": "Banking",
    "BANKBARODA.NS": "Banking", "PNB.NS": "Banking", "FEDERALBNK.NS": "Banking",
    "CANBK.NS": "Banking", "INDIANB.NS": "Banking", "IDFCFIRSTB.NS": "Banking",

    "BAJFINANCE.NS": "NBFC", "BAJAJFINSV.NS": "NBFC", "SBICARD.NS": "NBFC",
    "MUTHOOTFIN.NS": "NBFC", "ABCAPITAL.NS": "NBFC", "LICHSGFIN.NS": "NBFC",
    "MFSL.NS": "NBFC",

    "HDFCLIFE.NS": "Insurance", "SBILIFE.NS": "Insurance",
    "ICICIGI.NS": "Insurance", "ICICIPRULI.NS": "Insurance",

    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG", "DABUR.NS": "FMCG", "MARICO.NS": "FMCG",
    "COLPAL.NS": "FMCG", "GODREJCP.NS": "FMCG", "TATACONSUM.NS": "FMCG",
    "JUBLFOOD.NS": "FMCG", "ETERNAL.NS": "FMCG",

    "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma",
    "DIVISLAB.NS": "Pharma", "AUROPHARMA.NS": "Pharma", "LUPIN.NS": "Pharma",
    "BIOCON.NS": "Pharma", "TORNTPHARM.NS": "Pharma", "LAURUSLABS.NS": "Pharma",
    "ZYDUSLIFE.NS": "Pharma",

    "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto", "M&M.NS": "Auto",
    "BAJAJ-AUTO.NS": "Auto", "HEROMOTOCO.NS": "Auto", "EICHERMOT.NS": "Auto",
    "TVSMOTOR.NS": "Auto", "ASHOKLEY.NS": "Auto", "MOTHERSON.NS": "Auto",
    "BALKRISIND.NS": "Auto",

    "LT.NS": "Infra", "ADANIENT.NS": "Infra", "ADANIPORTS.NS": "Infra",
    "CONCOR.NS": "Infra", "GMRAIRPORT.NS": "Infra",
    "IRFC.NS": "Infra", "RVNL.NS": "Infra", "IRCTC.NS": "Infra",
    "INDUSTOWER.NS": "Infra",

    "TATASTEEL.NS": "Metals", "JSWSTEEL.NS": "Metals", "HINDALCO.NS": "Metals",
    "HINDZINC.NS": "Metals", "COALINDIA.NS": "Metals", "NMDC.NS": "Metals",
    "SAIL.NS": "Metals", "VEDL.NS": "Metals",

    "ULTRACEMCO.NS": "Cement", "SHREECEM.NS": "Cement", "AMBUJACEM.NS": "Cement",
    "GRASIM.NS": "Cement", "RAMCOCEM.NS": "Cement",

    "ASIANPAINT.NS": "Consumer", "TITAN.NS": "Consumer", "BERGEPAINT.NS": "Consumer",
    "HAVELLS.NS": "Consumer", "VOLTAS.NS": "Consumer", "GODREJPROP.NS": "Consumer",
    "TRENT.NS": "Consumer", "PAGEIND.NS": "Consumer", "OBEROIRLTY.NS": "Consumer",
    "SUPREMEIND.NS": "Consumer", "DLF.NS": "Consumer", "PIDILITIND.NS": "Consumer",
    "POLYCAB.NS": "Consumer",

    "BHARTIARTL.NS": "Telecom",

    "BOSCHLTD.NS": "Industrial", "SIEMENS.NS": "Industrial", "CUMMINSIND.NS": "Industrial",
    "EXIDEIND.NS": "Industrial", "SRF.NS": "Industrial", "UNITDSPR.NS": "Consumer",
    "UPL.NS": "Industrial",
}


def get_sector(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, "Other")


def get_display_name(symbol: str) -> str:
    return symbol.replace(".NS", "")
