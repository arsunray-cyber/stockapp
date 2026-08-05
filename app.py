import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import pipeline
import plotly.graph_objects as go
import plotly.express as px
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
import time
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# 1. Page configuration - Professional Web 2.0 Theme
# ============================================================
st.set_page_config(
    page_title="Indian Share Market AI Tool",
    layout="wide",
    page_icon="📈"
)

# Custom CSS for professional Web 2.0 theme
st.markdown("""
<style>
    /* Main theme colors - Professional Blue & Green palette */
    :root {
        --primary-color: #2563eb;
        --secondary-color: #10b981;
        --accent-color: #8b5cf6;
        --background-color: #f8fafc;
        --card-background: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --success-color: #22c55e;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
    }
    
    /* Global styles */
    .stApp {
        background-color: var(--background-color);
    }
    
    /* Card styling */
    .metric-card {
        background: var(--card-background);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        transform: translateY(-2px);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        color: white;
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.2);
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-top: -20px;
    }
    
    /* Tooltip styling */
    .tooltip-container {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted var(--text-secondary);
        cursor: help;
    }
    
    /* Metric value styling */
    .stMetricValue {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--primary-color) !important;
    }
    
    .stMetricLabel {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
    }
    
    /* Dataframe styling */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid var(--primary-color);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #eff6ff, #f0f9ff);
        border-left: 4px solid var(--primary-color);
        padding: 15px 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
        border-left: 4px solid var(--success-color);
        padding: 15px 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fffbeb, #fef3c7);
        border-left: 4px solid var(--warning-color);
        padding: 15px 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-color);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Helper function for tooltips
# ============================================================
def tooltip(text, explanation):
    """Create a tooltip with explanation"""
    return f"""
    <span class="tooltip-container" title="{explanation}">
        {text} ℹ️
    </span>
    """

# ============================================================
# 2. AI Sentiment Model
# ============================================================
@st.cache_resource
def load_sentiment_model():
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception:
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

sentiment_analyzer = load_sentiment_model()

# ============================================================
# 3. App Header with Professional Styling
# ============================================================
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.5rem;">📊 Indian Share Market AI Tool</h1>
    <p style="margin: 10px 0 0 0; opacity: 0.9;">
        Advanced market scanner with AI-powered sentiment analysis, technical indicators, 
        and price predictions for NSE/BSE
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 4. Exchange Selection
# ============================================================
exchange = st.sidebar.radio(
    "Select Market / Exchange:",
    ["NSE", "BSE"],
    horizontal=True,
    help="Choose National Stock Exchange (NSE) or Bombay Stock Exchange (BSE)"
)

suffix = ".NS" if exchange == "NSE" else ".BO"
exchange_full = (
    "National Stock Exchange (NSE)"
    if exchange == "NSE"
    else "Bombay Stock Exchange (BSE)"
)

# Sidebar info box
st.sidebar.markdown("""
<div class="info-box">
    <strong>💡 Quick Tips:</strong><br>
    • Use <b>Large Cap</b> for stable, established companies<br>
    • Use <b>Mid Cap</b> for growing companies with moderate risk<br>
    • Use <b>Small Cap</b> for high-growth potential with higher risk<br>
    • <b>Nifty Sectors</b> help analyze specific industries
</div>
""", unsafe_allow_html=True)

# ============================================================
# 5. Default Cap Segment Baskets - COMPREHENSIVE LISTS
# ============================================================
# These baskets contain representative stocks from each segment.
# Users can also add custom symbols from the UI in Tab 2.

BASE_BASKETS = {
    # ========== MARKET CAP SEGMENTS ==========
    "Large Cap": [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", 
        "BHARTIARTL", "ITC", "LT", "HINDUNILVR", "AXISBANK", "BAJFINANCE",
        "MARUTI", "SUNPHARMA", "ASIANPAINT", "KOTAKBANK", "TITAN",
        "ULTRACEMCO", "NESTLEIND", "BAJAJFINSV", "POWERGRID", "NTPC",
        "ONGC", "WIPRO", "HCLTECH", "ADANIENT", "ADANIGREEN", "TATAMOTORS",
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA", "GRASIM",
        "CIPLA", "DRREDDY", "DIVISLAB", "EICHERMOT", "M&M", "BRITANNIA",
        "GODREJCP", "DABUR", "COLPAL", "PIDILITIND", "BERGEPAINT",
        "SIEMENS", "ABB", "SCHAEFFLER", "CUMMINS", "THERMAX",
    ],
    "Mid Cap": [
        "PERSISTENT", "KPITTECH", "CUMMINSIND", "BAJAJELEC", "FEDERALBNK",
        "IDFCFIRSTB", "AUBANK", "IEX", "BSE", "BALAMINES", "ALKYLAMINE",
        "APOLLOTYRE", "ASHOKLEY", "BANKBARODA", "BEL", "BHEL", "COCHINSHIP",
        "CONCOR", "CROMPTON", "CREDITACC", "DEEPAKNTR", "DHANI", "DLF",
        "ESCORTS", "EXIDEIND", "GLENMARK", "GMRINFRA", "HAL", "HAVELLS",
        "HEG", "HONAUT", "HUDCO", "ICICIPRULI", "IDBI", "IDFC", "IGL",
        "INDHOTEL", "INDIAMART", "IRCTC", "ISEC", "JINDALSTEL", "JUBLFOOD",
        "KEI", "LALPATHLAB", "LAURUSLABS", "LICHSGFIN", "LTIM", "LUPIN",
        "MANAPPURAM", "MCX", "METROPOLIS", "MFSL", "MOTHERSON", "NAM-INDIA",
        "NAUKRI", "OFSS", "PAGEIND", "PEL", "PFC", "PNB", "POLYCAB",
        "POONAWALLA", "PVRINOX", "RAMCOCEM", "RBLBANK", "RECLTD", "SAIL",
        "SANOFI", "SHRIRAMFIN", "SJVN", "SOLARINDS", "STARHEALTH", "SUNTV",
        "SUPREMEIND", "SYNGENE", "TATACOMM", "TATATECH", "TORNTPHARM",
        "TRENT", "UCOBANK", "UNITDSPR", "VOLTAS", "WIPRO", "ZYDUSWELL",
    ],
    "Small Cap": [
        "TANLA", "PGEL", "HAPPSTMNTHS", "GRAVITA", "GANESHHOUC", "GABRIEL",
        "JINDALSAW", "MAZDOCK", "SHYAMCENT", "TITAGARH", "AARTIDRUGS",
        "AARTIIND", "ACRYCIL", "ADF", "ADVENZYMES", "AGARIND", "AGROPHOS",
        "AKSHARCHEM", "ALBERTDAVD", "ALICON", "ALPSINDUS", "AMBIKCO",
        "ANANTRAJ", "ANDHRAPAP", "ANSALAPI", "ANTGRAPHIC", "APCOTEXIND",
        "ARCHIDPLY", "ARIHANTCAP", "ARIHANTSUP", "ARMANFIN", "AROGRANITE",
        "ARPITA", "ASAHIINDIA", "ASHAPURO", "ASHNOOR", "ASAL", "ASTERDM",
        "ASTRAL", "ASTRON", "ATGL", "ATUL", "AURIONPRO", "AVADHSUGAR",
        "BAFNAPH", "BALAMINES", "BALMLAWRIE", "BANARISUG", "BANCOINDIA",
        "BANG", "BASML", "BAYERCROP", "BEARDSELL", "BEDMUTHA", "BEPL",
        "BESTSTEEL", "BGRENERGY", "BHAGERIA", "BHARATRAS", "BHARATWIRE",
        "BILPOWER", "BINANIIND", "BIRLACABLE", "BIRLACORPN", "BKMINDST",
        "BLACKROSE", "BLISSGVS", "BLUECHIP", "BLUEDART", "BLUESTARCO",
        "BODALCHEM", "BORORENEW", "BPL", "BRFL", "BRIGADE", "BRIGHTCOM",
        "BSL", "BURNPUR", "CALSOFT", "CAMLINFINE", "CAPACITE", "CAPLIPOINT",
        "CAPTRUST", "CARERATING", "CASTEXTECH", "CATVISION", "CEATLTD",
        "CELEBRITY", "CELLO", "CENTEXT", "CENTRALBK", "CENTUM", "CERA",
        "CEREBRAINT", "CESC", "CGCL", "CHALET", "CHEMCON", "CHEMFAB",
        "CHENNPETRO", "CHOICEIN", "CINEVISTA", "CLEDUCATE", "CLNENERGY",
        "CMMIPL", "CMSINFO", "COASTCORP", "COMPINFO", "COMPUSOFT",
        "CONTROLPR", "CORPORATE", "COSMOFILMS", "CREATIVEYE", "CRESSANDA",
        "CRISIL", "CROWN", "CSBBANK", "CSSL", "CTE", "CYBERMEDIA",
        "DATAMATICS", "DATAPATTNS", "DBL", "DBREALTY", "DCAL", "DCBBANK",
        "DCM", "DCMNVL", "DCW", "DELPHIFX", "DENORA", "DEVIT", "DGCONTENT",
        "DHAMPURSUG", "DHANUKA", "DHARSUGAR", "DHOOTIND", "DIAMONDYD",
        "DIGJAMLTD", "DISHTV", "DIVGIITTS", "DJML", "DKENTER", "DODLA",
        "DOLLAR", "DOLPHIN", "DOMSIND", "DPABHUSHAN", "DQE", "DREAMFOLKS",
        "DRL", "DSSL", "DUCON", "DWARKESH", "DYNAMATECH", "E2E", "EASEMYTRIP",
        "ECLERX", "ECOSMOBLTY", "EDSYS", "EIHAHOTELS", "EIMCOELECO",
        "ELGIEQUIP", "ELNET", "EMAMILTD", "EMKAYGLOBAL", "EMMBI", "EMSLIMITED",
        "ENDURANCE", "ENERGYDEV", "ENGIL", "EPACK", "EQUITASBNK", "ERIS",
        "ESAFSFB", "ESSARSHPNG", "ESTER", "ETHOSLTD", "EUROTEXIND",
        "EXCEL", "EXCELINDUS", "EXPLEOSOL", "FACT", "FAIRCHEM", "FCL",
        "FDC", "FERTIZING", "FGP", "FILATEX", "FINEORG", "FLEXITUFF",
        "FLUOROCHEM", "FMGOETZE", "FONEGA", "FORTIS", "FOSECOIND",
        "FRETAIL", "FSC", "FUSION", "GALAXYSURF", "GALLISPAT", "GANDHAR",
        "GANGESSECUR", "GARFIBRES", "GATEWAY", "GAYAPROJ", "GBFL",
        "GCL", "GEECEE", "GENESYS", "GENUSPOWER", "GEOJITFNC", "GHCL",
        "GICRE", "GILLANDERS", "GINISHREE", "GKWLIMITED", "GLFL",
        "GLOBALVECT", "GLOBUSSPR", "GMBREW", "GMDCLTD", "GMMPFAUDLR",
        "GNA", "GNFC", "GOACARBON", "GOCOLORS", "GODFRYPHLP", "GODHA",
        "GOKEX", "GOKUL", "GOLDIAM", "GOODLUCK", "GOPAL", "GORANI",
        "GOTREKHARA", "GPIL", "GPPL", "GRPLTD", "GRSE", "GRUH", "GSFC",
        "GSKCONS", "GTPL", "GUFICBIO", "GUJALKALI", "GUJAPOLLO", "GUJGASLTD",
        "GUJRAFFIA", "GUJTHEM", "GULFPETRO", "GULSHAN", "GVKPIL", "HANUNG",
        "HARIOMPIPE", "HARRMALAYA", "HATHWAY", "HATSUN", "HAVISHA",
        "HBLENGINE", "HBLPOWER", "HCC", "HDIL", "HECPROJECT", "HEMIPROP",
        "HERANBA", "HERITGFOOD", "HFCL", "HIMATSEINV", "HIMSFUT", "HINDCOMPOS",
        "HINDCON", "HINDCOPPER", "HINDMOTORS", "HINDOILEXP", "HINDPETRO",
        "HINDSYNTH", "HIRECT", "HITECH", "HITECHGEAR", "HLVLTD", "HMVL",
        "HONDAPOWER", "HOVS", "HSCL", "HTMEDIA", "HUBTOWN", "HYBRIDFIN",
        "IBREALEST", "IBULHSGFIN", "ICDS", "ICIL", "ICRA", "ID", "IFBIND",
        "IFCI", "IFGLEXPOR", "IGARASHI", "IGL", "IIFL", "IKIO", "IL&FSENGG",
        "IL&FSTRANS", "IMPAL", "INCREDIBLE", "INDCARD", "INDCOUNT", "INDELCAP",
        "INDELMAT", "INDEPENDENT", "INDIANCARD", "INDIANHUME", "INDIGO",
        "INDOTHAI", "INDRAIL", "INDSWFTLAB", "INDTERRAIN", "INDUSINDBK",
        "INFIBEAM", "INNOVATORS", "INSECTICID", "INTELLECT", "INTENSE",
        "IOB", "IOC", "IOLCP", "IRB", "IRC", "IRCON", "IRFC", "IRIS",
        "IRMENERGY", "ISFT", "ISGEC", "ITDCEM", "IVP", "IVZINVBNK",
        "IWIND", "JAGRAN", "JAIBALA", "JAICORPLTD", "JAMNAAUTO", "JAYAGROGN",
        "JAYBARMARU", "JAYSREETEA", "JBMA", "JBFIND", "JCHAC", "JCTL",
        "JHS", "JINDALPHOT", "JINDRILL", "JINDWORLD", "JIOSL", "JKCEMENT",
        "JKIL", "JKLAKSHMI", "JKPAPER", "JKTYRE", "JMA", "JMCPROJECT",
        "JMFINANCIL", "JOCIL", "JPASSOCIAT", "JPPOWER", "JSWENERGY",
        "JSWHL", "JTEKTINDIA", "JUBLINDS", "JUBLPHARMA", "JUSTDIAL",
        "JVLAGRO", "JWL", "KABRAEXTRU", "KAJARIRATE", "KAKATCEM",
        "KALAMANDIR", "KALPATPOWER", "KAMAHOLD", "KAMOPAINTS", "KANANIIND",
        "KANORICHEM", "KANSAINER", "KAPSTON", "KARMAENG", "KARURVYSYA",
        "KAUSHALYA", "KBCGLOBAL", "KBIL", "KDDL", "KEC", "KECL",
        "KEYCORPSER", "KFS", "KHANDWALA", "KHADIM", "KHAITANLTD",
        "KHANDSE", "KICL", "KILITCH", "KINGFA", "KIRIINDUS", "KIRLOSIND",
        "KIRLOSENG", "KIRLOSBROS", "KIRLFY", "KOHINOOR", "KOKUYOCMLN",
        "KOLTEPATIL", "KOTYARK", "KPIGREEN", "KRBL", "KRETTOSYS",
        "KRSNAA", "KSB", "KSERASERA", "KSHITIJPOL", "KSK", "KSL",
        "KTKBANK", "Kuantum", "KVERMA", "LABH", "LAKPRE", "LAKSHMIEFL",
        "LAMBODHARA", "LANDMARK", "LAOPALA", "LASA", "LATENTVIEW",
        "LAURUSLABS", "LAXMICOT", "LCCINFOTEC", "LEMONTREE", "LEMERITE",
        "LEMITAL", "LGBFORGE", "LGHL", "LIBERTSHOE", "LINCOLN", "LINC",
        "LLOYDSME", "LODHA", "LOKESHMACH", "LOTUSEYE", "LOVABLE",
        "LOYALTEX", "LPS", "LUMAXIND", "LUMAXTECH", "LUXIND", "LYKALABS",
        "MADRASFERT", "MADHAV", "MADHUCRIST", "MAGNUM", "MAHASTEEL",
        "MAHLIFE", "MAHLOG", "MAILTASK", "MANAKALUCO", "MANAKCOAT",
        "MANAKIND", "MANAKSIA", "MANAKSTEEL", "MANDHANA", "MANGALAM",
        "MANGCHEFER", "MANINDS", "MANOMAY", "MANORG", "MANPASAND",
        "MANSHREE", "MAPMYINDIA", "MARALOVER", "MARICO", "MARINE",
        "MARKSANS", "MARUTI", "MASFIN", "MASKINVEST", "MASTEK",
        "MATRIMONY", "MAWANASUG", "MAXHEALTH", "MAXVIL", "MAYURUNIQ",
        "MCDOWELL-N", "MCLEODRUSS", "MEDANTA", "MEDICAMEN", "MEDPLUS",
        "MEDICO", "MEGH", "MENONBE", "MEP", "METALFORGE", "METKORE",
        "MIDCAPCAP", "MINDACORP", "MINDTECK", "MIRZAINT", "MITCON",
        "MITTAL", "MKPL", "MMFL", "MMTC", "MODERN", "MODIRUBBER",
        "MODISONLTD", "MOHOTAIND", "MONARCH", "MONTECARLO", "MOREPENLAB",
        "MOTILALOFS", "MOTOGENFIN", "MPHASIS", "MPSLTD", "MRF", "MRPL",
        "MTNL", "MUDRA", "MUKANDLTD", "MUKKA", "MULTIBASE", "MULTIPLUS",
        "MUNJALAU", "MUNJALSHOW", "MURUDCERA", "MUTHOOTFIN", "MVKAGRO",
        "NACLIND", "NAGREEKCAP", "NAGREEKEXP", "NAHARPOLY", "NAHARSPING",
        "NANDAN", "NANDANI", "NARMADA", "NATCOPHARM", "NATIONALUM",
        "NAVKARCORP", "NAVINFLUOR", "NAVKARCL", "NBCC", "NBIFIN",
        "NCC", "NDGL", "NDL", "NDRAUTO", "NDTV", "NELCAST", "NELCO",
        "NEOGEN", "NERLE", "NETWORK18", "NEWGEN", "NEXTMEDIA",
        "NFL", "NHPC", "NICCO", "NIFTYBEES", "NIITLTD", "NIITMTS",
        "NILAINFRA", "NILASPACES", "NILKAMAL", "NINTEC", "NIRAJCEMENT",
        "NIRAJISPAT", "NITCO", "NITINSPIN", "NITIRAJ", "NKIND",
        "NLCINDIA", "NOCIL", "NORMALLY", "NOVARTIND", "NRB", "NRBBEARING",
        "NSEBEES", "NTL", "NUCLEUS", "NURECA", "NXTDIGITAL", "OBEROI",
        "ODIGMA", "OIL", "OLECTRA", "OMAXAUTO", "OMAXE", "OMINFRAL",
        "OMKARCHEM", "ONELIFECAP", "ONMOBILE", "OPTIEMUS", "ORBTEXP",
        "ORCHPHARMA", "ORIENTBELL", "ORIENTCEM", "ORIENTHOT", "ORIENTLTD",
        "ORISSAMINE", "OSIAHYPER", "OSWALAGRO", "OSWALMIN", "OSWALYARN",
        "PAEL", "PAGEIND", "PAISALO", "PALREDTEC", "PANACEABIO",
        "PANAMAPET", "PARABDRUGS", "PARACABLES", "PARAGMIL", "PARAS",
        "PAREKH", "PARIN", "PARSVNATH", "PARTYCRUSERS", "PASUPTAC",
        "PATANJALI", "PATINTLOG", "PAVNAIND", "PAYTM", "PBAINFRA",
        "PBAINFRA", "PBM", "PDSL", "PEARLPOLY", "PEGASUS", "PENIND",
        "PENPEBS", "PENTAMED", "PENTOCER", "PERFECT", "PERMIA",
        "PERSISTENT", "PETRONET", "PFIZER", "PFL", "PFOCUS", "PGHL",
        "PGIL", "PHOENIXLTD", "PHOTOSON", "PIDILITIND", "PIIND",
        "PILANIINVS", "PILITA", "PIONEEREMB", "PIPAVAVDOC", "PIRPHYTO",
        "PKTEA", "PLASTIBLEN", "PLATIND", "PLAZACABLE", "PLCINDUS",
        "PMC", "PMCFIN", "PNBHOUSING", "PNCINFRA", "PODDARHOUS",
        "PODDARMENT", "POKARNA", "POLYMED", "POLYPLEX", "PONNIERODE",
        "POWERINDIA", "POWERMECH", "PPAP", "PPL", "PRAJIND",
        "PRAKASH", "PRAKASHSTL", "PRAMARA", "PRAXIS", "PRECAM",
        "PRECISION", "PRECOT", "PREMIERPOL", "PREMIERENRG", "PREMIUM",
        "PRESTIGE", "PRICOLLTD", "PRIMESECU", "PRIMO", "PRISMJOHNSON",
        "PRITHVI", "PRITI", "PRITIKAUTO", "PRIVISCL", "PRUDENT",
        "PSB", "PSPPROJECT", "PTL", "PUNJABCHEM", "PUNJLLOYD",
        "PURVA", "PVP", "PVRINOX", "PYRAMID", "QUICKHEAL", "QUINTEGRA",
        "RADHIKAJWE", "RADIANTCMS", "RAIN", "RAJRAYON", "RAJRATAN",
        "RAJSHREE", "RAJTV", "RAJVIR", "RALLIS", "RAMANEWS",
        "RAMCOCEM", "RAMCOSYS", "RAMKY", "RAMRAT", "RANASUG",
        "RANEENGINE", "RANEHOLDIN", "RATNAMANI", "RATNABHU",
        "RAYMOND", "RBLBANK", "RCCPL", "RCF", "RDBREALTY", "RDEL",
        "REGENCERAM", "RELCNX100", "RELIANCE", "REMSONSIND",
        "RENUKA", "REPCOHOME", "RESPONIND", "RETAIL", "REVATHI",
        "RGL", "RIA", "RICOAUTO", "RITES", "RIVERDYN", "RKEC",
        "RMGALLOY", "RML", "ROHLTLTD", "ROLLT", "ROML", "ROSSARI",
        "ROTO", "RPPL", "RPOWER", "RSWM", "RSYSTEMS", "RTNINFRA",
        "RTNPOWER", "RUCHIRA", "RUPA", "SADBHAV", "SADBHIN",
        "SAFARI", "SAGCEM", "SAH", "SAHYADRI", "SAIL", "SAKAR",
        "SAKHTISUG", "SAKSOFT", "SALONA", "SALZER", "SAMHI",
        "SAMPRE", "SAMSKARA", "SANCO", "SANDUMA", "SANGAMIND",
        "SANGHIIND", "SANGHVIMOV", "SANOFI", "SANSERA", "SAPL",
        "SARDAEN", "SAREGAMA", "SARLAPOLY", "SARTHAKMET",
        "SASKEN", "SATIA", "SATIN", "SATRAPETRO", "SAVERA",
        "SAWACA", "SBCL", "SBGLP", "SBICARD", "SBIETF", "SBL",
        "SCHAND", "SCHNEIDER", "SEAMECLTD", "SECMARK", "SECURCRED",
        "SEIL", "SELAN", "SEL", "SEMAC", "SEQUENT", "SERVALL",
        "SESHAPAPER", "SETCO", "SEZAL", "SFL", "SGFL", "SHAH",
        "SHAIVAL", "SHAKTIPUMP", "SHALBY", "SHALPAINTS", "SHARDACROP",
        "SHARDAMOTR", "SHARIABEES", "SHARPINDIA", "SHETH", "SHILPA",
        "SHIVALIK", "SHIVAMAUTO", "SHIVTEX", "SHK", "SHOPERSTOP",
        "SHRADHA", "SHREECEM", "SHREEPUSHK", "SHREYANIND", "SHREYAS",
        "SHRIPISTON", "SHRIRAM", "SHRIRAMCIT", "SHRIRAMEPC",
        "SHYAMCENT", "SHYAMTEL", "SICAL", "SIGACHI", "SIGIND",
        "SIGNET", "SIGNPOST", "SIKKO", "SILGO", "SILINV", "SILLYMONKS",
        "SIMBHALS", "SIMPLEXINF", "SINDHUTRAD", "SINTERCOM",
        "SIRCA", "SITINET", "SIVAINDS", "SJS", "SJVN", "SKFINDIA",
        "SKIPPER", "SKMEGGPROD", "SMARTLINK", "SMCGLOBAL", "SMLISUZU",
        "SMSLIFE", "SNOWMAN", "SOBHA", "SOFTTECH", "SOLARINDS",
        "SOMANYCERA", "SOMATEX", "SOMICONVEY", "SOMIEXPR", "SONACOMS",
        "SONAMLTD", "SONATSOFTW", "SORILINFRA", "SOTL", "SOUTHBANK",
        "SPAL", "SPANDANA", "SPARC", "SPCENET", "SPECIALITY",
        "SPENCERS", "SPIC", "SPIRITCITY", "SPMLINFRA", "SPORTKING",
        "SPRING", "SRF", "SRGHFL", "SRHHYPOLTD", "SRIADHI", "SRIVALS",
        "SRPL", "SRSREAL", "SSEMLTD", "SSWL", "STAMPEDE", "STAR",
        "STARCEMENT", "STARHEALTH", "STARLOG", "STARLITE", "STARLTD",
        "STARPAPER", "STARTUP", "STATEGAS", "STLTECH", "STOVEKRAFT",
        "STRINFRA", "STRTECH", "STUDBE", "STYLEBAIN", "SUJANA",
        "SUMEETINDS", "SUMIT", "SUMMIT", "SUMSEC", "SUNCLAY",
        "SUNDARAM", "SUNDARMFIN", "SUNDRMBRAK", "SUNFLAG",
        "SUNILHEALTH", "SUNPHARMA", "SUNTECK", "SUPERSPIN",
        "SUPRAJIT", "SUPREMEENG", "SUPREMEIND", "SUPRIYA",
        "SURAJEST", "SURANASOL", "SURANATP", "SURESH", "SURYALAXMI",
        "SURYAMINI", "SURYAROSNI", "SURYODAY", "SUVEN", "SVA",
        "SVARTCORP", "SVELTD", "SVGLOBAL", "SWANENERGY", "SWARAJENG",
        "SWELECTES", "SWORDINFO", "SYBLY", "SYLPH", "SYNGENE",
        "SYRMA", "TAINWALCHM", "TAJGVK", "TALBROAUTO", "TANLA",
        "TARC", "TARMAT", "TASTYBITE", "TATACHEM", "TATACOFFEE",
        "TATACOMM", "TATAELXSI", "TATAINVEST", "TATAMETALI",
        "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TATATECH",
        "TATVA", "TBOtek", "TCI", "TCIEXP", "TCIDEVELOP", "TCNSBRANDS",
        "TCPLPACK", "TEAMLEASE", "TECHM", "TECHNOE", "TECILCHEM",
        "TEGA", "TEMBO", "TERASOFT", "TEXINFRA", "TEXMOPIPES",
        "TFI", "TFL", "TGIF", "THANGAMAYL", "THEINVEST", "THEMISMED",
        "THERMAX", "THIRUSUGAR", "THOMASCOOK", "THYROCARE",
        "TI", "TIDEWATER", "TIIL", "TIINDIA", "TIMESGTY",
        "TIMETECHNO", "TIMKEN", "TIPSINDLTD", "TIPSFILM",
        "TIRUMALCEM", "TITAN", "TMRVL", "TNPL", "TNTELE",
        "TOKYOPLAST", "TORNTPOWER", "TOTAL", "TPHQ", "TPLPLASTEH",
        "TRACXN", "TRANSWARRANT", "TRANSGENE", "TRIDENT",
        "TRIGYN", "TRIL", "TRIMURTHI", "TRIVENI", "TRITURBINE",
        "TRU", "TRUST", "TTKHLTCARE", "TTKPRESTIG", "TTL",
        "TVTODAY", "TWL", "TXPL", "TYCHE", "TYROON", "UBEIND",
        "UCAL", "UCOBANK", "UDS", "UFLEX", "UGARSUGAR",
        "UGROCAP", "UJJIVAN", "UJJIVANSFB", "ULTRACEMCO",
        "UMANGDAIRY", "UMESLTD", "UNIABEAL", "UNICHEMLAB",
        "UNIDT", "UNIENTER", "UNIPARTS", "UNITECH", "UNITY",
        "UNIVCABLE", "UNIVPHOTO", "UNIVSTARCH", "UNOMINDA",
        "UPL", "URAVI", "URJA", "USHAMART", "USK", "UTTAMSUGAR",
        "UTTAMVALUE", "V2RETAIL", "VAIBHAVGBL", "VAISHALI",
        "VALIANTORG", "VARDHACRLC", "VARDMNPOLY", "VARROC",
        "VASCONEQ", "VASWANI", "VBL", "VEDL", "VENKEYS",
        "VENUSPIPES", "VENUSREM", "VERANDA", "VERTOZ",
        "VESUVIUS", "VETO", "VGUARD", "VHL", "VIDHIING",
        "VIJAYA", "VIJIFIN", "VIKASECO", "VIMTALABS",
        "VINEETLAB", "VINNYOVERS", "VINYLINDIA", "VIPIND",
        "VIPULLTD", "VIRINCHI", "VISAKAIND", "VISESHINFO",
        "VIVIDHA", "VLSFINANCE", "VMART", "VOLTAMP",
        "VOLTAS", "VPRPL", "VRLLOG", "VSSL", "VSTIND",
        "VSTTILLERS", "VTL", "WAAREEENER", "WABCOINDIA",
        "WALCHANNAG", "WARRENTEA", "WEBELSOLAR", "WEIZFOREX",
        "WEIZMANN", "WELCORP", "WELINV", "WELSPUNIND",
        "WENDT", "WESTLIFE", "WHIRLPOOL", "WILLAMAGOR",
        "WINDLAS", "WINDMACHIN", "WINSOME", "WINSOL",
        "WOCKPHARMA", "WONDERLA", "WSI", "WSTCSTPAPR",
        "WTEGRN", "XCHANGING", "XPROINDIA", "YESBANK",
        "YORK", "YSL", "ZENTEC", "ZIMLAB", "ZODIAC",
        "ZODJRDMRKJ", "ZUARI", "ZUARIGLOB", "ZYDUSWELL",
    ],
    # ========== NIFTY SECTOR INDICES (as ETFs/Index stocks) ==========
    "Nifty Bank": ["BANKNIFTY", "ICICIBANK", "HDFCBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BANKBARODA", "PNB", "FEDERALBNK"],
    "Nifty IT": ["NIFTYIT", "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "PERSISTENT", "OFSS", "COFORGE"],
    "Nifty Pharma": ["NIFTYPHARM", "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "TORNTPHARM", "LUPIN", "BIOCON", "AUROPHARMA", "ALKEM"],
    "Nifty Auto": ["NIFTYAUTO", "MARUTI", "TATAMOTORS", "M&M", "EICHERMOT", "BAJAJ-AUTO", "HEROMOTOCO", "ASHOKLEY", "APOLLOTYRE", "MOTHERSON"],
    "Nifty FMCG": ["NIFTYFMCG", "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "GODREJCP", "COLPAL", "MARICO", "TATACONSUM"],
    "Nifty Metal": ["NIFTYMETAL", "TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL", "COALINDIA", "NMDC", "SAIL", "JINDALSTEL", "NATIONALUM"],
    "Nifty Realty": ["NIFTYREALTY", "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "BRIGADE", "SOBHA", "PHOENIXLTD", "CREST", "MACROTECH"],
    "Nifty Energy": ["NIFTYENERGY", "RELIANCE", "ONGC", "NTPC", "POWERGRID", "IOC", "BPCL", "HPCL", "ADANIGREEN", "ADANIPOWER"],
    "Nifty Infra": ["NIFTYINFRA", "LT", "ADANIENT", "ULTRACEMCO", "GRASIM", "ADANIPORTS", "GMRINFRA", "IRB", "KNR", "PNCINFRA"],
}

MARKET_BASKETS = {
    "NSE": {
        cap: [f"{symbol}.NS" for symbol in symbols]
        for cap, symbols in BASE_BASKETS.items()
    },
    "BSE": {
        cap: [f"{symbol}.BO" for symbol in symbols]
        for cap, symbols in BASE_BASKETS.items()
    },
}

st.sidebar.caption(
    "Default cap segment baskets are still used, but you can now add custom symbols "
    "directly from the scanner tab."
)

st.info(f"📈 Selected market: **{exchange_full}**")

# ============================================================
# 6. Helper Functions
# ============================================================
def parse_custom_symbols(input_text, default_suffix):
    """
    Parses comma-separated symbols entered by the user.
    If symbol does not contain .NS or .BO, selected exchange suffix is added.
    """
    if not input_text:
        return []

    # Allow comma or semicolon separated input
    normalized_text = input_text.replace(";", ",").replace("|", ",")

    symbols = []

    for raw_symbol in normalized_text.split(","):
        symbol = raw_symbol.strip().upper()
        symbol = symbol.replace(" ", "")

        if not symbol:
            continue

        if symbol.endswith((".NS", ".BO")):
            symbols.append(symbol)
        else:
            symbols.append(f"{symbol}{default_suffix}")

    # Remove duplicates while preserving order
    return list(dict.fromkeys(symbols))


def clean_symbol(symbol):
    """
    Removes NSE/BSE suffix for display.
    """
    return symbol.replace(".NS", "").replace(".BO", "")


def add_indicators(df):
    """
    Adds comprehensive technical indicators to price dataframe:
    - RSI, SMA50, SMA200
    - EMA12, EMA26, MACD
    - Bollinger Bands
    - ADX (Trend Strength)
    - Stochastic Oscillator
    - Williams %R
    - CCI
    - MFI (Money Flow Index)
    """
    # Safety check: ensure we have enough data
    if df is None or len(df) < 30:
        if df is not None:
            df = df.copy()
            df["RSI"] = np.nan
            df["SMA50"] = np.nan
            df["SMA200"] = np.nan
            df["EMA12"] = np.nan
            df["EMA26"] = np.nan
            df["MACD"] = np.nan
            df["MACD_Signal"] = np.nan
            df["BB_Upper"] = np.nan
            df["BB_Middle"] = np.nan
            df["BB_Lower"] = np.nan
            df["ADX"] = np.nan
            df["Stoch_K"] = np.nan
            df["Stoch_D"] = np.nan
            df["Williams_R"] = np.nan
            df["CCI"] = np.nan
            df["MFI"] = np.nan
        return df

    from ta.trend import EMAIndicator, MACD, ADXIndicator, CCIIndicator
    from ta.volatility import BollingerBands
    from ta.momentum import StochasticOscillator, WilliamsRIndicator
    from ta.volume import MFIIndicator
    
    df = df.copy()
    
    try:
        # RSI
        df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
        
        # Simple Moving Averages
        df["SMA50"] = df["Close"].rolling(window=50).mean()
        df["SMA200"] = df["Close"].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df["EMA12"] = EMAIndicator(close=df["Close"], window=12).ema_indicator()
        df["EMA26"] = EMAIndicator(close=df["Close"], window=26).ema_indicator()
        
        # MACD
        macd = MACD(close=df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        
        # Bollinger Bands
        bb = BollingerBands(close=df["Close"])
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Middle"] = bb.bollinger_mavg()
        df["BB_Lower"] = bb.bollinger_lband()
        
        # ADX (Average Directional Index)
        adx = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"])
        df["ADX"] = adx.adx()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"])
        df["Stoch_K"] = stoch.stoch()
        df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
        
        # Williams %R
        williams = WilliamsRIndicator(high=df["High"], low=df["Low"], close=df["Close"])
        df["Williams_R"] = williams.williams_r()
        
        # CCI (Commodity Channel Index)
        cci = CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"])
        df["CCI"] = cci.cci()
        
        # MFI (Money Flow Index) - requires volume
        if "Volume" in df.columns:
            mfi = MFIIndicator(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"])
            df["MFI"] = mfi.money_flow_index()
        else:
            df["MFI"] = np.nan
            
    except Exception as e:
        st.warning(f"Could not calculate some indicators: {str(e)}")
        # Ensure all columns exist even if calculation failed
        for col in ["RSI", "SMA50", "SMA200", "EMA12", "EMA26", "MACD", "MACD_Signal", 
                    "BB_Upper", "BB_Middle", "BB_Lower", "ADX", "Stoch_K", "Stoch_D", 
                    "Williams_R", "CCI", "MFI"]:
            if col not in df.columns:
                df[col] = np.nan
    
    return df


def evaluate_stock(df):
    """
    Generates a technical score and predicts UP/DOWN/NEUTRAL.

    Positive score = bullish bias
    Negative score = bearish bias
    """
    signals = []
    score = 0.0

    current_price = float(df["Close"].iloc[-1])
    prev_price = float(df["Close"].iloc[-2])

    current_rsi = df["RSI"].iloc[-1]
    prev_rsi = df["RSI"].iloc[-2]

    current_volume = df["Volume"].iloc[-1]
    avg_volume_20d = df["Volume"].iloc[-21:-1].mean()

    sma50_curr = df["SMA50"].iloc[-1]
    sma200_curr = df["SMA200"].iloc[-1]
    sma50_prev = df["SMA50"].iloc[-2]
    sma200_prev = df["SMA200"].iloc[-2]

    # Additional indicators for comprehensive analysis
    ema12 = df["EMA12"].iloc[-1] if "EMA12" in df.columns else None
    ema26 = df["EMA26"].iloc[-1] if "EMA26" in df.columns else None
    macd = df["MACD"].iloc[-1] if "MACD" in df.columns else None
    macd_signal = df["MACD_Signal"].iloc[-1] if "MACD_Signal" in df.columns else None
    
    # Bollinger Bands
    bb_upper = df["BB_Upper"].iloc[-1] if "BB_Upper" in df.columns else None
    bb_lower = df["BB_Lower"].iloc[-1] if "BB_Lower" in df.columns else None
    bb_middle = df["BB_Middle"].iloc[-1] if "BB_Middle" in df.columns else None
    
    # ADX for trend strength
    adx = df["ADX"].iloc[-1] if "ADX" in df.columns else None
    
    # Stochastic Oscillator
    stoch_k = df["Stoch_K"].iloc[-1] if "Stoch_K" in df.columns else None
    stoch_d = df["Stoch_D"].iloc[-1] if "Stoch_D" in df.columns else None
    
    # Williams %R
    williams_r = df["Williams_R"].iloc[-1] if "Williams_R" in df.columns else None
    
    # CCI
    cci = df["CCI"].iloc[-1] if "CCI" in df.columns else None
    
    # MFI (Money Flow Index)
    mfi = df["MFI"].iloc[-1] if "MFI" in df.columns else None

    price_change_pct = (
        ((current_price - prev_price) / prev_price) * 100
        if prev_price != 0
        else 0.0
    )

    vol_multiplier = None
    if (
        pd.notna(current_volume)
        and pd.notna(avg_volume_20d)
        and avg_volume_20d > 0
    ):
        vol_multiplier = float(current_volume / avg_volume_20d)

    # --------------------------------------------------------
    # Volume breakout / breakdown
    # --------------------------------------------------------
    if vol_multiplier is not None and vol_multiplier >= 2.0:
        if current_price > prev_price:
            signals.append("🔥 High-volume price breakout")
            score += 30
        elif current_price < prev_price:
            signals.append("🩸 High-volume distribution / breakdown")
            score -= 30
        else:
            signals.append("⚠️ Unusual volume")

    # --------------------------------------------------------
    # RSI momentum signals
    # --------------------------------------------------------
    if pd.notna(current_rsi) and pd.notna(prev_rsi):
        if current_rsi > 55 and prev_rsi <= 55:
            signals.append("⚡ Bullish RSI cross above 55")
            score += 25

        elif current_rsi < 45 and prev_rsi >= 45:
            signals.append("⚠️ Bearish RSI cross below 45")
            score -= 25

        elif 30 <= current_rsi <= 45 and prev_rsi < 30:
            signals.append("🛡️ Oversold recovery")
            score += 15

        elif 55 <= current_rsi <= 70 and prev_rsi > 70:
            signals.append("🔻 Overbought pullback risk")
            score -= 15

        if current_rsi >= 80:
            signals.append("🥵 Extremely overbought")
            score -= 10
        elif current_rsi <= 20:
            signals.append("🧊 Extremely oversold")
            score += 10

    # --------------------------------------------------------
    # MACD signals
    # --------------------------------------------------------
    if pd.notna(macd) and pd.notna(macd_signal):
        if macd > macd_signal and macd > 0:
            signals.append("📊 MACD Bullish Crossover")
            score += 20
        elif macd < macd_signal and macd < 0:
            signals.append("📉 MACD Bearish Crossover")
            score -= 20
        
        # MACD divergence
        if macd > 0 and macd_signal > 0:
            signals.append("📈 MACD in Positive Territory")
            score += 10
        elif macd < 0 and macd_signal < 0:
            signals.append("📉 MACD in Negative Territory")
            score -= 10

    # --------------------------------------------------------
    # EMA Crossover (12/26)
    # --------------------------------------------------------
    if pd.notna(ema12) and pd.notna(ema26):
        if ema12 > ema26:
            signals.append("⚡ Bullish EMA Cross (12>26)")
            score += 15
        elif ema12 < ema26:
            signals.append("⚠️ Bearish EMA Cross (12<26)")
            score -= 15

    # --------------------------------------------------------
    # Bollinger Bands signals
    # --------------------------------------------------------
    if pd.notna(bb_upper) and pd.notna(bb_lower) and pd.notna(bb_middle):
        if current_price > bb_upper:
            signals.append("🔥 Price breaking above Upper BB")
            score += 15
        elif current_price < bb_lower:
            signals.append("🩸 Price breaking below Lower BB")
            score -= 15
        elif current_price < bb_middle and current_price > bb_lower:
            signals.append("📉 Price near lower band - potential bounce")
            score += 10
        elif current_price > bb_middle and current_price < bb_upper:
            signals.append("📈 Price near upper band - potential resistance")
            score -= 10

    # --------------------------------------------------------
    # Stochastic Oscillator signals
    # --------------------------------------------------------
    if pd.notna(stoch_k) and pd.notna(stoch_d):
        if stoch_k > 80 and stoch_d > 80:
            signals.append("🥵 Stochastic Overbought")
            score -= 15
        elif stoch_k < 20 and stoch_d < 20:
            signals.append("🧊 Stochastic Oversold")
            score += 15
        
        if stoch_k > stoch_d and stoch_k < 80:
            signals.append("⚡ Stochastic Bullish Cross")
            score += 15
        elif stoch_k < stoch_d and stoch_k > 20:
            signals.append("⚠️ Stochastic Bearish Cross")
            score -= 15

    # --------------------------------------------------------
    # Williams %R signals
    # --------------------------------------------------------
    if pd.notna(williams_r):
        if williams_r > -20:
            signals.append("🥵 Williams %R Overbought")
            score -= 10
        elif williams_r < -80:
            signals.append("🧊 Williams %R Oversold")
            score += 10

    # --------------------------------------------------------
    # CCI signals
    # --------------------------------------------------------
    if pd.notna(cci):
        if cci > 100:
            signals.append("🔥 CCI Strong Bullish")
            score += 15
        elif cci < -100:
            signals.append("🩸 CCI Strong Bearish")
            score -= 15
        elif cci > 0:
            signals.append("📈 CCI Positive Momentum")
            score += 5
        elif cci < 0:
            signals.append("📉 CCI Negative Momentum")
            score -= 5

    # --------------------------------------------------------
    # MFI (Money Flow Index) signals
    # --------------------------------------------------------
    if pd.notna(mfi):
        if mfi > 80:
            signals.append("💰 MFI Overbought (Money flowing out)")
            score -= 15
        elif mfi < 20:
            signals.append("💰 MFI Oversold (Money flowing in)")
            score += 15
        elif mfi > 50:
            signals.append("💰 MFI Bullish Money Flow")
            score += 10
        elif mfi < 50:
            signals.append("💰 MFI Bearish Money Flow")
            score -= 10

    # --------------------------------------------------------
    # ADX Trend Strength
    # --------------------------------------------------------
    if pd.notna(adx):
        if adx > 25:
            signals.append(f"💪 Strong Trend (ADX: {adx:.1f})")
            if ema12 and ema26 and ema12 > ema26:
                score += 15
            elif ema12 and ema26 and ema12 < ema26:
                score -= 15
        else:
            signals.append("📊 Weak/No Clear Trend")

    # --------------------------------------------------------
    # Trend / Moving average signals
    # --------------------------------------------------------
    if pd.notna(sma50_curr) and pd.notna(sma200_curr):
        if sma50_curr > sma200_curr:
            if (
                pd.notna(sma50_prev)
                and pd.notna(sma200_prev)
                and sma50_prev <= sma200_prev
            ):
                signals.append("🌟 Golden Cross Confirmation")
                score += 25

            if current_price > sma50_curr:
                signals.append("📈 Price above 50SMA in bullish trend")
                score += 15

        elif sma50_curr < sma200_curr:
            if (
                pd.notna(sma50_prev)
                and pd.notna(sma200_prev)
                and sma50_prev >= sma200_prev
            ):
                signals.append("💀 Death Cross Confirmation")
                score -= 25

            if current_price < sma50_curr:
                signals.append("📉 Price below 50SMA in bearish trend")
                score -= 15

        # Extra trend alignment score
        if current_price > sma50_curr > sma200_curr:
            score += 10
        elif current_price < sma50_curr < sma200_curr:
            score -= 10

    # --------------------------------------------------------
    # Final direction prediction
    # --------------------------------------------------------
    if score >= 25:
        direction = "📈 UP"
    elif score <= -25:
        direction = "📉 DOWN"
    else:
        direction = "⚖️ NEUTRAL"

    if score >= 65:
        ranking = "🔥 STRONG UP"
    elif score >= 25:
        ranking = "📈 BUY / UP"
    elif score <= -65:
        ranking = "🔥 STRONG DOWN"
    elif score <= -25:
        ranking = "📉 SELL / DOWN"
    else:
        ranking = "⚖️ HOLD / MIXED"

    return {
        "score": int(round(score)),
        "direction": direction,
        "ranking": ranking,
        "signals": signals,
        "price": current_price,
        "change_pct": price_change_pct,
        "rsi": current_rsi,
        "vol_multiplier": vol_multiplier,
    }


def calculate_predicted_price(df, analysis_result):
    """
    Calculate predicted price based on technical analysis using multiple methods:
    1. RSI-based mean reversion
    2. Moving average convergence
    3. Volume-weighted momentum
    4. Support/Resistance levels
    
    Returns a dictionary with predicted price and confidence level.
    """
    current_price = float(df["Close"].iloc[-1])
    
    # Get technical indicators
    rsi = analysis_result.get("rsi", 50)
    if pd.isna(rsi):
        rsi = 50
    
    sma50 = df["SMA50"].iloc[-1] if pd.notna(df["SMA50"].iloc[-1]) else current_price
    sma200 = df["SMA200"].iloc[-1] if pd.notna(df["SMA200"].iloc[-1]) else current_price
    
    # Calculate recent volatility (ATR-like measure)
    high_low_range = df["High"].iloc[-20:].mean() - df["Low"].iloc[-20:].mean()
    avg_volatility = high_low_range / current_price if current_price > 0 else 0.02
    
    # Method 1: RSI-based prediction (mean reversion)
    if rsi < 30:  # Oversold - expect bounce
        rsi_prediction = current_price * (1 + avg_volatility * 0.5)
        rsi_confidence = 0.6
    elif rsi > 70:  # Overbought - expect pullback
        rsi_prediction = current_price * (1 - avg_volatility * 0.5)
        rsi_confidence = 0.6
    else:  # Neutral zone
        rsi_prediction = current_price
        rsi_confidence = 0.3
    
    # Method 2: Moving average convergence
    if sma50 > sma200:  # Bullish trend
        ma_gap = (sma50 - sma200) / sma200
        ma_prediction = current_price * (1 + min(ma_gap, 0.05))
        ma_confidence = 0.5
    elif sma50 < sma200:  # Bearish trend
        ma_gap = (sma200 - sma50) / sma200
        ma_prediction = current_price * (1 - min(ma_gap, 0.05))
        ma_confidence = 0.5
    else:
        ma_prediction = current_price
        ma_confidence = 0.3
    
    # Method 3: Score-based momentum
    score = analysis_result.get("score", 0)
    momentum_factor = score / 100.0  # Normalize score
    momentum_prediction = current_price * (1 + momentum_factor * avg_volatility)
    momentum_confidence = min(abs(score) / 50.0, 0.7)
    
    # Weighted average of predictions
    total_weight = rsi_confidence + ma_confidence + momentum_confidence
    weighted_prediction = (
        rsi_prediction * rsi_confidence +
        ma_prediction * ma_confidence +
        momentum_prediction * momentum_confidence
    ) / total_weight
    
    # Calculate confidence level
    overall_confidence = (rsi_confidence + ma_confidence + momentum_confidence) / 3
    
    # Determine price target range
    if analysis_result.get("direction") == "📈 UP":
        target_up = weighted_prediction * (1 + avg_volatility)
        target_down = current_price * (1 - avg_volatility * 0.5)
    elif analysis_result.get("direction") == "📉 DOWN":
        target_up = current_price * (1 + avg_volatility * 0.5)
        target_down = weighted_prediction * (1 - avg_volatility)
    else:
        target_up = current_price * (1 + avg_volatility * 0.3)
        target_down = current_price * (1 - avg_volatility * 0.3)
    
    # Round to 2 decimal places
    predicted_price = round(weighted_prediction, 2)
    upside_potential = round((predicted_price - current_price) / current_price * 100, 2)
    
    return {
        "predicted_price": predicted_price,
        "current_price": round(current_price, 2),
        "upside_potential": upside_potential,
        "target_high": round(target_up, 2),
        "target_low": round(target_down, 2),
        "confidence": round(overall_confidence * 100, 1),
        "method_weights": {
            "rsi": round(rsi_confidence, 2),
            "moving_average": round(ma_confidence, 2),
            "momentum": round(momentum_confidence, 2)
        }
    }


# ============================================================
# 8. Dashboard Tabs
# ============================================================
tab1, tab2 = st.tabs(
    [
        "🎯 Single Stock Deep Dive",
        "⚡ AI Multi-Stock Scanner & Predictions",
    ]
)

# ============================================================
# TAB 1: Single Stock Analysis - Enhanced with Comprehensive Technical Analysis
# ============================================================
with tab1:
    st.subheader("🎯 Individual Stock Diagnostic Panel")
    
    # Tooltip helper function
    def tooltip(text, explanation):
        return f"""
        <div class="tooltip-container">
            {text}
            <span class="tooltip-text">{explanation}</span>
        </div>
        """
    
    t_col1, t_col2 = st.columns(2)

    with t_col1:
        ticker_input = (
            st.text_input(
                f"Enter {exchange} Stock Symbol:",
                value="RELIANCE",
                help=tooltip("Stock Symbol", "The trading symbol of the company on NSE (.NS) or BSE (.BO)")
            )
            .upper()
            .strip()
        )

    with t_col2:
        time_period = st.selectbox(
            "Select Historical Data Period:",
            ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years"],
            help=tooltip("Time Period", "Historical data range for technical analysis")
        )

    period_mapping = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
    }

    yf_period = period_mapping[time_period]
    full_ticker = f"{ticker_input}{suffix}"

    if st.button("🔍 Analyze Stock", key="btn_deep", type="primary"):
        if not ticker_input:
            st.error("Please enter a stock symbol.")
        else:
            with st.spinner(f"Fetching comprehensive data tracks for {ticker_input}..."):
                try:
                    stock = yf.Ticker(full_ticker)
                    hist_data = stock.history(period=yf_period)
                    info = stock.info

                    if hist_data.empty:
                        st.error(
                            f"No data found. Please check if the {exchange} symbol is valid."
                        )
                    else:
                        # Add all technical indicators
                        hist_data = add_indicators(hist_data)
                        analysis = evaluate_stock(hist_data)
                        predicted = calculate_predicted_price(hist_data, analysis)
                        
                        current_price = float(hist_data["Close"].iloc[-1])
                        
                        # ===================== TOP SECTION: Key Metrics with Tooltips =====================
                        st.markdown("### 📊 Key Stock Fundamentals")
                        
                        f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
                        
                        with f_col1:
                            st.metric(
                                label=tooltip("Current Price", "Latest traded price"),
                                value=f"₹{info.get('currentPrice', current_price):.2f}" if info.get('currentPrice') else f"₹{current_price:.2f}",
                                delta=f"{analysis['change_pct']:.2f}%"
                            )
                        
                        with f_col2:
                            st.metric(
                                label=tooltip("P/E Ratio", "Price-to-Earnings ratio - valuation metric"),
                                value=f"{round(info.get('trailingPE', 0), 2)}" if info.get('trailingPE') else "N/A"
                            )
                        
                        with f_col3:
                            st.metric(
                                label=tooltip("Market Cap", "Total market value of company's shares"),
                                value=f"₹{round(info.get('marketCap', 0) / 10000000, 2):,} Cr" if info.get("marketCap") else "N/A"
                            )
                        
                        with f_col4:
                            rsi_val_display = f"{round(float(analysis['rsi']), 1)}" if pd.notna(analysis['rsi']) else "N/A"
                            rsi_delta = "Overbought (>70)" if pd.notna(analysis['rsi']) and analysis['rsi'] > 70 else "Oversold (<30)" if pd.notna(analysis['rsi']) and analysis['rsi'] < 30 else "Neutral (30-70)"
                            st.metric(
                                label=tooltip("RSI (14)", "Relative Strength Index - momentum oscillator (0-100)"),
                                value=rsi_val_display,
                                delta=rsi_delta
                            )
                        
                        with f_col5:
                            st.metric(
                                label=tooltip("Signal Score", "Composite technical score (-100 to +100)"),
                                value=f"{analysis['score']}",
                                delta=analysis['direction']
                            )
                        
                        # ===================== PRICE PREDICTIONS SECTION =====================
                        st.markdown("### 🎯 AI Price Predictions")
                        
                        pred_col1, pred_col2, pred_col3 = st.columns(3)
                        
                        with pred_col1:
                            st.markdown("""
                            <div class="metric-card" style="background: linear-gradient(135deg, #fef3c7, #fde68a);">
                                <h4 style="margin:0; color: #92400e;">📈 Short-Term (1-2 Weeks)</h4>
                                <p style="font-size: 2rem; font-weight: bold; margin: 10px 0; color: #78350f;">₹{:.2f}</p>
                                <p style="margin: 5px 0; color: #92400e;">Target Range: ₹{:.2f} - ₹{:.2f}</p>
                                <p style="margin: 5px 0; font-size: 0.9rem; color: #78350f;">Confidence: {:.0f}%</p>
                            </div>
                            """.format(
                                predicted['predicted_price'],
                                predicted['target_low'],
                                predicted['target_high'],
                                predicted['confidence']
                            ), unsafe_allow_html=True)
                        
                        with pred_col2:
                            # Mid-term prediction (adjust based on trend)
                            mid_term_factor = 1 + (analysis['score'] / 200)  # Adjusted factor
                            mid_term_price = current_price * mid_term_factor
                            mid_term_high = current_price * (mid_term_factor + 0.05)
                            mid_term_low = current_price * (mid_term_factor - 0.05)
                            
                            st.markdown("""
                            <div class="metric-card" style="background: linear-gradient(135deg, #dbeafe, #bfdbfe);">
                                <h4 style="margin:0; color: #1e40af;">📊 Mid-Term (1-3 Months)</h4>
                                <p style="font-size: 2rem; font-weight: bold; margin: 10px 0; color: #1e3a8a;">₹{:.2f}</p>
                                <p style="margin: 5px 0; color: #1e40af;">Target Range: ₹{:.2f} - ₹{:.2f}</p>
                                <p style="margin: 5px 0; font-size: 0.9rem; color: #1e40af;">Based on Trend Analysis</p>
                            </div>
                            """.format(
                                mid_term_price,
                                mid_term_low,
                                mid_term_high
                            ), unsafe_allow_html=True)
                        
                        with pred_col3:
                            # Long-term prediction (based on fundamentals and long-term trend)
                            lt_factor = 1 + (analysis['score'] / 100) * 0.1  # Conservative long-term
                            lt_price = current_price * lt_factor
                            lt_high = current_price * (lt_factor + 0.1)
                            lt_low = current_price * max(0.8, lt_factor - 0.1)
                            
                            st.markdown("""
                            <div class="metric-card" style="background: linear-gradient(135deg, #d1fae5, #a7f3d0);">
                                <h4 style="margin:0; color: #065f46;">📉 Long-Term (6-12 Months)</h4>
                                <p style="font-size: 2rem; font-weight: bold; margin: 10px 0; color: #064e3b;">₹{:.2f}</p>
                                <p style="margin: 5px 0; color: #065f46;">Target Range: ₹{:.2f} - ₹{:.2f}</p>
                                <p style="margin: 5px 0; font-size: 0.9rem; color: #065f46;">Fundamental + Technical</p>
                            </div>
                            """.format(
                                lt_price,
                                lt_low,
                                lt_high
                            ), unsafe_allow_html=True)
                        
                        # ===================== MAIN CHARTS SECTION =====================
                        st.markdown("### 📈 Technical Analysis Charts")
                        
                        chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
                            "Price & Moving Averages",
                            "Volume Analysis",
                            "Oscillators Dashboard",
                            "Sector Comparison"
                        ])
                        
                        with chart_tab1:
                            # Main price chart with multiple indicators
                            fig_price = go.Figure()
                            
                            # Candlestick
                            fig_price.add_trace(go.Candlestick(
                                x=hist_data.index,
                                open=hist_data["Open"],
                                high=hist_data["High"],
                                low=hist_data["Low"],
                                close=hist_data["Close"],
                                name="Price",
                                increasing_line_color='#10b981',
                                decreasing_line_color='#ef4444'
                            ))
                            
                            # Moving Averages
                            if "SMA50" in hist_data.columns and pd.notna(hist_data["SMA50"]).any():
                                fig_price.add_trace(go.Scatter(
                                    x=hist_data.index,
                                    y=hist_data["SMA50"],
                                    name="SMA 50",
                                    line=dict(color="#f59e0b", width=2)
                                ))
                            
                            if "SMA200" in hist_data.columns and pd.notna(hist_data["SMA200"]).any():
                                fig_price.add_trace(go.Scatter(
                                    x=hist_data.index,
                                    y=hist_data["SMA200"],
                                    name="SMA 200",
                                    line=dict(color="#8b5cf6", width=2)
                                ))
                            
                            if "EMA12" in hist_data.columns and pd.notna(hist_data["EMA12"]).any():
                                fig_price.add_trace(go.Scatter(
                                    x=hist_data.index,
                                    y=hist_data["EMA12"],
                                    name="EMA 12",
                                    line=dict(color="#06b6d4", width=1.5, dash='dash')
                                ))
                            
                            # Bollinger Bands
                            if "BB_Upper" in hist_data.columns and pd.notna(hist_data["BB_Upper"]).any():
                                fig_price.add_trace(go.Scatter(
                                    x=hist_data.index,
                                    y=hist_data["BB_Upper"],
                                    name="BB Upper",
                                    line=dict(color="#ef4444", width=1, dash='dot'),
                                    fill=None
                                ))
                                fig_price.add_trace(go.Scatter(
                                    x=hist_data.index,
                                    y=hist_data["BB_Lower"],
                                    name="BB Lower",
                                    line=dict(color="#10b981", width=1, dash='dot'),
                                    fill='tonexty'
                                ))
                            
                            fig_price.update_layout(
                                title="Price Action with Moving Averages & Bollinger Bands",
                                xaxis_rangeslider_visible=False,
                                height=500,
                                template="plotly_white",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02)
                            )
                            
                            st.plotly_chart(fig_price, use_container_width=True)
                        
                        with chart_tab2:
                            # Volume chart
                            fig_vol = go.Figure()
                            
                            colors = ['#10b981' if hist_data["Close"].iloc[i] >= hist_data["Open"].iloc[i] else '#ef4444' 
                                     for i in range(len(hist_data))]
                            
                            fig_vol.add_trace(go.Bar(
                                x=hist_data.index,
                                y=hist_data["Volume"],
                                name="Volume",
                                marker_color=colors
                            ))
                            
                            # Volume MA
                            vol_ma = hist_data["Volume"].rolling(window=20).mean()
                            fig_vol.add_trace(go.Scatter(
                                x=hist_data.index,
                                y=vol_ma,
                                name="Volume MA (20)",
                                line=dict(color="#f59e0b", width=2)
                            ))
                            
                            fig_vol.update_layout(
                                title="Volume Analysis",
                                xaxis_rangeslider_visible=False,
                                height=400,
                                template="plotly_white"
                            )
                            
                            st.plotly_chart(fig_vol, use_container_width=True)
                        
                        with chart_tab3:
                            # Oscillators dashboard
                            osc_col1, osc_col2 = st.columns(2)
                            
                            with osc_col1:
                                # RSI Chart
                                fig_rsi = go.Figure()
                                
                                fig_rsi.add_trace(go.Scatter(
                                    x=hist_data.index,
                                    y=hist_data["RSI"],
                                    name="RSI",
                                    line=dict(color="#8b5cf6", width=2)
                                ))
                                
                                # Overbought/Oversold lines
                                fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought")
                                fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10b981", annotation_text="Oversold")
                                fig_rsi.add_hline(y=50, line_dash="dot", line_color="#6b7280")
                                
                                fig_rsi.update_layout(
                                    title="RSI (Relative Strength Index)",
                                    xaxis_rangeslider_visible=False,
                                    height=300,
                                    template="plotly_white",
                                    yaxis=dict(range=[0, 100])
                                )
                                
                                st.plotly_chart(fig_rsi, use_container_width=True)
                            
                            with osc_col2:
                                # MACD Chart
                                if "MACD" in hist_data.columns and "MACD_Signal" in hist_data.columns:
                                    fig_macd = go.Figure()
                                    
                                    fig_macd.add_trace(go.Scatter(
                                        x=hist_data.index,
                                        y=hist_data["MACD"],
                                        name="MACD",
                                        line=dict(color="#06b6d4", width=2)
                                    ))
                                    
                                    fig_macd.add_trace(go.Scatter(
                                        x=hist_data.index,
                                        y=hist_data["MACD_Signal"],
                                        name="Signal",
                                        line=dict(color="#f59e0b", width=2)
                                    ))
                                    
                                    # Histogram
                                    macd_hist = hist_data["MACD"] - hist_data["MACD_Signal"]
                                    fig_macd.add_trace(go.Bar(
                                        x=hist_data.index,
                                        y=macd_hist,
                                        name="Histogram",
                                        marker_color=['#10b981' if v > 0 else '#ef4444' for v in macd_hist],
                                        opacity=0.5
                                    ))
                                    
                                    fig_macd.update_layout(
                                        title="MACD (Moving Average Convergence Divergence)",
                                        xaxis_rangeslider_visible=False,
                                        height=300,
                                        template="plotly_white"
                                    )
                                    
                                    st.plotly_chart(fig_macd, use_container_width=True)
                        
                        with chart_tab4:
                            # Additional oscillators
                            more_osc_col1, more_osc_col2 = st.columns(2)
                            
                            with more_osc_col1:
                                # Stochastic
                                if "Stoch_K" in hist_data.columns and "Stoch_D" in hist_data.columns:
                                    fig_stoch = go.Figure()
                                    
                                    fig_stoch.add_trace(go.Scatter(
                                        x=hist_data.index,
                                        y=hist_data["Stoch_K"],
                                        name="%K",
                                        line=dict(color="#06b6d4", width=2)
                                    ))
                                    
                                    fig_stoch.add_trace(go.Scatter(
                                        x=hist_data.index,
                                        y=hist_data["Stoch_D"],
                                        name="%D",
                                        line=dict(color="#f59e0b", width=2)
                                    ))
                                    
                                    fig_stoch.add_hline(y=80, line_dash="dash", line_color="#ef4444")
                                    fig_stoch.add_hline(y=20, line_dash="dash", line_color="#10b981")
                                    
                                    fig_stoch.update_layout(
                                        title="Stochastic Oscillator",
                                        xaxis_rangeslider_visible=False,
                                        height=300,
                                        template="plotly_white",
                                        yaxis=dict(range=[0, 100])
                                    )
                                    
                                    st.plotly_chart(fig_stoch, use_container_width=True)
                            
                            with more_osc_col2:
                                # Williams %R or CCI
                                if "Williams_R" in hist_data.columns:
                                    fig_williams = go.Figure()
                                    
                                    fig_williams.add_trace(go.Scatter(
                                        x=hist_data.index,
                                        y=hist_data["Williams_R"],
                                        name="Williams %R",
                                        line=dict(color="#8b5cf6", width=2)
                                    ))
                                    
                                    fig_williams.add_hline(y=-20, line_dash="dash", line_color="#ef4444")
                                    fig_williams.add_hline(y=-80, line_dash="dash", line_color="#10b981")
                                    
                                    fig_williams.update_layout(
                                        title="Williams %R",
                                        xaxis_rangeslider_visible=False,
                                        height=300,
                                        template="plotly_white",
                                        yaxis=dict(range=[-100, 0])
                                    )
                                    
                                    st.plotly_chart(fig_williams, use_container_width=True)
                        
                        # ===================== TECHNICAL INDICATORS SUMMARY TABLE =====================
                        st.markdown("### 📋 Comprehensive Technical Indicators Summary")
                        
                        tech_col1, tech_col2 = st.columns(2)
                        
                        with tech_col1:
                            st.subheader("📊 Trend Indicators")
                            
                            trend_data = []
                            
                            # SMA signals
                            sma50_val = hist_data["SMA50"].iloc[-1] if "SMA50" in hist_data.columns else None
                            sma200_val = hist_data["SMA200"].iloc[-1] if "SMA200" in hist_data.columns else None
                            
                            if pd.notna(sma50_val):
                                trend_data.append({
                                    "Indicator": "SMA 50",
                                    "Value": f"₹{sma50_val:.2f}",
                                    "Signal": "Bullish" if current_price > sma50_val else "Bearish"
                                })
                            
                            if pd.notna(sma200_val):
                                trend_data.append({
                                    "Indicator": "SMA 200",
                                    "Value": f"₹{sma200_val:.2f}",
                                    "Signal": "Bullish" if current_price > sma200_val else "Bearish"
                                })
                            
                            # EMA signals
                            ema12_val = hist_data["EMA12"].iloc[-1] if "EMA12" in hist_data.columns else None
                            ema26_val = hist_data["EMA26"].iloc[-1] if "EMA26" in hist_data.columns else None
                            
                            if pd.notna(ema12_val):
                                trend_data.append({
                                    "Indicator": "EMA 12",
                                    "Value": f"₹{ema12_val:.2f}",
                                    "Signal": "Bullish" if current_price > ema12_val else "Bearish"
                                })
                            
                            if pd.notna(ema26_val):
                                trend_data.append({
                                    "Indicator": "EMA 26",
                                    "Value": f"₹{ema26_val:.2f}",
                                    "Signal": "Bullish" if current_price > ema26_val else "Bearish"
                                })
                            
                            # MACD
                            macd_val = hist_data["MACD"].iloc[-1] if "MACD" in hist_data.columns else None
                            macd_sig = hist_data["MACD_Signal"].iloc[-1] if "MACD_Signal" in hist_data.columns else None
                            
                            if pd.notna(macd_val) and pd.notna(macd_sig):
                                trend_data.append({
                                    "Indicator": "MACD",
                                    "Value": f"{macd_val:.2f}",
                                    "Signal": "Bullish" if macd_val > macd_sig else "Bearish"
                                })
                            
                            if trend_data:
                                trend_df = pd.DataFrame(trend_data)
                                st.dataframe(trend_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No trend indicator data available")
                        
                        with tech_col2:
                            st.subheader("📈 Momentum Indicators")
                            
                            momentum_data = []
                            
                            # RSI
                            rsi_val = hist_data["RSI"].iloc[-1] if "RSI" in hist_data.columns else None
                            if pd.notna(rsi_val):
                                if rsi_val > 70:
                                    rsi_signal = "Overbought"
                                elif rsi_val < 30:
                                    rsi_signal = "Oversold"
                                else:
                                    rsi_signal = "Neutral"
                                momentum_data.append({
                                    "Indicator": "RSI (14)",
                                    "Value": f"{rsi_val:.2f}",
                                    "Signal": rsi_signal
                                })
                            
                            # Stochastic
                            stoch_k = hist_data["Stoch_K"].iloc[-1] if "Stoch_K" in hist_data.columns else None
                            stoch_d = hist_data["Stoch_D"].iloc[-1] if "Stoch_D" in hist_data.columns else None
                            
                            if pd.notna(stoch_k):
                                if stoch_k > 80:
                                    stoch_signal = "Overbought"
                                elif stoch_k < 20:
                                    stoch_signal = "Oversold"
                                else:
                                    stoch_signal = "Neutral"
                                momentum_data.append({
                                    "Indicator": "Stochastic %K",
                                    "Value": f"{stoch_k:.2f}",
                                    "Signal": stoch_signal
                                })
                            
                            # Williams %R
                            williams = hist_data["Williams_R"].iloc[-1] if "Williams_R" in hist_data.columns else None
                            if pd.notna(williams):
                                if williams > -20:
                                    will_signal = "Overbought"
                                elif williams < -80:
                                    will_signal = "Oversold"
                                else:
                                    will_signal = "Neutral"
                                momentum_data.append({
                                    "Indicator": "Williams %R",
                                    "Value": f"{williams:.2f}",
                                    "Signal": will_signal
                                })
                            
                            # CCI
                            cci_val = hist_data["CCI"].iloc[-1] if "CCI" in hist_data.columns else None
                            if pd.notna(cci_val):
                                if cci_val > 100:
                                    cci_signal = "Overbought"
                                elif cci_val < -100:
                                    cci_signal = "Oversold"
                                else:
                                    cci_signal = "Neutral"
                                momentum_data.append({
                                    "Indicator": "CCI",
                                    "Value": f"{cci_val:.2f}",
                                    "Signal": cci_signal
                                })
                            
                            # MFI
                            mfi_val = hist_data["MFI"].iloc[-1] if "MFI" in hist_data.columns else None
                            if pd.notna(mfi_val):
                                if mfi_val > 80:
                                    mfi_signal = "Overbought"
                                elif mfi_val < 20:
                                    mfi_signal = "Oversold"
                                else:
                                    mfi_signal = "Neutral"
                                momentum_data.append({
                                    "Indicator": "MFI",
                                    "Value": f"{mfi_val:.2f}",
                                    "Signal": mfi_signal
                                })
                            
                            if momentum_data:
                                momentum_df = pd.DataFrame(momentum_data)
                                st.dataframe(momentum_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No momentum indicator data available")
                        
                        # ===================== DETECTED SIGNALS =====================
                        st.markdown("### 🚨 Detected Technical Signals")
                        
                        if analysis["signals"]:
                            signals_display = ""
                            for signal in analysis["signals"]:
                                if "🔥" in signal or "⚡" in signal or "🌟" in signal or "💪" in signal:
                                    signals_display += f"✅ {signal}<br>"
                                elif "🩸" in signal or "💀" in signal or "⚠️" in signal or "🔻" in signal:
                                    signals_display += f"❌ {signal}<br>"
                                else:
                                    signals_display += f"• {signal}<br>"
                            
                            st.markdown(f"""
                            <div style="padding: 15px; background-color: #f0fdf4; border-left: 4px solid {'#10b981' if analysis['score'] > 0 else '#ef4444'}; border-radius: 8px; margin: 10px 0;">
                                <h4 style="margin-top: 0; color: #166534;">Technical Analysis Summary</h4>
                                <p><strong>Overall Verdict:</strong> {analysis['ranking']}</p>
                                <p><strong>Signal Score:</strong> {analysis['score']} / 100</p>
                                <hr style="border-color: #e5e7eb;">
                                <p><strong>Key Triggers Detected:</strong></p>
                                <p>{signals_display}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("No strong directional signals detected at this time.")
                        
                        # ===================== AI NEWS SENTIMENT =====================
                        st.markdown("### 🤖 AI News Sentiment Engine")
                        
                        news_list = stock.news
                        
                        if not news_list:
                            st.info("No recent headlines found for this ticker to process.")
                        else:
                            positive_count = 0
                            negative_count = 0
                            neutral_count = 0
                            
                            st.markdown("**Latest Headlines Analysis:**")
                            
                            for article in news_list[:5]:
                                if not isinstance(article, dict):
                                    continue
                                
                                title = article.get("title", "")
                                publisher = article.get("publisher", "Unknown Source")
                                
                                if not title and isinstance(article.get("content"), dict):
                                    title = article.get("content", {}).get("title", "")
                                
                                if not title:
                                    continue
                                
                                result = sentiment_analyzer(title)[0]
                                label = result["label"].lower()
                                score = round(result["score"] * 100, 1)
                                
                                if "pos" in label:
                                    bg_color = "#e1f5fe"
                                    badge = "🟢 POSITIVE"
                                    positive_count += 1
                                elif "neg" in label:
                                    bg_color = "#ffebee"
                                    badge = "🔴 NEGATIVE"
                                    negative_count += 1
                                else:
                                    bg_color = "#f5f5f5"
                                    badge = "⚪ NEUTRAL"
                                    neutral_count += 1
                                
                                st.markdown(
                                    f"""
                                    <div style="background-color:{bg_color}; padding:12px; border-radius:8px; margin-bottom:10px; border-left: 3px solid {'#10b981' if 'pos' in label else '#ef4444' if 'neg' in label else '#6b7280'};">
                                        <small style="color:gray;">{publisher}</small><br>
                                        <strong>{title}</strong><br>
                                        <span style="font-size:12px; font-weight:bold;">
                                            AI Assessment: {badge} ({score}%)
                                        </span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            
                            st.markdown("---")
                            
                            # Sentiment summary cards
                            sent_col1, sent_col2, sent_col3 = st.columns(3)
                            
                            sent_col1.metric("Positive News", positive_count)
                            sent_col2.metric("Negative News", negative_count)
                            sent_col3.metric("Neutral News", neutral_count)
                            
                            if positive_count > negative_count:
                                st.success("### 🟢 Overall Short-Term Sentiment: **BULLISH**")
                            elif negative_count > positive_count:
                                st.error("### 🔴 Overall Short-Term Sentiment: **BEARISH**")
                            else:
                                st.warning("### ⚪ Overall Short-Term Sentiment: **NEUTRAL / MIXED**")
                
                except Exception as e:
                    st.error(f"An unexpected data connection error occurred: {e}")
                    st.exception(e)


# ============================================================
# TAB 2: Multi Stock Scanner + Up/Down Prediction
# ============================================================
with tab2:
    st.subheader("AI Automated Technical Trend & Breakout Prediction Engine")

    st.write(
        "Select exchange from the sidebar, choose a cap segment, and optionally add custom symbols. "
        "The system will score stocks technically and predict possible upside or downside moves."
    )

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        # Get all available segment names from BASE_BASKETS keys
        all_segments = list(BASE_BASKETS.keys())
        segment = st.selectbox(
            "Select Market Segment:",
            all_segments
        )

    with filter_col2:
        scan_mode = st.selectbox(
            "Scanner Mode:",
            [
                "Segment Basket + Custom Symbols",
                "Segment Basket Only",
                "Custom Symbols Only",
            ]
        )

    custom_symbols_input = st.text_input(
        "Custom Symbols (comma-separated, optional):",
        placeholder="Example: RELIANCE, TCS, INFY, HDFCBANK",
        help=(
            "Enter one or more stock symbols separated by commas. "
            "If you do not add .NS or .BO, the selected exchange suffix will be added automatically."
        )
    )

    selected_basket = MARKET_BASKETS[exchange][segment]
    custom_basket = parse_custom_symbols(custom_symbols_input, suffix)

    if scan_mode == "Segment Basket + Custom Symbols":
        basket = list(dict.fromkeys(selected_basket + custom_basket))
    elif scan_mode == "Segment Basket Only":
        basket = selected_basket
    else:
        basket = custom_basket

    st.caption(
        f"Scanner mode: **{scan_mode}** | "
        f"Market: **{exchange}** | "
        f"Segment: **{segment}** | "
        f"Total symbols to scan: **{len(basket)}**"
    )

    with st.expander("Preview Scanner List", expanded=False):
        if basket:
            st.write(", ".join(basket))
        else:
            st.write("No symbols selected.")

    st.warning(
        "Predictions are based on technical scoring and are for educational purposes only. "
        "They should not be treated as financial advice."
    )

    if st.button("Launch System-Wide Market Scan", key="btn_scan"):
        if not basket:
            st.warning(
                "No symbols found to scan. Please enter custom symbols or choose a segment basket mode."
            )
        else:
            scan_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, ticker in enumerate(basket):
                status_text.text(f"Scanning data tracks for {ticker}...")
                progress_bar.progress((idx + 1) / len(basket))

                try:
                    ticker_obj = yf.Ticker(ticker)
                    df = ticker_obj.history(period="1y")

                    if len(df) < 55:
                        continue

                    df = add_indicators(df)
                    analysis = evaluate_stock(df)

                    ticker_exchange = (
                        "NSE"
                        if ticker.endswith(".NS")
                        else "BSE"
                        if ticker.endswith(".BO")
                        else exchange
                    )

                    display_segment = (
                        segment
                        if ticker in selected_basket
                        else "Custom"
                    )

                    rsi_display = (
                        round(float(analysis["rsi"]), 1)
                        if pd.notna(analysis["rsi"])
                        else "N/A"
                    )

                    vol_display = (
                        f"{analysis['vol_multiplier']:.1f}x"
                        if analysis["vol_multiplier"] is not None
                        else "N/A"
                    )

                    scan_results.append(
                        {
                            "Ticker": clean_symbol(ticker),
                            "Exchange": ticker_exchange,
                            "Segment": display_segment,
                            "Predicted Move": analysis["direction"],
                            "Signal Score": analysis["score"],
                            "AI Verdict": analysis["ranking"],
                            "Price": f"₹{analysis['price']:.2f}",
                            "Change": f"{analysis['change_pct']:.2f}%",
                            "RSI": rsi_display,
                            "Volume Multiplier": vol_display,
                            "Technical Structural Triggers": (
                                ", ".join(analysis["signals"])
                                if analysis["signals"]
                                else "No strong directional trigger"
                            ),
                        }
                    )

                    time.sleep(0.35)

                except Exception:
                    continue

            status_text.text("Scan sequence completed successfully.")

            if not scan_results:
                st.warning(
                    "No usable data returned for the selected scan list. "
                    "Please check symbol names or try NSE/BSE suffixes manually."
                )
            else:
                results_df = pd.DataFrame(scan_results)
                results_df = results_df.sort_values("Signal Score", ascending=False)

                if len(results_df) < len(basket):
                    st.warning(
                        "Some symbols did not return usable data. "
                        "This can happen due to incorrect symbol names or missing exchange data."
                    )

                up_df = results_df[results_df["Signal Score"] >= 25].copy()
                down_df = results_df[results_df["Signal Score"] <= -25].copy()
                neutral_df = results_df[
                    (results_df["Signal Score"] > -25)
                    & (results_df["Signal Score"] < 25)
                ].copy()

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                metric_col1.metric("Scanned Stocks", len(results_df))
                metric_col2.metric("Predicted UP", len(up_df))
                metric_col3.metric("Predicted DOWN", len(down_df))
                metric_col4.metric("Neutral", len(neutral_df))

                column_config = {
                    "Ticker": st.column_config.TextColumn("Ticker"),
                    "Exchange": st.column_config.TextColumn("Exchange"),
                    "Segment": st.column_config.TextColumn("Segment"),
                    "Predicted Move": st.column_config.TextColumn("Predicted Move"),
                    "Signal Score": st.column_config.NumberColumn(
                        "Signal Score",
                        format="%d"
                    ),
                    "AI Verdict": st.column_config.TextColumn("AI Verdict"),
                    "Price": st.column_config.TextColumn("Price"),
                    "Change": st.column_config.TextColumn("1D Change"),
                    "RSI": st.column_config.TextColumn("RSI"),
                    "Volume Multiplier": st.column_config.TextColumn("Volume x"),
                    "Technical Structural Triggers": st.column_config.TextColumn(
                        "Triggers Flagged"
                    ),
                }

                st.markdown("### 📈 Predicted Upside Moves")

                if up_df.empty:
                    st.info("No stocks predicted UP for this selected scan list.")
                else:
                    st.dataframe(
                        up_df,
                        column_config=column_config,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.markdown("### 📉 Predicted Downside Moves")

                if down_df.empty:
                    st.info("No stocks predicted DOWN for this selected scan list.")
                else:
                    st.dataframe(
                        down_df,
                        column_config=column_config,
                        use_container_width=True,
                        hide_index=True,
                    )

                with st.expander("⚖️ Neutral / Mixed Stocks", expanded=False):
                    if neutral_df.empty:
                        st.info("No neutral stocks found.")
                    else:
                        st.dataframe(
                            neutral_df,
                            column_config=column_config,
                            use_container_width=True,
                            hide_index=True,
                        )

                with st.expander("🧾 Full Scan Results", expanded=False):
                    st.dataframe(
                        results_df,
                        column_config=column_config,
                        use_container_width=True,
                        hide_index=True,
                    )