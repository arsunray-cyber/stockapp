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
    Adds RSI, SMA50, SMA200 to price dataframe.
    """
    df = df.copy()
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()
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
# TAB 1: Single Stock Analysis
# ============================================================
with tab1:
    st.subheader("Individual Stock Diagnostic Panel")

    t_col1, t_col2 = st.columns(2)

    with t_col1:
        ticker_input = (
            st.text_input(
                f"Enter {exchange} Stock Symbol:",
                value="RELIANCE"
            )
            .upper()
            .strip()
        )

    with t_col2:
        time_period = st.selectbox(
            "Select Historical Data Period:",
            ["1 Month", "3 Months", "6 Months", "1 Year"]
        )

    period_mapping = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
    }

    yf_period = period_mapping[time_period]
    full_ticker = f"{ticker_input}{suffix}"

    if st.button("Analyze Stock", key="btn_deep"):
        if not ticker_input:
            st.error("Please enter a stock symbol.")
        else:
            with st.spinner(f"Fetching data tracks for {ticker_input}..."):
                try:
                    stock = yf.Ticker(full_ticker)
                    hist_data = stock.history(period=yf_period)
                    info = stock.info

                    if hist_data.empty:
                        st.error(
                            f"No data found. Please check if the {exchange} symbol is valid."
                        )
                    else:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(
                                f"#### Price Channels: {info.get('longName', ticker_input)}"
                            )

                            hist_data["MA20"] = (
                                hist_data["Close"].rolling(window=20).mean()
                            )
                            hist_data["MA50"] = (
                                hist_data["Close"].rolling(window=50).mean()
                            )

                            fig = go.Figure()

                            fig.add_trace(
                                go.Candlestick(
                                    x=hist_data.index,
                                    open=hist_data["Open"],
                                    high=hist_data["High"],
                                    low=hist_data["Low"],
                                    close=hist_data["Close"],
                                    name="Price",
                                )
                            )

                            fig.add_trace(
                                go.Scatter(
                                    x=hist_data.index,
                                    y=hist_data["MA20"],
                                    name="20 Day MA",
                                    line=dict(color="orange", width=1.5),
                                )
                            )

                            fig.add_trace(
                                go.Scatter(
                                    x=hist_data.index,
                                    y=hist_data["MA50"],
                                    name="50 Day MA",
                                    line=dict(color="blue", width=1.5),
                                )
                            )

                            fig.update_layout(
                                xaxis_rangeslider_visible=False,
                                height=400,
                                margin=dict(l=20, r=20, t=20, b=20),
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            st.markdown("### Key Stock Fundamentals")

                            f_col1, f_col2, f_col3 = st.columns(3)

                            f_col1.metric(
                                "Current Price",
                                f"₹{info.get('currentPrice', 'N/A')}"
                            )

                            f_col2.metric(
                                "Trailing P/E Ratio",
                                f"{round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else 'N/A'}"
                            )

                            f_col3.metric(
                                "Market Cap (Cr)",
                                f"₹{round(info.get('marketCap', 0) / 10000000, 2):,}"
                                if info.get("marketCap")
                                else "N/A"
                            )

                        with col2:
                            st.markdown("#### AI News Sentiment Engine")

                            news_list = stock.news

                            if not news_list:
                                st.info(
                                    "No recent headlines found for this ticker to process."
                                )
                            else:
                                positive_count = 0
                                negative_count = 0

                                st.markdown("**Latest Headlines Analysis:**")

                                for article in news_list[:4]:
                                    if not isinstance(article, dict):
                                        continue

                                    title = article.get("title", "")
                                    publisher = article.get("publisher", "Unknown Source")

                                    # Some yfinance news structures vary,
                                    # so try fallback title extraction.
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

                                    st.markdown(
                                        f"""
                                        <div style="background-color:{bg_color}; padding:10px; border-radius:5px; margin-bottom:10px;">
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
                                st.markdown("### 🤖 Final AI Sentiment Scorecard")

                                if positive_count > negative_count:
                                    st.success("Overall Short-Term Sentiment: **BULLISH**")
                                elif negative_count > negative_count:
                                    st.error("Overall Short-Term Sentiment: **BEARISH**")
                                else:
                                    st.warning("Overall Short-Term Sentiment: **NEUTRAL / MIXED**")

                except Exception as e:
                    st.error(f"An unexpected data connection error occurred: {e}")


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