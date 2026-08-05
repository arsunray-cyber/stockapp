\# 📊 Indian Share Market AI Tool



A Streamlit-based Indian stock market analysis dashboard for \*\*NSE\*\* and \*\*BSE\*\* with technical scanning, market cap segment selection, custom symbol input, interactive charts, and AI-powered news sentiment analysis.



This tool is designed for educational and research purposes only.



\---



\## ✨ Features



\### 1. Market Selection



Users can select between:



\- \*\*NSE\*\* - National Stock Exchange

\- \*\*BSE\*\* - Bombay Stock Exchange



The application automatically uses the correct Yahoo Finance suffix:



| Exchange | Yahoo Finance Suffix |

|---|---|

| NSE | `.NS` |

| BSE | `.BO` |



\---



\### 2. Single Stock Deep Dive



For any selected stock, users can analyze:



\- Candlestick chart

\- 20-day moving average

\- 50-day moving average

\- Current price

\- Trailing P/E ratio

\- Market cap

\- Latest news headlines

\- AI-based news sentiment

\- Overall sentiment scorecard



Sentiment categories:



\- 🟢 Positive

\- 🔴 Negative

\- ⚪ Neutral



Final sentiment output:



\- \*\*BULLISH\*\*

\- \*\*BEARISH\*\*

\- \*\*NEUTRAL / MIXED\*\*



\---



\### 3. Multi-Stock Scanner



The scanner tab allows users to scan multiple stocks at once.



Users can select:



\- Large Cap

\- Mid Cap

\- Small Cap



Scanner modes:



| Mode | Description |

|---|---|

| Segment Basket + Custom Symbols | Scans selected cap segment plus user-entered symbols |

| Segment Basket Only | Scans only the selected cap segment |

| Custom Symbols Only | Scans only symbols entered by the user |



\---



\### 4. Custom Symbol Input



Users can enter custom stock symbols directly from the dashboard.



Example:



```text

RELIANCE, TCS, INFY, HDFCBANK

