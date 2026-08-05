import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import pipeline
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
import time

# ============================================================
# 1. Page configuration
# ============================================================
st.set_page_config(
    page_title="Indian Share Market AI Tool",
    layout="wide"
)

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
# 3. App Header
# ============================================================
st.title("📊 Indian Share Market AI Tool")
st.caption(
    "Market-wise scanner for NSE/BSE with cap segment selection, custom symbol input, "
    "technical scoring, interactive charts, and AI news sentiment."
)

# ============================================================
# 4. Exchange Selection
# ============================================================
exchange = st.sidebar.radio(
    "Select Market / Exchange:",
    ["NSE", "BSE"],
    horizontal=True
)

suffix = ".NS" if exchange == "NSE" else ".BO"
exchange_full = (
    "National Stock Exchange (NSE)"
    if exchange == "NSE"
    else "Bombay Stock Exchange (BSE)"
)

# ============================================================
# 5. Default Cap Segment Baskets
# ============================================================
# These default baskets remain available, but users can now
# also add custom symbols from the UI in Tab 2.

BASE_BASKETS = {
    "Large Cap": [
        "RELIANCE",
        "TCS",
        "HDFCBANK",
        "ICICIBANK",
        "INFY",
        "SBIN",
        "BHARTIARTL",
        "ITC",
        "LT",
        "HINDUNILVR",
    ],
    "Mid Cap": [
        "PERSISTENT",
        "KPITTECH",
        "CUMMINSIND",
        "BAJAJELEC",
        "FEDERALBNK",
        "IDFCFIRSTB",
        "AUBANK",
        "IEX",
        "BSE",
        "BALAMINES",
    ],
    "Small Cap": [
        "TANLA",
        "PGEL",
        "HAPPSTMNTH",
        "GRAVITA",
        "GANESHHOUC",
        "GABRIEL",
        "JINDALSAW",
        "MAZDOCK",
        "SHYAMCENT",
        "TITAGARH",
    ],
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


# ============================================================
# 7. Dashboard Tabs
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
        segment = st.selectbox(
            "Select Market Segment:",
            ["Large Cap", "Mid Cap", "Small Cap"]
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