import streamlit as st 
import pandas as pd
import plotly.express as px
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# Page configuration
st.set_page_config(page_title='Trading Price Predictor', layout='wide')
st.title('Trading Price Predictor with Extended Features')

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler()
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

# Sidebar inputs
st.sidebar.header('Input Parameters')
ticker_symbol = st.sidebar.text_input('Enter Stock Symbol', 'AAPL')
start_date = st.sidebar.date_input('Start Date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End Date', date.today())

@st.cache_data
def download_data(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            st.error(f"No data found for {symbol}")
            return None
        # Ensure data has required columns
        required_columns = ['Close', 'High', 'Low']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Missing required columns: {[col for col in required_columns if col not in df.columns]}")
            return None
        return df
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

def validate_data(data):
    """Validate data for technical analysis calculations"""
    if data is None:
        return False
    required_columns = ['Close', 'High', 'Low']
    return all(col in data.columns for col in required_columns)

def plot_chart(data, column, title):
    try:
        if isinstance(column, str):
            y_data = data[column] if isinstance(column, str) else column
        else:
            y_data = column
        fig = px.line(data, x=data.index, y=y_data, title=title)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error plotting chart: {str(e)}")

def calculate_indicators(data):
    """Calculate all technical indicators at once"""
    if not validate_data(data):
        return None, None, None, None, None, None
    
    try:
        # Convert data to Series if it's not already
        close_prices = data['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.squeeze()
        
        high_prices = data['High']
        if isinstance(high_prices, pd.DataFrame):
            high_prices = high_prices.squeeze()
            
        low_prices = data['Low']
        if isinstance(low_prices, pd.DataFrame):
            low_prices = low_prices.squeeze()

        # Bollinger Bands (using 20-day SMA)
        bb_indicator = BollingerBands(close=close_prices, window=20, window_dev=2)
        bb = pd.DataFrame({
            'Close': close_prices,
            'bb_h': bb_indicator.bollinger_hband(),
            'bb_l': bb_indicator.bollinger_lband()
        })

        # MACD (12, 26, 9 are standard parameters)
        macd_indicator = MACD(
            close=close_prices,
            window_slow=26,
            window_fast=12,
            window_sign=9
        )
        macd = macd_indicator.macd()
        
        # RSI (14 days is standard)
        rsi_indicator = RSIIndicator(close=close_prices, window=14)
        rsi = rsi_indicator.rsi()
        
        # SMA with explicit parameters
        sma_indicator = SMAIndicator(close=close_prices, window=20)
        sma = sma_indicator.sma_indicator()
        
        # EMA with explicit parameters
        ema_indicator = EMAIndicator(close=close_prices, window=20)
        ema = ema_indicator.ema_indicator()
        
        # Ichimoku
        ichimoku = IchimokuIndicator(
            high=high_prices,
            low=low_prices,
            window1=9,
            window2=26,
            window3=52
        )
        ichimoku_data = pd.DataFrame({
            'Close': close_prices,
            'ichimoku_a': ichimoku.ichimoku_a(),
            'ichimoku_b': ichimoku.ichimoku_b(),
            'ichimoku_base': ichimoku.ichimoku_base_line()
        })
        
        return bb, macd, rsi, sma, ema, ichimoku_data
    
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return None, None, None, None, None, None

def tech_indicators(data):
    if not validate_data(data):
        st.error("No valid data available for technical analysis")
        return

    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', 
                     ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA', 'Ichimoku'])

    try:
        bb, macd, rsi, sma, ema, ichimoku_data = calculate_indicators(data)
        
        if all(v is not None for v in [bb, macd, rsi, sma, ema, ichimoku_data]):
            if option == 'Close':
                plot_chart(data, 'Close', 'Closing Price')
            elif option == 'BB':
                fig = px.line(bb, x=bb.index, y=['Close', 'bb_h', 'bb_l'], 
                             title='Bollinger Bands')
                st.plotly_chart(fig)
            elif option == 'MACD':
                plot_chart(pd.DataFrame({'MACD': macd}, index=data.index), 'MACD', 'MACD')
            elif option == 'RSI':
                plot_chart(pd.DataFrame({'RSI': rsi}, index=data.index), 'RSI', 'RSI')
            elif option == 'SMA':
                plot_chart(pd.DataFrame({'SMA': sma}, index=data.index), 'SMA', 'Simple Moving Average')
            elif option == 'EMA':
                plot_chart(pd.DataFrame({'EMA': ema}, index=data.index), 'EMA', 'Exponential Moving Average')
            elif option == 'Ichimoku':
                fig = px.line(ichimoku_data, x=ichimoku_data.index, 
                             y=['Close', 'ichimoku_a', 'ichimoku_b', 'ichimoku_base'], 
                             title='Ichimoku Cloud')
                st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error in technical indicators: {str(e)}")

def sentiment_analysis(symbol):
    st.header(f"News Sentiment for {symbol}")
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news[:5] if ticker.news else []
        
        if not news:
            st.warning("No recent news found for this symbol")
            return
            
        analyzer = SentimentIntensityAnalyzer()
        
        for article in news:
            headline = article.get('title', '')
            if headline:
                sentiment_score = analyzer.polarity_scores(headline)
                st.write(f"Headline: {headline}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Positive: {sentiment_score['pos']:.2f}")
                with col2:
                    st.write(f"Neutral: {sentiment_score['neu']:.2f}")
                with col3:
                    st.write(f"Negative: {sentiment_score['neg']:.2f}")
                st.write("---")
                
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")

def backtest_strategy(data):
    if not validate_data(data):
        st.error("No valid data available for backtesting")
        return

    st.header("Backtesting Strategy")
    try:
        # Ensure we're working with a Series object
        close_series = data['Close'].squeeze()
        rsi = RSIIndicator(close_series).rsi()
        
        # Allow user to adjust RSI thresholds
        col1, col2 = st.columns(2)
        with col1:
            oversold = st.slider("Oversold threshold", 20, 40, 30)
        with col2:
            overbought = st.slider("Overbought threshold", 60, 80, 70)
        
        buy_signals = rsi < oversold
        sell_signals = rsi > overbought

        buy_dates = data.index[buy_signals]
        sell_dates = data.index[sell_signals]

        # Calculate potential returns
        if len(buy_dates) > 0 and len(sell_dates) > 0:
            returns = []
            for buy_date in buy_dates:
                # Find next sell date after buy date
                next_sells = sell_dates[sell_dates > buy_date]
                if len(next_sells) > 0:
                    sell_date = next_sells[0]
                    buy_price = data.loc[buy_date, 'Close']
                    sell_price = data.loc[sell_date, 'Close']
                    returns.append((sell_price - buy_price) / buy_price * 100)
            
            if returns:
                st.write(f"Average Return per Trade: {np.mean(returns):.2f}%")
                st.write(f"Number of Trades: {len(returns)}")

        # Display signals
        col1, col2 = st.columns(2)
        with col1:
            if len(buy_dates) > 0:
                st.write(f"Buy Signals (RSI < {oversold}):")
                st.write(buy_dates.strftime('%Y-%m-%d').tolist())
            else:
                st.write("No buy signals found in this period")

        with col2:
            if len(sell_dates) > 0:
                st.write(f"Sell Signals (RSI > {overbought}):")
                st.write(sell_dates.strftime('%Y-%m-%d').tolist())
            else:
                st.write("No sell signals found in this period")

    except Exception as e:
        st.error(f"Error in backtesting: {str(e)}")

def portfolio_performance(data):
    if not validate_data(data):
        st.error("No valid data available for portfolio analysis")
        return

    st.header("Portfolio Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        shares = st.number_input("Enter number of shares", min_value=1, value=100)
    with col2:
        investment_date = st.date_input(
            "Investment Date",
            min_value=data.index[0].date(),
            max_value=data.index[-1].date(),
            value=data.index[-1].date()
        )
    
    if st.button("Add to Portfolio"):
        try:
            # Convert to float to ensure we're working with scalar values
            price = float(data.loc[str(investment_date), 'Close'])
            st.session_state.portfolio[ticker_symbol] = {
                'shares': int(shares),
                'price': price,
                'date': investment_date
            }
            st.success(f"Added {shares} shares of {ticker_symbol} to portfolio")
        except Exception as e:
            st.error(f"Error adding to portfolio: {str(e)}")
    
    if st.session_state.portfolio:
        total_value = 0.0
        total_cost = 0.0
        
        for stock, info in st.session_state.portfolio.items():
            try:
                # Ensure we're working with scalar values
                current_price = float(data['Close'].iloc[-1])
                shares = int(info['shares'])
                entry_price = float(info['price'])
                
                # Calculate position metrics
                position_value = current_price * shares
                cost_basis = entry_price * shares
                roi = ((current_price - entry_price) / entry_price * 100)
                
                # Update portfolio totals
                total_value += position_value
                total_cost += cost_basis
                
                # Display position information
                st.write(f"### {stock}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Shares: {shares:,}")
                    st.write(f"Entry Date: {info['date']}")
                with col2:
                    st.write(f"Entry Price: ${entry_price:,.2f}")
                    st.write(f"Current Price: ${current_price:,.2f}")
                with col3:
                    st.write(f"Position Value: ${position_value:,.2f}")
                    st.write(f"ROI: {roi:.2f}%")
                st.write("---")
            
            except Exception as e:
                st.error(f"Error calculating performance for {stock}: {str(e)}")
        
        # Calculate portfolio summary using scalar values
        if total_cost > 0:
            total_roi = ((total_value - total_cost) / total_cost * 100)
        else:
            total_roi = 0.0
            
        # Display portfolio summary
        st.write("## Portfolio Summary")
        st.write(f"Total Value: ${total_value:,.2f}")
        st.write(f"Total Cost: ${total_cost:,.2f}")
        st.write(f"Total ROI: {total_roi:.2f}%")
        
    else:
        st.info("Portfolio is empty. Add some stocks to track performance.")

def stock_info(symbol):
    st.header('Stock Information')
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Basic Info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("### Company Info")
            st.write(f"**Name:** {info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        
        with col2:
            st.write("### Market Data")
            st.write(f"**Market Cap:** ${info.get('marketCap', 0):,.2f}")
            st.write(f"**Volume:** {info.get('volume', 'N/A'):,}")
            st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
        
        with col3:
            st.write("### Dividends & Yields")
            st.write(f"**Dividend Yield:** {info.get('dividendYield', 0) * 100:.2f}%")
            st.write(f"**Forward Yield:** {info.get('dividendRate', 'N/A')}")
            st.write(f"**Payout Ratio:** {info.get('payoutRatio', 'N/A')}")
        
        # Additional metrics
        st.write("### Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
        with col2:
            st.metric("52 Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
        with col3:
            st.metric("50 Day Average", f"${info.get('fiftyDayAverage', 'N/A')}")
        with col4:
            st.metric("200 Day Average", f"${info.get('twoHundredDayAverage', 'N/A')}")
        
        # Business description
        if info.get('longBusinessSummary'):
            st.write("### Business Summary")
            st.write(info['longBusinessSummary'])
            
    except Exception as e:
        st.error(f"Error fetching stock info: {str(e)}")

def main():
    # Download data
    data = download_data(ticker_symbol, start_date, end_date)
    st.session_state.data = data
    
    # Main menu
    menu_option = st.sidebar.selectbox(
        'Select Feature',
        ['Stock Info', 'Visualize', 'Recent Data', 'Backtest Strategy', 
         'Portfolio Performance', 'News Sentiment']
    )
    
    # Display selected feature
    if menu_option == 'Visualize':
        tech_indicators(data)
    elif menu_option == 'Recent Data':
        if data is not None:
            st.header('Recent Data')
            st.dataframe(data.tail(10))
        else:
            st.error("No data available to display")
    elif menu_option == 'Stock Info':
        stock_info(ticker_symbol)
    elif menu_option == 'Backtest Strategy':
        backtest_strategy(data)
    elif menu_option == 'Portfolio Performance':
        portfolio_performance(data)
    elif menu_option == 'News Sentiment':
        sentiment_analysis(ticker_symbol)

if __name__ == '__main__':
    main()
