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
        return df
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

def plot_chart(data, column, title):
    try:
        fig = px.line(data, x=data.index, y=column, title=title)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error plotting chart: {str(e)}")

def tech_indicators(data):
    if data is None:
        st.error("No data available for technical analysis")
        return

    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', 
                     ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA', 'Ichimoku'])

    try:
        # Bollinger bands with specified window
        bb_indicator = BollingerBands(data['Close'], window=20, window_dev=2)
        bb = data.copy()
        bb['bb_h'] = bb_indicator.bollinger_hband()
        bb['bb_l'] = bb_indicator.bollinger_lband()
        bb = bb[['Close', 'bb_h', 'bb_l']]

        # Other indicators with specified windows
        macd = MACD(
            data['Close'],
            window_slow=26,
            window_fast=12,
            window_sign=9
        ).macd()
        
        rsi = RSIIndicator(
            data['Close'],
            window=14
        ).rsi()
        
        sma = SMAIndicator(
            data['Close'],
            window=14
        ).sma_indicator()
        
        ema = EMAIndicator(
            data['Close'],
            window=14
        ).ema_indicator()

        # Ichimoku with default periods
        ichimoku = IchimokuIndicator(
            high=data['High'],
            low=data['Low'],
            window1=9,
            window2=26,
            window3=52
        )
        ichimoku_data = data.copy()
        ichimoku_data['ichimoku_a'] = ichimoku.ichimoku_a()
        ichimoku_data['ichimoku_b'] = ichimoku.ichimoku_b()
        ichimoku_data['ichimoku_base_line'] = ichimoku.ichimoku_base_line()

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
                         y=['Close', 'ichimoku_a', 'ichimoku_b'], 
                         title='Ichimoku Cloud')
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        st.error("Please make sure your data contains the required columns (Close, High, Low)")

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
                st.write(f"Sentiment Score: {sentiment_score}")
                st.write("---")
                
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")

def backtest_strategy(data):
    if data is None:
        st.error("No data available for backtesting")
        return

    st.header("Backtesting Strategy")
    try:
        rsi = RSIIndicator(data['Close']).rsi()
        buy_signals = rsi < 30
        sell_signals = rsi > 70

        buy_dates = data.index[buy_signals]
        sell_dates = data.index[sell_signals]

        if len(buy_dates) > 0:
            st.write("Buy Signals (RSI < 30):")
            st.write(buy_dates.strftime('%Y-%m-%d').tolist())
        else:
            st.write("No buy signals found in this period")

        if len(sell_dates) > 0:
            st.write("Sell Signals (RSI > 70):")
            st.write(sell_dates.strftime('%Y-%m-%d').tolist())
        else:
            st.write("No sell signals found in this period")

    except Exception as e:
        st.error(f"Error in backtesting: {str(e)}")

def portfolio_performance(data):
    if data is None:
        st.error("No data available for portfolio analysis")
        return

    st.header("Portfolio Performance")
    shares = st.number_input("Enter number of shares", min_value=1, value=100)
    if st.button("Add to Portfolio"):
        st.session_state.portfolio[ticker_symbol] = {
            'shares': shares,
            'price': data['Close'].iloc[-1]
        }
    
    if st.session_state.portfolio:
        for stock, info in st.session_state.portfolio.items():
            current_price = data['Close'].iloc[-1]
            roi = ((current_price - info['price']) / info['price'] * 100)
            st.write(f"{stock}:")
            st.write(f"Shares: {info['shares']}")
            st.write(f"Initial Price: ${info['price']:.2f}")
            st.write(f"Current Price: ${current_price:.2f}")
            st.write(f"ROI: {roi:.2f}%")
            st.write("---")
    else:
        st.write("Portfolio is empty. Add some stocks to track performance.")

def stock_info(symbol):
    st.header('Stock Information')
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Company Name:** {info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        with col2:
            st.write(f"**Market Cap:** {info.get('marketCap', 'N/A')}")
            st.write(f"**Dividend Yield:** {info.get('dividendYield', 'N/A')}")
            st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
    except Exception as e:
        st.error(f"Error fetching stock info: {str(e)}")

def main():
    # Download data
    data = download_data(ticker_symbol, start_date, end_date)
    st.session_state.data = data
    
    # Main menu
    menu_option = st.sidebar.selectbox(
        'Select Feature',
        ['Visualize', 'Recent Data', 'Stock Info', 'Backtest Strategy', 
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