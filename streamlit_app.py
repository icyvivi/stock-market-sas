import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile

from matplotlib.pyplot import axis
import streamlit as st  # streamlit library
import pandas as pd  # pandas library
import yfinance as yf  # yfinance library
import datetime  # datetime library
from datetime import date
from plotly import graph_objs as go  # plotly library
from plotly.subplots import make_subplots
#from prophet import Prophet  # prophet library
# plotly library for prophet model plotting
from prophet.plot import plot_plotly
import time  # time library
from streamlit_option_menu import option_menu  # select_options library

# Page title
st.set_page_config(page_title='ML model builder', layout="wide", #initial_sidebar_state="expanded", 
page_icon='🏗️')
st.title('🏗️ ML model builder')

st.sidebar.write('''# SAS ''')
with st.sidebar: 
        sidebar_menu_list = ["Stocks Performance Comparison", "Real-Time Stock Price", "Stock Prediction", 'About']
        selected = option_menu("Data Analytics", sidebar_menu_list)

# read csv file
stock_df = pd.read_csv("tickers_list.csv")
#dict_csv = pd.read_csv('tickers_list.csv', header=None, index_col=0).to_dict()[1]  # read csv file
#dict_csv = stock_df[['trading_name', 'ticker']].sort_values(['trading_name']).set_index('trading_name').to_dict()['ticker']
stock_df[['trading_name', 'ticker']].to_csv('ticker_mapping.csv')
dict_csv = pd.read_csv('ticker_mapping.csv', header=None, index_col=0).to_dict()[1]

# Stock Performance Comparison Section Starts Here
if(selected == sidebar_menu_list[0]):  # if user selects 'Stocks Performance Comparison'
    st.subheader("Stocks Performance Comparison")
    #stock_df["Company Name"]
    tickers = stock_df["trading_name"].sort_values()
    # tickers = stock_df["ticker"]
    #tickers = ['D05.SI', 'NVDA', 'NIO']
    # dropdown for selecting assets
    dropdown = st.multiselect('Pick your assets', tickers)

    with st.spinner('Loading...'):  # spinner while loading
        time.sleep(2)

    symb_list = []  # list for storing symbols
    for i in dropdown:  # for each asset selected
        val = dict_csv.get(i)  # get symbol from csv file
        symb_list.append(val)  # append symbol to list
    #ticker = symb_list[0]  # forcing to 1 ticker first


    def download_ticker(ticker):
        df = yf.Ticker(ticker).history(period='max', interval='1d', auto_adjust=False, back_adjust=False).reset_index()
        df['ticker'] = ticker
        df['datetime'] = pd.to_datetime(df['Date'])
        df['date'] = df['datetime'].dt.date
        # df['time'] = df['datetime'].dt.time
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['prediction_target'] = df.groupby(['ticker'], as_index=False)['Adj Close'].pct_change(22).shift(-22)
        df.dropna(inplace=True)
        df.drop(columns=['Date', 'ticker', 'datetime', 'date'], inplace=True)
        return df

    def relativeret(df):  # function for calculating relative return
        rel = df.pct_change()  # calculate relative return
        cumret = (1+rel).cumprod() - 1  # calculate cumulative return
        cumret = cumret.fillna(0)  # fill NaN values with 0
        return cumret  # return cumulative return



    if len(dropdown) > 0:  # if user selects atleast one asset
        start = datetime.date(2000, 1, 1)
        end = datetime.date.today()
        # df = relativeret(yf.download(symb_list, start, end))['Adj Close']  # download data from yfinance
        downloaded_ticker = download_ticker('D05.SI')
        df = relativeret(downloaded_ticker)['Adj Close']  # download data from yfinance
        closingPrice = downloaded_ticker['Adj Close']
        volume = downloaded_ticker['Volume']
        # download data from yfinance
        raw_df = relativeret(downloaded_ticker)
        raw_df.reset_index(inplace=True)  # reset index

        #closingPrice = downloaded_ticker['Adj Close']  # download data from yfinance
        #volume = downloaded_ticker['Volume']
        
        st.subheader('Raw Data {}'.format(dropdown))
        st.write(raw_df)  # display raw data
        chart = ('Line Chart', 'Area Chart', 'Bar Chart')  # chart types
        # dropdown for selecting chart type
        dropdown1 = st.selectbox('Pick your chart', chart)
        with st.spinner('Loading...'):  # spinner while loading
            time.sleep(2)

        st.subheader('Relative Returns {}'.format(dropdown))
                
        if (dropdown1) == 'Line Chart':  # if user selects 'Line Chart'
            st.line_chart(df)  # display line chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.line_chart(closingPrice)  # display line chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.line_chart(volume)  # display line chart

        elif (dropdown1) == 'Area Chart':  # if user selects 'Area Chart'
            st.area_chart(df)  # display area chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.area_chart(closingPrice)  # display area chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.area_chart(volume)  # display area chart

        elif (dropdown1) == 'Bar Chart':  # if user selects 'Bar Chart'
            st.bar_chart(df)  # display bar chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.bar_chart(closingPrice)  # display bar chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.bar_chart(volume)  # display bar chart

        else:
            st.line_chart(df, width=1000, height=800,
                          use_container_width=False)  # display line chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.line_chart(closingPrice)  # display line chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.line_chart(volume)  # display line chart

    else:  # if user doesn't select any asset
        #dropdown = ['NVDA']
        st.write('Please select at least one asset')  # display message
# Stock Performance Comparison Section Ends Here