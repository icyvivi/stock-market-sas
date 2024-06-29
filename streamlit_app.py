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
        sidebar_menu_list = ["Stock Price", "Stocks Performance Comparison", "Stock Prediction", 'About']
        selected = option_menu("Data Analytics", sidebar_menu_list)

# read csv file
stock_df = pd.read_csv("tickers_list.csv")
stock_df[['trading_name', 'ticker']].to_csv('ticker_mapping.csv', index=False)
dict_csv = pd.read_csv('ticker_mapping.csv', header=None, index_col=0).to_dict()[1]

# Stock Performance Comparison Section Starts Here
if(selected == sidebar_menu_list[0]):  # if user selects 'Stocks Performance Comparison'
    st.subheader("Stocks Performance")
    tickers = stock_df["trading_name"].sort_values()
    # dropdown for selecting assets
    # dropdown = st.multiselect('Select a company of interest', tickers)
    
    dropdown = []
    dropdown.append(st.selectbox('Select a company of interest :', tickers))

    with st.spinner('Loading...'):  # spinner while loading
        time.sleep(2)

    symb_list = []  # list for storing symbols
    for i in dropdown:  # for each asset selected
        val = dict_csv.get(i)  # get symbol from csv file
        symb_list.append(val)  # append symbol to list
    ticker = symb_list[0] # forcing to 1 ticker first
    st.write(f'Ticker : {ticker}')


    def download_ticker(ticker):
        df = yf.Ticker(ticker).history(period='max', interval='1d', auto_adjust=False, back_adjust=False).reset_index()
        df['ticker'] = ticker
        df['datetime'] = pd.to_datetime(df['Date'])
        df['date_local'] = df['datetime'].dt.date
        # df['time'] = df['datetime'].dt.time
        df['year'] = df['datetime'].dt.year.astype(int)
        df['month'] = df['datetime'].dt.month.astype(int)
        df['day'] = df['datetime'].dt.day.astype(int)
        df['transaction_estimate'] = df['Adj Close'] * df['Volume']
        # df['prediction_target'] = df.groupby(['ticker'], as_index=False)['Adj Close'].pct_change(22).shift(-22)
        # df.dropna(inplace=True)
        #df.drop(columns=['Date', 
        #                 #'ticker', 
        #                 'datetime', 'date'], inplace=True)
        return df

    import plotly.express as px
    if len(dropdown) > 0:  # if user selects at least one asset
        start = datetime.date(1900, 1, 1)
        end = datetime.date.today()
        df = download_ticker(ticker)
        #closingPrice = df['Adj Close']
        #volume = df['Volume']

        st.subheader('OHLCV')
        st.dataframe(df.sort_values(['Date'], ascending=False).reset_index(drop=True), height=210, hide_index=False, use_container_width=True)
        # st.write(df)  # display raw data
        #chart = ('Line Chart', 'Area Chart', 'Bar Chart')  # chart types
        # dropdown for selecting chart type
        #dropdown1 = st.selectbox('Pick your chart', chart)
        #with st.spinner('Loading...'):  # spinner while loading
        #    time.sleep(2)

        #st.subheader('{}'.format(ticker))
        #plotly_fig = px.line(df, x='Date', y='Adj Close', 
        #                     template='plotly_dark',
        #                     title=ticker)
        fig_date = df['date_local'] #df['Date']
        fig_open = df['Open']
        fig_high = df['High']
        fig_low = df['Low']
        fig_close = df['Close']
        fig_price = df['Adj Close']
        fig_vol = df['Volume']

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=fig_date,
                             open=fig_open, high=fig_high, low=fig_low, close=fig_close,
                             increasing_line_color='green', decreasing_line_color='red',
                             name='OHLC'),
              row=1, col=1)
        fig.add_trace(go.Bar(x=fig_date, y=fig_vol,
                     marker=dict(color=np.where(fig_close.diff() >= 0, 'green', 'red')),
                     name='Volume'),
              row=2, col=1)


        # Update layout for better readability
        fig.update_layout(title = ticker,
                          #title=f"{str(df['Company'].unique()[0])} triggers since {str(model_last_date.date())} (Score : Buy {df_backtest[df_backtest['trade_action'] == 'buy']['f1-score'].iloc[0]:.0%} , Sell {df_backtest[df_backtest['trade_action'] == 'sell']['f1-score'].iloc[0]:.0%})",
                        #xaxis_title="Date",
                        yaxis_title="Price",
                        yaxis2_title="Volume",
                        legend_title="Legends",
                        plot_bgcolor='black',
                        paper_bgcolor='black',
                        font=dict(color='white'),
                        showlegend=True,
                        xaxis_rangeslider_visible=False)

        # Remove gridlines
        fig.update_xaxes(showgrid=False, type='category')
        fig.update_yaxes(showgrid=False, row=1, col=1)
        fig.update_yaxes(showgrid=False, row=2, col=1)
        st.write(fig)
        #st.line_chart(df, x='Date', y='Adj Close', # width=1000, height=800, 
        #              use_container_width=False)  # display line chart
        #st.bar_chart(df, x='Date', y='Volume')  # display bar chart

    else:  # if user doesn't select any asset
        #dropdown = ['NVDA']
        st.write('Please select a company')  # display message
        pass
# Stock Performance Comparison Section Ends Here