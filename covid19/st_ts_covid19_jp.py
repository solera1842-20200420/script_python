"""
Plot Covid19 NewCases in Japan
"""

# starandard libarary
import datetime as dt
import numpy as np
import os
import pandas as pd
# import pandas_datareader as pdr

# for plot
# import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
# import seaborn as sns
import streamlit as st

# for ml, statistics
import statsmodels.api as sm
import statsmodels.formula.api as smf
#from statsmodels.tsa import seasonal

# dataset
DATA_DIR ='./data/mhlw'
RAW_DATA = 'newly_confirmed_cases_daily.csv'
@st.cache
def load_dataset():
    df = pd.read_csv(os.path.join(DATA_DIR, RAW_DATA))  # read csv
    return df

# daily data in japan
def data_jp():
    df = load_dataset()
    df = df.query('Prefecture == "ALL"')  # exclude data each prefecture
    df = df.rename(columns={'Newly confirmed cases': 'NewCases'})  # rename column
    df = df.drop(columns={'Prefecture'})
    return df

# daily data in 7days
def last_7days():  # last 7days
    df = data_jp()
    df = df.tail(7)
    return df

# function: trend
# 14days
def get_trend14():
    df = data_jp()
    d_newcases = df['NewCases'].astype(float)
    stl_14 = sm.tsa.seasonal_decompose(d_newcases, period=14)
    trend_14 = stl_14.trend
    return trend_14

# 60dyas
def get_trend60():
    df = data_jp()
    d_newcases = df['NewCases'].astype(float)
    stl_60 = sm.tsa.seasonal_decompose(d_newcases, period=60)
    trend_60 = stl_60.trend
    return trend_60

# function: plot daily data
def plot_ts():
    df = data_jp()
    fig_ts = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': True}]])
    fig_ts.add_trace(go.Scatter(x=df['Date'],
        y=df['NewCases'],  # daily data
            name="Daily"))
    fig_ts.add_trace(go.Scatter(x=df['Date'],
        y=df['NewCases'].rolling(14).mean(),  # SMA14days
            name="SMA: 14days"))
    fig_ts.add_trace(go.Scatter(x=df['Date'],
        y=df['NewCases'].rolling(30).mean(),  # SMA30days
            name="SMA: 30days"))
    fig_ts.add_trace(go.Scatter(x=df['Date'],
        y=df['NewCases'].rolling(60).mean(),  # SMA60days
            name="SMA: 60days"))

    fig_ts.update_layout(title_text="NewlyConfirmedCases: Daily, 14days, 30days, 60days'",
        width=900, height=500)
    return fig_ts

# function: plot seasnoal, trend, resid
def plot_stl():
    trend_14 = get_trend14()
    trend_60 = get_trend60()
    fig_stl = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': True}]])
    fig_stl.add_trace(go.Scatter(x=trend_14.index, y=trend_14, name="Trend: 14days"))
    fig_stl.add_trace(go.Scatter(x=trend_60.index, y=trend_60, name="Trend: 60days"))
    fig_stl.update_layout(title_text="Trend: 14days, 60days'",
                          width=700, height=300)
    return fig_stl

# funtion: plot rate of change
def plot_PctChange():
    df = data_jp()
    fig_PctChange = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': True}]])
    fig_PctChange.add_trace(go.Scatter(x=df['Date'],
        y=df['NewCases'].pct_change(),  # rate of change(daily data)
            name="RateOfChange_daily"))
    fig_PctChange.add_trace(go.Scatter(x=df['Date'],
        y=df['NewCases'].pct_change(7),  # rate of change(7days)
            name="RateOfChange_daily_7days"))

    fig_PctChange.add_trace(go.Scatter(x=df['Date'],
        y=df['NewCases'].pct_change(14),  # rate of change(14days)
            name="RateOfChange_14days"))

    fig_PctChange.add_trace(go.Scatter(x=df['Date'],
        y=df['NewCases'].pct_change(30),  # rate of change(30days)
            name="RateOfChange_30days"))

    fig_PctChange.update_layout(title_text="RateOfChange: Daily, 7days, 14days, 30days",
                                width=700, height=300)
    return fig_PctChange

"""
Display Dataframe and Plot
"""
st.title('Covid19 TimeSeries Analysis in Japan')
st.write(
    last_7days()
)

st.write('時系列データをline plot')
# plot_ts()
print("Date:", dt.datetime.today())

# plot sma
st.write(
    plot_ts()
)

# plot trend
st.write(
    plot_stl()
)

# plot rate of change
st.write(
    plot_PctChange()
)

# scipt end...s