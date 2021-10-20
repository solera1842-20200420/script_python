# Nikkei225のplot

# starandard libarary
import datetime as dt
import numpy as np
import os
import pandas as pd
import pandas_datareader as pdr

# for plot
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import seaborn as sns
import streamlit as st

# for ml, statistics
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa

# dataset
@st.cache
def load_dataset():
    start ='2019-01-01'
    end = dt.date.today()
    information_source='yahoo'
    ticker_symbol='^N225'
    df = pdr.DataReader(ticker_symbol, information_source, start, end)
    return df

def show_df():
    data = load_dataset()
    df = st.write(data)
    df = st.write(df)
    return df
show_df()

# df = load_dataset()
# print(df.columns)

def plot_ts():
    df = load_dataset()
    # df = df
    plot_line = px.line(data_frame=df, x=df.index, y=df['Close'], title="Close")
    return plot_line

def plot_PctChange():
    df = load_dataset()
    plot_line = px.line(data_frame=df, x=df.index, y=df['Close'].pct_change(), title="Rate of Change")
    return plot_line

st.write('時系列データをline plot')
# plot_ts()
st.write(
    plot_ts()
)
st.write(
    plot_PctChange()
)
