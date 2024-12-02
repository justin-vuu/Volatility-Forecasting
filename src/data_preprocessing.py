import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(ticker):
    btc = yf.Ticker(ticker)
    df = btc.history(period='max')
    return df

def compute_volatility(df, window=30, n_future=7):
    df['returns'] = 100 * df.Close.pct_change().dropna()
    df['log_returns'] = np.log(df.Close / df.Close.shift(1))

    df['vol_current'] = df.log_returns.rolling(window=window).apply(realized_volatility_daily)
    df['vol_future'] = df.log_returns.shift(-n_future).rolling(window=window).apply(realized_volatility_daily)

    df.dropna(inplace=True)
    return df

def realized_volatility_daily(series_log_return):
    n = len(series_log_return)
    return np.sqrt(np.sum(series_log_return ** 2) / (n-1))
