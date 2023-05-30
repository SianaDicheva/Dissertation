import pandas as pd
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt


# Define functions for Williams indicator and Lempel-Ziv complexity
def williams_indicator(high, low, close):
    hh = high.rolling(window=14).max()
    ll = low.rolling(window=14).min()
    res = -100 * ((hh - close) / (hh - ll))
    return res

def lempel_ziv_complexity(binary_sequence):
    """Lempel-Ziv complexity for a binary sequence, in simple Python code."""
    u, v, w = 0, 1, 1
    v_max = 1
    length = len(binary_sequence)
    complexity = 1
    while True:
        if binary_sequence[u + v - 1] == binary_sequence[w + v - 1]:
            v += 1
            if w + v >= length:
                complexity += 1
                break
        else:
            if v > v_max:
                v_max = v
            u += 1
            if u == w:
                complexity += 1
                w += v_max
                if w > length:
                    break
                else:
                    u = 0
                    v = 1
                    v_max = 1
            else:
                v = 1
    return complexity

# Load data
df = pd.read_csv('./financial_data/BTCUSD3y.csv')

# Calculate Williams indicator
williams = williams_indicator(df['High'], df['Low'], df['Close'])

# Calculate binary sequence
binary_sequence = [1 if x <= -20 else 0 for x in williams]

# Calculate Lempel-Ziv complexity for each window of 21 days
window_size = 21
lz_complexities = []
for i in range(window_size//2, len(binary_sequence), window_size):
    window = binary_sequence[max(0, i-window_size//2):min(len(binary_sequence), i+window_size//2)]
    lz_complexity = lempel_ziv_complexity(window)
    lz_complexities.append(lz_complexity)

# Create dates corresponding to midpoint of each window
dates = pd.date_range(start='2020-01-11', end='2022-12-31', freq='21D') + pd.Timedelta(days=10)

# Plot Lempel-Ziv complexity over time
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dates, lz_complexities, color='blue', linewidth=2, label='Lempel-Ziv complexity')
# Set title and labels
ax.set_title('Lempel-Ziv Complexity of BTC-USD in 2022', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Complexity', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=12)
plt.xticks(rotation=45)
plt.show()

df = pd.read_csv("./financial_data/BTCUSD3y.csv", parse_dates=['Date'])
df = df.iloc[::-1]  # Reverse the dataframe to plot from earliest to latest date

# Compute Williams %R indicator
period = 14
df['HH'] = df['High'].rolling(period).max()
df['LL'] = df['Low'].rolling(period).min()
df['W%R'] = -100 * (df['HH'] - df['Close']) / (df['HH'] - df['LL'])

# Plot the candlestick chart with Williams %R indicator
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the candlestick chart
ohlc = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()
ohlc['Date'] = ohlc['Date'].apply(mdates.date2num)
candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='g', colordown='r')
ax.xaxis_date()
ax.grid(True)

# Plot Williams %R indicator
ax2 = ax.twinx()
ax2.plot(df['Date'], df['W%R'], color='blue', label='Williams %R')
ax2.axhline(-20, color='green', linestyle='--')
ax2.axhline(-80, color='red', linestyle='--')
ax2.fill_between(df['Date'], -20, df['W%R'], where=(df['W%R'] >= -20), alpha=0.1, color='green')
ax2.fill_between(df['Date'], -80, df['W%R'], where=(df['W%R'] <= -80), alpha=0.1, color='red')
ax2.set_ylim(-100, 0)
ax2.invert_yaxis()
ax2.grid(False)

# Set the x-axis label and title
ax.set_xlabel('Date')
ax.set_title('Bitcoin Candlestick Chart with Williams %R Indicator')


plt.show()


fig, ax1 = plt.subplots(figsize=(12, 6))

# Compute Williams %R indicator
period = 14
df['HH'] = df['High'].rolling(period).max()
df['LL'] = df['Low'].rolling(period).min()
df['W%R'] = -100 * (df['HH'] - df['Close']) / (df['HH'] - df['LL'])

# Plot Bitcoin's closing prices
ax1.plot(df['Date'], df['Close'], color='blue', label='Closing price')

# Set title and labels for primary y-axis
ax1.set_title('Bitcoin Closing Prices and Williams %R in 2022', fontsize=16)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Price (USD)', fontsize=12)

# Create secondary y-axis for Williams %R
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['W%R'], color='red', label='Williams %R')
ax2.set_ylabel('Williams %R', fontsize=12)

# Add grid and legends
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper left', fontsize=12)
ax2.legend(loc='upper right', fontsize=12)

plt.show()











