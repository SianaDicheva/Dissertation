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



# Function to calculate Lempel-Ziv complexity
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


# Read BTC-USD CSV file into a pandas DataFrame
df_btc = pd.read_csv('././financial_data/BTCUSD3y.csv')

# Convert Date column to datetime format
df_btc['Date'] = pd.to_datetime(df_btc['Date'])

# Read USD-EUR CSV file into a pandas DataFrame
df_eth = pd.read_csv('././financial_data/ETHUSD3y.csv')

# Convert Date column to datetime format
df_eth['Date'] = pd.to_datetime(df_eth['Date'])

# Read USD-EUR CSV file into a pandas DataFrame
df_usdt = pd.read_csv('././financial_data/USDTUSD3y.csv')

# Convert Date column to datetime format
df_usdt['Date'] = pd.to_datetime(df_usdt['Date'])

# Group the data by month
months_btc = []
complexities_btc = []
for month, group in df_btc.groupby(pd.Grouper(key='Date', freq='M')):
    # Convert the price difference to binary sequence
    diff = (group['Close'] - group['Open']) / group['Open']
    # Convert the price difference to binary sequence
    binary_sequence = np.where(diff > 0, 1, 0)
    # binary_sequence = np.where(group['Open'] > group['Close'], 1, 0)
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    months_btc.append(month)
    complexities_btc.append(complexity)


# Group the data by month
months_eur = []
complexities_eur = []
for month, group in df_eth.groupby(pd.Grouper(key='Date', freq='M')):
    # Convert the price difference to binary sequence
    diff = (group['Close'] - group['Open']) / group['Open']
    # Convert the price difference to binary sequence
    binary_sequence = np.where(diff > 0, 1, 0)
    # binary_sequence = np.where(group['Open'] > group['Close'], 1, 0)
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    months_eur.append(month)
    complexities_eur.append(complexity)

# Group the data by month
months_usdt = []
complexities_ustd = []
for month, group in df_usdt.groupby(pd.Grouper(key='Date', freq='M')):
    diff = (group['Close'] - group['Open']) / group['Open']
    # Convert the price difference to binary sequence
    binary_sequence = np.where(diff > 0, 1, 0)
    # Convert the price difference to binary sequence
    # binary_sequence = np.where(group['Open'] > group['Close'], 1, 0)
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    months_usdt.append(month)
    complexities_ustd.append(complexity)

print(34)
# Plot the complexities of BTC-USD and USD-EUR on the same graph
plt.plot(months_btc, complexities_btc, label='BTC-USD')
plt.plot(months_eur, complexities_eur, label='USD-ETH')
plt.plot(months_usdt, complexities_ustd, label='USD-USTD')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Complexity', fontsize=12)
plt.title('Lempel-Ziv Complexity of BTC-USD, ETH-USD, USDT-USD in 2020-2022')
plt.legend(loc='upper left', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

print(43)




# Group the data by month
months_btc = []
binary_seqs_btc = []
for month, group in df_btc.groupby(pd.Grouper(key='Date', freq='M')):
    # Convert the price difference to binary sequence
    binary_sequence = np.where(group['Open'] > group['Close'], 1, 0)
    months_btc.append(month)
    binary_seqs_btc.append(binary_sequence)

# Group the data by month
months_eth = []
binary_seqs_eth = []
for month, group in df_eth.groupby(pd.Grouper(key='Date', freq='M')):
    # Convert the price difference to binary sequence
    binary_sequence = np.where(group['Open'] > group['Close'], 1, 0)
    months_eth.append(month)
    binary_seqs_eth.append(binary_sequence)

# Group the data by month
months_usdt = []
binary_seqs_usdt = []
for month, group in df_usdt.groupby(pd.Grouper(key='Date', freq='M')):
    # Convert the price difference to binary sequence
    binary_sequence = np.where(group['Open'] > group['Close'], 1, 0)
    months_usdt.append(month)
    binary_seqs_usdt.append(binary_sequence)

# Calculate Lempel-Ziv complexities for each currency and month
complexities = []
for i in range(len(months_btc)):
    row = []
    for binary_seq in [binary_seqs_btc[i], binary_seqs_eth[i], binary_seqs_usdt[i]]:
        complexity = lempel_ziv_complexity(binary_seq)
        row.append(complexity)
    complexities.append(row)

# Convert complexities to a pandas DataFrame
df_complexities = pd.DataFrame(complexities, columns=['BTC', 'ETH', 'USDT'], index=months_btc)

# fig, ax = plt.subplots(figsize=(10, 6))
# df_complexities.plot(ax=ax)
# ax.set_xlabel('Month')
# ax.set_ylabel('Lempel-Ziv Complexity')
# ax.set_title('Lempel-Ziv Complexity of BTC, ETH, and USDT Prices')
# plt.show()


# new stuff

# Define functions for Williams indicator and Lempel-Ziv complexity
def williams_indicator(high, low, close):
    hh = high.rolling(window=14).max()
    ll = low.rolling(window=14).min()
    res = -100 * ((hh - close) / (hh - ll))
    return res


# Calculate Williams indicator
williams_btc = williams_indicator(df_btc['High'], df_btc['Low'], df_btc['Close'])

# Calculate binary sequence
binary_sequence_btc = [1 if x <= -20 else 0 for x in williams_btc]

# Calculate Lempel-Ziv complexity for each window of 21 days
window_size = 21
lz_complexities_btc = []
for i in range(window_size//2, len(binary_sequence_btc), window_size):
    window = binary_sequence_btc[max(0, i-window_size//2):min(len(binary_sequence_btc), i+window_size//2)]
    lz_complexity = lempel_ziv_complexity(window)
    lz_complexities_btc.append(lz_complexity)

# Create dates corresponding to midpoint of each window
dates = pd.date_range(start='2020-01-11', end='2022-12-31', freq='21D') + pd.Timedelta(days=10)

# Calculate Williams indicator
williams_usdt = williams_indicator(df_usdt['High'], df_usdt['Low'], df_usdt['Close'])
# Calculate binary sequence

binary_sequence_usdt = [1 if x <= -20 else 0 for x in williams_usdt]

# Calculate Lempel-Ziv complexity for each window of 21 days
window_size = 21
lz_complexities_usdt = []
for i in range(window_size//2, len(binary_sequence_usdt), window_size):
    window = binary_sequence_usdt[max(0, i-window_size//2):min(len(binary_sequence_usdt), i+window_size//2)]
    lz_complexity_eth = lempel_ziv_complexity(window)
    lz_complexities_usdt.append(lz_complexity_eth)

# Create dates corresponding to midpoint of each window

dates_usdt = pd.date_range(start='2020-01-11', end='2022-12-31', freq='21D') + pd.Timedelta(days=10)

# Calculate Williams indicator
williams_eth = williams_indicator(df_eth['High'], df_eth['Low'], df_eth['Close'])
# Calculate binary sequence

binary_sequence_eth = [1 if x <= -20 else 0 for x in williams_eth]

# Calculate Lempel-Ziv complexity for each window of 21 days
window_size = 21
lz_complexities_eth = []
for i in range(window_size//2, len(binary_sequence_eth), window_size):
    window = binary_sequence_eth[max(0, i-window_size//2):min(len(binary_sequence_eth), i+window_size//2)]
    lz_complexity_eth = lempel_ziv_complexity(window)
    lz_complexities_eth.append(lz_complexity_eth)

# Create dates corresponding to midpoint of each window

dates_eth = pd.date_range(start='2020-01-11', end='2022-12-31', freq='21D') + pd.Timedelta(days=10)

# Plot Lempel-Ziv complexity over time
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dates, lz_complexities_btc, color='blue', linewidth=2, label='BTC')
ax.plot(dates_eth, lz_complexities_eth, color='red', linewidth=2, label='ETH')
ax.plot(dates_usdt, lz_complexities_usdt, color='green', linewidth=2, label='USDT')
# Set title and labels
ax.set_title('Lempel-Ziv Complexity of BTC-USD, ETH-USD, USDT-USD in 2020-2022', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Complexity', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# standart deviations

# Group the data by month
months_btc = []
complexities_btc = []
for month, group in df_btc.groupby(pd.Grouper(key='Date', freq='M')):
    diff = (group['Close'] - group['Open']) / group['Open']
    mean_diff_eur = np.mean(diff)
    std_diff_eur = np.std(diff)
    
    # Convert the price difference to binary sequence
    binary_sequence = (diff.abs() > 2* std_diff_eur + mean_diff_eur).astype(int).tolist()
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    months_btc.append(month)
    complexities_btc.append(complexity)


# Group the data by month
months_eur = []
complexities_eur = []
for month, group in df_eth.groupby(pd.Grouper(key='Date', freq='M')):
    diff = (group['Close'] - group['Open']) / group['Open']
    mean_diff_eur = np.mean(diff)
    std_diff_eur = np.std(diff)
    
    # Convert the price difference to binary sequence
    binary_sequence = (diff.abs() > 2* std_diff_eur + mean_diff_eur).astype(int).tolist()
    
    # Convert the price difference to binary sequence
    # binary_sequence = ((group['Open'] - group['Close']).abs() > 2* std_diff_eur + mean_diff_eur).astype(int).tolist()
    
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    months_eur.append(month)
    complexities_eur.append(complexity)

# Group the data by month
months_usdt = []
complexities_ustd = []
for month, group in df_usdt.groupby(pd.Grouper(key='Date', freq='M')):
    diff = (group['Close'] - group['Open']) / group['Open']
    mean_diff_eur = np.mean(diff)
    std_diff_eur = np.std(diff)
    
    # Convert the price difference to binary sequence
    binary_sequence = (diff.abs() > 2* std_diff_eur + mean_diff_eur).astype(int).tolist()
    
    # # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    months_usdt.append(month)
    complexities_ustd.append(complexity)

# Plot Lempel-Ziv complexity over time
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(months_btc, complexities_btc, color='blue', linewidth=2, label='BTC')
ax.plot(months_eth, complexities_eur, color='red', linewidth=2, label='ETH')
ax.plot(months_usdt, complexities_ustd, color='green', linewidth=2, label='USDT')
# Set title and labels
ax.set_title('Lempel-Ziv Complexity of BTC-USD, ETH-USD, USDT-USD in 2020-2022', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Complexity', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=12)
plt.xticks(rotation=45)
plt.show()




# # Group the data by month and currency
# data = {}
# currencies = ['BTC-USD', 'ETH-USD', 'USDT-USD']
# for currency, df in zip(currencies, [df_btc, df_eth, df_usdt]):
#     for month, group in df.groupby(pd.Grouper(key='Date', freq='M')):
#         # Convert the price difference to binary sequence
#         binary_sequence = np.where(group['Open'] > group['Close'], 1, 0)
#         # Calculate the Lempel-Ziv complexity of the binary sequence
#         complexity = lempel_ziv_complexity(binary_sequence)
#         if month not in data:
#             data[month] = {}
#         data[month][currency] = complexity

# # Create a 2D matrix of Lempel-Ziv complexities
# complexities = []
# months = sorted(data.keys())
# for month in months:
#     row = []
#     for currency in currencies:
#         row.append(data[month].get(currency, 0))
#     complexities.append(row)


# # different plot
# # Group the data by month and currency
# data = {}
# currencies = ['BTC-USD', 'ETH-USD', 'USDT-USD']
# for currency, df in zip(currencies, [df_btc, df_eth, df_usdt]):
#     for month, group in df.groupby(pd.Grouper(key='Date', freq='M')):
#         # Convert the price difference to binary sequence
#         binary_sequence = np.where(group['Open'] > group['Close'], 1, 0)
#         # Calculate the Lempel-Ziv complexity of the binary sequence
#         complexity = lempel_ziv_complexity(binary_sequence)
#         if month not in data:
#             data[month] = {}
#         data[month][currency] = complexity

# # Create a line plot of Lempel-Ziv complexities over time for each currency
# plt.figure(figsize=(12, 6))
# for currency in currencies:
#     complexities = []
#     months = []
#     for month in sorted(data.keys()):
#         if currency in data[month]:
#             complexities.append(data[month][currency])
#             months.append(month)
#     plt.plot(months, complexities, label=currency)

# # plt.xlabel('Time')
# # plt.ylabel('Lempel-Ziv Complexity')
# # plt.title('Lempel-Ziv Complexity of Financial Data')
# # plt.legend()
# # plt.show()