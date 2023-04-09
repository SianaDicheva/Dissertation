import pandas as pd
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import numpy as np

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
df = pd.read_csv('./financial_data/BTCUSD.csv')

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
dates = pd.date_range(start='2022-01-11', end='2022-12-31', freq='21D') + pd.Timedelta(days=10)

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


# plots eur-usd willams
# Load data
df1 = pd.read_csv('./financial_data/ETHUSD.csv')

# Calculate Williams indicator
williams_eur = williams_indicator(df1['High'], df1['Low'], df1['Close'])
# Calculate binary sequence

binary_sequence_eur = [1 if x <= -20 else 0 for x in williams_eur]

# Calculate Lempel-Ziv complexity for each window of 21 days
window_size = 21
lz_complexities_eur = []
for i in range(window_size//2, len(binary_sequence_eur), window_size):
    window = binary_sequence_eur[max(0, i-window_size//2):min(len(binary_sequence_eur), i+window_size//2)]
    lz_complexity_eth = lempel_ziv_complexity(window)
    lz_complexities_eur.append(lz_complexity_eth)

# Create dates corresponding to midpoint of each window

dates_eth = pd.date_range(start='2022-01-11', end='2022-12-31', freq='21D') + pd.Timedelta(days=10)

# Plot Lempel-Ziv complexity over time
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dates, lz_complexities, color='blue', linewidth=2, label='BTC')
ax.plot(dates_eth, lz_complexities_eur, color='red', linewidth=2, label='ETH')
# Set title and labels
ax.set_title('Lempel-Ziv Complexity of BTC-USD in 2022', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Complexity', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=12)
plt.xticks(rotation=45)
plt.show()