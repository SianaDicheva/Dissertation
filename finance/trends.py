import pandas as pd
import numpy as np
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


# read the data
df = pd.read_csv('./financial_data/BTCUSD.csv')

# calculate the Williams indicator
def williams(data, period):
    high = data['High'].rolling(window=period).max()
    low = data['Low'].rolling(window=period).min()
    williams = ((high - data['Close']) / (high - low)) * -100
    return williams

df['Williams'] = williams(df, 14)

# set window sizes
windows = [30, 60, 90, 120]

# create subplots
fig, axs = plt.subplots(nrows=len(windows), ncols=1, figsize=(12, 10), sharex=True)

for i, window in enumerate(windows):
    # calculate the williams indicator and add it to the dataframe
    df['Williams'] = williams(df, window)

    # generate the binary sequence
    binary_sequence = np.where(df['Williams'] < -20, 1, 0)

    # calculate the Lempel-Ziv complexity
    complexity = lempel_ziv_complexity(binary_sequence)

    # plot the binary sequence and the corresponding LZ complexity
    axs[i].plot(df['Date'], binary_sequence, color='blue', label='Williams indicator')
    axs[i].set_ylabel('Binary sequence')
    axs[i].twinx().plot(df['Date'], complexity, color='red', label='LZ complexity')
    axs[i].set_ylabel('LZ complexity')
    axs[i].set_title(f'Window size: {window}')
    axs[i].legend()

# add x-axis label to the last subplot
axs[-1].set_xlabel('Date')

plt.tight_layout()
plt.show()


# read the data
df = pd.read_csv('./financial_data/BTCUSD.csv')

# set threshold values
thresholds = [-20, -30, -40, -50]

# create subplots
fig, axs = plt.subplots(nrows=len(thresholds), ncols=1, figsize=(12, 10), sharex=True)

for i, threshold in enumerate(thresholds):
    # calculate the Williams indicator and add it to the dataframe
    high = df['High'].rolling(window=14).max()
    low = df['Low'].rolling(window=14).min()
    df[f'Williams_{threshold}'] = ((high - df['Close']) / (high - low)) * -100

    # generate the binary sequence
    binary_sequence = np.where(df[f'Williams_{threshold}'] < threshold, 1, 0)

    # calculate the Lempel-Ziv complexity
    complexity = lempel_ziv_complexity(binary_sequence)

    # plot the binary sequence and the corresponding LZ complexity
    axs[i].plot(df['Date'], binary_sequence, color='blue', label='Williams indicator')
    axs[i].set_ylabel('Binary sequence')
    axs[i].twinx().plot(df['Date'], complexity, color='red', label='LZ complexity')
    axs[i].set_ylabel('LZ complexity')
    axs[i].set_title(f'Threshold: {threshold}')
    axs[i].legend()

# add x-axis label to the last subplot
axs[-1].set_xlabel('Date')

plt.tight_layout()
plt.show


import pandas as pd
import matplotlib.pyplot as plt

# load the BTC-USD data
data = pd.read_csv('btc-usd-data.csv')

# calculate the Williams %R indicator
n = 14
high = data['High'].rolling(window=n).max()
low = data['Low'].rolling(window=n).min()
williams_r = -100 * (high - data['Close']) / (high - low)

# define different moving average windows to use
windows = [7, 14, 21, 30]

# calculate the LZ complexity for each window
lz_complexities = []
for window in windows:
    # apply moving average
    close_ma = data['Close'].rolling(window=window).mean()
    # generate binary sequence based on Williams %R indicator
    binary_sequence = [1 if x <= -20 else 0 for x in williams_r]
    # calculate LZ complexity for binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    lz_complexities.append(complexity)

# plot the LZ complexity for each window
plt.plot(windows, lz_complexities, marker='o')
plt.xlabel('Moving Average Window')
plt.ylabel('LZ Complexity')
plt.title('LZ Complexity vs Moving Average Window')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("./financial_data/BTCUSD.csv")

# Calculate Williams indicator
df["HH"] = df["High"].rolling(14).max()
df["LL"] = df["Low"].rolling(14).min()
df["Williams"] = ((df["HH"] - df["Close"]) / (df["HH"] - df["LL"])) * -100

# Calculate Lempel-Ziv complexity
binary_sequence = [1 if x >= -20 else 0 for x in df["Williams"]]
lz_complexity = lempel_ziv_complexity(binary_sequence)

# Plot complexity and Williams indicator
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Williams', color=color)
ax1.plot(df['Date'], df['Williams'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('LZ Complexity', color=color)
ax2.plot(df['Date'], lz_complexity, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Trend Analysis")
plt.legend()
plt.show()
