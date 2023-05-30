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

# Group the data by month and calculate the Lempel-Ziv complexity for each month
months = []
complexities_btc = []
complexities_eth = []
complexities_usdt = []
for month, group_btc, group_eth, group_usdt in zip(df_btc.groupby(pd.Grouper(key='Date', freq='M')),
                                                   df_eth.groupby(pd.Grouper(key='Date', freq='M')),
                                                   df_usdt.groupby(pd.Grouper(key='Date', freq='M'))):
    # Convert the price difference to binary sequence for each currency
    binary_sequence_btc = np.where(group_btc['Open'] > group_btc['Close'], 1, 0)
    binary_sequence_eth = np.where(group_eth['Open'] > group_eth['Close'], 1, 0)
    binary_sequence_usdt = np.where(group_usdt['Open'] > group_usdt['Close'], 1, 0)

    # Calculate the Lempel-Ziv complexity of the binary sequence for each currency
    complexity_btc = lempel_ziv_complexity(binary_sequence_btc)
    complexity_eth = lempel_ziv_complexity(binary_sequence_eth)
    complexity_usdt = lempel_ziv_complexity(binary_sequence_usdt)

    # Append the complexity values to the corresponding list
    complexities_btc.append(complexity_btc)
    complexities_eth.append(complexity_eth)
    complexities_usdt.append(complexity_usdt)

    # Append the month to the months list
    months.append(month)

# Calculate the average complexity across all three currencies for each month
complexities_avg = np.mean([complexities_btc, complexities_eth, complexities_usdt])


