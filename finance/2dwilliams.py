import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Define functions for Williams indicator and Lempel-Ziv complexity
def williams_indicator(high, low, close):
    hh = high.rolling(window=14).max()
    ll = low.rolling(window=14).min()
    res = -100 * ((hh - close) / (hh - ll))
    return res

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

# Standart deviation

# Calculate Williams indicator
# williams = williams_indicator(df['High'], df['Low'], df['Close'])

# Calculate binary sequence
# binary_sequence = [1 if x <= -20 else 0 for x in williams]

# Calculate Lempel-Ziv complexity for each window of 21 days
# window_size = 21
# lz_complexities = []
# for i in range(window_size//2, len(binary_sequence), window_size):
#     window = binary_sequence[max(0, i-window_size//2):min(len(binary_sequence), i+window_size//2)]
#     lz_complexity = lempel_ziv_complexity(window)
#     lz_complexities.append(lz_complexity)

# Group the data by month
months_data = {}
for month, group in df_btc.groupby(pd.Grouper(key='Date', freq='M')):
    # Calculate Williams indicator
    williams = williams_indicator(group['High'], group['Low'], group['Close'])
    # Calculate binary sequence
    binary_sequence = [1 if x <= -20 else 0 for x in williams]
    # Calculate Lempel-Ziv complexity for each window of 21 days
    window_size = 21
    lz_complexities = []
    for i in range(window_size//2, len(binary_sequence), window_size):
        window = binary_sequence[max(0, i-window_size//2):min(len(binary_sequence), i+window_size//2)]
        lz_complexity = lempel_ziv_complexity(window)
        lz_complexities.append(lz_complexity)
    # Save binary sequences and corresponding Lempel-Ziv complexities for each month
    # complexity = lempel_ziv_complexity(binary_sequence)
    months_data[month]=pd.DataFrame(data= binary_sequence ).transpose() 


# Group the data by month
months_eur = []
complexities_eur = []
for month, group in df_eth.groupby(pd.Grouper(key='Date', freq='M')):
    # Calculate Williams indicator
    williams = williams_indicator(group['High'], group['Low'], group['Close'])
    # Calculate binary sequence
    binary_sequence = [1 if x <= -20 else 0 for x in williams]
    # Calculate Lempel-Ziv complexity for each window of 21 days
    window_size = 21
    lz_complexities = []
    for i in range(window_size//2, len(binary_sequence), window_size):
        window = binary_sequence[max(0, i-window_size//2):min(len(binary_sequence), i+window_size//2)]
        lz_complexity = lempel_ziv_complexity(window)
        lz_complexities.append(lz_complexity)
    # Save binary sequences and corresponding Lempel-Ziv complexities for each month
    # complexity = lempel_ziv_complexity(binary_sequence)
    month_df = months_data[month]
    month_df = pd.concat( [month_df, pd.DataFrame( binary_sequence).transpose() ], axis=0)
    months_data[month] = month_df

# Group the data by month
months_usdt = []
complexities_ustd = []
for month, group in df_usdt.groupby(pd.Grouper(key='Date', freq='M')):
    # Calculate Williams indicator
    williams = williams_indicator(group['High'], group['Low'], group['Close'])
    # Calculate binary sequence
    binary_sequence = [1 if x <= -20 else 0 for x in williams]
    # Calculate Lempel-Ziv complexity for each window of 21 days
    window_size = 21
    lz_complexities = []
    for i in range(window_size//2, len(binary_sequence), window_size):
        window = binary_sequence[max(0, i-window_size//2):min(len(binary_sequence), i+window_size//2)]
        lz_complexity = lempel_ziv_complexity(window)
        lz_complexities.append(lz_complexity)
    # Save binary sequences and corresponding Lempel-Ziv complexities for each month
    # complexity = lempel_ziv_complexity(binary_sequence)
    month_df = months_data[month]
    month_df = pd.concat( [month_df, pd.DataFrame( binary_sequence).transpose() ], axis=0)
    months_data[month] = month_df



complexities_all = []
for month, matrix in months_data.items():   
    print(f"{month} matrix ={matrix}" )
    complexity = lempel_ziv_complexity(matrix.to_numpy(). flatten())  
    complexities_all.append(complexity)

# Create a list of months for x-axis labels
months = list(months_data.keys())

# Create a line plot of Lempel-Ziv complexities vs. months
plt.plot(months, complexities_all)

# Set the x-axis label to "Month"
plt.xlabel("Month")

# Set the y-axis label to "Lempel-Ziv Complexity"
plt.ylabel("Lempel-Ziv Complexity")

# Set the title to "Lempel-Ziv Complexity vs. Month"
plt.title("Lempel-Ziv Complexity vs. Month")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
# Display the plot
plt.show()