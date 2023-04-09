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
df_btc = pd.read_csv('././financial_data/BTCUSD.csv')

# Convert Date column to datetime format
df_btc['Date'] = pd.to_datetime(df_btc['Date'])

# Group the data by month
months_btc = []
complexities_btc = []
for month, group in df_btc.groupby(pd.Grouper(key='Date', freq='M')):
    # Convert the price difference to binary sequence
    binary_sequence = np.where(group['Open'] > group['Close'], 1, 0)
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    months_btc.append(month)
    complexities_btc.append(complexity)

# Read USD-EUR CSV file into a pandas DataFrame
df_eur = pd.read_csv('././financial_data/ETHUSD.csv')

# Convert Date column to datetime format
df_eur['Date'] = pd.to_datetime(df_eur['Date'])

# Group the data by month
months_eur = []
complexities_eur = []
for month, group in df_eur.groupby(pd.Grouper(key='Date', freq='M')):
    # Convert the price difference to binary sequence
    binary_sequence = np.where(group['Open'] > group['Close'], 1, 0)
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    months_eur.append(month)
    complexities_eur.append(complexity)

# Plot the complexities of BTC-USD and USD-EUR on the same graph
plt.plot(months_btc, complexities_btc, label='BTC-USD')
plt.plot(months_eur, complexities_eur, label='USD-ETH')
plt.xlabel('Months')
plt.ylabel('Lempel-Ziv Complexity')
plt.title('Comparison of BTC-USD and USD-ETH')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Read BTC-USD CSV file into a pandas DataFrame
df_btc = pd.read_csv('./financial_data/BTCUSD.csv')

# Convert Date column to datetime format
df_btc['Date'] = pd.to_datetime(df_btc['Date'])

# Calculate mean and standard deviation of opening-closing price difference
mean_diff_btc = np.mean(df_btc['Open'] - df_btc['Close'])
std_diff_btc = np.std(df_btc['Open'] - df_btc['Close'])

# Group the data by month
months_btc = []
complexities_btc = []
for month, group in df_btc.groupby(pd.Grouper(key='Date', freq='M')):
    # Convert the price difference to binary sequence
    binary_sequence = ((group['Open'] - group['Close']).abs() >  std_diff_btc + mean_diff_btc).astype(int).tolist()
    # binary_sequence =  (group['Open'] > group['Close']).astype(int)
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    months_btc.append(month)
    complexities_btc.append(complexity)

# Read USD-EUR CSV file into a pandas DataFrame
df_eur = pd.read_csv('./financial_data/ETHUSD.csv')

df_eur['Date'] = pd.to_datetime(df_eur['Date'])


# Calculate mean and standard deviation of opening-closing price difference
mean_diff_eur = np.mean(df_eur['Open'] - df_eur['Close'])
std_diff_eur = np.std(df_eur['Open'] - df_eur['Close'])

# Group the data by month
months_eur = []
complexities_eur = []
for month, group in df_eur.groupby(pd.Grouper(key='Date', freq='M')):
    # Convert the price difference to binary sequence
    binary_sequence = ((group['Open'] - group['Close']).abs() >  std_diff_eur + mean_diff_eur).astype(int).tolist()
    # binary_sequence =  (group['Open'] > group['Close']).astype(int)
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    months_eur.append(month)
    complexities_eur.append(complexity)

# Plot the complexities of BTC-USD and USD-EUR on the same graph
plt.plot(months_btc, complexities_btc, label='BTC-USD')
plt.plot(months_eur, complexities_eur, label='USD-ETH')
plt.xlabel('Months')
plt.ylabel('Lempel-Ziv Complexity')
plt.title('Comparison of BTC-USD and USD-EUR')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Read BTC-USD CSV file into a pandas DataFrame
df_btc = pd.read_csv('./financial_data/BTCUSD.csv')

# Convert Date column to datetime format
df_btc['Date'] = pd.to_datetime(df_btc['Date'])

# Read USD-EUR CSV file into a pandas DataFrame
df_eur = pd.read_csv('./financial_data/ETHUSD.csv')

# Convert Date column to datetime format
df_eur['Date'] = pd.to_datetime(df_eur['Date'])

# Group the data by month for BTC-USD
months_btc = []
complexities_btc = []
for month, group in df_btc.groupby(pd.Grouper(key='Date', freq='M')):
    # Calculate mean and standard deviation of opening-closing price difference for the month
    mean_diff_btc = np.mean(group['Open'] - group['Close'])
    std_diff_btc = np.std(group['Open'] - group['Close'])
    
    # Convert the price difference to binary sequence
    binary_sequence = ((group['Open'] - group['Close']).abs() >  std_diff_btc + mean_diff_btc).astype(int).tolist()
    
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    
    months_btc.append(month)
    complexities_btc.append(complexity)

# Group the data by month for USD-EUR
months_eur = []
complexities_eur = []
for month, group in df_eur.groupby(pd.Grouper(key='Date', freq='M')):
    # Calculate mean and standard deviation of opening-closing price difference for the month
    mean_diff_eur = np.mean(group['Open'] - group['Close'])
    std_diff_eur = np.std(group['Open'] - group['Close'])
    
    # Convert the price difference to binary sequence
    binary_sequence = ((group['Open'] - group['Close']).abs() >  std_diff_eur + mean_diff_eur).astype(int).tolist()
    
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    
    months_eur.append(month)
    complexities_eur.append(complexity)

# Plot the complexities of BTC-USD and USD-EUR on the same graph
plt.plot(months_btc, complexities_btc, label='BTC-USD')
plt.plot(months_eur, complexities_eur, label='USD-ETH')
plt.xlabel('Months')
plt.ylabel('Lempel-Ziv Complexity')
plt.title('Comparison of BTC-USD and USD-ETH')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

