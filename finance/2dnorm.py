import pandas as pd
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
months_data = {}
for month, group in df_btc.groupby(pd.Grouper(key='Date', freq='M')):
    # Convert the price difference to binary sequence
    diff = (group['Close'] - group['Open']) / group['Open']
    binary_sequence = np.where(diff > 0, 1, 0)
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    months_data[month]=pd.DataFrame(data= binary_sequence ).transpose() 
    months_btc.append(month)
    complexities_btc.append(complexity)


# Group the data by month
months_eur = []
complexities_eur = []
for month, group in df_eth.groupby(pd.Grouper(key='Date', freq='M')):
    # Convert the price difference to binary sequence
    diff = (group['Close'] - group['Open']) / group['Open']
    binary_sequence = np.where(diff > 0, 1, 0)
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    month_df = months_data[month]
    month_df = pd.concat( [month_df, pd.DataFrame( binary_sequence).transpose() ], axis=0)
    months_data[month] = month_df
    months_eur.append(month)
    complexities_eur.append(complexity)

# Group the data by month
months_usdt = []
complexities_ustd = []
for month, group in df_usdt.groupby(pd.Grouper(key='Date', freq='M')):
    # Convert the price difference to binary sequence
    diff = (group['Close'] - group['Open']) / group['Open']
    binary_sequence = np.where(diff > 0, 1, 0)
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    month_df = months_data[month]
    month_df = pd.concat( [month_df, pd.DataFrame( binary_sequence).transpose() ], axis=0 )
    months_data[month] = month_df
    months_usdt.append(month)
    complexities_ustd.append(complexity)


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


# Group the data by month
months_data = {}
for month, group in df_btc.groupby(pd.Grouper(key='Date', freq='M')):
    # Calculate the opening-closing price difference
    price_diff = (group['Close'] - group['Open']) / group['Open']
    # Calculate the mean and standard deviation of the opening-closing price difference for all months
    mean_diff = price_diff.mean()
    std_diff = price_diff.std()
    # Binarize the data based on whether the opening-closing price difference is greater than the mean plus two standard deviations or less than the mean minus two standard deviations
    binary_sequence = np.where(price_diff > mean_diff + 2*std_diff, 1, np.where(price_diff < mean_diff - 2*std_diff, 0, 0))
    binary_sequence = binary_sequence[~np.isnan(binary_sequence)]
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    months_data[month]=pd.DataFrame(data= binary_sequence ).transpose() 


# Group the data by month
months_eur = []
complexities_eur = []
for month, group in df_eth.groupby(pd.Grouper(key='Date', freq='M')):
    # Calculate the opening-closing price difference
    price_diff = (group['Close'] - group['Open']) / group['Open']
    # Calculate the mean and standard deviation of the opening-closing price difference for all months
    mean_diff = price_diff.mean()
    std_diff = price_diff.std()
    # Binarize the data based on whether the opening-closing price difference is greater than the mean plus two standard deviations or less than the mean minus two standard deviations
    binary_sequence = np.where(price_diff > mean_diff + 2*std_diff, 1, np.where(price_diff < mean_diff - 2*std_diff, 0, 0))
    binary_sequence = binary_sequence[~np.isnan(binary_sequence)]
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
    month_df = months_data[month]
    month_df = pd.concat( [month_df, pd.DataFrame( binary_sequence).transpose() ], axis=0)
    months_data[month] = month_df

# Group the data by month
months_usdt = []
complexities_ustd = []
for month, group in df_usdt.groupby(pd.Grouper(key='Date', freq='M')):
    # Convert the price difference to binary sequence
    # Calculate the opening-closing price difference
    price_diff = (group['Close'] - group['Open']) / group['Open']
    # Calculate the mean and standard deviation of the opening-closing price difference for all months
    
    mean_diff = price_diff.mean()
    std_diff = price_diff.std()
    # Binarize the data based on whether the opening-closing price difference is greater than the mean plus two standard deviations or less than the mean minus two standard deviations
    binary_sequence = np.where(price_diff > mean_diff + 2*std_diff, 1, np.where(price_diff < mean_diff - 2*std_diff, 0, 0))
    binary_sequence = binary_sequence[~np.isnan(binary_sequence)]
    # Calculate the Lempel-Ziv complexity of the binary sequence
    complexity = lempel_ziv_complexity(binary_sequence)
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