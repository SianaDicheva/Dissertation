import math
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# Read in the CSV files for BTC-USD, ETH-USD, and USDT-USD
df_btc = pd.read_csv('././financial_data/BTCUSD3y.csv')
df_eth = pd.read_csv('././financial_data/ETHUSD3y.csv')
df_usdt = pd.read_csv('././financial_data/USDTUSD3y.csv')

# Merge the dataframes on the 'Date' column
df = pd.merge(df_btc[['Date', 'Close']], df_eth[['Date', 'Close']], on='Date', suffixes=('_BTC', '_ETH'))
df = pd.merge(df, df_usdt[['Date', 'Close']], on='Date')
df.columns = ['Date', 'Close_BTC', 'Close_ETH', 'Close_USDT']

# Calculate the log returns for each cryptocurrency
df['Log_Return_BTC'] = df['Close_BTC'].apply(lambda x: math.log(x)) - df['Close_BTC'].apply(lambda x: math.log(x)).shift(1)
df['Log_Return_ETH'] = df['Close_ETH'].apply(lambda x: math.log(x)) - df['Close_ETH'].apply(lambda x: math.log(x)).shift(1)
df['Log_Return_USDT'] = df['Close_USDT'].apply(lambda x: math.log(x)) - df['Close_USDT'].apply(lambda x: math.log(x)).shift(1)

# Drop any rows with missing data
df.dropna(inplace=True)

# Calculate the correlation matrix
corr_matrix = df[['Log_Return_BTC', 'Log_Return_ETH', 'Log_Return_USDT']].corr()

# Plot the correlation matrix as a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix: BTC, ETH, and USDT')
plt.show()


# Create a new binary feature indicating whether opening price is smaller than closing price
# df_btc['Open_Less_Close'] = df_btc['Open'] < df_btc['Close']
# df_eth['Open_Less_Close'] = df_eth['Open'] < df_eth['Close']
# df_usdt['Open_Less_Close'] = df_usdt['Open'] < df_usdt['Close']

# Concatenate the three dataframes together
df = pd.concat([df_btc, df_eth, df_usdt])

# Drop any rows with missing data
df.dropna(inplace=True)

# Drop the 'Date' column (not needed for clustering or plotting)
df.drop('Date', axis=1, inplace=True)

# Perform k-means clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0).fit(df)

# Plot the clusters using the first two features (Open and High)
plt.scatter(df['Open'], df['Close'], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Open')
plt.ylabel('High')
plt.title('K-Means Clustering of Financial Data')
plt.show()


plt.scatter(df['Open'], df['High'], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='x', color='red')
plt.xlabel('Open')
plt.ylabel('High')
plt.title('K-Means Clustering of Financial Data with Centroids')
plt.show()


# Perform k-means clustering for each currency separately with 2 clusters
kmeans_btc = KMeans(n_clusters=2, random_state=0).fit(df_btc[['Open', 'High', 'Low', 'Close']])
kmeans_eth = KMeans(n_clusters=2, random_state=0).fit(df_eth[['Open', 'High', 'Low', 'Close']])
kmeans_usdt = KMeans(n_clusters=2, random_state=0).fit(df_usdt[['Open', 'High', 'Low', 'Close']])

# Plot the clusters for each currency separately
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(df_btc['Open'], df_btc['High'], c=kmeans_btc.labels_, cmap='viridis')
plt.xlabel('Open')
plt.ylabel('High')
plt.title('BTC-USD Clustering')

plt.subplot(1, 3, 2)
plt.scatter(df_eth['Open'], df_eth['High'], c=kmeans_eth.labels_, cmap='viridis')
plt.xlabel('Open')
plt.ylabel('High')
plt.title('ETH-USD Clustering')

plt.subplot(1, 3, 3)
plt.scatter(df_usdt['Open'], df_usdt['High'], c=kmeans_usdt.labels_, cmap='viridis')
plt.xlabel('Open')
plt.ylabel('High')
plt.title('USDT-USD Clustering')

plt.tight_layout()
plt.show()

# Merge the dataframes into one
df = pd.concat([df_btc, df_eth, df_usdt], keys=['BTC', 'ETH', 'USDT'])

# Perform k-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(df[['Open', 'High', 'Low', 'Close']])

# Extract the cluster labels for each currency
labels_btc = kmeans.labels_[0:len(df_btc)]
labels_eth = kmeans.labels_[len(df_btc):len(df_btc)+len(df_eth)]
labels_usdt = kmeans.labels_[-len(df_usdt):]

# Plot the clusters for each currency separately
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(df_btc['Open'], df_btc['Close'], c=labels_btc, cmap='viridis')
plt.xlabel('Open')
plt.ylabel('High')
plt.title('BTC-USD Clustering')

plt.subplot(1, 3, 2)
plt.scatter(df_eth['Open'], df_eth['Close'], c=labels_eth, cmap='viridis')
plt.xlabel('Open')
plt.ylabel('High')
plt.title('ETH-USD Clustering')

plt.subplot(1, 3, 3)
plt.scatter(df_usdt['Open'], df_usdt['Close'], c=labels_usdt, cmap='viridis')
plt.xlabel('Open')
plt.ylabel('High')
plt.title('USDT-USD Clustering')

plt.tight_layout()
plt.show()



# Merge the dataframes into one
df = pd.concat([df_btc, df_eth, df_usdt], keys=['BTC', 'ETH', 'USDT'])

# Perform k-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(df[['Open', 'High', 'Low', 'Close']])

# Extract the cluster labels for each currency
labels_btc = kmeans.labels_[0:len(df_btc)]
labels_eth = kmeans.labels_[len(df_btc):len(df_btc)+len(df_eth)]
labels_usdt = kmeans.labels_[-len(df_usdt):]

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Open'], df['High'], df['Low'], c=[labels_btc, labels_eth, labels_usdt], cmap='viridis')
ax.set_xlabel('Open')
ax.set_ylabel('High')
ax.set_zlabel('Low')
ax.set_title('3D Clustering of BTC-USD, ETH-USD, and USDT-USD')
plt.show()
 

# NEW 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

# Read BTC-USD CSV file into a pandas DataFrame
df_btc = pd.read_csv('././financial_data/BTCUSD3y.csv')

# Keep only rows where Open < Close
df_btc = df_btc[df_btc['Open'] < df_btc['Close']]

# Cluster BTC data
X_btc = df_btc[['Open', 'High', 'Low']]
kmeans_btc = KMeans(n_clusters=3)
kmeans_btc.fit(X_btc)
labels_btc = kmeans_btc.labels_

# Read ETH-USD CSV file into a pandas DataFrame
df_eth = pd.read_csv('././financial_data/ETHUSD3y.csv')

# Keep only rows where Open < Close
df_eth = df_eth[df_eth['Open'] < df_eth['Close']]

# Cluster ETH data
X_eth = df_eth[['Open', 'High', 'Low']]
kmeans_eth = KMeans(n_clusters=3)
kmeans_eth.fit(X_eth)
labels_eth = kmeans_eth.labels_

# Read USDT-USD CSV file into a pandas DataFrame
df_usdt = pd.read_csv('././financial_data/USDTUSD3y.csv')

# Keep only rows where Open < Close
df_usdt = df_usdt[df_usdt['Open'] < df_usdt['Close']]

# Cluster USDT data
X_usdt = df_usdt[['Open', 'High', 'Low']]
kmeans_usdt = KMeans(n_clusters=3)
kmeans_usdt.fit(X_usdt)
labels_usdt = kmeans_usdt.labels_

# Plot 3D scatter plot for BTC data
fig_btc = plt.figure()
ax_btc = fig_btc.add_subplot(111, projection='3d')
ax_btc.scatter(X_btc['Open'], X_btc['High'], X_btc['Low'], c=labels_btc, cmap='viridis')
ax_btc.set_xlabel('Open')
ax_btc.set_ylabel('High')
ax_btc.set_zlabel('Low')
ax_btc.set_title('BTC Clustering')

# Plot 3D scatter plot for ETH data
fig_eth = plt.figure()
ax_eth = fig_eth.add_subplot(111, projection='3d')
ax_eth.scatter(X_eth['Open'], X_eth['High'], X_eth['Low'], c=labels_eth, cmap='viridis')
ax_eth.set_xlabel('Open')
ax_eth.set_ylabel('High')
ax_eth.set_zlabel('Low')
ax_eth.set_title('ETH Clustering')

# Plot 3D scatter plot for USDT data
fig_usdt = plt.figure()
ax_usdt = fig_usdt.add_subplot(111, projection='3d')
ax_usdt.scatter(X_usdt['Open'], X_usdt['High'], X_usdt['Low'], c=labels_usdt, cmap='viridis')
ax_usdt.set_xlabel('Open')
ax_usdt.set_ylabel('High')
ax_usdt.set_zlabel('Low')
ax_usdt.set_title('USDT Clustering')

plt.show()


# Create a bubble plot
fig, ax = plt.subplots(figsize=(10, 7))

dates = pd.date_range(start='2022-01-11', end='2022-12-31', freq='21D') + pd.Timedelta(days=10)


# Add a bubble for each currency
ax.scatter(df_btc['Date'], df_btc['Close'], s=df_btc['Open']*0.1, alpha=0.5, color='blue',label='BTC')
ax.scatter(df_eth['Date'], df_eth['Close'], s=df_eth['Open']*0.1, alpha=0.5, color='orange',label='ETH')
ax.scatter(df_usdt['Date'], df_usdt['Close'], s=df_usdt['Open']*0.1, alpha=0.5, color='green', label='USTD')

# set the x-axis ticks to display every month
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Set axis labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price')
ax.set_title('Bubble Plot of BTC, ETH, and USDT')

ax.legend()

# Show the plot
plt.show()

df_btc['Date'] = pd.to_datetime(df_btc['Date'])
df_eth['Date'] = pd.to_datetime(df_eth['Date'])
df_usdt['Date'] = pd.to_datetime(df_usdt['Date'])

# Create a bubble plot
fig, ax = plt.subplots(figsize=(10, 7))

# Add a bubble for each currency
ax.scatter(df_btc['Date'], df_btc['Close'], s=100, alpha=0.5, color='blue',label='BTC')
ax.scatter(df_eth['Date'], df_eth['Close'], s=100, alpha=0.5, color='orange',label='ETH')
ax.scatter(df_usdt['Date'], df_usdt['Close'], s=100, alpha=0.5, color='green', label='USTD')

# set the x-axis ticks to display every 60 days
ax.xaxis.set_major_locator(mdates.DayLocator(interval=90))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Set axis labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price')
ax.set_title('Bubble Plot of BTC, ETH, and USDT')

ax.legend()
# Show the plot
plt.show()

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Add a scatter plot for each currency
ax.scatter(df_btc['Open'], df_btc['Close'], df_btc['Volume'], alpha=0.5, c='blue', label='BTC')
ax.scatter(df_eth['Open'], df_eth['Close'], df_eth['Volume'], alpha=0.5, c='orange', label='ETH')
ax.scatter(df_usdt['Open'], df_usdt['Close'], df_usdt['Volume'], alpha=0.5, c='green', label='USDT')


# Set axis labels and title
ax.set_xlabel('Opening Price')
ax.set_ylabel('Closing Price')
ax.set_zlabel('Volume')
ax.set_title('3D Scatter Plot of BTC, ETH, and USDT')

# Add a legend
ax.legend()
plt.show()
