import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Read the data from CSV file
df = pd.read_csv('./financial_data/BTCUSD.csv')

# Calculate the Williams indicator
df['HH'] = df['High'].rolling(window=14).max()
df['LL'] = df['Low'].rolling(window=14).min()
df['Williams'] = ((df['HH'] - df['Close']) / (df['HH'] - df['LL'])) * -100

# Calculate the binary sequence for Williams indicator values
df['Binary'] = [1 if x < -20 else 0 for x in df['Williams']]

# Calculate the LZ complexity
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

df['LZ Complexity'] = df['Binary'].apply(lempel_ziv_complexity)

# Plot the Williams indicator and LZ complexity over time
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df['Date'], df['Williams'], label='Williams Indicator')
ax.plot(df['Date'], df['LZ Complexity'], label='LZ Complexity')
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Williams Indicator and LZ Complexity Over Time')
ax.legend()

plt.show()

# Use linear regression analysis to quantify the strength and direction of trends
slope_williams, intercept_williams, r_value_williams, p_value_williams, std_err_williams = linregress(df.index, df['Williams'])
slope_lz, intercept_lz, r_value_lz, p_value_lz, std_err_lz = linregress(df.index, df['LZ Complexity'])

print('Slope of Williams indicator:', slope_williams)
print('Slope of LZ complexity:', slope_lz)
print('Correlation coefficient of Williams indicator:', r_value_williams)
print('Correlation coefficient of LZ complexity:', r_value_lz)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('./financial_data/BTCUSD.csv')

# Calculate Williams indicator
period = 14
df['HH'] = df['High'].rolling(period).max()
df['LL'] = df['Low'].rolling(period).min()
df['W%R'] = -100 * (df['HH'] - df['Close']) / (df['HH'] - df['LL'])

# Calculate binary sequence using Williams indicator
df['Binary'] = np.where(df['W%R'] < -80, 1, 0)

# Calculate LZ complexity for binary sequence
binary_sequence = df['Binary'].values
complexity = lempel_ziv_complexity(binary_sequence)

# Calculate linear regression line for LZ complexity
x = np.arange(len(df))
y = df['LZ Complexity'].values.reshape(-1, 1)
reg = LinearRegression().fit(x, y)
trend = reg.predict(x)

# Plot Williams indicator and LZ complexity with linear regression line
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Williams %R', color=color)
ax1.plot(df['Date'], df['W%R'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('LZ Complexity', color=color)
ax2.plot(df['Date'], df['LZ Complexity'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax3 = ax2.twinx()

color = 'tab:green'
ax3.set_ylabel('LZ Complexity Trend', color=color)
ax3.plot(df['Date'], trend, color=color)
ax3.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.legend()
plt.show()
