import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the Bitcoin data
data = pd.read_csv('./financial_data/BTCUSD.csv')

# Calculate the Williams indicator
high = data['High']
low = data['Low']
close = data['Close']
williams = ((high - close.rolling(window=14).max()) / (high - low.rolling(window=14).min())) * -100

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

# Calculate the LZ complexity
binary_sequence = [1 if w >= 0 else 0 for w in williams]
lz_complexity = lempel_ziv_complexity(binary_sequence)

# Combine the Williams and LZ complexity values into a feature matrix
X = np.column_stack((williams, lz_complexity))

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
labels = kmeans.labels_

# Plot the results
plt.scatter(williams, lz_complexity, c=labels, cmap='viridis')
plt.xlabel('Williams Indicator')
plt.ylabel('LZ Complexity')
plt.title('K-means Clustering Results')
plt.show()
