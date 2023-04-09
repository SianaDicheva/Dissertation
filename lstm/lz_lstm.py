import numpy as np
import zlib
import matplotlib.pyplot as plt

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


# Load numpy files
n_epochs = 10
outputs = [np.load(f"output_{i:02d}.npy") for i in range(1, n_epochs+1)]

# Apply threshold of 0.5 to create binary list
binary_list = []
for output in outputs:
    binary_list.append([1 if o > 0.5 else 0 for o in output])

# Calculate Lempel-Ziv complexity of binary list
lz_complexity = []
for i in range(n_epochs):
    compressed = lempel_ziv_complexity(binary_list[i])
    lz_complexity.append(len(compressed))

# Plot complexity as a function of epoch
plt.plot(range(1, n_epochs+1), lz_complexity)
plt.xlabel('Epoch')
plt.ylabel('Lempel-Ziv complexity')
plt.show()
