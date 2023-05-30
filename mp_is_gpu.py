import cupy as cp
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


# function to calculate Lempel-Ziv complexity for a given file
def calculate_LZ_complexity(file_name):
    with open("D:\Education\Dissertation\code\data55" + "/"+file_name, "r") as text_file:
        # read whole file to a string
        data = text_file.read()
        
        # convert string to binary sequence
        binary_sequence = cp.asarray([int(d) for d in data if d.isdigit()])

        
        # calculate Lempel-Ziv complexity
        n = lempel_ziv_complexity(binary_sequence)
        
        return n

# define the list of file names to process
file_names = []
for x in os.listdir("D:\Education\Dissertation\code\data55"):
    if x.endswith(".txt"):
        # Prints only text file present in My Folder
        file_names.append(x)
        print(x)

# calculate Lempel-Ziv complexity for each file using CuPy in parallel
with cp.cuda.Device(0):
    n_list = cp.array(list(map(calculate_LZ_complexity, file_names)))
    
# plot Lempel-Ziv complexity against temperature
temp = np.arange(0.1, 5.1, 0.1)
T_list = np.round(temp,3)
plt.axvline(x =  2.26918, color = 'b', label = 'Critical temperature')
plt.plot(T_list, n_list.get(), 'o-')
plt.xlabel('Temperature')
plt.ylabel('Lempel-Ziv Complexity')
plt.legend()
plt.show()