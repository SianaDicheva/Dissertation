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

s = '1001111011000010'
n = lempel_ziv_complexity(s)

file_names =[]

for x in os.listdir("D:\Education\Dissertation\code\ising_model_data"):
    if x.endswith(".txt"):
        # Prints only text file present in My Folder
        file_names.append(x)
        # print(x)

# define a list of file names
# file_names = ["D:\Education\Dissertation\code\ising_model_data\spin_matrix_T0.5.txt",
#               "D:\Education\Dissertation\code\ising_model_data\spin_matrix_T1.0.txt",
#               "D:\Education\Dissertation\code\ising_model_data\spin_matrix_T2.0.txt",
#               "D:\Education\Dissertation\code\ising_model_data\spin_matrix_T2.2.txt",
#               "D:\Education\Dissertation\code\ising_model_data\spin_matrix_T2.3.txt",
#               "D:\Education\Dissertation\code\ising_model_data\spin_matrix_T2.4.txt",
#               "D:\Education\Dissertation\code\ising_model_data\spin_matrix_T2.5.txt",
#               "D:\Education\Dissertation\code\ising_model_data\spin_matrix_T2.6.txt",
#               "D:\Education\Dissertation\code\ising_model_data\spin_matrix_T3.5.txt",
#               "D:\Education\Dissertation\code\ising_model_data\spin_matrix_T5.0.txt"]

# file_names = ["D:\Education\Dissertation\code\ising_model_data\*.txt"]

# define a list of temperatures
# T_list = [0.5, 1.0, 2.0, 2.2, 2.3, 2.4, 2.5, 2.6, 3.5, 5.0]

temp = np.arange(0.1, 5.1, 0.1)
T_list = np.round(temp,3)

# initialize a list to store Lempel-Ziv complexity values
n_list = []

# iterate over file names
for file_name in file_names:
    # open file in read mode
    print(file_name)
    with open("D:\Education\Dissertation\code\ising_model_data" + "/"+file_name, "r") as text_file:
        # read whole file to a string
        data = text_file.read()
        
        # calculate Lempel-Ziv complexity
        n = lempel_ziv_complexity(data)
        
        # append Lempel-Ziv complexity value to list
        n_list.append(n)

# plot Lempel-Ziv complexity against temperature
plt.axvline(x =  2.26918, color = 'b', label = 'Critical temperature')
plt.plot(T_list, n_list, 'o-')
plt.xlabel('Temperature')
plt.ylabel('Lempel-Ziv Complexity')
plt.legend()
plt.show()


# https://perso.crans.org/besson/publis/Lempel-Ziv_Complexity.git/Short_study_of_the_Lempel-Ziv_complexity.html
# implementation thats working with no mistakes


# def convert_text_to_binary(text):
#     binary_sequence = ''
#     for char in text:
#         binary_sequence += format(ord(char), '08b')
#     return binary_sequence

# text_response = "This is a sample response."
# binary_sequence = convert_text_to_binary(text_response)
# complexity = lempel_ziv_complexity(binary_sequence)
# print("Lempel-Ziv complexity:", complexity)

df = pd.read_csv('./financial_data/BTC-USD.csv')

text_file = open('./financial_data/binary_output.txt', "r")
    # read whole file to a string
data = text_file.read()
        
    # calculate Lempel-Ziv complexity
n = lempel_ziv_complexity(data)
print (n)