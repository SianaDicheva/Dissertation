import csv
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional
from keras.callbacks import ModelCheckpoint


import nltk


"""
In a synchronous update, all spins are updated simultaneously in one step, so the energy function and 
the Metropolis update are applied to all spins in the same way at each step. 
The result is a new configuration of spins that represents the next state of the system. 
This process is repeated multiple times to simulate the evolution of the system over time.
"""

def initialize_spin_matrix(N, M):
    return np.random.choice([-1, 1], size=(N, M))

def calculate_energy(spin_matrix, N, M, J):
    total_energy = 0
    for i in range(N):
        for j in range(M):
            total_energy += spin_matrix[i, j] * (
                spin_matrix[(i + 1) % N, j] +
                spin_matrix[i, (j + 1) % M]
            )
    return -J * total_energy/2

def calculate_magnetization(spin_matrix):
    return np.abs(np.sum(spin_matrix)/(N*M))

def convert_to_binary(spin_matrix):
    binary_spin_matrix = np.where(spin_matrix == -1, 0, 1)
    return binary_spin_matrix

def ising_model(N, M, T, J, n_steps):
    spin_matrix = initialize_spin_matrix(N, M)
    energies = []
    temperatures = []
    # magnetizations =[]
    for step in range(n_steps):
        old_spin_matrix = spin_matrix
        for i in range(N):
            for j in range(M):
                delta_E = 2 * J * old_spin_matrix[i, j] * (
                    old_spin_matrix[(i + 1) % N, j] +
                    old_spin_matrix[i, (j + 1) % M] +
                    old_spin_matrix[(i - 1 + N) % N, j] +
                    old_spin_matrix[i, (j - 1 + M) % M]
                )
                beta = 1 / T
                if np.random.uniform() < np.exp(-beta * delta_E):
                    spin_matrix[i, j] *= -1
        energies.append(calculate_energy(spin_matrix, N, M, J))
        magnetizations.append(calculate_magnetization(spin_matrix))

        temperatures.append(T)
    binary_spin_matrix = convert_to_binary(spin_matrix)
    filename = f"./ising_model_data/spin_matrix_T{T}.txt"
    np.savetxt(filename, binary_spin_matrix, fmt='%d')
    return energies, temperatures, spin_matrix


# def ising_model(N, M, T, J, n_steps):
#     spin_matrix = initialize_spin_matrix(N, M)
#     energies = []
#     spins = []
#     for step in range(n_steps):
#         old_spin_matrix = spin_matrix
#         for i in range(N):
#             for j in range(M):
#                 delta_E = 2 * J * old_spin_matrix[i, j] * (
#                     old_spin_matrix[(i + 1) % N, j] +
#                     old_spin_matrix[i, (j + 1) % M] +
#                     old_spin_matrix[(i - 1 + N) % N, j] +
#                     old_spin_matrix[i, (j - 1 + M) % M]
#                 )
#                 beta = 1 / T
#                 if np.random.uniform() < np.exp(-beta * delta_E):
#                     spin_matrix[i, j] *= -1
#         energies.append(calculate_energy(spin_matrix, N, M, J))
#         spins.append(np.sum(spin_matrix)/(N*M))
        
#     binary_spin_matrix = convert_to_binary(spin_matrix)
#     filename = f"./ising_model_data/spin_matrix_T{T}.txt"
#     np.savetxt(filename, binary_spin_matrix, fmt='%d')
    
#     return spin_matrix, energies, spins

N = 100
M = 100

temp = np.arange(0.1, 5.1, 0.1)
T_list = np.round(temp,2)

# T_list = [0.5, 1.0, 2.0,2.2,2.27, 2.3, 2.4, 2.5, 2.6,3.5, 5.0]
J = 1
n_steps = 100

magnetizations = []
energies1 =[]
for T in T_list:
    energies, temperatures, spin_matrix = ising_model(N, M, T, J, n_steps)
    energie1 = calculate_energy(spin_matrix, N, M, J)
    energies1.append(energie1)
    magnetization = calculate_magnetization(spin_matrix)
    magnetizations.append(magnetization)


    plt.clf()
    plt.imshow(spin_matrix, cmap='gray', vmin=0, vmax=1)
    # plt.title("Final state of spin matrix for T = {}".format(T))
    plt.savefig("./ising_model_plots/im_step{}.png".format(T))
    # plt.show()

    # plt.clf()
    # plt.plot(energies)
    # plt.xlabel("Step")
    # plt.ylabel("Energy")
    # plt.title("Energy vs. Step for T = {}".format(T))
    # plt.savefig("./ising_model_plots/im_temp{}.png".format(T))
    # #plt.show()

# temperatures = [0.5, 2.27, 5.0]

# for T in temperatures:
#     spin_matrix, energies, spins = ising_model(N=50, M=50, T=T, J=1, n_steps=1000)
#     plt.plot(range(len(spins)), spins, label = 'T = {0}'.format(T))
    
# plt.legend(loc = 'best')
# plt.xlabel('nSteps')
# plt.ylabel('Average Spin')
# plt.ylim(-1.2, 1.2)
# plt.savefig('./ising_model_plots/average-spin.png')
# plt.show()



# for T in T_list:
#     energies, temperatures, spin_matrix = ising_model(N, M, T, J, n_steps)
#     magnetization = calculate_magnetization(spin_matrix)
#     magnetizations.append(magnetization)

# plt.clf()
# plt.plot(T_list, magnetizations)
# plt.xlabel("Temperature")
# plt.ylabel("Magnetization")
# plt.title("Magnetization vs. Temperature")
# plt.savefig("plots/ex_mag.png")
# # plt.show()

# plt.clf()
# plt.plot(T_list, magnetizations)
# plt.xlabel("Temperature")
# plt.ylabel("Magnetisation")
# plt.title("Magnetisation vs. Temperature")
# plt.savefig("./ising_model_plots/im_mag_abs.png")
# # plt.show()

# plt.clf()
# plt.plot(T_list, energies1)
# plt.xlabel("Temperature (T)")
# plt.ylabel("Energy")
# plt.title("Energy vs. Temperature")
# plt.savefig("./ising_model_plots/im_energy_2.png")
# # plt.show()

# Ns = [10, 20, 50, 100, 250]  # System Size
# T_Tcs = np.linspace(0.5, 3.7, 30)  # T/Tc
# Tc = 2.268  # Onsager's Tc

# def ising_model_magnetization(n, t):
#     spins = np.random.choice([-1, 1], size=(n, n))
#     beta = 1 / (t * Tc)
#     for i in range(5000):
#         row, col = np.random.randint(n, size=2)
#         energy_change = 2 * spins[row, col] * (spins[(row-1)%n, col] + spins[(row+1)%n, col] + spins[row, (col-1)%n] + spins[row, (col+1)%n])
#         if energy_change < 0 or np.random.rand() < np.exp(-beta * energy_change):
#             spins[row, col] *= -1
#     return np.abs(np.sum(spins)) / n**2

# for n in Ns:
#     avgspins = []
#     for i, T_Tc in enumerate(T_Tcs):
#         T = T_Tc * Tc
#         avgspin = ising_model_magnetization(n, T)
#         avgspins.append(avgspin)
#     plt.plot(T_Tcs, avgspins, 'o-', label='L = {0}'.format(n))

# plt.xlabel('T/T$_{c}$', fontsize=16)
# plt.ylabel('<M$_{L}$>', fontsize=16)
# plt.legend()
# plt.savefig('ising_model_plots/magnetization19.png')
# plt.show()