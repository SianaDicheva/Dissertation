import numpy as np
import matplotlib.pyplot as plt

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
    return -J * total_energy

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
        # magnetizations.append(calculate_magnetization(spin_matrix))

        temperatures.append(T)
    binary_spin_matrix = convert_to_binary(spin_matrix)
    filename = f"ising_model_data/spin_matrix_T{T}.txt"
    np.savetxt(filename, binary_spin_matrix, fmt='%d')
    print(filename)
    return energies, temperatures, spin_matrix

N = 5
M = 5


temp = np.arange(0.1, 5.1, 0.1)
T_list = np.round(temp,2)
# [0.5, 1.0, 2.0,2.2, 2.3, 2.4, 2.5, 2.6,3.5, 5.0]
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

    # plt.imshow(spin_matrix, cmap='gray', vmin=0, vmax=1)
    # plt.title("Final state of spin matrix for T = {}".format(T))
    # plt.savefig("plots/im_step{}.png".format(T))
    # #plt.show()

    # plt.clf()
    # plt.plot(energies)
    # plt.xlabel("Step")
    # plt.ylabel("Energy")
    # plt.title("Energy vs. Step for T = {}".format(T))
    # plt.savefig("plots/im_temp{}.png".format(T))
    # #plt.show()



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
# plt.show()

# plt.clf()
# plt.plot(T_list, magnetizations)
# plt.xlabel("Temperature")
# plt.ylabel("Magnetization")
# plt.title("Magnetization vs. Temperature")
# plt.savefig("plots/im_mag_abs.png")
# # plt.show()

plt.clf()
plt.plot(T_list, energies1)
plt.xlabel("Temperature")
plt.ylabel("Energy")
plt.title("Energy vs. Temperature")
plt.savefig("plots/im_energy.png")
plt.show()