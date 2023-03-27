import numpy as np
import matplotlib.pyplot as plt

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
    return (np.abs(np.sum(spin_matrix))/(N*M))

def ising_model(N, M, T, J, n_steps):
    spin_matrix = initialize_spin_matrix(N, M)
    energies = []
    magnetizations = []
    
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
    avg_energy = np.mean(energies)
    avg_magnetization = np.mean((magnetizations))
    return avg_energy, avg_magnetization, magnetizations
    # return energies, magnetizations,spin_matrix


N = 100
M = 100
temp = np.arange(0.1, 5.1, 0.3)
T_list = np.round(temp,3)
# T_list = [1.0, 2.2, 2.3, 2.4, 2.5, 3.5, 5.0]
J = 1
n_steps = 50
n_runs = 5 


colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']


magnetizations = np.zeros ( shape=( len(T_list) ,n_runs),dtype=float)

magnetizations_all = []

energies =[]
avg_magnetizations =[]
avg_energies =[]

x=0
for T in T_list:
    energies_T = []
    magnetizations_T = []
    for run in range(n_runs):
        energies_run, magnetizations_run,spin_matrix = ising_model(N, M, T, J, n_steps)
        energies_T.append(energies_run)
        magnetizations_T.append(magnetizations_run)
    energies.append(energies_T)
    #new_row = np.array( magnetizations_T)
    magnetizations[x]=magnetizations_T
    #np.append( arr= magnetizations, values= [new_row], axis=0 ) 
    avg_energies.append(np.mean(energies_T))
    avg_magnetizations.append(np.mean(np.abs(magnetizations_T)))
    magnetizations_all.append(magnetizations)
    x=x+1

#fig, ax = plt.subplots()
plt.clf()
#mag_arr = np.array(magnetizations)
plt.figure(figsize=(12,10))
for i in  range(len(magnetizations)): # range(len(T_list)):
    #ll = int(mag_arr.shape(1))
    for j in range(n_runs):
        y = magnetizations[:,j]
        plt.scatter(  x= T_list, y= y,  color=colors[j] )# label=f"Run {i+1}")
    # plt.plot(T_list[i]*np.ones(n_runs), avg_magnetizations[i]*np.ones(n_runs), 'o', color=colors[i])
    # plt.plot(T_list[i], avg_magnetizations[i], 'o', color=colors[i], label='T = {}'.format(T_list[i]))
plt.plot(  T_list,avg_magnetizations, label="Average Magnetization", linewidth=2.5, linestyle='--')

plt.xlabel("Temperature")
plt.ylabel("Magnetization")
plt.title("Average Magnetization vs. Temperature")
plt.legend()
plt.savefig("ising_model_plots/im_mag_avg_5.png")
plt.clf()



