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
    return (np.abs(np.sum(spin_matrix)))

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
T_list = [1.0, 2.2, 2.3, 2.4, 2.5, 3.5, 5.0]
J = 1
n_steps = 50
n_runs = 5 

# avg_magnetizations = []
# magnetizations_all = []

# # for T in T_list:
# #     avg_energy, avg_magnetization, magnetizations = ising_model(N, M, T, J, n_steps, n_runs)
# #     avg_magnetizations.append(avg_magnetization)
# #     magnetizations_all.append(magnetizations)

# # fig, ax = plt.subplots()

# # # # Plot individual magnetizations with rolling average
# # # smoothed_magnetizations = []
# # # for i in range(n_runs):
# # #     magnetizations_smoothed = np.convolve(magnetizations_all[i], np.ones(5)/5, mode='valid')
# # #     smoothed_magnetizations.append(magnetizations_smoothed)
# # #     ax.plot(T_list[2:-2], magnetizations_smoothed, alpha=0.5, linewidth=0.5, color='gray')
# # # avg_smoothed_magnetizations = np.mean(smoothed_magnetizations, axis=0)
# # # ax.plot(T_list, avg_magnetizations, label="Average Magnetization", linewidth=2.5, linestyle='--')
# # # ax.plot(T_list[2:-2], avg_smoothed_magnetizations, label="Smoothed Magnetization", linewidth=2.5, linestyle='--')
# # # ax.set_xlabel("Temperature")
# # # ax.set_ylabel("Magnetization")
# # # ax.set_title("Magnetization vs. Temperature")
# # # ax.legend()
# # # plt.savefig("ising_model_plots/im_mag_smoothed.png")
# # # plt.clf()

# # # Plot individual magnetizations with rolling average
# # for i in range(n_runs):
# #     magnetizations_for_run_i = [magnetizations_all[j][i] for j in range(len(T_list))]
# #     ax.scatter(T_list, magnetizations_for_run_i, label=f"Run {i+1}")
# # ax.plot(T_list, avg_magnetizations, label="Average Magnetization", linewidth=2.5, linestyle='--')
# # ax.set_xlabel("Temperature")
# # ax.set_ylabel("Magnetization")
# # ax.set_title("Magnetization vs. Temperature")
# # ax.legend()
# # plt.savefig("ising_model_plots/im_mag_smoothed.png")
# # plt.clf()

# for T in T_list:
#     avg_energy, avg_magnetization, magnetizations = ising_model(N, M, T, J, n_steps, n_runs)
#     avg_magnetizations.append(avg_magnetization)
#     magnetizations_all.append(magnetizations)

# fig, ax = plt.subplots()
# for i in range(n_runs):
#     ax.scatter(T_list, [magnetizations_all[j][i] for j in range(len(T_list))], label=f"Run {i+1}")
# ax.plot(T_list, avg_magnetizations, label="Average Magnetization", linewidth=2.5, linestyle='--')
# ax.set_xlabel("Temperature")
# ax.set_ylabel("Magnetization")
# ax.set_title("Magnetization vs. Temperature")
# ax.legend()
# plt.savefig("ising_model_plots/im_mag_4.png")
# plt.clf()

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
plt.savefig("ising_model_plots/im_mag_avg_3.png")
plt.clf()



# magnetizations = []

# magnetizations_all = []

# energies =[]
# avg_magnetizations =[]
# avg_energies =[]


# for T in T_list:
#     energies_T = []
#     magnetizations_T = []
#     for run in range(n_runs):
#         energies_run, magnetizations_run,spin_matrix = ising_model(N, M, T, J, n_steps)
#         energies_T.append(energies_run)
#         magnetizations_T.append(magnetizations_run)
#     energies.append(energies_T)
#     magnetizations.append(magnetizations_T)
#     avg_energies.append(np.mean(energies_T))
#     avg_magnetizations.append(np.mean(np.abs(magnetizations_T)))
#     magnetizations_all.append(magnetizations)

# fig, ax = plt.subplots()


# plt.figure(figsize=(8,6))
# for i in range(len(T_list)):
#     plt.plot(T_list[i]*np.ones(n_runs), avg_magnetizations[i]*np.ones(n_runs), 'o', color=colors[i])
#     ax.scatter( T_list[i], [magnetizations_all[j][i] for j in range(len(T_list))], label=f"Run {i+1}")
#     plt.plot(T_list[i], avg_magnetizations[i], 'o', color=colors[i], label='T = {}'.format(T_list[i]))
# plt.plot(T_list, avg_magnetizations, label="Average Magnetization", linewidth=2.5, linestyle='--')

# plt.xlabel("Temperature")
# plt.ylabel("Magnetization")
# plt.title("Average Magnetization vs. Temperature")
# plt.legend()
# plt.savefig("ising_model_plots/im_mag_avg_3.png")
# plt.clf()

# magnetizations = []
# energies =[]
# avg_magnetizations =[]
# avg_energies =[]
# for T in T_list:
#     energies_T, magnetizations_T,spin_matrix = ising_model(N, M, T, J, n_steps)
#     energies.append(energies_T)
#     magnetizations.append(magnetizations_T)
#     avg_energies.append(np.mean(energies_T))
#     avg_magnetizations.append(np.mean(np.abs(magnetizations_T)))

# plt.plot(T_list, avg_energies)
# plt.xlabel("Temperature")
# plt.ylabel("Energy")
# plt.title("Average Magnetization vs. Temperature")
# plt.savefig("ising_model_plots/im_mag_avg_1.png")
# plt.clf()

# plt.plot(T_list, avg_magnetizations)
# plt.xlabel("Temperature")
# plt.ylabel("Energy")
# plt.title("Energy vs. Temperature")
# plt.savefig("ising_model_plots/im_mag_1.png")
# plt.clf()



# for T in T_list:
#     energies = []
#     magnetizations = []
#     for i in range(10):
#         energy, magnetization = ising_model(N, M, T, J, n_steps)
#         energies.append(energy)
#         magnetizations.append(magnetization)
#     avg_energies.append(np.mean(energies))
#     avg_magnetizations.append(np.mean(magnetizations))

# plt.plot(T_list, avg_energies, 'o-', label='Energy')
# plt.plot(T_list, avg_magnetizations, 'o-', label='Magnetization')
# plt.legend()
# plt.xlabel('Temperature')
# plt.ylabel('Average Energy/Magnetization')
# plt.show()

