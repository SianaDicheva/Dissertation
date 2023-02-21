import os
import numpy as np
import matplotlib.pyplot as plt

def energy(spins, J, H):
    """Calculate the energy of the current spin configuration"""
    N = spins.shape[0]
    return -J * np.sum(spins[:-1, :] * spins[1:, :]) - J * np.sum(spins[:, :-1] * spins[:, 1:]) - H * np.sum(spins)

def metropolis_update(spins, T, J, H):
    """Apply Metropolis update to all spins in the system"""
    N = spins.shape[0]
    new_spins = spins.copy()
    for i in range(N):
        for j in range(N):
            spin = spins[i, j]
            delta_E = 2 * J * spin * (spins[(i-1)%N, j] + spins[(i+1)%N, j] + spins[i, (j-1)%N] + spins[i, (j+1)%N]) + 2 * H * spin
            if delta_E <= 0 or np.random.random() < np.exp(-delta_E/T):
                new_spins[i, j] = 1 - spin
    return new_spins

def save_plot(T,step,spins):
    plt.imshow(spins, cmap='gray', vmin=0, vmax=1)
    plt.title("T = {} Step = {}".format(T, step))
    plt.savefig("plots/temp{}_step{}.png".format(int(T),step))

def clear_plots():
    my_path = 'plots/'
    for file_name in os.listdir(my_path):
        if file_name.endswith('.png'):
            os.remove(my_path + file_name)

def simulate(N, T, J, H, steps, save_plot_step: int):
    """Simulate Ising Model for given number of steps"""
    spins = np.random.choice([0, 1], size=(N, N))

    for i in range(steps):
        # Update all spins in the grid simultaneously.
        spins = metropolis_update(spins, J, H, T)
        #if i %save_plot_step ==0 :
            #save_plot(T=T,step=i,spins=spins)
    return spins


# Set parameters
N = 20
T = 10.0
J = 1.0
H = 0.0
steps = 100

#clear plots 
#clear_plots()

# Simulate Ising Model
spin_configs = simulate(N, T, J, H, steps,10)

def magnetization(spins):
    return np.mean(spins)

# Initialize the spins
L = 16
spins = np.random.choice([0, 1], size=(L, L))

# Define the range of temperatures
T_min = 1.0
T_max = 5.0
n_temps = 5
temperatures = np.linspace(T_min, T_max, n_temps)

# Run the simulations and compute the magnetization at each temperature
magnetizations = []
for T in temperatures:
    spins = simulate(N, T, J, H, steps,10)
    mag = magnetization(spins)
    magnetizations.append(mag)

# Plot the temperature vs magnetization
plt.plot(temperatures, magnetizations)
plt.xlabel("Temperature")
plt.ylabel("Magnetization")
plt.show()


def simulate(T, spins, n_steps):
    L = spins.shape[0]
    for step in range(n_steps):
        for i in range(L):
            for j in range(L):
                s = spins[i, j]
                nb = spins[(i+1)%L, j] + spins[i, (j+1)%L] + spins[(i-1)%L, j] + spins[i, (j-1)%L]
                cost = 2 * s * nb
                if cost < 0:
                    s = 1 - s
                elif np.random.uniform() < np.exp(-cost / T):
                    s = 1 - s
                spins[i, j] = s
        plt.imshow(spins, cmap='gray', vmin=0, vmax=1)
        plt.title("T = {} Step = {}".format(T, step))
        plt.savefig("plots/temp{}_step{}.png".format(T,step))
        #plt.show()
    return spins

# Initialize the spins
L = 50
spins = np.random.choice([0, 1], size=(L, L))

# Define the temperature and number of time steps
T = 4
n_steps = 50

# Run the simulation and plot the spin configurations at each time step
spins = simulate(T, spins, n_steps)

