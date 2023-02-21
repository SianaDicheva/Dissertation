import numpy as np
import matplotlib.pyplot as plt

"""
In a synchronous update, all spins are updated simultaneously in one step, so the energy function and 
the Metropolis update are applied to all spins in the same way at each step. 
The result is a new configuration of spins that represents the next state of the system. 
This process is repeated multiple times to simulate the evolution of the system over time.
"""


def energy(spins, J, H):
    """
    Calculate the energy of the system given the current configuration of spins.

    Args:
    - spins: numpy array representing the current configuration of spins.
    - J: interaction constant between neighboring spins.
    - H: external magnetic field.

    Returns:
    - E: total energy of the system.
    """
    N = spins.shape[0]
    # Calculate the energy due to interaction between neighboring spins.
    E = -J * np.sum(spins[:, :-1] * spins[:, 1:]) - J * np.sum(spins[:-1, :] * spins[1:, :])
    # Calculate the energy due to the external magnetic field.
    E -= H * np.sum(spins)
    return E

def metropolis_update(spins, J, H, T):
    """
    Update all spins in the grid simultaneously using the Metropolis algorithm.

    Args:
    - spins: numpy array representing the current configuration of spins.
    - J: interaction constant between neighboring spins.
    - H: external magnetic field.
    - T: temperature.

    Returns:
    - spins: numpy array representing the updated configuration of spins.
    """
    N = spins.shape[0]
    for i in range(N):
        for j in range(N):
            # Calculate the change in energy due to flipping this spin.
            delta_E = 2 * J * spins[i, j] * (spins[(i + 1) % N, j] + spins[i, (j + 1) % N] + spins[(i - 1 + N) % N, j] + spins[i, (j - 1 + N) % N]) + 2 * H * spins[i, j]
            # Flip the spin with probability determined by the Metropolis algorithm.
            if delta_E <= 0:
                spins[i, j] = -spins[i, j]
            elif np.random.rand() < np.exp(-delta_E / T):
                spins[i, j] = -spins[i, j]
    return spins

def simulate(N, J, H, T, steps):
    """
    Simulate the Ising Model for a given number of steps with synchronous updates

    - N: number of spins in each direction (grid will have NxN spins).
    - J: interaction constant between neighboring spins.
    - H: external magnetic field.
    - T: temperature.
    - steps: number of steps to simulate.
    - spins: numpy array representing the final configuration of spins.
    """
    # Initialize the spins randomly.
    spins = np.random.choice([0, 1], size=(N, N))
    for i in range(steps):
        # Update all spins in the grid simultaneously.
        spins = metropolis_update(spins, J, H, T)
    return spins

# Set the parameters for the simulation.
N = 50  # Number of spins in each direction (grid will have NxN spins).
J = 1  # Interaction constant between neighboring spins.
H = 0  # External magnetic field.
T = 2  # Temperature.
steps = 100  # Number of steps to simulate.

# Run the simulation.
spins = simulate(N, J, H, T, steps)

# Plot the final configuration of spins.
plt.imshow(spins, cmap='gray', vmin=0, vmax=1)
plt.show()

# Set the parameters for the simulation.
N = 50  # Number of spins in each direction (grid will have NxN spins).
J = 1  # Interaction constant between neighboring spins.
H = 0  # External magnetic field.
T_min = 1  # Minimum temperature.
T_max = 4  # Maximum temperature.
steps = 100  # Number of steps to simulate.

# Array to store the temperatures.
temperatures = np.linspace(T_min, T_max, num=100)

# Array to store the average magnetizations.
magnetizations = np.zeros_like(temperatures)

# Loop over temperatures.
for i, T in enumerate(temperatures):
    # Run the simulation.
    spins = simulate(N, J, H, T, steps)
    # Calculate the average magnetization.
    magnetizations[i] = np.mean(spins)

# Plot the average magnetization as a function of temperature.
# plt.plot(temperatures, magnetizations)
# plt.xlabel('Temperature')
# plt.ylabel('Average Magnetization')
# plt.show()