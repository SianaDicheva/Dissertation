import matplotlib.pyplot as plt
import subprocess

import lstm_noisy

# Create an empty list to store the results
results_list = []

# # Run the original file 10 times
# for _ in range(10):
#     # Capture the standard output using the `redirect_stdout` context manager
#     import io
#     from contextlib import redirect_stdout

#     with io.StringIO() as buf, redirect_stdout(buf):
#         # Call the function from the original file
#         lstm_noisy.printcompl()
#         printed_result = buf.getvalue().strip()

#     # Add the printed result to the list
#     results_list.append(printed_result)

# # Print the results
# print(results_list)

# Run the file 10 times
for _ in range(10):
    # Execute the Python file using subprocess
    process = subprocess.Popen(['python', 'lstm_noisy.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the process to finish and capture the output
    stdout, stderr = process.communicate()

    # Decode the output as a string
    output = stdout.decode().strip()

    # Add the output to the list
    results_list.append(output)

# Print the results
print(results_list)

# Data from table
data = [
    [4, 5, 6, 9, 7, 13, 13, 14, 15, 14],
    [4, 15, 12, 5, 7, 11, 9, 13, 13, 16],
    [2, 9, 18, 19, 19, 11, 11, 12, 11, 13],
    [4, 9, 9, 12, 12, 14, 17, 11, 9, 10],
    [5, 6, 9, 6, 8, 8, 9, 9, 10, 14],
    [6, 13, 14, 14, 11, 16, 16, 16, 16, 16],
    [5, 8, 10, 13, 15, 13, 16, 15, 14, 14],
    [6, 2, 5, 8, 11, 8, 7, 10, 7, 8],
    [4, 9, 6, 7, 10, 10, 11, 11, 11, 11],
    [3, 8, 2, 7, 8, 10, 9, 9, 6, 6],
]

data1 = [
      [3, 4, 5, 5, 4, 5, 6, 6, 6, 6],
      [5, 4, 6, 4, 3, 5, 7, 7, 6, 4],
      [4, 4, 7, 4, 5, 4, 8, 7, 8, 7],
      [4, 4, 6, 6, 5, 6, 5, 6, 6, 5],
      [3, 3, 6, 5, 5, 6, 8, 2, 4, 4],
      [4, 5, 7, 5, 3, 3, 5, 7, 5, 5],
      [3, 4, 4, 3, 5, 5, 7, 8, 6, 6],
      [4, 5, 5, 4, 5, 6, 6, 5, 7, 7],
      [3, 6, 7, 5, 4, 4, 6, 5, 8, 7],
      [4, 3, 3, 3, 4, 5, 3, 6, 7, 5],
]

# Calculate average for each epoch
average = [sum(run)/len(run) for run in zip(*data)]

print(average)
average_noisy = [sum(run)/len(run) for run in zip(*data1)]
print(average_noisy)

# # Plot the data
# plt.plot(range(1,11), average, label='Average', linewidth=2)
# for i, run in enumerate(data):
#     plt.plot(range(1,11), run, linestyle='--', alpha=0.5, label=f'Run {i+1}')

# plt.legend(fontsize='small')
# plt.xlabel('Epoch')
# plt.ylabel('LZ complexity')
# plt.title('LZ complexity through learning of LSTM network')
# plt.show()

# Plot the data
plt.plot(range(1,11), average_noisy, label='Average', linewidth=2)
for i, run in enumerate(data1):
    plt.plot(range(1,11), run, linestyle='--', alpha=0.5, label=f'Run {i+1}')

plt.legend(fontsize='small')
plt.xlabel('Epoch')
plt.ylabel('LZ complexity')
plt.title('LZ complexity through learning of LSTM network')
plt.show()