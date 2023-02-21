def compress(data):
    # Initialize the dictionary to contain all single-character substrings
    dictionary = {chr(i): i for i in range(256)}
    next_index = 256
    output = []

    # Initialize the input index
    index = 0
    while index < len(data):
        # Find the longest match between the current input index and the substrings in the dictionary
        length = 1
        while index + length <= len(data) and data[index:index+length] in dictionary:
            length += 1
        length -= 1

        # Output the index of the matching substring in the dictionary
        output.append(dictionary[data[index:index+length]])

        # Add the next character from the input string to the output string and to the dictionary
        if index + length < len(data):
            dictionary[data[index:index+length+1]] = next_index
            next_index += 1

        # Increment the input index
        index += length

    # Return the compressed output
    return output

def decompress(compressed):
    # Initialize the dictionary to contain all single-character substrings
    dictionary = {i: chr(i) for i in range(256)}
    next_index = 256
    output = []

    # Loop through the compressed output
    for index in compressed:
        # Get the next substring from the dictionary
        substring = dictionary[index]

        # Add the substring to the output string
        output.append(substring)

        # Add the next character from the input string to the dictionary
        if next_index < 2**16:
            dictionary[next_index] = substring + substring[0]
            next_index += 1

    # Return the decompressed output
    return ''.join(output)


def ising_to_string(config):
    """Converts a 2D Ising model configuration to a string."""
    s = ""
    for i in range(config.shape[0]):
        for j in range(config.shape[1]):
            if config[i,j] == 1:
                s += "1"
            else:
                s += "0"
    return s