import numpy as np

def conditional_entropy(y, x):
    # Convert input to numpy arrays if they are not already
    y = np.array(y)
    x = np.array(x)
    
    # Get unique values and their counts for X and Y
    unique_x, counts_x = np.unique(x, return_counts=True)
    unique_y, counts_y = np.unique(y, return_counts=True)
    
    # Initialize conditional entropy
    h_y_given_x = 0.0
    
    # Calculate conditional entropy
    for xi, cx in zip(unique_x, counts_x):
        # Get the subset of Y corresponding to xi in X
        y_given_xi = y[x == xi]
        
        # Get unique values and their counts for the subset of Y given xi in X
        unique_y_given_xi, counts_y_given_xi = np.unique(y_given_xi, return_counts=True)
        
        # Calculate conditional probabilities P(Y|X) for xi
        p_y_given_xi = counts_y_given_xi / cx
        
        # Calculate conditional entropy contribution for xi
        h_y_given_xi = np.sum(-p_y_given_xi * np.log2(p_y_given_xi))
        
        # Weighted sum based on P(X=xi)
        h_y_given_x += h_y_given_xi * (cx / len(x))
    
    return h_y_given_x

# Example usage:
# Assuming X and Y are two discrete random variables in the form of lists or arrays
X = [1087, 16, 64, 96, 255, 120]
Y = [195, 8, 0, 0.00893998146057129, 0.011049985588562010, 0.00969004631042480]
# Calculate H(Y|X)
conditional_entropy_value = conditional_entropy(Y, X)
print("Conditional Entropy H(Y|X) =", conditional_entropy_value)


def hamming_distance(str1, str2):
    # Calculate the Hamming distance between two strings
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def max_hamming_distance(string_array):
    n = len(string_array)
    max_distance = 0
    
    # Iterate through all possible rotations of the array
    for i in range(n):
        rotated_array = string_array[i:] + string_array[:i]
        distance = hamming_distance(string_array, rotated_array)
        max_distance = max(max_distance, distance)
    
    return max_distance

# Example usage:
string_array_example = ["abcde", "bcdea", "cdeab", "deabc", "eabcd"]
max_distance = max_hamming_distance(string_array_example)
print("Maximum Hamming Distance of the string array:", max_distance)


def hamming_distance(str1, str2):
    # Calculate the Hamming distance between two strings
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def print_all_distances(string_array):
    n = len(string_array)
    
    for i in range(n):
        for j in range(i + 1, n):
            distance = hamming_distance(string_array[i], string_array[j])
            print(f"Hamming distance between '{string_array[i]}' and '{string_array[j]}': {distance}")

# Example usage:
string_array_example = ["abcde", "bcdea", "cdeab", "deabc", "eabcd"]
print_all_distances(string_array_example)
