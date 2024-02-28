import os
import csv
from subprocess import Popen, PIPE
import itertools
import torch
import numpy as np

def colorize(text, color):
    colors = {
        "blue": "\033[94m",
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "end": "\033[0m",
    }
    return colors[color] + text + colors["end"]

def print_colored(text, color):
    print(colorize(text, color))

def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def get_interventions_dict(all_activations, top_heads, directions):
    device = "cuda"
    directions = normalize_vectors(directions)
    interventions_dict = {}
    for (layer, head), val_acc in top_heads:
        dir = directions[layer, head]                     # Assuming normalized
        activations = all_activations[:, layer, head, :]  # N x 128
        proj_vals = activations @ dir.T
        proj_val_std = np.std(proj_vals)
    
        # Check if the layer key exists in the dictionary, if not, initialize an empty list
        if layer not in interventions_dict:
            interventions_dict[layer] = []
        dir = torch.tensor(dir).to(device)
        # Append the tuple (head, val_acc, proj_val_std) to the list corresponding to the layer
        interventions_dict[layer].append((head, dir, proj_val_std, val_acc))
    return interventions_dict
    
def blend_directions(a, b, t):
    v = (1 - t) * a + t * b
    return v
    
def normalize_vectors(arr):
    """
    Normalizes the last dimension of a numpy array
    """
    # Calculate the norm (magnitude) of each (128,) vector
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)

    # Avoid division by zero by setting zero norms to one
    norms[norms == 0] = 1

    # Normalize the vectors
    normalized_arr = arr / norms
    return normalized_arr

def calculate_cosine_similarity(vector_a, vector_b):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
    vector_a (numpy.array): The first vector.
    vector_b (numpy.array): The second vector.

    Returns:
    float: The cosine similarity between vector_a and vector_b.
    """
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    cosine_similarity = dot_product / (norm_a * norm_b)
    return cosine_similarity

def diff_directions(d1, d2):
    # Ensure d1 and d2 have the same shape
    if d1.shape != d2.shape:
        raise ValueError("d1 and d2 must have the same shape")

    N, M, _ = d1.shape
    result = np.zeros_like(d1)

    for i in range(N):
        for j in range(M):
            # Normalize d2[i, j]
            d2_norm = d2[i, j] / np.linalg.norm(d2[i, j])

            # Project d1[i, j] onto d2[i, j]
            proj_d1_on_d2 = np.dot(d2_norm, d1[i, j]) * d2_norm

            # Subtract the projection from d1[i, j]
            result[i, j] = d1[i, j] - proj_d1_on_d2
    result = normalize_vectors(result)
    return result


def edit_csv_row(filename, row_to_edit, new_data):
    if not os.path.exists(filename):
        raise Exception('No csv file found', filename)
    # Read the CSV file and store the data in a list
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data = [row for row in reader]

    # Update the data in the desired row
    if len(data) > row_to_edit:
        data[row_to_edit] = new_data
    else:
        assert row_to_edit == len(data)
        data.append(new_data)

    # Write the updated data back to the CSV file
    print('writing to', filename)
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerows(data)

def push_data(data_dir: str, repo_url: str):

    # Get the current working directory
    cwd = os.getcwd()

    # Change the working directory to the data directory
    os.chdir(data_dir)

    # pull changes from GitHub
    p = Popen(['git', 'pull'], stdout=PIPE, stderr=PIPE) # needs to be '.' to add all files from data directory
    p.communicate()

    # Stage all changes (can send them somwhere better than github i guess?)
    p = Popen(['git', 'add', '.'], stdout=PIPE, stderr=PIPE) # needs to be '.' to add all files from data directory
    p.communicate()

    # Commit changes
    p = Popen(['git', 'commit', '-m', 'auto-commit-csv-change'], stdout=PIPE, stderr=PIPE)
    p.communicate()

    # Push changes to GitHub
    p = Popen(['git', 'push', repo_url], stdout=PIPE, stderr=PIPE)
    p.communicate()

    # Change back to the original working directory
    os.chdir(cwd)


def get_num_items(file_name: str) -> int:
    # Open the CSV file in append mode
    csv_file = f'{file_name}'
    print(csv_file)
    if not os.path.exists(csv_file):
        return 0
    num_rows = 0
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            num_rows += 1
    return num_rows

def get_vars_from_out(out:str, var_list: list, stage_1=False):
    # Get the variables from the output
    var_dict = {}
    for lines in out.splitlines():
        for var in var_list:
            if f'{var}:' in lines:
                if stage_1:
                    var_dict[var] = ": ".join(lines.split(": ")[1:])
                else:
                    var_dict[var] = lines.split(': ')[1].strip()
    return var_dict

def find_largest_k_items(arr, k):
    """
    Find the largest k items in a numpy array.

    Parameters:
    arr (numpy.array): A numpy array from which to find the largest k items.
    k (int): The number of largest items to find.

    Returns:
    list of tuples: A list of tuples, each containing the index and value of one of the k largest items.
    """
    # Flatten the array and get the indices of the largest k values
    indices = np.unravel_index(np.argsort(arr.ravel())[-k:], arr.shape)
    
    # Zip the indices together and get the corresponding values
    largest_items = [(index, arr[index]) for index in zip(*indices)]
    return largest_items[::-1]

