import numpy as np

# Converts the weird data organization into a normal list of matrices
def to_list(file):
    data = np.load(file)
    return data
