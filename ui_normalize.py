import numpy as np

def apply_normalization(y):
    if np.max(np.abs(y)) == 0:
        return y
    return y / np.max(np.abs(y))
