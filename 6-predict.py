import numpy as np

def predict(d, w, b):
    p = np.dot(d, w) + b
    return np.where(p > 0, 1, -1)

