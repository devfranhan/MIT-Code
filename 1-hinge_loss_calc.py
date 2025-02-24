import numpy as np

def loss(vec, lbl, w, b):
    m = lbl * (np.dot(w, vec) + b)
    return max(0, 1 - m)



# trying to test with different values for vec and lbl
# might need to check np.dot behavior
# print(loss(np.array([1, 2]), 1, np.array([0.5, -0.5]), 0))
