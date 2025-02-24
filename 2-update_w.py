import numpy as np

def update(vec, lbl, w, b):
    pred = lbl * (np.dot(w, vec) + b)
    if pred <= 0:
        w += lbl * vec
        b += lbl
    return w, b


