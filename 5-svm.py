import numpy as np

def svm(d, lbls, t, l):
    w = np.zeros(d.shape[1])
    b = 0
    step = 0

    for _ in range(t):
        for i in range(len(lbls)):
            step += 1
            eta = 1 / np.sqrt(step)
            w = (1 - eta * l) * w
            if lbls[i] * (np.dot(w, d[i]) + b) <= 1:
                w += eta * lbls[i] * d[i]
                b += eta * lbls[i]

    return w, b



