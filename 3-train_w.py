import numpy as np

def train(data, lbls, t):
    w = np.zeros(data.shape[1])
    b = 0

    for _ in range(t):
        for i in range(len(lbls)):
            w, b = update(data[i], lbls[i], w, b)

    return w, b



# improve learning speed for some datasets
