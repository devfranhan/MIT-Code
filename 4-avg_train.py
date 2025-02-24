import numpy as np

def avg_train(d, lbls, t):
    w = np.zeros(d.shape[1])
    b = 0
    w_sum = np.zeros_like(w)
    b_sum = 0
    count = 0

    for _ in range(t):
        for i in range(len(lbls)):
            w, b = update(d[i], lbls[i], w, b)
            w_sum += w
            b_sum += b
            count += 1

    return w_sum / count, b_sum / count


# avoiding division by zero
