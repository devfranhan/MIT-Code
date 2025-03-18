import numpy as np

bf, bi, bo, bc = -100, 100, 0, 0
Wfh, Wfx, Wih, Wix, Woh, Wox, Wch, Wcx = 0, 0, 0, 100, 0, 50, 0, 100

def sigmoid(x):
    return 1 if x >= 1 else 0

def tanh(x):
    return 1 if x >= 1 else (-1 if x <= -1 else 0)

def lstm_states(inputs, h0=0, c0=1):
    h = [h0]
    c = c0
    for xt in inputs:
        ft = sigmoid(Wfh * h[-1] + Wfx * xt + bf)
        it = sigmoid(Wih * h[-1] + Wix * xt + bi)
        ot = sigmoid(Woh * h[-1] + Wox * xt + bo)
        c = ft * c + it * tanh(Wch * h[-1] + Wcx * xt + bc)
        h.append(ot * tanh(c))
    
    h = np.array(h)
    h[np.abs(h) == 0.5] = 0
    return h.astype(int).tolist()

# sequence_1 = [0, 0, 1, 1, 0, 1, 0]
# h_values_1 = lstm_states(sequence_1)

# sequence_2 = [1, 1, 0, 1, 1]
# h_values_2 = lstm_states(sequence_2)

# h_values_1, h_values_2