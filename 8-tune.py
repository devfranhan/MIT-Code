import numpy as np
import project1 as p1
import utils

T = [1, 5, 10, 15, 25, 50]
L = [0.001, 0.01, 0.1, 1, 10]

tune_res = utils.tune_perceptron(T, train_bow_features, train_labels, val_bow_features, val_labels)
best_T = T[np.argmax(tune_res[1])]


