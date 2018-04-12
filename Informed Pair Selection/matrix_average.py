import numpy as np

def matrix_mean(x, y):
    label_set = {}
    for example in range(len(x)):
        label = y[example]
        if label in label_set:
            label_set[label].append([x[example]])
        else:
            label_set[label] = [x[example]]
    for key in label_set:
        label_set[key] = np.array(label_set[key]).mean(0)
    return label_set
