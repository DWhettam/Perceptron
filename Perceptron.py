import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filename, num_columns):
        df = (pd.read_csv(filename)).iloc[:, -1]
        data_matrix = np.zeros((num_columns, df.shape[0]), dtype = float)
        for i in range(num_columns):
            data_matrix[i, num_columns-(i+1):] = df[:-(num_columns-(i+1)) or None]
        return data_matrix

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = load_data('data.csv', 5)
learning_rate = 0.1
weights = np.random.normal(0, 1, x.shape[0])
epochs = 1
predictions = []

for epoch in range(epochs):
    for idx, data in enumerate(x.T[1:]):
        predictions.append(sigmoid(np.dot(x.T, weights)))
        error = x.T[idx + 1, 2] - predictions[-1][idx]
        weights += (learning_rate * data * error)

plt.plot(sigmoid(np.dot(x.T, weights)))
plt.show()
