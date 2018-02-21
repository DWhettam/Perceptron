import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_data(filename):
        df = pd.read_csv(filename)
        data = df.iloc[:, -1]
        return data

def transform_data(x):
    num_columns = 3
    data_matrix = np.zeros((num_columns, x.shape[0]), dtype = float)
    data_matrix[0,num_columns-1:] = x[:-(num_columns-1) or None]
    data_matrix[1,num_columns-2:] = x[:-(num_columns-2) or None]
    data_matrix[2,num_columns-3:] = x[:-(num_columns-3) or None]
    return data_matrix


x = load_data('Initial_data.csv')
x = transform_data(x)
learning_rate = 0.1
n = x.shape[0]
weights = np.random.uniform(-0.5, 0.5, n)
sigmoid_input = 0
epochs = 1
predictions = []
iterations = []

for epoch in range(epochs):
    for idx, data in enumerate(x.T):
        if idx == x.shape[1] -1:
            break
        for i, value in enumerate(data):
            sigmoid_input += (value * weights[i])
        prediction = sigmoid(sigmoid_input)
        error = x.T[idx + 1, 2] - prediction
        delta = learning_rate * data * error
        weights += delta

        predictions.append(prediction)
        iterations.append(idx)

plt.plot(iterations, predictions)
plt.show()
