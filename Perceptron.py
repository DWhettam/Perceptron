import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_data(filename):
        df = pd.read_csv(filename)
        labels = df.iloc[:, -1]
        features = df.iloc[:, :-1]
        return labels, labels


x_test, y_test = load_data('Initial_data.csv')

learning_rate = 0.000002
n = x_test.shape[0]
weights = np.random.uniform(-0.5, 0.5, n)
error = []
prediction = []

for i in range(10):
    previous_error = error
    prediction = sigmoid(np.dot(x_test, weights))
    print(prediction)
    error = y_test - prediction
    delta = (np.dot(x_test.T, error) * learning_rate)
    weights += delta

print("predictions: ", prediction)
