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
weights = np.random.normal(0, 0.5, x.shape[0])
epochs = 1
predictions = []
error_squared = []

for epoch in range(epochs):
    for idx, data in enumerate(x.T[1:]):
        predictions.append(sigmoid(np.dot(x.T, weights)))
        error = x.T[idx + 1, 2] - predictions[-1][idx]
        error_squared.append(np.square(error))
        weights += (learning_rate * data * error)

plt.figure(0)
plt1 = plt.subplot2grid((3,3), (0,0), colspan=3)
plt1.plot(sigmoid(np.dot(x.T, weights)))
plt1.set_title("Predicted Robot Movements Over Time")
plt1.set_xlabel("Iterations")
plt1.set_ylabel("Robot Position")

plt2 = plt.subplot2grid((3,3), (1,0), colspan=3)
plt2.plot(x[-1])
plt2.set_title("Actual Robot Movements Over Time")
plt2.set_xlabel("Iterations")
plt2.set_ylabel("Robot Position")

plt3 = plt.subplot2grid((3,3), (2,0), colspan=3)
plt3.plot(error_squared)
plt3.set_title("Squared Error Over Time")
plt3.set_xlabel("Iterations")
plt3.set_ylabel("Squared Error")
MSE = (np.sum(error_squared))/len(error_squared)
plt3.text(1500, 0.2, 'Mean Squared Error: %f' %MSE)

plt.tight_layout()
plt.show()
