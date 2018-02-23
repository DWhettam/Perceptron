import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(filename, num_columns):
        df = (pd.read_csv(filename)).iloc[:, -1]
        x_train, x_test = train_test_split(df, test_size = 0.3, random_state = None, shuffle = False, stratify = None)
        train_matrix = np.zeros((num_columns, x_train.shape[0]), dtype = float)
        test_matrix = np.zeros((num_columns, x_test.shape[0]), dtype = float)
        data_matrix = np.zeros((num_columns, df.shape[0]), dtype = float)
        for i in range(num_columns):
            train_matrix[i, num_columns-(i+1):] = x_train[:-(num_columns-(i+1)) or None]
            test_matrix[i, num_columns-(i+1):] = x_test[:-(num_columns-(i+1)) or None]
            data_matrix[i, num_columns-(i+1):] = df[:-(num_columns-(i+1)) or None]
        return train_matrix, test_matrix, data_matrix


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_train, x_test, x_complete = load_data('data.csv', 3)
learning_rate = 0.1
weights = np.random.normal(0, 0.5, x_train.shape[0])
epochs = 1
predictions = []
error_squared = []

for epoch in range(epochs):
    for idx, data in enumerate(x_train.T[1:]):
        predictions.append(sigmoid(np.dot(x_train.T, weights)))
        error = x_train.T[idx + 1, 2] - predictions[-1][idx]
        error_squared.append(np.square(error))
        weights += (learning_rate * data * error)

plt.figure(0)
plt1 = plt.subplot2grid((3,3), (0,0), colspan=3)
plt1.plot(sigmoid(np.dot(x_train.T, weights)))
plt1.plot(sigmoid(np.dot(x_test.T, weights)))
plt1.set_title("Predicted Robot Movements Over Time")
plt1.set_xlabel("Iterations")
plt1.set_ylabel("Robot Position")

plt2 = plt.subplot2grid((3,3), (1,0), colspan=3)
plt2.plot(x_complete[-1])
plt2.set_title("Actual Robot Movements Over Time")
plt2.set_xlabel("Iterations")
plt2.set_ylabel("Robot Position")

plt3 = plt.subplot2grid((3,3), (2,0), colspan=3)
plt3.plot(error_squared)
plt3.set_title("Squared Error Over Time")
plt3.set_xlabel("Iterations")
plt3.set_ylabel("Squared Error")
MSE = (np.sum(error_squared))/len(error_squared)
plt3.text(1000, 0.06, 'Mean Squared Error: %f' %MSE)

plt.tight_layout()
plt.show()
