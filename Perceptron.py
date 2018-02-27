import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(filename, num_columns):
        df = (pd.read_csv(filename)).iloc[:, -1]
        #split data into train and test sets
        x_train, x_test = train_test_split(df, test_size = 0.3, random_state = None, shuffle = False, stratify = None)
        train_matrix = np.zeros((num_columns, x_train.shape[0]), dtype = float)
        test_matrix = np.zeros((num_columns, x_test.shape[0]), dtype = float)
        data_matrix = np.zeros((num_columns, df.shape[0]), dtype = float)
        #Populate matrices with specified number of columns
        for i in range(num_columns):
            train_matrix[i, num_columns-(i+1):] = x_train[:-(num_columns-(i+1)) or None]
            test_matrix[i, num_columns-(i+1):] = x_test[:-(num_columns-(i+1)) or None]
            data_matrix[i, num_columns-(i+1):] = df[:-(num_columns-(i+1)) or None]
        return train_matrix, test_matrix, data_matrix

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_train, x_test, x_complete = load_data('data.csv', 3)
learning_rate = 0.1
#initialise weights with random gaussian distribution
weights = np.random.normal(0, 0.5, x_complete.shape[0])
epochs = 1
train_mse = []
test_mse = []

#training
for epoch in range(epochs):
    predictions = []
    test_predictions = []
    train_error_list = []
    test_error_list = []
    for idx, data in enumerate(x_train.T[1:]):
        #Perceptron
        predictions.append(sigmoid(np.dot(x_train.T, weights)))
        error = x_train.T[idx + 1, -1] - predictions[-1][idx]
        weights += (learning_rate * data * error)

        #Getting train + test errors
        train_error_list.append(error)
        if idx < len(x_test.T) -1:
            test_predictions.append(sigmoid(np.dot(x_test.T, weights)))
            test_error_list.append(x_test.T[idx + 1, -1] - test_predictions[-1][idx])
        #Calculating MSE
        train_mse.append((sum(np.square(train_error_list)))/len(train_error_list))
        test_mse.append((sum(np.square(test_error_list)))/len(test_error_list))

#Printing error values to console
train_MSE = train_mse[-1]
test_MSE = test_mse[-1]
print("training MSE: ", train_MSE)
print("testing MSE: ", test_MSE)
#Plotting predictions, actual and error
plt.figure(0)
plt1 = plt.subplot2grid((3,3), (0,0), colspan=3)
plt1.plot(sigmoid(np.dot(x_complete.T, weights)), label = 'Predicted')
plt1.set_title("Predicted Robot Movements Over Time")
plt1.set_xlabel("Iterations")
plt1.set_ylabel("Robot Position")

plt2 = plt.subplot2grid((3,3), (1,0), colspan=3)
plt2.plot(x_complete[-1])
plt2.set_title("Actual Robot Movements Over Time")
plt2.set_xlabel("Iterations")
plt2.set_ylabel("Robot Position")

plt3 = plt.subplot2grid((3,3), (2,0), colspan=3)
train, = plt3.plot(train_mse, label = 'Train')
test, = plt3.plot(test_mse, label = 'Test')
plt3.set_ylim(0, 0.005)
plt3.set_title("MSE Over Time")
plt3.set_xlabel("Iterations")
plt3.set_ylabel("Mean Squared Error")

plt.legend([test, train], ['Test', 'Train'])
plt.tight_layout()
plt.show()
