import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(filename, num_columns):
        df = (pd.read_csv(filename)).iloc[:, -1]
        #split data into train and test sets
        x_train, x_test = train_test_split(df, test_size = 0.3, random_state = None, shuffle = False, stratify = None)
        train_matrix = np.zeros((x_train.shape[0], num_columns), dtype = float)
        test_matrix = np.zeros((x_test.shape[0], num_columns), dtype = float)
        data_matrix = np.zeros((df.shape[0], num_columns), dtype = float)
        #Populate matrices with specified number of columns
        for i in range(num_columns):
            train_matrix[num_columns-(i+1):, i] = x_train[:-(num_columns-(i+1)) or None]
            test_matrix[num_columns-(i+1):, i] = x_test[:-(num_columns-(i+1)) or None]
            data_matrix[num_columns-(i+1):, i] = df[:-(num_columns-(i+1)) or None]
        train_matrix = np.c_[np.ones(train_matrix.shape[0]), train_matrix]
        test_matrix = np.c_[np.ones(test_matrix.shape[0]), test_matrix]
        data_matrix = np.c_[np.ones(data_matrix.shape[0]), data_matrix]
        return train_matrix, test_matrix, data_matrix

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# def step_function(x):
#     threshold = 0.2
#     true_false_values = np.all(np.less(np.subtract(x, threshold), 0))
#     if np.etrue_false_values == True:
#         return 1
#     else:
#         return 0




x_train, x_test, x_complete = load_data('data.csv', 3)
learning_rate = 0.5
#initialise weights with random gaussian distribution
weights = np.random.normal(0, 0.5, x_complete.shape[1])
epochs = 100
train_mse = []
test_mse = []

#training
for epoch in range(epochs):
    train_square_error_list = []
    test_square_error_list = []
    for idx, data in enumerate(x_train[:-1]):
        #perceptron
        prediction = sigmoid(np.dot(data, weights))
        error = x_train[idx + 1, -1] - prediction
        weights += (learning_rate * data * error)

        #Getting train + test errors
        train_square_error_list.append(np.square(error))
        if idx < len(x_test) -1:
            test_prediction = sigmoid(np.dot(x_test[idx], weights))
            test_square_error_list.append(np.square(x_test[idx + 1, -1] - test_prediction))
        #Calculating MSE
    train_mse.append((sum(train_square_error_list))/len(train_square_error_list))
    test_mse.append((sum(test_square_error_list))/len(test_square_error_list))

#Printing error values to console
print("training MSE: ", train_mse[-1])
print("testing MSE: ", test_mse[-1])
#Plotting predictions, actual and error
plt.figure(0)
plt1 = plt.subplot2grid((2,3), (0,0), colspan=3)
predicted, = plt1.plot(sigmoid(np.dot(x_complete, weights)), label = 'Predicted')
actual, = plt1.plot(x_complete[:,-1], label = 'Actual')
plt1.set_title("Robot Movements Over Time")
plt1.set_xlabel("Iterations")
plt1.set_ylabel("Robot Position")
plt1.legend([predicted, actual], ['Predicted', 'Actual'])

plt3 = plt.subplot2grid((2,3), (1,0), colspan=3)
train, = plt3.plot(train_mse, label = 'Train')
test, = plt3.plot(test_mse, label = 'Test')
plt3.set_title("MSE Over Time")
plt3.set_xlabel("Epochs")
plt3.set_ylabel("Mean Squared Error")
plt3.legend([test, train], ['Test', 'Train'])

plt.tight_layout()
plt.show()
