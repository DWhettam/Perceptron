import numpy as np
import csv
import random
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def load_data(filename):

        label = []
        data = []

        data = np.genfromtxt(filename, delimeter = ',', dtype = None)
        
        for row in csv_file:                        
                        value = row[:-1]
                        # Adds out the class value
                        data.append(value)

                        # Removes the label and saves the data
                        del row[:-1]
                        label.append(row) 
                    
                        
        return data, label
        
x_test, y_test = load_data('TestData.csv')
x_train, y_train = load_data('TrainingData.csv')

n = len(x_test[0])
learning_rate = 0.02
weights = []

for i in range(n):
        weights.append(round(random.uniform(-0.5,0.5),2))

y_predict = []

print(y_test)
