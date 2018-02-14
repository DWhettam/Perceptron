
import pandas as pd
import random
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def load_data(filename):
        df = pd.read_csv(filename)
        labels = df.iloc[:, -1]
        features = df.iloc[:, :-1]
        print("data:", df)
        print("features", features)

        return features, labels


x_test, y_test = load_data('TestData.csv')
learning_rate = 0.02
weights = []


y_predict = []

print(y_test)
