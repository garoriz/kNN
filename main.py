from sklearn.datasets import load_iris
import numpy as np


class Iris:
    sepal_length = 0
    sepal_width = 0
    petal_length = 0
    petal_width = 0
    class_name = ""

    def __init__(self, sepal_length, sepal_width, petal_length, petal_width):
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def k_nearest_neighbors(X_train, y_train, x_test, k):
    distances = []
    for i in range(len(X_train)):
        distance = euclidean_distance(x_test, X_train[i])
        distances.append((distance, y_train[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = [distance[1] for distance in distances[:k]]
    prediction = max(set(neighbors), key=neighbors.count)
    return prediction


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_normalized = (X - mean) / std

    import random

    train_ratio = 0.8

    num_samples = len(X)

    num_train_samples = int(train_ratio * num_samples)
    num_test_samples = num_samples - num_train_samples

    indices = list(range(num_samples))
    random.shuffle(indices)

    X_train = [X[i] for i in indices[:num_train_samples]]
    y_train = [y[i] for i in indices[:num_train_samples]]

    X_test = [X[i] for i in indices[num_train_samples:]]
    y_test = [y[i] for i in indices[num_train_samples:]]

    optimal_k = np.sqrt(len(X))
