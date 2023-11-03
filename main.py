from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


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


def plot_projection(data):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.scatter(data[:, 0], data[:, 1], c=y)
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.title('Проекция на оси до нормализации')

    plt.subplot(2, 3, 2)
    plt.scatter(data[:, 0], data[:, 2], c=y)
    plt.xlabel('sepal length (cm)')
    plt.ylabel('petal length (cm)')
    plt.title('Проекция на оси до нормализации')

    plt.subplot(2, 3, 3)
    plt.scatter(data[:, 0], data[:, 3], c=y)
    plt.xlabel('sepal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.title('Проекция на оси до нормализации')

    plt.subplot(2, 3, 4)
    plt.scatter(data[:, 1], data[:, 2], c=y)
    plt.xlabel('sepal width (cm)')
    plt.ylabel('petal length (cm)')
    plt.title('Проекция на оси до нормализации')

    plt.subplot(2, 3, 5)
    plt.scatter(data[:, 1], data[:, 3], c=y)
    plt.xlabel('sepal width (cm)')
    plt.ylabel('petal width (cm)')
    plt.title('Проекция на оси до нормализации')

    plt.subplot(2, 3, 6)
    plt.scatter(data[:, 2], data[:, 3], c=y)
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.title('Проекция на оси до нормализации')

    plt.tight_layout()
    plt.show()

    plt.subplot(2, 3, 1)
    plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=y)
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.title('Проекция на оси после нормализации')

    plt.subplot(2, 3, 2)
    plt.scatter(X_normalized[:, 0], X_normalized[:, 2], c=y)
    plt.xlabel('sepal length (cm)')
    plt.ylabel('petal length (cm)')
    plt.title('Проекция на оси после нормализации')

    plt.subplot(2, 3, 3)
    plt.scatter(X_normalized[:, 0], X_normalized[:, 3], c=y)
    plt.xlabel('sepal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.title('Проекция на оси после нормализации')

    plt.subplot(2, 3, 4)
    plt.scatter(X_normalized[:, 1], X_normalized[:, 2], c=y)
    plt.xlabel('sepal width (cm)')
    plt.ylabel('petal length (cm)')
    plt.title('Проекция на оси после нормализации')

    plt.subplot(2, 3, 5)
    plt.scatter(X_normalized[:, 1], X_normalized[:, 3], c=y)
    plt.xlabel('sepal width (cm)')
    plt.ylabel('petal width (cm)')
    plt.title('Проекция на оси после нормализации')

    plt.subplot(2, 3, 6)
    plt.scatter(X_normalized[:, 2], X_normalized[:, 3], c=y)
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.title('Проекция на оси после нормализации')

    plt.tight_layout()
    plt.show()


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

    plot_projection(X)
