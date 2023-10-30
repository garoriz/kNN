from sklearn.datasets import load_iris


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


if __name__ == '__main__':
    data = load_iris()
    print(data)
