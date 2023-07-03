# You must implement a test executable named test.py that imports your framework and
# • Generates a training and a test set of 1, 000 points sampled uniformly in [0, 1]2
# , each with a
# label 0 if outside the disk centered at (0.5, 0.5) of radius 1/√2π, and 1 inside,
# • builds a network with two input units, one output unit, three hidden layers of 25 units,
# • trains it with MSE, logging the loss,
# • computes and prints the final train and the test errors


import unittest
from matplotlib import pyplot as plt

import numpy as np
from data import generate_disc_data
# from sklearn.model_selection import train_test_split
from linearlayer import LinearLayer
from lossfunctions import MSE
from optimizer import SGD

from sequential import Sequential


class TestUniformDisk(unittest.TestCase):
    def __init__(self, methodName: str = "runEndToEndTest") -> None:
        super().__init__(methodName)

        self.network = Sequential(
            [LinearLayer(2, 1)])  # , LinearLayer(2, 2), LinearLayer(2, 1)

        self.optimizer = SGD(lr=0.001)
        self.loss_function = MSE()
        self.training_samples = 1000
        self.X, self.y = generate_disc_data(self.training_samples)

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=42)

    def test_trains_network_with_MSE(self):
        """
        """
        for epoch in range(50):
            average_loss = 0
            for datapoint, target in zip(self.X, self.y):
                output = self.network.forward(datapoint)
                loss = self.loss_function(output, target)
                mse_derivative = self.loss_function.backward(output, target)
                self.network.backward(mse_derivative)
                self.optimizer.step(self.network)
                average_loss += loss
            print(">> Epoch: ", epoch, "Avg Loss: ", round(
                average_loss/self.training_samples, 4))

        plot_decision_boundary(self.network, self.X, self.y)


def plot_decision_boundary(network, X, y):
    x_values = np.linspace(0, 1, 100)
    y_values = np.linspace(0, 1, 100)
    xx, yy = np.meshgrid(x_values, y_values)
    coordinates = np.column_stack((xx.flatten(), yy.flatten()))

    predictions = []
    for coordinate in coordinates:
        predictions.append(network.forward(coordinate))

    predictions = np.array(predictions).reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', label='Training Set')
    # circle = plt.Circle(np.array([0.5, 0.5]), 1 / (np.sqrt(2) * np.pi), color='gray', alpha=0.3)
    # plt.gca().add_patch(circle)

    plt.contourf(xx, yy, predictions, cmap='bwr', alpha=0.8)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Decision Boundary')
    plt.show()


if __name__ == "__main__":
    unittest.main()
