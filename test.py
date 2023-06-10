import unittest
from unittest.mock import Mock

import numpy as np

from main import LinearLayer

class LinearNetwork():
    def __init__(self) -> None:
        self.linear_layer = LinearLayer(2, 1)

    def forward(self, x):
        x = self.linear_layer(x)
        return x

class TestModules(unittest.TestCase):

    def test_forward_pass(self):
        """
        We are performing one forward step and basically doing the calculation:
        y = x * w.T + b
        y = [0,1] * [1,2].T + [1]
        y = [3]
        """
        network = LinearNetwork()
        network.linear_layer.set_parameters(np.array([[1,2]]), np.array([[1]]))
        data = np.array([[0,1]])

        result = network.forward(data)

        self.assertTrue(np.array_equal(result, np.array([[3]])))




if __name__ == "__main__":
    unittest.main()
