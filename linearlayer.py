import numpy as np
from lossfunctions import MSE
from module import Module


class LinearLayer(Module):
    def __init__(self, input_neurons, output_neurons) -> None:
        # TODO how do we want to initialize the weights and bias?
        # self.weights = np.random.normal(0, 1, size=(input_neurons, output_neurons))
        # self.bias = np.random.normal(0, 1, size=(input_neurons, output_neurons))
        self.weights = np.ones((output_neurons, input_neurons))
        self.bias = np.ones((output_neurons, 1))

        self.gradient_weights = None
        self.gradient_bias = None

        self.input_data = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.input_data = x
        return np.dot(x, self.weights.T) + self.bias

    def backward(self, previous_gradient):
        self.gradient_weights = previous_gradient @ self.input_data
        self.gradient_bias = previous_gradient
        return previous_gradient @ self.weights

    def get_parameters(self):
        return self.weights, self.bias

    def set_parameters(self, weights, bias):
        assert weights.shape == self.weights.shape, f"Your provided weights {weights.shape} != {self.weights.shape}"
        assert bias.shape == self.bias.shape, f"Your provided bias {bias.shape} != {self.bias.shape}"
        self.weights = weights
        self.bias = bias


class LinearNetwork():
    def __init__(self) -> None:
        self.linear_layer = LinearLayer(2, 1)

    def forward(self, x):
        x = self.linear_layer(x)
        return x


if __name__ == "__main__":
    network = LinearNetwork()
    loss_function = MSE()
    params = network.linear_layer.get_parameters()
    input = np.array([[1, 2]])
    target = np.array([[1]])
    output = network.forward(input)

    loss = loss_function(output, target)
    mse_derivative = loss_function.backward(output, target)

    linear_gradient = network.linear_layer.backward(mse_derivative)
    print(linear_gradient)
    print(output)