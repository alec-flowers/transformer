import numpy as np

class LinearNetwork():
    def __init__(self) -> None:
        self.linear_layer = LinearLayer(2, 1)

    def forward(self, x):
        x = self.linear_layer(x)
        return x


class LinearLayer():
    def __init__(self, input_neurons, output_neurons) -> None:
        # TODO how do we want to initialize the weights and bias?
        # self.weights = np.random.normal(0, 1, size=(input_neurons, output_neurons))
        # self.bias = np.random.normal(0, 1, size=(input_neurons, output_neurons))
        self.weights = np.ones((output_neurons, input_neurons))
        self.bias = np.ones((output_neurons, 1))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return np.dot(x, self.weights.T) + self.bias

    def get_parameters(self):
        return self.weights, self.bias

    def set_parameters(self, weights, bias):
        assert weights.shape == self.weights.shape, f"Your provided weights {weights.shape} != {self.weights.shape}"
        assert bias.shape == self.bias.shape, f"Your provided bias {bias.shape} != {self.bias.shape}"
        self.weights = weights
        self.bias = bias



if __name__ == "__main__":
    network = LinearNetwork()
    params = network.linear_layer.get_parameters()
    input = np.array([[1,2]])
    output = network.forward(input)
    print(output)
