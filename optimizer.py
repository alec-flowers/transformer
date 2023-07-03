class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, network):
        layers = network.get_layers()
        for layer in layers:
            weights, bias = layer.get_parameters()
            gradient_weights, gradient_bias = layer.get_gradients()

            # print(gradient_weights, weights)
            # print(gradient_bias, bias)

            weights -= self.lr * gradient_weights
            bias -= self.lr * gradient_bias

            layer.set_parameters(weights, bias)
