import numpy as np
from .base import AbstractModel

NUM_NEURONS=0
ACTIV_FUNC=1

class MLPRegressor(AbstractModel):
    def __init__(self, layers, input_width):
        super(MLPRegressor, self).__init__(None)
        self.layers = layers
        self.create_layers(self.layers, input_width)

    # weights are initialized to 0.5
    def create_layers(self, layers, input_width):
        self.params = []

        prev_layer_width = input_width
        for i in range(len(layers)):
            curr_layer_width = layers[i][NUM_NEURONS] + 1

            layer_weights = np.random.rand(prev_layer_width, curr_layer_width)
            self.params.append(layer_weights)
            
            prev_layer_width = curr_layer_width

    def fit(self, inputs, outputs):
        pass

    def predict(self, x):
        if not self.trained:
            self.outputs = []

        z = x

        for layer_idx in range(len(self.layers)):
            W = self.params[layer_idx]
            f = self.layers[layer_idx][ACTIV_FUNC]

            print("z shape:", z.shape, "w shape:", W.shape, "out shape:", z.shape[0], W.shape[1])

            z = f(z @ W)

            # keep track of each layer's output during
            # training, so we can backpropagate error
            if not self.trained:
                self.outputs.append(z)

        return z
