import numpy as np
from .base import AbstractModel

NUM_NEURONS=0
ACTIV_FUNC=1

class MLPRegressor(AbstractModel):
    def __init__(self, layers, input_width):
        super(MLPRegressor, self).__init__(None)
        self.layers = layers
        self.params = self.create_layers(self.layers, input_width)

    # weights are initialized to 0.5
    def create_layers(self, layers, input_width):
        W = []

        prev_layer_width = input_width
        #shape = (input_width, self.layers[0][NUM_NEURONS])

        for i in range(len(layers)):
            curr_layer_width = layers[i][NUM_NEURONS]
            shape = (prev_layer_width, curr_layer_width)
            W.append(np.full((shape), 0.5))
            prev_layer_width = curr_layer_width

            #shape = (layers[i-1][NUM_NEURONS], layers[i][NUM_NEURONS])

        return W

    def fit(self, inputs, outputs):
        pass

    def predict(self, x):
        z = x

        for layer_idx in range(len(self.layers)):
            W = self.params[layer_idx]
            f = self.layers[layer_idx][ACTIV_FUNC]

            print(W.shape, z.shape)

            z = f(z @ W)

        return z