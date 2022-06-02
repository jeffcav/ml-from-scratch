import numpy as np
from .base import AbstractModel

class MLPRegressor(AbstractModel):
    def __init__(self, layers=()):
        super(MLPRegressor, self).__init__(None)
        self.layers = layers
        self.params = self.create_layers(layers)

        for l in self.params:
            print(l.shape)


    def create_layers(self, layers):
        W = []
        for i in range(len(layers) - 1):
            shape = (layers[i][0], layers[i+1][0])
            W.append(np.zeros((shape)))

        return W

    def fit(self, inputs, outputs):
        pass

    def predict(self, x):
        z = x

        for layer_idx in range(len(self.layers) - 1):
            W = self.params[layer_idx]
            f = self.layers[layer_idx][1]

            z = f(W.T @ z)

        return z