import numpy as np
from .base import AbstractModel

NUM_NEURONS_IDX=0
ACTIV_FUNC_IDX=1

class MLPRegressor(AbstractModel):
    def __init__(self, layers, input_width, solver):
        super(MLPRegressor, self).__init__(None)
        self.solver = solver
        self.layers = layers
        self.params = self.create_layers(self.layers, input_width)

    def create_layers(self, layers, input_width):
        W = []

        prev_layer_width = input_width
        for i in range(len(layers)):
            layer_width = layers[i][NUM_NEURONS_IDX]
            
            layer_weights = np.random.rand(prev_layer_width + 1, layer_width) # +1 for bias
            prev_layer_width = layer_width
            
            W.append(layer_weights)
        return W

    def fit(self, inputs, outputs):
        inputs = np.c_[np.ones(inputs.shape[0]), inputs]
        errors = self.solver.solve(self, inputs, outputs)

        self.trained = True
        return errors

    def predict(self, x):
        if not self.trained:
            self.inputs = []
            self.linear_outputs = []
            self.nonlinear_outputs = []

        z = x

        for layer_idx in range(len(self.layers)):
            # append bias = 1 to all layers when trained
            # or to all but the input layer when training
            if self.trained or (layer_idx > 0 and not self.trained):
                z = np.c_[np.ones(z.shape[0]), z]

            W = self.params[layer_idx]
            f = self.layers[layer_idx][ACTIV_FUNC_IDX]

            # keep track of each layer's outputs (raw and non-linear)
            # during training, so we can more easily backpropagate error
            if not self.trained:
                self.inputs.append(z)

            u = z @ W
            if not self.trained:
                self.linear_outputs.append(u)

            z = f(u)
            if not self.trained:
                self.nonlinear_outputs.append(z)

        return z
