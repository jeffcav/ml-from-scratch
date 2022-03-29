class Model:
    PARAMS_WEIGHTS_IDX = 0
    PARAMS_ACTIVATION_IDX = 0

    def __init__(self):
        self.params = None
        pass

    def fit(self, inputs, outputs, optimizer, epochs, learning_rate, regularization):
        pass

    def predict(self, x):
        pass

    def weights(self, layer):
        return self.params[layer][0]

    def activation_function(self, layer):
        return self.params[layer][1]
