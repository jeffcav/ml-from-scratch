import numpy as np
from .. import functions as functions
from . import abstract

class LinearRegression(abstract.Model):
    def __init__(self):
        self.params = None
        pass

    def fit(self, inputs, outputs, optimizer, epochs, learning_rate, regularization):
        self.training = True
        
        # initialize model parameters
        num_params = inputs.shape[1] + 1
        weights = np.zeros((1, num_params))
        self.params = [(weights, functions.Identity)]

        # train model
        optimizer(self, inputs, outputs, epochs, learning_rate, regularization)

        self.training = False

    def predict(self, x):
        # add column of ones to input matrix
        if not self.training:
            x = np.c_(np.ones(x.shape[0], x))

        return x @ self.params[0][0].T
