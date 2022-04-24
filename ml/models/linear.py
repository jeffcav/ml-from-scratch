import numpy as np
from abc import abstractmethod

from . import base
from .. import functions as functions

class AbstractLinear(base.AbstractModel):
    def __init__(self, solver, params=([0], functions.Identity)):
        super(AbstractLinear, self).__init__(params)
        self.solver = solver

    @abstractmethod
    def fit(self, inputs, outputs):
        pass

    @abstractmethod
    def predict(self, inputs):
        pass

    def get_weights(self):
        return self.params[0]

    def get_activation_function(self):
        return self.params[1]

    def set_weights(self, weights):
        for i in range(len(weights)):
            self.params[0][i] = weights[i]

class LinearRegression(AbstractLinear):
    def __init__(self, solver, scaler=None):
        super(LinearRegression, self).__init__(solver)
        self.scaler = scaler

    def fit(self, inputs, outputs):
        # append column on 1's
        num_samples = inputs.shape[0]
        inputs = np.c_[np.ones(num_samples), inputs]

        # initialize model parameters
        num_params = inputs.shape[1]
        weights = np.zeros((1, num_params))
        self.params = (weights, functions.Identity)

        # train model
        errors = self.solver.solve(self, inputs, outputs)
        self.trained = True

        return errors

    def predict(self, x):
        # add column of 1's
        if self.trained:
            num_samples = x.shape[0]
            x = np.c_[np.ones(num_samples), x]

        return x @ self.get_weights().T


class PolynomialRegression(AbstractLinear):
    def __init__(self, solver, degree, scaler=None):
        super(PolynomialRegression, self).__init__(solver)
        self.degree = degree
        self.scaler = scaler

    def make_polynomial_features(self, x):
        p = x.copy()

        for degree in range(2, self.degree + 1):
            p = np.append(p, x ** degree, axis=1)

        return p

    def scale_dataset(self, inputs, outputs):
        if self.scaler is None:
            self.inputs_scaler = None
            self.outputs_scaler = None
            return inputs, outputs
        
        self.inputs_scaler = self.scaler()
        self.inputs_scaler.fit(inputs)
        inputs = self.inputs_scaler.transform(inputs)

        self.outputs_scaler = self.scaler()
        self.outputs_scaler.fit(outputs)
        outputs = self.outputs_scaler.transform(outputs)

        return inputs, outputs

    def fit(self, inputs, outputs):
        # compute polynomial features, scaler and add column of 1's
        inputs = self.make_polynomial_features(inputs)
        inputs, outputs = self.scale_dataset(inputs, outputs)
        inputs = np.c_[np.ones(inputs.shape[0]), inputs]

        # initialize model parameters
        num_params = inputs.shape[1]
        weights = np.zeros((1, num_params))
        self.params = (weights, functions.Identity)

        # train model
        errors = self.solver.solve(self, inputs, outputs)
        self.trained = True

        return errors

    def predict(self, x):
        if self.trained:
            x = self.make_polynomial_features(x)

        if self.inputs_scaler is not None and self.trained:
            x = self.inputs_scaler.transform(x)

        # add column of ones to the input matrix
        if self.trained:
            x = np.c_[np.ones(x.shape[0]), x]

        y = x @ self.get_weights().T

        if self.outputs_scaler is not None and self.trained:
            return self.outputs_scaler.inverse_transform(y)
        return y
