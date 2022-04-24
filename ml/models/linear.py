import numpy as np
from abc import abstractmethod

from . import base
from .. import functions as functions
from ml.algorithms.normalization import IdentityScaler

class AbstractLinear(base.AbstractModel):
    def default_activation_function(self, x):
        return x

    def default_transformation_function(self, x):
        return x

    def __init__(self, solver, transformation_function=None, activation_function=None, dataScaler=IdentityScaler):

        params = None
        super(AbstractLinear, self).__init__(params)

        # function that transforms inputs
        if transformation_function is None:
            self.transformation_function = self.default_transformation_function
        else:
            self.transformation_function = transformation_function

        # function that transforms outputs
        if activation_function is None:
            self.activation_function = self.default_activation_function
        else:
            self.activation_function = activation_function

        self.trained = False
        self.solver = solver        
        self.inputs_scaler = dataScaler()
        self.outputs_scaler = dataScaler()


    def fit(self, inputs, outputs):
        inputs = self.transformation_function(inputs)
        
        inputs = self.inputs_scaler.fit(inputs)
        outputs = self.outputs_scaler.fit(outputs)
        
        inputs = np.c_[np.ones(inputs.shape[0]), inputs]

        # initialize model parameters
        num_params = inputs.shape[1]
        weights = np.zeros((1, num_params))
        
        self.params = weights
        
        errors = self.solver.solve(self, inputs, outputs)
        self.trained = True

        return errors

    def predict(self, x):
        if self.trained:
            x = self.transformation_function(x)

        if self.trained:
            x = self.inputs_scaler.transform(x)

        if self.trained:
            x = np.c_[np.ones(x.shape[0]), x]

        y = self.activation_function(x @ self.params.T)

        if self.trained:
            return self.outputs_scaler.inverse_transform(y)

        return y

    def get_activation_function(self):
        return self.activation_function

    def get_transformation_function(self):
        return self.transformation_function

    def set_weights(self, weights):
        self.params = weights.copy()

class LinearRegression(AbstractLinear):
    def __init__(self, solver, dataScaler=IdentityScaler):
        super(LinearRegression, self).__init__(solver, dataScaler=dataScaler)

class PolynomialRegression(AbstractLinear):
    def polynomial_features(self, x):
        p = x.copy()

        for degree in range(2, self.degree + 1):
            p = np.append(p, x ** degree, axis=1)

        return p

    def __init__(self, solver, degree, dataScaler=IdentityScaler):
        super(PolynomialRegression, self).__init__(solver, transformation_function = self.polynomial_features, dataScaler=dataScaler)
        self.degree = degree
