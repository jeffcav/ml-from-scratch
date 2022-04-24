import numpy as np
from abc import abstractmethod

from .. import models

def rmse(y, y_hat):
    return np.sqrt(((y-y_hat)**2).mean())

class AbstractSolver:
    def __init__(self):
        pass
    
    @abstractmethod
    def solve(self, model, inputs, outputs):
        pass


class OrdinaryLeastSquares(AbstractSolver):
    def __init__(self, regularization=0.0):
        super().__init__()
        self.regularization = regularization

    def solve(self, model, inputs, outputs):
        """ 
        Runs the Ordinary Least Squares algorithm and updates model parameters *after* trainning

        Parameters:
            model (ml.linear.AbstractLinear): any linear model
            inputs (numpy.Array): array of inputs with potentially multiple features
            outputs (numpy.Array): array of outputs for each input
        """

        n = inputs.shape[1]
        W = ((np.linalg.pinv(inputs.T @ inputs + (self.regularization * np.identity(n)))) @ inputs.T) @ outputs
        model.set_weights(W.T)

class AbstractGradientDescent(AbstractSolver):
    def __init__(self, epochs, learning_rate, regularization) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization

    @abstractmethod
    def solve(self, model, inputs, outputs):
        pass

class GradientDescent(AbstractGradientDescent):
    def __init__(self, epochs, learning_rate, regularization) -> None:
        super(GradientDescent, self).__init__(epochs, learning_rate, regularization)

    def solve(self, model, inputs, outputs):
        """
        Runs the Grandient Descent algorithm and modifies model parameters *during* trainning

        Parameters:
            model (ml.abstract.Model): model with initialized parameters and a predict(x) function.
            inputs (numpy.Array): array of inputs with potentially multiple features
            outputs (numpy.Array): array of outputs for each input
        Returns:
            errors (array): trainning errors at each epoch

        """

        weights = model.get_weights()
        activation = model.get_activation_function()

        errors = []
        for _ in range(self.epochs):
            predictions = model.predict(inputs)
            error = outputs - predictions

            regularization_term = self.regularization * weights
            regularization_term[0,0] = 0.0
            
            gradients = (inputs * error).mean(axis=0, keepdims=True) - regularization_term
            weights += self.learning_rate * gradients

            epoch_error = rmse(outputs, predictions)
            errors.append(epoch_error)

        return errors

class StochasticGradientDescent(AbstractGradientDescent):
    def __init__(self, epochs, learning_rate, regularization) -> None:
        super(StochasticGradientDescent, self).__init__(epochs, learning_rate, regularization)

    def solve(self, model, inputs, outputs):
        """
        Runs the Stochastic Grandient Descent algorithm and updates model parameters *during* trainning

        Parameters:
            model (ml.abstract.Model): model with initialized parameters and a predict(x) function.
            inputs (numpy.Array): array of inputs with potentially multiple features
            outputs (numpy.Array): array of outputs for each input
        Returns:
            errors (array): trainning errors at each epoch

        """
        
        weights = model.get_weights()
        activation = model.get_activation_function()

        errors = []
        for _ in range(self.epochs):
            # shuffle data
            shuffle = np.random.permutation(inputs.shape[0])
            inputs = inputs[shuffle]
            outputs = outputs[shuffle]

            for i in range(inputs.shape[0]):
                prediction = model.predict(inputs[i])
                error = outputs[i] - prediction

                regularization_term = self.regularization * weights
                regularization_term[0,0] = 0.0

                gradients = (inputs[i] * error) - regularization_term
                weights += (self.learning_rate * gradients)

            predictions = model.predict(inputs)
            epoch_error = rmse(outputs, predictions)
            errors.append(epoch_error)

        return errors
