import numpy as np
from abc import abstractmethod

from .. import models
from .. functions.metrics.regression import RMSE

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
        model.params = (((np.linalg.pinv(inputs.T @ inputs + (self.regularization * np.identity(n)))) @ inputs.T) @ outputs).T

class AbstractGradientDescent(AbstractSolver):
    def __init__(self, epochs, learning_rate, regularization, metrics) -> None:
        self.epochs = epochs
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.regularization = regularization

    @abstractmethod
    def solve(self, model, inputs, outputs):
        pass

class GradientDescent(AbstractGradientDescent):
    def __init__(self, epochs, learning_rate, regularization, metrics=RMSE()) -> None:
        super(GradientDescent, self).__init__(epochs, learning_rate, regularization, metrics)

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

        training_measurements = []
        for _ in range(self.epochs):
            predictions = model.predict(inputs)
            error = outputs - predictions

            regularization_term = self.regularization * model.params
            regularization_term[0,0] = 0.0

            gradients = (inputs * error).mean(axis=0, keepdims=True) - regularization_term
            model.params += self.learning_rate * gradients

            # keep track of training quality
            predictions = model.predict(inputs)
            epoch_measurement = self.metrics.measure(outputs, predictions)
            training_measurements.append(epoch_measurement)

        return training_measurements

class StochasticGradientDescent(AbstractGradientDescent):
    def __init__(self, epochs, learning_rate, regularization, metrics=RMSE()) -> None:
        super(StochasticGradientDescent, self).__init__(epochs, learning_rate, regularization, metrics)

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

        training_measurements = []
        for _ in range(self.epochs):
            # shuffle data
            shuffle = np.random.permutation(inputs.shape[0])
            inputs = inputs[shuffle]
            outputs = outputs[shuffle]

            for i in range(inputs.shape[0]):
                prediction = model.predict(inputs[i])
                error = outputs[i] - prediction

                regularization_term = self.regularization * model.params
                regularization_term[0,0] = 0.0

                gradients = (inputs[i] * error) - regularization_term
                model.params += (self.learning_rate * gradients)

            # keep track of training quality
            predictions = model.predict(inputs)
            epoch_measurement = self.metrics.measure(outputs, predictions)
            training_measurements.append(epoch_measurement)

        return training_measurements
