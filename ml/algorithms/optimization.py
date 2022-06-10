import numpy as np
from abc import abstractmethod

from .. import models
from ..models.nn import ACTIV_FUNC_IDX
from .. functions.metrics.regression import RMSE

import matplotlib.pyplot as plt

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
    def __init__(self, epochs, learning_rate, regularization, metrics, momentum=0) -> None:
        self.epochs = epochs
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.momentum = momentum

    def __repr__(self):
        return f"Epochs={self.epochs}, LearningRate={self.learning_rate}, Regularization={self.regularization}, Momentum={self.momentum}"
    
    def __str__(self):
        return f"Epochs={self.epochs}, LearningRate={self.learning_rate}, Regularization={self.regularization}, Momentum={self.momentum}"

    @abstractmethod
    def solve(self, model, inputs, outputs, inputs_test=None, outputs_test=None):
        pass

class GradientDescent(AbstractGradientDescent):
    def __init__(self, epochs, learning_rate, regularization, metrics=RMSE()) -> None:
        super(GradientDescent, self).__init__(epochs, learning_rate, regularization, metrics)

    def solve(self, model, inputs, outputs, inputs_test=None, outputs_test=None):
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
        test_measurements = []
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

            # measure test error (if applicable)
            if inputs_test is not None:
                test_predictions = model.predict(inputs_test)
                epoch_measurement = self.metrics.measure(outputs_test, test_predictions)
                test_measurements.append(epoch_measurement)

        if inputs_test is not None:
            return training_measurements, test_measurements
        return training_measurements, None

class StochasticGradientDescent(AbstractGradientDescent):
    def __init__(self, epochs, learning_rate, regularization, metrics=RMSE()) -> None:
        super(StochasticGradientDescent, self).__init__(epochs, learning_rate, regularization, metrics)

    def solve(self, model, inputs, outputs, inputs_test=None, outputs_test=None):
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
        test_measurements = []
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

            # measure test error (if applicable)
            if inputs_test is not None:
                test_predictions = model.predict(inputs_test)
                epoch_measurement = self.metrics.measure(outputs_test, test_predictions)
                test_measurements.append(epoch_measurement)

        if inputs_test is not None:
            return training_measurements, test_measurements
        return training_measurements, None


class BackpropSGD(AbstractGradientDescent):
    def __init__(self, epochs, learning_rate, regularization, momentum=0, batch_size=1, metrics=RMSE()) -> None:
        super(BackpropSGD, self).__init__(epochs, learning_rate, regularization, metrics, momentum=momentum)
        self.batch_size = batch_size

    def backprop_output_layer(self, x, y, y_estimated, weights, last_step):
        delta = y - y_estimated

        regularization_term = self.regularization * weights
        regularization_term[0,0] = 0.0

        mean_xerror = (x * delta).mean(axis=0, keepdims=True)

        adjust = mean_xerror.T - regularization_term
        step = (self.learning_rate * adjust) + (self.momentum * last_step)
        return step, delta

    def backprop_hidden_layer(self, func, linear_output, input, delta_nxt_layer, weights, weights_nxt_layer, last_step):
        grad = func.grad(linear_output)

        delta = grad * (delta_nxt_layer @ weights_nxt_layer[1:,:].T)

        regularization_term = self.regularization * weights
        regularization_term[0,0] = 0.0

        step = self.learning_rate*(input.T@delta) + (self.momentum * last_step)

        return step, delta

    def solve(self, model, inputs, outputs, inputs_test=None, outputs_test=None):
        """
        Runs the Backpropagation with mini-batch Stochastic Grandient
        Descent (SGD) and modifies model parameters *during* trainning

        Parameters:
            model (ml.abstract.Model): model with initialized parameters and a predict(x) function.
            inputs (numpy.Array): array of inputs with potentially multiple features
            outputs (numpy.Array): array of outputs for each input
        Returns:
            errors (array): trainning errors at each epoch

        """

        training_measurements = []
        test_measurements = []
        for _ in range(self.epochs):

            # shuffle and batch input data
            shuffle = np.random.permutation(inputs.shape[0])
            num_batches = inputs.shape[0]/self.batch_size
            batches = np.array_split(shuffle, num_batches)

            for batch in batches:
                X = inputs[batch]
                Y = outputs[batch]

                steps = [0] * len(model.params)

                # feedforward
                predictions = model.predict(X)

                # compute gradients of the output layer
                output_layer_idx = len(model.params) - 1
                i = model.inputs[output_layer_idx]
                step, delta = self.backprop_output_layer(i, Y, 
                                                    predictions, model.params[output_layer_idx], 
                                                    steps[output_layer_idx])
                steps[output_layer_idx] = step

                # compute gradients of the hidden layers
                for layer_idx in range(output_layer_idx - 1, -1, -1):

                    layer_inputs = model.inputs[layer_idx]
                    layer_linear_outputs = model.linear_outputs[layer_idx]
                    func = model.layers[layer_idx][ACTIV_FUNC_IDX]

                    step, delta = self.backprop_hidden_layer(func, layer_linear_outputs,
                                                layer_inputs, delta,
                                                model.params[layer_idx],
                                                model.params[layer_idx+1],
                                                steps[layer_idx])
                    steps[layer_idx] = step

                # update weights
                for layer_idx in range(output_layer_idx, -1, -1):
                    model.params[layer_idx] += steps[layer_idx]

            # measure train error
            predictions = model.predict(inputs)
            epoch_measurement = self.metrics.measure(outputs, predictions)
            training_measurements.append(epoch_measurement)

            # measure test error (if applicable)
            if inputs_test is not None:
                test_predictions = model.predict(inputs_test)
                epoch_measurement = self.metrics.measure(outputs_test, test_predictions)
                test_measurements.append(epoch_measurement)

        if inputs_test is not None:
            return training_measurements, test_measurements
        return training_measurements, None
