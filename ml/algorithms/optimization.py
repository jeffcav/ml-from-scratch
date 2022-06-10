import numpy as np
from abc import abstractmethod

from .. import models
from ..models.nn import ACTIV_FUNC_IDX
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


class BackpropGD(AbstractGradientDescent):
    def __init__(self, epochs, learning_rate, regularization, batch_size=1, metrics=RMSE()) -> None:
        super(BackpropGD, self).__init__(epochs, learning_rate, regularization, metrics)
        self.batch_size = batch_size

    def backprop_output_layer(self, x, y, y_estimated, weights):
        delta = y - y_estimated

        regularization_term = self.regularization * weights
        regularization_term[0,0] = 0.0

        mean_xerror = (x * delta).mean(axis=0, keepdims=True)
        adjust = mean_xerror.T - regularization_term
        step = (self.learning_rate * adjust)

        return step, delta

    def backprop_hidden_layer(self, func, linear_output, input, delta_nxt_layer, weights, weights_nxt_layer):
        grad = func.grad(linear_output)

        print("[SHAPES]\n\t",
            "grad:", grad.shape,
            "delta_nxt_layer:", delta_nxt_layer.shape,
            "weights_nxt_layer:", weights_nxt_layer.shape
        )

        delta = grad * (delta_nxt_layer @ weights_nxt_layer[1:,:].T)

        regularization_term = self.regularization * weights
        regularization_term[0,0] = 0.0

        step = self.learning_rate*(input.T@delta)

        return step, delta

    def solve(self, model, inputs, outputs):
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
        for _ in range(self.epochs):
            # shuffle and batch input data
            shuffle = np.random.permutation(inputs.shape[0])
            num_batches = inputs.shape[0]/self.batch_size
            batches = np.array_split(shuffle, num_batches)

            for batch in batches:
                X = inputs[batch]
                Y = outputs[batch]

                # feedforward
                predictions = model.predict(X)

                output_layer_idx = len(model.params) - 1
                i = model.inputs[output_layer_idx]
                step, delta = self.backprop_output_layer(i, Y, predictions, model.params[output_layer_idx])
                model.params[output_layer_idx] += step

                for layer_idx in range(output_layer_idx - 1, -1, -1):
                    layer_inputs = model.inputs[layer_idx]
                    layer_linear_outputs = model.linear_outputs[layer_idx]
                    func = model.layers[layer_idx][ACTIV_FUNC_IDX]

                    step, delta = self.backprop_hidden_layer(func, layer_linear_outputs,
                                    layer_inputs, delta,
                                    model.params[layer_idx],
                                    model.params[layer_idx+1])
                    model.params[layer_idx] += step

            # keep track of training quality
            predictions = model.predict(inputs)
            epoch_measurement = self.metrics.measure(outputs, predictions)
            training_measurements.append(epoch_measurement)

        return training_measurements
