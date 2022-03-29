import numpy as np
from .. import models

class GradientDescent:
    def __init__(self) -> None:
        pass

    def __call__(self, model, inputs, outputs, epochs, lr, regularization):
        """ Runs the Grandient Descent algorithm and modifies model during trainning

        Parameters:
            model (ml.abstract.Model): model with initialized parameters and a predict(x) function.
                Examples: 
                    f(x) = w0 + w1x -> model.params = [([w0, w1], Identity())]
                    g(x) = sigmoid(w0 + w1x) -> model.params = [([w0, w1], Sigmoid())]
            
            inputs (numpy.Array): array of inputs with potentially multiple features
            
            outputs (numpy.Array): array of outputs for each input
            
            lr (float): learning rate
            
            epochs (int): number of epochs

        Returns:
            errors (array): trainning errors at each epoch

        """
        
        # extract model's weights and activation funtion
        # support for multiple layers is planned for future releases
        layer = 0
        weights = model.weights(layer)
        activation = model.activation_function(layer)

        # Add column of ones to the input
        inputs = np.c_[np.ones(inputs.shape[0]), inputs]

        errors = []
        for _ in range(epochs):
            predictions = model.predict(inputs)
            error = outputs - predictions
            
            # support for the gradient of more activation
            # functions is planned for future releases
            gradients = (inputs * error).mean(axis=0)
            weights += lr * gradients

            errors.append(error)
            
        return errors

class StochasticGradientDescent:
    def __init__(self) -> None:
        pass

    def __call__(self, model, inputs, outputs, epochs, lr, regularization):
        """ Runs the Grandient Descent algorithm and modifies model during trainning

        Parameters:
            model (ml.abstract.Model): model with initialized parameters and a predict(x) function.
                Examples: 
                    f(x) = w0 + w1x -> model.params = [([w0, w1], Identity())]
                    g(x) = sigmoid(w0 + w1x) -> model.params = [([w0, w1], Sigmoid())]
            
            inputs (numpy.Array): array of inputs with potentially multiple features
            
            outputs (numpy.Array): array of outputs for each input
            
            lr (float): learning rate
            
            epochs (int): number of epochs

        Returns:
            errors (array): trainning errors at each epoch

        """
        
        # extract model's weights and activation funtion
        # support for multiple layers is planned for future releases
        layer = 0
        weights = model.weights(layer)
        activation = model.activation_function(layer)

        # Add column of ones to the input
        inputs = np.c_[np.ones(inputs.shape[0]), inputs]

        errors = []
        for _ in range(epochs):
            # shuffle data
            shuffle = np.random.permutation(inputs.shape[0])
            inputs = inputs[shuffle]
            outputs = outputs[shuffle]

            for i in range(inputs.shape[0]):
                prediction = model.predict(inputs[i])
                error = outputs[i] - prediction

                gradients = (inputs[i] * error)
                weights += (lr * gradients)

            # store errors
            errors.append(error)

        return errors
