import numpy as np
from .base import AbstractModel

from ml.functions.distance import EuclideanDistance
from ml.algorithms.normalization import IdentityScaler

class KNN(AbstractModel):
    def __init__(self, k, distance=EuclideanDistance, data_scaler=IdentityScaler):
        super(KNN, self).__init__(None)

        self.k = k
        self.distance = distance()
        self.inputs_scaler = data_scaler()

    def fit(self, inputs, outputs):
        self.inputs = self.inputs_scaler.fit(inputs).transform(inputs)
        self.outputs = outputs

    def predict(self, x):
        predictions = []
        for xi in x:
            distances = self.distance.measure(self.inputs_scaler.transform(xi), self.inputs)

            indexes_of_k_nearest = distances.argsort()[:self.k]
            classes_of_k_nearest = self.outputs[indexes_of_k_nearest]
            predictions.append(np.bincount(classes_of_k_nearest).argmax())

        return np.array(predictions)
