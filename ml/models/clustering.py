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
        preds = []
        for xi in x:
            dists = self.distance.measure(self.inputs_scaler.transform(xi), self.inputs)
            preds.append(np.bincount((self.outputs[dists.argsort()[:self.k]])).argmax())
        return np.array(preds)
