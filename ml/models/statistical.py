
import numpy as np
from .base import AbstractModel
from ml.algorithms.normalization import IdentityScaler

class GaussianDiscriminantAnalysis(AbstractModel):

    def __init__(self, dataScaler=IdentityScaler):
        params = None
        super(GaussianDiscriminantAnalysis, self).__init__(params)

    def compute_classes_probabilities(self, outputs):
        _, count_class_k = np.unique(outputs, return_counts=True)
        return count_class_k / outputs.shape[0]

    def compute_cov_matrices(self, inputs, outputs):
        cov_matrices = np.array([np.cov(inputs[outputs == 0], rowvar=False)])
        for class_k in range(1, np.max(outputs) + 1):
            covariance_matrix = np.cov(inputs[outputs == class_k], rowvar=False)
            cov_matrices = np.concatenate([cov_matrices, [covariance_matrix]])

        return cov_matrices

    def compute_means(self, inputs, outputs):
        means = np.array(np.mean(inputs[outputs == 0], axis=0, keepdims=True))
        for class_k in range(1, np.max(outputs) + 1):
            mean_k = np.mean(inputs[outputs == class_k], axis=0, keepdims=True)
            means = np.concatenate([means, mean_k])

        return means


    def fit(self, inputs, outputs):
        self.prob_classes = self.compute_classes_probabilities(outputs)
        self.cov_matrices = self.compute_cov_matrices(inputs, outputs)
        self.means = self.compute_means(inputs, outputs)

        self.trained = True

    def predict(self, x):
        p0 = np.log(self.prob_classes)

        p1 = -1/2 * np.log(np.linalg.det(self.cov_matrices))
        
        distance_from_mean = x - self.means
        p2 = -1/2 * (distance_from_mean @ np.linalg.pinv(self.cov_matrices) @ distance_from_mean.T)

        return np.argmax(p0 + p1 + p2)