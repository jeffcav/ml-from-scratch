
import numpy as np
from .base import AbstractModel

class GaussianDiscriminantAnalysis(AbstractModel):

    def __init__(self):
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
        self.num_classes = len(np.unique(outputs))
        self.prob_classes = self.compute_classes_probabilities(outputs)
        self.cov_matrices = self.compute_cov_matrices(inputs, outputs)
        self.means = self.compute_means(inputs, outputs)

        self.trained = True

    def predict(self, x):
        y_pred = []

        for i in range(x.shape[0]):
            probs = []

            for k in range(self.num_classes):
                p0 = np.log(self.prob_classes[k])

                p1 = -0.5 * np.log(np.linalg.det(self.cov_matrices[k]))

                distance_from_mean = x[i] - self.means[k]
                p2 = -(0.5) * (distance_from_mean @ (np.linalg.pinv(self.cov_matrices[k]) @ distance_from_mean.T))

                probs.append(p0+p1+p2)

            y_pred.append(np.argmax(probs))

        return np.array(y_pred)


class GaussianNaiveBayes(AbstractModel):
    def __init__(self):
        params = None
        super(GaussianNaiveBayes, self).__init__(params)

    def compute_classes_probabilities(self, outputs):
        _, count_class_k = np.unique(outputs, return_counts=True)
        return count_class_k / outputs.shape[0]

    def compute_variance_matrices(self, inputs, outputs):
        variances = np.array([(inputs[outputs == 0]).var(axis=0, ddof=1)])
        for class_k in range(1, np.max(outputs) + 1):
            variance = inputs[outputs == class_k].var(axis=0, ddof=1)
            variances = np.concatenate([variances, [variance]])

        return variances

    def compute_means(self, inputs, outputs):
        means = np.array(np.mean(inputs[outputs == 0], axis=0, keepdims=True))
        for class_k in range(1, np.max(outputs) + 1):
            mean_k = np.mean(inputs[outputs == class_k], axis=0, keepdims=True)
            means = np.concatenate([means, mean_k])

        return means

    def fit(self, inputs, outputs):
        self.num_classes = len(np.unique(outputs))
        self.prob_classes = self.compute_classes_probabilities(outputs)
        self.variances = self.compute_variance_matrices(inputs, outputs)
        self.means = self.compute_means(inputs, outputs)

        self.trained = True

    def predict(self, x):
        y_pred = []

        for i in range(x.shape[0]):
            probs = []

            for k in range(self.num_classes):
                p0 = np.log(self.prob_classes[k])

                p1 = 0
                for d in range(x.shape[1]):
                    p1 += np.log(2 * np.pi * self.variances[k][d])
                p1 *= -0.5

                p2 = 0
                for d in range(x.shape[1]):
                    p2 += ((x[i][d] - self.means[k][d])**2)/self.variances[k][d]
                p2 *= -0.5

                probs.append(p0+p1+p2)

            y_pred.append(np.argmax(probs))

        return np.array(y_pred)
