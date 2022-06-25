from calendar import c
from random import sample
import numpy as np
from .base import AbstractModel

import matplotlib.pyplot as plt

from ml.functions.distance import EuclideanDistance
from ml.algorithms.normalization import IdentityScaler

class KNN(AbstractModel):
    def __init__(self, num_clusters, distance=EuclideanDistance, data_scaler=IdentityScaler):
        super(KNN, self).__init__(None)

        self.num_clusters = num_clusters
        self.distance = distance()
        self.inputs_scaler = data_scaler()

    def fit(self, inputs, outputs):
        self.inputs = self.inputs_scaler.fit(inputs).transform(inputs)
        self.outputs = outputs

    def predict(self, x):
        predictions = []
        for xi in x:
            distances = self.distance.measure(self.inputs_scaler.transform(xi), self.inputs)

            indexes_of_k_nearest = distances.argsort()[:self.num_clusters]
            classes_of_k_nearest = self.outputs[indexes_of_k_nearest]
            predictions.append(np.bincount(classes_of_k_nearest).argmax())

        return np.array(predictions)

class KMeans(AbstractModel):
    def __init__(self, k, max_iter, distance=EuclideanDistance):
        super(KMeans, self).__init__(None)

        self.max_iter = max_iter
        self.num_clusters = k
        self.distance = distance()

    # finds good-enough centroids which better clusters inputs
    def fit(self, inputs):
        num_samples, num_features = inputs.shape
        
        self.errors = []
        self.output_centroids = []

        # choose 'num_clusters' elements from the inputs matrix,
        # then build a 3D matrix with cluster index as the first dimension
        initial_centroids = inputs[np.random.choice(num_samples, self.num_clusters, replace=False)]
        self.centroids = initial_centroids.reshape(self.num_clusters, 1, num_features)

        for i in range(self.max_iter):
            clusters = [[] for _ in range(self.num_clusters)]

            # assign samples to closest centroid
            distances = self.distance.measure(inputs, self.centroids, axis=2)
            closest_centroid_idxs = np.argmin(distances, axis=0, keepdims=True)

            # fill clusters with sample indexes
            for sample_idx, closest_centroid in enumerate(closest_centroid_idxs.T):
                clusters[closest_centroid[0]].append(sample_idx)

            # update centroids
            old_centroids = self.centroids.copy()
            for cluster_idx in range(self.num_clusters):
                self.centroids[cluster_idx] = np.mean(inputs[clusters[cluster_idx]], axis=0)

            # record error and centroids in current iteration
            current_err = self.quantization_error(inputs)
            self.errors.append(current_err)
            self.output_centroids.append(self.centroids)

            if self.is_converged(old_centroids, self.centroids):
                print("Solution converged.")
                break

        self.trained = True
        return self.errors, self.output_centroids

    def is_converged(self, old_centroids, curr_centroids):
        distances = [self.distance.measure(old_centroids[i, 0], curr_centroids[i, 0], axis=0) for i in range(self.num_clusters)]
        
        if np.sum(distances) == 0:
            return True
        else:
            return False

    def quantization_error(self, x):
        norm = np.linalg.norm(x - self.centroids, axis=1)
        return np.sum(norm**2)

    def db_index(self, inputs):
        pass
        # mini_delta = np.abs(inputs-self.centroids).
        # return (1/self.num_clusters)
