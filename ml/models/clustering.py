import numpy as np
from .base import AbstractModel

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
            self.clusters = [[] for _ in range(self.num_clusters)]

            # assign samples to closest centroid
            distances = self.distance.measure(inputs, self.centroids, axis=2)
            closest_centroid_idxs = np.argmin(distances, axis=0, keepdims=True)

            # fill clusters with sample indexes
            for sample_idx, closest_centroid in enumerate(closest_centroid_idxs.T):
                self.clusters[closest_centroid[0]].append(sample_idx)

            # update centroids
            old_centroids = self.centroids.copy()
            for cluster_idx in range(self.num_clusters):
                self.centroids[cluster_idx] = np.mean(inputs[self.clusters[cluster_idx]], axis=0)

            # record error and centroids in current iteration
            current_err = self.quantization_error(inputs)
            self.errors.append(current_err)
            self.output_centroids.append(self.centroids)

            if self.is_converged(old_centroids, self.centroids):
                break

        self.trained = True
        return self.errors, self.output_centroids

    def is_converged(self, old_centroids, curr_centroids):
        distances = [self.distance.measure(old_centroids[i, 0], curr_centroids[i, 0], axis=0) for i in range(self.num_clusters)]
        return (np.sum(distances) == 0)

    def quantization_error(self, x):
        norm = np.linalg.norm(x - self.centroids, axis=1)
        return np.sum(norm**2)

    def db_index(self, inputs):
        mini_delta = np.zeros(self.num_clusters)
        big_delta = np.zeros((self.num_clusters, self.num_clusters))

        for ki in range(self.num_clusters):
            dif = inputs[self.clusters[ki]] - self.centroids[ki][0]
            norm = np.linalg.norm(dif, axis=1, keepdims=True)
            mini_delta[ki] = (np.mean(norm, axis=1, keepdims=True))[0]

            for kj in range(self.num_clusters):
                if ki != kj:
                    big_delta[ki, kj] = EuclideanDistance().measure(self.centroids[ki][0], self.centroids[kj][0], axis=0)

        db = 0
        for ki in range(self.num_clusters):
            max = 0

            for kj in range(self.num_clusters):
                if ki != kj:
                    current = (mini_delta[ki] + mini_delta[kj])/big_delta[ki, kj]
                    if current > max:
                        max = current
            db += max
        return db/self.num_clusters
