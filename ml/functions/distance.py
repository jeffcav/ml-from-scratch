import numpy as np

class EuclideanDistance():
    def measure(self, a, b):
        return np.sqrt(((a-b)**2).sum(axis=1))

class MahalanobisDistance():
    def measure(self, x_new, x):
        cov = np.linalg.pinv(np.cov(x, rowvar=False))
        dif = x_new-x
        dists = np.sqrt(np.diagonal(dif@(cov@dif.T)))

        return dists
