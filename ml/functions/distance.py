import numpy as np
class EuclideanDistance():
    def __init__(self) -> None:
        pass
    def measure(self, a, b):
        return np.sqrt(((a-b)**2).sum(axis=1))
