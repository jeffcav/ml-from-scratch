import numpy as np

from .abstract import AbstractMeasure

class BinaryAccuracy(AbstractMeasure):
    def measure(self, a, b):
        return np.mean(np.round(a) == np.round(b))
