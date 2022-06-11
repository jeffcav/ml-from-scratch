import numpy as np

from .abstract import AbstractMeasure

class RMSE(AbstractMeasure):
    def measure(self, a, b):
        return np.sqrt(((a-b)**2).mean())

class MSE(AbstractMeasure):
    def measure(self, a, b):
        return ((a-b)**2).mean()

class MAE(AbstractMeasure):
    def measure(self, a, b):
        return np.abs(a-b).mean()

class MRE(AbstractMeasure):
    def measure(self, a, b):
        np.abs((a-b)/a).mean()
