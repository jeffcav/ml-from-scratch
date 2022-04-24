import numpy as np

def mse(a, b):
    return ((a-b)**2).mean()

def rmse(a, b):
    return np.sqrt(((a-b)**2).mean())

class Stats:
    def __init__(self) -> None:
        self._mse = []
        self._rmse = []

    def add(self, output, prediction):
        self._mse.appen(mse(output, prediction))
        self._rmse.appen(rmse(output, prediction))
