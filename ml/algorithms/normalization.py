
from abc import abstractmethod

class AbstractScaler():
    @abstractmethod
    def fit(self, x):
        return self

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def inverse_transform(self, x):
        pass

class MinMaxScaler(AbstractScaler):
    def __init__(self):
        self.max = 1
        self.min = 0

    def fit(self, x):
        self.max = x.max(axis=0)
        self.min = x.min(axis=0)
        return self

    def transform(self, x):
        return (x - self.min) / (self.max - self.min)

    def inverse_transform(self, x):
        return (x * (self.max - self.min)) + self.min

class IdentityScaler(AbstractScaler):
    def __init__(self):
        pass

    def fit(self, x):
        return self

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x
