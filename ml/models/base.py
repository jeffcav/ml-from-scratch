from abc import abstractmethod

class AbstractModel:
    def __init__(self, params=None):
        self.params = params
        self.trained = False

    @abstractmethod
    def fit(self, inputs, outputs):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def get_params(self):
        return self.params
