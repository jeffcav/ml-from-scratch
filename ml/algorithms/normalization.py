
class MinMaxScaler():
    def __init__(self):
        self.max = 1
        self.min = 0

    def fit(self, x):
        self.max = x.max(axis=0)
        self.min = x.min(axis=0)

    def transform(self, x):
        return (x - self.min) / (self.max - self.min)

    def inverse_transform(self, x):
        return (x * (self.max - self.min)) + self.min
