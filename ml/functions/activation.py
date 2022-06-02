import numpy as np

class Identity:
    def __call__(self, x):
        return x
    
    def grad(self, x):
        return 1

class Sigmoid:
    def __call__(self, x):
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig
    
    def grad(self, x):
        sig = self.__call__(x)
        return sig - (sig*sig)

class Tanh:
    def __call__(self, x):
        z = np.exp(2*x)
        return (z - 1) / (z + 1)

    def grad(self, x):
        t = self.__call__(x)
        return 1 - (t*t)

class Relu:
    def __call__(self, x):
        # I found this is faster
        # than using np.maximum
        return (abs(x) + x) / 2
