class Func:
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        pass

    def grad(self, x):
        pass

class Identity(Func):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x):
        return x

    def grad(self, x):
        return 1
