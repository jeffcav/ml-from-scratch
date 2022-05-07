from abc import abstractmethod

class AbstractMeasure:
    @abstractmethod
    def measure(self, y1, y2):
        pass
