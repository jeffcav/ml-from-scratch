import numpy as np

from .abstract import AbstractMeasure

class BinaryAccuracy(AbstractMeasure):
    def measure(self, a, b):
        return np.mean(np.round(a) == np.round(b))

class F1Score(AbstractMeasure):
    def measure(self, y_truth, y_estimated):
        true_positives = np.sum(np.logical_and((y_estimated == 1), (y_truth == 1)))
        false_positives = np.sum(np.logical_and((y_estimated == 1), (y_truth == 0)))
        false_negatives = np.sum(np.logical_and((y_estimated == 0), (y_truth == 1)))

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        f1 = 2 * ((precision * recall) / (precision + recall))
        return f1
