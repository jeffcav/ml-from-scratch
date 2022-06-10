import numpy as np

from .abstract import AbstractMeasure

class BinaryAccuracy(AbstractMeasure):
    def measure(self, y_truth, y_estimated):
        y_estimated = np.round(y_estimated)
        return np.mean(np.round(y_truth) == np.round(y_estimated))

class ClassificationError(AbstractMeasure):
    def measure(self, y_truth, y_estimated):
        return np.mean(np.round(y_truth) != np.round(y_estimated))

class Recall(AbstractMeasure):
    def measure(self, y_truth, y_estimated):
        y_estimated = np.round(y_estimated)

        true_positives = np.sum(np.logical_and((y_estimated == 1), (y_truth == 1)))
        false_negatives = np.sum(np.logical_and((y_estimated == 0), (y_truth == 1)))

        p = true_positives + false_negatives
        if p > 1.0e-7:
            return true_positives / p
        return 0.0


class Precision(AbstractMeasure):
    def measure(self, y_truth, y_estimated):
        y_estimated = np.round(y_estimated)

        true_positives = np.sum(np.logical_and((y_estimated == 1), (y_truth == 1)))
        false_positives = np.sum(np.logical_and((y_estimated == 1), (y_truth == 0)))

        pp = true_positives + false_positives
        if pp > 1.0e-7:
            return true_positives / pp
        return 0.0


class F1Score(AbstractMeasure):
    def measure(self, y_truth, y_estimated):
        y_estimated = np.round(y_estimated)

        precision = Precision().measure(y_truth, y_estimated)
        recall = Recall().measure(y_truth, y_estimated)

        if precision+recall > 1.0e-7:
            return 2 * ((precision * recall) / (precision + recall))
        return 0.0
