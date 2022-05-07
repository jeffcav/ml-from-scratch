import numpy as np

from .functions.metrics.classification import BinaryAccuracy, F1Score, Precision, Recall

class ClassificationStats:
    def __init__(self) -> None:
        self.n = 0

        self.stats = {
            "values": {
                "f1": [],
                "recall": [],
                "precision": [],
                "binary_accuracy": []
            }
        }

    def update_values(self, expected, predicted):
        self.stats["values"]["f1"].append(
            F1Score().measure(expected, np.round(predicted))
        )

        self.stats["values"]["recall"].append(
            Recall().measure(expected, np.round(predicted))
        )

        self.stats["values"]["precision"].append(
            Precision().measure(expected, np.round(predicted))
        )

        self.stats["values"]["binary_accuracy"].append(
            BinaryAccuracy().measure(expected, np.round(predicted))
        )

    def add(self, expected, predicted):
        self.update_values(expected, predicted)
