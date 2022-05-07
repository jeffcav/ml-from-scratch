import numpy as np

from .functions.metrics.regression import RMSE

class Kfold:
    def __init__(self, num_folds) -> None:
        self.num_folds = num_folds

    def split(self, num_samples, shuffle=True):
        """
        Returns train and test indexes for each fold.
        Tries to evenly distributes samples across folds:
            num_samples % self.num_folds folds contain num_samples // self.num_folds + 1 samples,
            other contain num_samples // self.num_folds
        """
        if shuffle:
            idx = np.random.permutation(num_samples)
        else:
            idx = np.arange(num_samples)

        fold_len = num_samples // self.num_folds
        num_bigger_folds = num_samples % self.num_folds
        
        begin = 0
        for i in range(self.num_folds):
            end = begin + fold_len + (1 if i < num_bigger_folds else 0)
            
            yield np.delete(idx, np.arange(begin, end)), np.arange(begin, end)
            
            begin = end

class GridSearchCV:
    MODEL_IDX = 0
    MODEL_ARGS_IDX = 1

    def __init__(self, stats_generator=None) -> None:
        self.candidates = []

        self.stats_generator = stats_generator
        self.stats = []

    def add(self, model, model_args):
        self.candidates.append((model, model_args))

        if self.stats_generator is not None:
            self.stats.append(self.stats_generator())

    def search(self, inputs, outputs, num_folds=5, score=RMSE(), score_minimize=True, shuffle=True):
        model_cost = np.full(len(self.candidates), 0.0)

        kfold = Kfold(num_folds)
        for train_idx, test_idx in kfold.split(inputs.shape[0], shuffle):
            X_train = inputs[train_idx]
            Y_train = outputs[train_idx]

            X_test = inputs[test_idx]
            Y_test = outputs[test_idx]

            for i in range(len(self.candidates)):
                candidate = self.candidates[i]

                model = candidate[GridSearchCV.MODEL_IDX]
                model_args = candidate[GridSearchCV.MODEL_ARGS_IDX]

                m = model(**model_args)
                m.fit(X_train, Y_train)

                Y_predicted = m.predict(X_test)
                model_cost[i] += score.measure(Y_test, Y_predicted)

                if self.stats_generator is not None:
                    self.stats[i].add(Y_test, Y_predicted)

        mean_cost = model_cost / num_folds
        if score_minimize:
            self.index_of_best = np.argmin(mean_cost)
        else:
            self.index_of_best = np.argmax(mean_cost)

        return self.candidates[self.index_of_best]
