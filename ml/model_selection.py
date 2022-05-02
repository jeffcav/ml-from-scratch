from wsgiref.handlers import format_date_time
import numpy as np
from ml.stats import rmse

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

class GridSearch:
    def __init__(self) -> None:
        self.entries = []

    def add(self, model, model_args):
        self.entries.append((model, model_args))

    def search(self, X_train, y_train, num_folds):
        for i in range(len(self.entries)):
            model = self.entries[i][0]
            model_args = self.entries[i][1]

            m = model(**model_args)
            m.fit(X_train, y_train)

            y_hat = m.predict(X_cv)
            cv_rmse = rmse(y_cv, y_hat)
            print(cv_rmse)
