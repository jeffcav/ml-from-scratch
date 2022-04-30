import numpy as np
from ml.stats import rmse

class GridSearch:
    def __init__(self) -> None:
        self.entries = []

    def add(self, model, model_args):
        self.entries.append((model, model_args))

    def make_dataset(self, k_index, cv_len, X, Y):
        """
        This function expects that all K-folds have same length
        """

        if (k_index * cv_len) > X.shape[0]:
            raise ValueError("K-folds are not of same length")
        
        cv_begin = (k_index * cv_len)
        cv_frontier = cv_begin + cv_len
        
        needs_permutation = ((k_index * cv_len) + cv_len) < X.shape[0]
        if needs_permutation:
            permutations = np.arange(X.shape[0])
        
            # put current fold to the end
            tmp = permutations[cv_begin:cv_begin+cv_len]
            permutations[cv_begin:cv_begin+cv_len] = permutations[-cv_len:]
            permutations[-cv_len:] = tmp

            X = X[permutations]
            Y = Y[permutations]
        
        return X[0:cv_frontier,:], Y[0:cv_frontier,:], X[cv_frontier:-1,:], Y[cv_frontier:-1,:]

    def search(self, X_train, y_train, num_folds):
        for i in range(len(self.entries)):
            model = self.entries[i][0]
            model_args = self.entries[i][1]

            m = model(**model_args)
            m.fit(X_train, y_train)

            y_hat = m.predict(X_cv)
            cv_rmse = rmse(y_cv, y_hat)
            print(cv_rmse)
