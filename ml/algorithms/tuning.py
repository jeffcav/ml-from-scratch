from ml.stats import rmse

class GridSearch:
    def __init__(self) -> None:
        self.entries = []

    def add(self, model, model_args):
        self.entries.append((model, model_args))

    def search(self, X_train, y_train, X_cv, y_cv, num_folds):
        for entry in self.entries:
            model = entry[0]
            model_args = entry[1]

            m = model(**model_args)
            m.fit(X_train, y_train)

            y_hat = m.predict(X_cv)
            cv_rmse = rmse(y_cv, y_hat)
            print(cv_rmse)
