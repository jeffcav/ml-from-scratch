import math
import numpy as np

def load_csv(csv_filename, no_output=False, delimiter=',', remove_duplicates=False, remove_inconsistent_labels=False):
    ds = np.genfromtxt(csv_filename, delimiter=delimiter)

    # returns X=ds,y=None
    if no_output:
        return ds, None

    # remove similar samples
    if remove_duplicates:
        ds = np.unique(ds, axis=0)

    X = ds[:,0:-1]
    y = ds[:,-1]
    y = y.reshape((y.shape[0], 1))

    # remove cases where features are similar but labels differ
    if remove_inconsistent_labels:
        X, idx = np.unique(ds, axis=0, return_index=True)
        y = y[idx]

    return X,y

def split_train_test(X, y, train_percentage, shuffle=True):
    if (shuffle):
        permutations = np.random.permutation(X.shape[0])
        X = X[permutations]
        y = y[permutations]

    mark = math.floor(X.shape[0]*train_percentage)

    X_train = X[0:mark,:]
    y_train = y[0:mark]
    X_test = X[mark:,:]
    y_test = y[mark:]

    return X_train, y_train, X_test, y_test
