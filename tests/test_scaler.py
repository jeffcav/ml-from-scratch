import numpy as np
from ml.algorithms.normalization import MinMaxScaler

def test_min_max_scaler():
    x = np.array([
        [3.0, 10.0],
        [6.0, 20.0], 
        [9.0, 30.0]
    ])

    scaler = MinMaxScaler()
    scaler.fit(x)

    x_scaled = scaler.transform(x)
    x_unscaled = scaler.inverse_transform(x_scaled)

    assert np.array_equiv(scaler.transform(x[0]), np.array([0.0, 0.0])) == True
    assert np.array_equiv(scaler.transform(x[1]), np.array([0.5, 0.5])) == True
    assert np.array_equiv(scaler.transform(x[2]), np.array([1.0, 1.0])) == True

    assert np.array_equiv(x, x_unscaled) == True, x_unscaled

