import pytest
import numpy as np
from ml.models.linear import LinearRegression, PolynomialRegression
from ml.algorithms.optimization import GradientDescent, OrdinaryLeastSquares, StochasticGradientDescent

@pytest.mark.parametrize(
    "solver", [
        OrdinaryLeastSquares(),
        GradientDescent(5000, 0.05, 0),
        StochasticGradientDescent(5000, 0.05, 0)]
)
def test_linear_regression(solver):
    expected = np.array([[2.0, 3.0]])

    x = np.array([
        [3.0], 
        [6.0], 
        [7.0]
    ])
    y = np.array([
        [11.0], 
        [20.0], 
        [23.0]
    ])

    linreg = LinearRegression(solver)
    linreg.fit(x, y)

    assert linreg.get_weights().shape == expected.shape
    assert np.allclose(linreg.get_weights(), expected) == True

@pytest.mark.parametrize(
    "solver", [
        OrdinaryLeastSquares(),
        GradientDescent(5000, 0.05, 0),
        StochasticGradientDescent(5000, 0.05, 0)]
)
def test_polynomial_regression(solver):
    expected = np.array([[2.0, 3.0]])

    x = np.array([
        [3.0], 
        [6.0], 
        [7.0]
    ])
    y = np.array([
        [11.0], 
        [20.0], 
        [23.0]
    ])

    polyreg = PolynomialRegression(solver, 1)
    polyreg.fit(x, y)

    assert polyreg.get_weights().shape == expected.shape
    assert np.allclose(polyreg.get_weights(), expected) == True

@pytest.mark.parametrize(
    "solver", [OrdinaryLeastSquares()]
)
def test_polynomial_regression_with_degree_2(solver):
    expected = np.array([[0.5, 0.3, 0.1]])

    x = np.array([
        [2.0], 
        [3.0],
        [5.0]
    ])
    y = np.array([
        [1.5], 
        [2.3],
        [4.5]
    ])

    polyreg = PolynomialRegression(solver, 2)
    polyreg.fit(x, y)

    assert polyreg.get_weights().shape == expected.shape
    assert np.allclose(polyreg.get_weights(), expected) == True
