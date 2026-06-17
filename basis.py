# basis.py
# Design matrix constructors — the only thing that changes between models.
#
# The key insight of this project:
#   The MH sampler is completely model-agnostic.
#   It only ever sees a design matrix X and a parameter vector beta.
#   Changing the model means changing how X is constructed — nothing else.
#
# All models are LINEAR IN BETA:
#   y = X @ beta + epsilon
#
# This means the same likelihood, prior, posterior and sampler work
# regardless of whether X contains raw x values, polynomial terms,
# or spline basis functions.

import numpy as np


def build_linear(x):
    """
    Design matrix for simple linear regression.

        y = beta_0 + beta_1 * x

    X = [1, x]   shape (n, 2)

    Parameters
    ----------
    x : ndarray, shape (n,)
        Input values.

    Returns
    -------
    X : ndarray, shape (n, 2)
        Column of ones and column of x values.
    """
    n = len(x)
    return np.column_stack([np.ones(n), x])


def build_polynomial(x, degree):
    """
    Design matrix for polynomial regression of given degree.

        y = beta_0 + beta_1*x + beta_2*x^2 + ... + beta_p*x^degree

    X = [1, x, x^2, ..., x^degree]   shape (n, degree+1)

    Despite being nonlinear in x, this model is still linear in beta —
    the same OLS formula and MH sampler apply without modification.

    Parameters
    ----------
    x      : ndarray, shape (n,)
    degree : int
        Polynomial degree. degree=1 gives linear, degree=2 quadratic etc.

    Returns
    -------
    X : ndarray, shape (n, degree+1)
    """
    return np.column_stack([x**i for i in range(degree + 1)])


def build_spline(x, knots):
    """
    Design matrix for cubic spline regression using truncated power basis.

        y = beta_0 + beta_1*x + beta_2*x^2 + beta_3*x^3
            + beta_4*(x-k1)+^3 + beta_5*(x-k2)+^3 + ...

    Where (x - k)+^3 = (x-k)^3 if x > k, else 0.

    Each knot term activates only beyond its knot position.
    This allows the curve to change shape at each knot while
    maintaining smoothness — value, first and second derivatives
    are continuous at every knot.

    X shape: (n, 4 + len(knots))

    Parameters
    ----------
    x     : ndarray, shape (n,)
    knots : list of float
        Knot positions. More knots = more flexibility.
        Recommended: place at quantiles of x — np.percentile(x, [20,40,60,80])

    Returns
    -------
    X : ndarray, shape (n, 4 + len(knots))
    """
    n    = len(x)
    cols = [np.ones(n), x, x**2, x**3]
    for k in knots:
        cols.append(np.where(x > k, (x - k)**3, 0.0))
    return np.column_stack(cols)


def auto_knots(x, n_knots=4):
    """
    Automatically place knots at evenly spaced quantiles of x.

    Parameters
    ----------
    x       : ndarray, shape (n,)
    n_knots : int
        Number of knots. More knots = more flexible curve.

    Returns
    -------
    knots : ndarray, shape (n_knots,)
    """
    percentiles = np.linspace(100/(n_knots+1), 100*n_knots/(n_knots+1), n_knots)
    return np.percentile(x, percentiles)
