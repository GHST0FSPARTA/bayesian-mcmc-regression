# model.py
# Bayesian regression model — likelihood, prior, posterior
#
# All functions work in log space to avoid floating point underflow.
# Products of probabilities become sums of log probabilities.
#
# Bayes theorem:
#   p(beta | X, y) ∝ p(y | X, beta) * p(beta)
#
# The normalising constant p(y|X) is intractable — it requires
# integrating over all possible beta values. It is never computed
# here because it cancels in the MH acceptance ratio.

import numpy as np


def log_likelihood(beta, X, y, sigma):
    """
    Log likelihood of observed data under Gaussian noise assumption.

    Each observation is modelled as:
        y_i = X_i @ beta + epsilon_i,   epsilon_i ~ N(0, sigma^2)

    Which means:
        y_i | beta ~ N(X_i @ beta, sigma^2)

    The Gaussian noise assumption is justified by the Central Limit
    Theorem — real noise is the sum of many small independent sources
    which converges to Gaussian regardless of individual distributions.

    Maximising log likelihood is equivalent to minimising sum of
    squared residuals — OLS is maximum likelihood estimation under
    this Gaussian noise model.

    Parameters
    ----------
    beta  : ndarray, shape (p,)
        Parameter vector — intercept and regression coefficients.
    X     : ndarray, shape (n, p)
        Design matrix — columns are basis functions of input x.
    y     : ndarray, shape (n,)
        Observed outputs.
    sigma : float
        Noise standard deviation. Treated as known — estimated from
        OLS residuals before running MCMC.

    Returns
    -------
    float
        Log p(y | X, beta, sigma)
    """
    residuals = y - X @ beta
    n         = len(y)
    return (
        - (n / 2) * np.log(2 * np.pi * sigma**2)
        - (1 / (2 * sigma**2)) * np.sum(residuals**2)
    )


def log_prior(beta, tau=10.0):
    """
    Log prior over parameters — independent Gaussian on each beta_j.

        beta_j ~ N(0, tau^2)   for each j

    tau controls how strongly the prior constrains parameters:
        tau = 0.5  : tight prior, parameters near zero expected
        tau = 10.0 : wide prior, data dominates almost entirely
        tau = 100  : essentially flat, no prior information

    With tau = 10, the prior assigns 99.7% of probability mass to
    beta_j in [-30, 30] — wide enough to not exclude any physically
    reasonable slope for standardised data.

    Parameters
    ----------
    beta : ndarray, shape (p,)
        Parameter vector.
    tau  : float
        Prior standard deviation. Default 10.0 — weakly informative.

    Returns
    -------
    float
        Log p(beta)
    """
    return -0.5 * np.sum(
        np.log(2 * np.pi * tau**2) + (beta**2 / tau**2)
    )


def log_posterior(beta, X, y, sigma, tau=10.0):
    """
    Log posterior — proportional to likelihood times prior.

        log p(beta | X, y) = log p(y | X, beta) + log p(beta) + const

    The constant is log(1/p(y|X)) — the intractable normalising term.
    It does not depend on beta so it cancels in the MH acceptance ratio:

        alpha = p(beta* | X, y) / p(beta_t | X, y)
              = [p(y|X,beta*) * p(beta*) / p(y|X)]
                / [p(y|X,beta_t) * p(beta_t) / p(y|X)]
              = p(y|X,beta*) * p(beta*) / p(y|X,beta_t) * p(beta_t)

    p(y|X) cancels exactly — never computed.

    Parameters
    ----------
    beta  : ndarray, shape (p,)
    X     : ndarray, shape (n, p)
    y     : ndarray, shape (n,)
    sigma : float
    tau   : float

    Returns
    -------
    float
        log p(beta | X, y)  up to an additive constant
    """
    return log_likelihood(beta, X, y, sigma) + log_prior(beta, tau)
