# mcmc.py
# Metropolis-Hastings MCMC sampler and convergence diagnostics.
#
# Core algorithm:
#   Constructs a Markov chain whose stationary distribution is the
#   posterior p(beta | X, y). Running the chain produces samples
#   from the posterior without ever computing the intractable
#   normalising constant p(y|X).
#
# Why it works:
#   The acceptance ratio alpha = p(beta*|X,y) / p(beta_t|X,y)
#   causes p(y|X) to cancel exactly. The chain satisfies detailed
#   balance — a sufficient condition for the stationary distribution
#   to be the posterior.

import numpy as np


def metropolis_hastings(log_posterior_fn, X, y,
                        n_samples=15000,
                        step_size=0.01,
                        initial_beta=None):
    """
    Metropolis-Hastings MCMC sampler.

    At each iteration:
        1. Propose beta* = beta_current + N(0, step_size^2 * I)
           — local Gaussian random walk, symmetric proposal
           — symmetry ensures q(beta*|beta_t) = q(beta_t|beta*)
           — this cancels in the acceptance ratio
        2. Compute log_alpha = log_post(beta*) - log_post(beta_t)
        3. Accept with probability alpha = exp(log_alpha):
           — if alpha >= 1 (uphill): always accept
           — if alpha < 1 (downhill): accept with probability alpha
           — implemented as: accept if log(u) < log_alpha, u~Uniform(0,1)
           — P(log(u) < log_alpha) = P(u < alpha) = alpha exactly
        4. Store beta_current regardless of accept/reject
           — rejected samples are stored intentionally
           — repetition of current position encodes probability 1-alpha
           — this makes histogram of samples proportional to posterior

    Step size tuning:
        Too small (rate > 0.5): chain barely moves, high autocorrelation
        Too large (rate < 0.2): proposals always rejected, chain stuck
        Target: acceptance rate between 0.2 and 0.5

    Parameters
    ----------
    log_posterior_fn : callable
        Function (beta, X, y) -> float. Log posterior up to a constant.
    X            : ndarray, shape (n, p)
        Design matrix.
    y            : ndarray, shape (n,)
        Observed outputs.
    n_samples    : int
        Total number of MCMC iterations including burn-in.
    step_size    : float
        Standard deviation of Gaussian proposal.
    initial_beta : ndarray, shape (p,), optional
        Starting point. Defaults to zeros.
        Recommended: initialise at OLS estimate to minimise burn-in.

    Returns
    -------
    samples : ndarray, shape (n_samples, p)
        All sampled beta vectors. Discard first burn_in rows before use.
    acceptance_rate : float
        Fraction of proposals accepted. Target: 0.2 to 0.5.
    """

    # ── Initialise ────────────────────────────────────────────────────────
    if initial_beta is None:
        initial_beta = np.zeros(X.shape[1])

    n_params         = len(initial_beta)
    samples          = np.zeros((n_samples, n_params))
    beta_current     = initial_beta.copy()
    log_post_current = log_posterior_fn(beta_current, X, y)
    n_accepted       = 0

    # ── Main loop ─────────────────────────────────────────────────────────
    for t in range(n_samples):

        # propose new beta — local Gaussian random walk
        # size inferred from beta dimension — works for any p
        proposal = beta_current + np.random.normal(0, step_size,
                                                    size=n_params)

        # evaluate log posterior at proposal
        log_post_proposal = log_posterior_fn(proposal, X, y)

        # log acceptance ratio — p(y|X) cancels here
        log_alpha = log_post_proposal - log_post_current

        # accept with probability alpha
        # log(u) < log(alpha) iff u < alpha, P(u < alpha) = alpha
        log_u = np.log(np.random.uniform(0, 1))

        if log_u < log_alpha:
            beta_current     = proposal
            log_post_current = log_post_proposal
            n_accepted      += 1

        # store current position — rejected samples kept intentionally
        samples[t] = beta_current

    acceptance_rate = n_accepted / n_samples
    return samples, acceptance_rate


def compute_acf(chain, max_lag=50):
    """
    Autocorrelation function up to max_lag.

    ACF at lag k measures correlation between sample t and sample t+k.
    Healthy chain: ACF drops below 0.05 quickly (lag 5-15).
    Slow decay indicates high autocorrelation — few effective samples.

    Parameters
    ----------
    chain   : ndarray, shape (n,)
        1D array of samples for one parameter.
    max_lag : int

    Returns
    -------
    acf : ndarray, shape (max_lag+1,)
        Autocorrelation at lags 0, 1, ..., max_lag.
        acf[0] = 1.0 always.
    """
    n    = len(chain)
    mean = chain.mean()
    var  = chain.var()
    acf  = []

    for lag in range(max_lag + 1):
        cov = np.mean((chain[:n-lag] - mean) * (chain[lag:] - mean))
        acf.append(cov / var)

    return np.array(acf)


def compute_ess(chain):
    """
    Effective sample size — number of approximately independent samples.

        ESS = N / (1 + 2 * sum(ACF[k] for k until ACF drops below 0.05))

    High autocorrelation → low ESS → fewer independent samples than
    the raw sample count suggests.

    Parameters
    ----------
    chain : ndarray, shape (n,)

    Returns
    -------
    ess : float
    """
    acf     = compute_acf(chain, max_lag=min(500, len(chain)//4))
    cutoff  = np.where(np.abs(acf) < 0.05)[0]
    cutoff  = cutoff[0] if len(cutoff) > 0 else len(acf)
    ess     = len(chain) / (1 + 2 * np.sum(acf[1:cutoff]))
    return max(ess, 1.0)


def gelman_rubin(chains):
    """
    Gelman-Rubin R-hat convergence diagnostic.

    Runs multiple chains from different starting points.
    Compares within-chain variance to between-chain variance.
    If all chains converged to the same distribution these are equal
    giving R-hat = 1.0.

    Interpretation:
        R-hat < 1.01  : converged
        R-hat < 1.05  : probably fine, run longer to confirm
        R-hat > 1.1   : not converged — do not use samples

    Parameters
    ----------
    chains : list of ndarray
        Each element is a 1D array of post-burn-in samples from one chain.

    Returns
    -------
    r_hat : float
    """
    m = len(chains)
    n = len(chains[0])

    chain_means = np.array([c.mean() for c in chains])
    grand_mean  = chain_means.mean()

    # between-chain variance B
    B = (n / (m - 1)) * np.sum((chain_means - grand_mean)**2)

    # within-chain variance W
    W = np.mean([c.var() for c in chains])

    # pooled variance estimate
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    r_hat   = np.sqrt(var_hat / W)

    return r_hat
