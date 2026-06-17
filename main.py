# main.py
# Bayesian Regression via Metropolis-Hastings MCMC
#
# Single entry point — user provides data and model choice.
# The sampler, likelihood, prior and posterior are model-agnostic.
# The model is entirely encoded in the design matrix X.
#
# Usage:
#   from main import fit_bayesian_regression
#   samples = fit_bayesian_regression(x, y, model='spline')

import numpy as np
import matplotlib.pyplot as plt

from model import log_posterior
from mcmc  import metropolis_hastings, compute_acf, compute_ess, gelman_rubin
from basis import build_linear, build_polynomial, build_spline, auto_knots


def fit_bayesian_regression(x, y,
                             model='linear',
                             degree=3,
                             knots=None,
                             n_samples=15000,
                             step_size=0.01,
                             burn_in=500,
                             tau=10.0,
                             plot=True):
    """
    Fit Bayesian regression to (x, y) data using MH-MCMC.

    The model is specified entirely by the design matrix:
        linear     : X = [1, x]
        polynomial : X = [1, x, x^2, ..., x^degree]
        spline     : X = [1, x, x^2, x^3, (x-k1)+^3, ...]

    The same sampler handles all models — passing a different model
    string changes X, not the inference machinery.

    Parameters
    ----------
    x         : ndarray, shape (n,)
        Input data. Recommend standardising: (x - x.mean()) / x.std()
    y         : ndarray, shape (n,)
        Output data.
    model     : str
        'linear', 'polynomial', or 'spline'.
    degree    : int
        Polynomial degree. Used only when model='polynomial'.
    knots     : list of float, optional
        Knot positions. Used only when model='spline'.
        If None, placed automatically at evenly spaced quantiles.
    n_samples : int
        Total MCMC iterations including burn-in.
    step_size : float
        Proposal standard deviation. Tune for acceptance rate 0.2-0.5.
    burn_in   : int
        Samples to discard before chain has converged.
        Recommended: initialise at OLS to minimise burn-in needed.
    tau       : float
        Prior standard deviation. Default 10.0 — weakly informative.
    plot      : bool
        Whether to generate diagnostic plots.

    Returns
    -------
    samples_clean : ndarray, shape (n_samples - burn_in, p)
        Posterior samples over all p parameters.
        Row i is one complete plausible beta vector.
        Column j is all samples for parameter j.
    """

    # ── Step 1: Build design matrix based on model choice ─────────────────
    if model == 'linear':
        X = build_linear(x)

    elif model == 'polynomial':
        X = build_polynomial(x, degree)

    elif model == 'spline':
        if knots is None:
            knots = auto_knots(x, n_knots=4)
            print(f"Auto knots placed at: {np.round(knots, 3)}")
        X = build_spline(x, knots)

    else:
        raise ValueError(
            f"Unknown model '{model}'. "
            f"Choose from: 'linear', 'polynomial', 'spline'."
        )

    n_params = X.shape[1]
    print(f"\nModel     : {model}")
    print(f"Parameters: {n_params}  (beta_0 to beta_{n_params-1})")
    print(f"Data      : {len(x)} points")

    # ── Step 2: Estimate sigma and initialise at OLS ──────────────────────
    # Starting at OLS means chain begins in high-probability region.
    # Burn-in is minimal — chain is already near the posterior.
    beta_ols  = np.linalg.inv(X.T @ X) @ X.T @ y
    sigma_est = (y - X @ beta_ols).std()
    r2_ols    = 1 - np.sum((y - X@beta_ols)**2) / np.sum((y-y.mean())**2)

    print(f"OLS R²    : {r2_ols:.4f}")
    print(f"σ estimate: {sigma_est:.4f}")

    # ── Step 3: Run MH sampler ────────────────────────────────────────────
    print(f"\nRunning MH sampler ({n_samples} iterations)...")

    # wrap log_posterior to fix sigma and tau
    def log_post_fn(beta, X, y):
        return log_posterior(beta, X, y, sigma=sigma_est, tau=tau)

    samples, acc_rate = metropolis_hastings(
        log_posterior_fn = log_post_fn,
        X            = X,
        y            = y,
        n_samples    = n_samples,
        step_size    = step_size,
        initial_beta = beta_ols
    )

    print(f"Acceptance rate: {acc_rate:.3f}  "
          f"({'✓ healthy' if 0.2 <= acc_rate <= 0.5 else '⚠ tune step_size'})")

    # ── Step 4: Remove burn-in ────────────────────────────────────────────
    samples_clean = samples[burn_in:]
    print(f"Usable samples: {len(samples_clean)}")

    # ── Step 5: Posterior summary ─────────────────────────────────────────
    print(f"\n── Posterior Summary ─────────────────────────────────────────")
    for j in range(n_params):
        s    = samples_clean[:, j]
        ess  = compute_ess(s)
        print(f"β{j:>2}: mean={s.mean():+.4f}  std={s.std():.4f}  "
              f"95% CI=[{np.percentile(s,2.5):.4f}, {np.percentile(s,97.5):.4f}]  "
              f"ESS={ess:.0f}")

    # ── Step 6: Plots ─────────────────────────────────────────────────────
    if plot:
        _plot_results(x, y, X, samples_clean, sigma_est, model, beta_ols)

    return samples_clean


def _plot_results(x, y, X, samples_clean, sigma_est, model, beta_ols):
    """Generate diagnostic plots — fit, residuals, posteriors, ACF."""

    n_params = X.shape[1]
    x_pred   = np.linspace(x.min(), x.max(), 300)

    # build prediction matrix for x_pred
    if model == 'linear':
        X_pred = build_linear(x_pred)
    elif model == 'polynomial':
        degree = n_params - 1
        X_pred = build_polynomial(x_pred, degree)
    else:
        # spline — infer knots from column count
        # columns: 4 base + len(knots) knot terms
        # knot positions not stored here so recompute from x
        knots  = auto_knots(x, n_knots=n_params-4)
        X_pred = build_spline(x_pred, knots)

    # predicted means shape (n_samples, 300)
    pred_means  = samples_clean @ X_pred.T
    credible_lo = np.percentile(pred_means, 2.5,  axis=0)
    credible_hi = np.percentile(pred_means, 97.5, axis=0)
    mean_pred   = pred_means.mean(axis=0)

    noise       = np.random.normal(0, sigma_est, size=pred_means.shape)
    pred_obs    = pred_means + noise
    pred_lo     = np.percentile(pred_obs, 2.5,  axis=0)
    pred_hi     = np.percentile(pred_obs, 97.5, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Plot 1: Fit ───────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.fill_between(x_pred, pred_lo, pred_hi,
                    alpha=0.15, color='steelblue', label='95% prediction interval')
    ax.fill_between(x_pred, credible_lo, credible_hi,
                    alpha=0.4,  color='steelblue', label='95% credible interval')
    ax.plot(x_pred, mean_pred, 'b-', linewidth=2, label='Posterior mean')
    ax.plot(x_pred, X_pred @ beta_ols, 'g--', linewidth=1.5, label='OLS')
    ax.scatter(x, y, color='black', s=15, alpha=0.4, zorder=5, label='Data')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Bayesian {model.capitalize()} Regression')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Residuals ─────────────────────────────────────────────────
    ax       = axes[0, 1]
    beta_mean = samples_clean.mean(axis=0)
    y_fit     = X @ beta_mean
    residuals = y - y_fit
    ax.scatter(y_fit, residuals, alpha=0.5, s=20, color='darkorange')
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot (should be random scatter)')
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Posterior distributions of first two parameters ──────────
    ax = axes[1, 0]
    colors = ['steelblue', 'darkorange', 'green', 'purple',
              'red', 'brown', 'pink', 'gray']
    for j in range(min(n_params, 4)):
        ax.hist(samples_clean[:, j], bins=50, alpha=0.5,
                color=colors[j], density=True, label=f'β{j}')
        ax.axvline(beta_ols[j], color=colors[j],
                   linestyle='--', linewidth=1.5)
    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Density')
    ax.set_title('Posterior distributions\n(dashed = OLS estimate)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot 4: ACF for first parameter ──────────────────────────────────
    ax      = axes[1, 1]
    acf     = compute_acf(samples_clean[:, 0], max_lag=50)
    lags    = np.arange(len(acf))
    ax.bar(lags, acf, color='steelblue', alpha=0.7, width=0.8)
    ax.axhline( 0.05, color='red', linestyle='--',
                linewidth=1, label='±0.05 threshold')
    ax.axhline(-0.05, color='red', linestyle='--', linewidth=1)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('ACF — β₀ (intercept)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Bayesian {model.capitalize()} Regression — MH-MCMC',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(f'results_{model}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: results_{model}.png")


# ── Demo — runs when file is executed directly ────────────────────────────
if __name__ == '__main__':

    np.random.seed(42)

    # generate nonlinear data
    n = 100
    x = np.linspace(0, 10, n)
    y = np.sin(x)*2 + 0.3*x + np.random.normal(0, 0.5, n)

    print("=" * 60)
    print("LINEAR MODEL")
    print("=" * 60)
    fit_bayesian_regression(x, y, model='linear', step_size=0.05)

    print("\n" + "=" * 60)
    print("POLYNOMIAL MODEL (degree 5)")
    print("=" * 60)
    fit_bayesian_regression(x, y, model='polynomial',
                             degree=5, step_size=0.001)

    print("\n" + "=" * 60)
    print("SPLINE MODEL (4 auto knots)")
    print("=" * 60)
    fit_bayesian_regression(x, y, model='spline', step_size=0.001)
