# Bayesian Regression via Metropolis-Hastings MCMC

Bayesian linear, polynomial and spline regression implemented from scratch in Python using only NumPy — posterior distributions over parameters, predictive uncertainty intervals, and convergence diagnostics on synthetic and real data.

---

## The Problem

Classical regression finds one best answer. OLS gives a single parameter estimate with no way to express uncertainty. A different sample of data would give different parameters — but OLS has no mechanism to capture that variability.

## The Solution

Bayesian regression treats parameters as random variables with probability distributions. Bayes theorem updates prior beliefs about parameters using observed data to produce a posterior distribution:

```
p(β | X, y) ∝ p(y | X, β) · p(β)
```

The posterior distribution over β gives:
- The most probable parameter values
- Credible intervals — where parameters plausibly lie
- Predictive uncertainty — honest confidence bands on predictions

## The Challenge

The denominator of Bayes theorem requires integrating over all possible β — analytically intractable in general. Metropolis-Hastings MCMC bypasses this by constructing a Markov chain whose stationary distribution is exactly the posterior. The intractable denominator cancels in the acceptance ratio and is never computed.

---

## Key Design — Model Agnostic Sampler

The sampler never knows what model it is fitting. The model is entirely encoded in the design matrix X:

```python
# same sampler, different model — just change X
samples = fit_bayesian_regression(x, y, model='linear')
samples = fit_bayesian_regression(x, y, model='polynomial', degree=5)
samples = fit_bayesian_regression(x, y, model='spline')
```

| Model | Design matrix X | Parameters |
|---|---|---|
| Linear | [1, x] | 2 |
| Polynomial degree p | [1, x, x², ..., xᵖ] | p+1 |
| Cubic spline (k knots) | [1, x, x², x³, (x-κ₁)₊³, ...] | 4+k |

All models are linear in β — same likelihood, same prior, same sampler.

---

## Implementation

```
bayesian-mcmc-regression/
│
├── main.py          ← user entry point — fit_bayesian_regression()
├── mcmc.py          ← MH sampler, ACF, ESS, Gelman-Rubin R-hat
├── model.py         ← log likelihood, log prior, log posterior
├── basis.py         ← design matrix constructors
└── requirements.txt
```

### Metropolis-Hastings Algorithm

```
1. Start at β⁽⁰⁾ = OLS estimate
2. Propose β* = β⁽ᵗ⁾ + ε,  ε ~ N(0, s²I)
3. Compute log α = log p(β*|X,y) − log p(β⁽ᵗ⁾|X,y)
4. Draw u ~ Uniform(0,1)
5. If log(u) < log α: β⁽ᵗ⁺¹⁾ = β*   (accept)
   Else:              β⁽ᵗ⁺¹⁾ = β⁽ᵗ⁾  (stay)
6. Store β⁽ᵗ⁺¹⁾ regardless — rejected samples encode probability 1−α
7. Repeat 15,000 times
```

The intractable p(y|X) cancels in step 3 — it appears identically in numerator and denominator of the acceptance ratio.

### Model

**Likelihood** — Gaussian noise assumption justified by the Central Limit Theorem:
```
y_i | β ~ N(X_i β, σ²)
```

**Prior** — weakly informative Gaussian, data dominated:
```
β_j ~ N(0, τ²),  τ = 10
```

**Posterior** — proportional to likelihood × prior:
```
p(β | X, y) ∝ p(y | X, β) · p(β)
```

---

## Usage

```python
import numpy as np
from main import fit_bayesian_regression

# your data
x = np.linspace(0, 10, 100)
y = np.sin(x)*2 + np.random.normal(0, 0.5, 100)

# fit — returns posterior samples shape (n_samples, n_params)
samples = fit_bayesian_regression(
    x, y,
    model     = 'spline',    # 'linear', 'polynomial', 'spline'
    n_samples = 15000,
    step_size = 0.001,       # tune for acceptance rate 0.2-0.5
    plot      = True
)

# posterior samples — one row per iteration, one column per parameter
print(samples.shape)               # (14500, 8) for spline with 4 knots
print(samples[:, 1].mean())        # posterior mean of β₁
print(np.percentile(samples[:,1], [2.5, 97.5]))  # 95% credible interval
```

---

## Convergence Diagnostics

Three diagnostics verify the chain has converged to the posterior:

**Trace plot** — chain should hover stably around a fixed value after burn-in. Drifting or trending indicates non-convergence.

**Autocorrelation (ACF)** — measures correlation between samples k steps apart. Should drop below 0.05 within 10-30 lags. Slow decay means high autocorrelation — fewer effective independent samples than raw count suggests.

**Gelman-Rubin R-hat** — runs multiple chains from different starting points. R-hat < 1.01 confirms all chains converged to the same distribution.

```
R-hat < 1.01  ✓  converged
R-hat > 1.1   ✗  not converged — run longer or tune step size
```

---

## Results

On synthetic nonlinear data (sine wave + linear trend, n=100, σ=0.5):

| Model | R² | Parameters | Acceptance Rate |
|---|---|---|---|
| Linear | 0.61 | 2 | 0.33 |
| Polynomial (degree 5) | 0.93 | 6 | 0.28 |
| Spline (4 knots) | 0.92 | 8 | 0.24 |

On California Housing real data (n=200, predicting house value from income):
- β₁ posterior mean = 0.724, 95% CI = [0.610, 0.839]
- Posterior mean and OLS agree to within 0.002 — expected with flat prior and 200 data points

---

## Installation

```bash
git clone https://github.com/yourusername/bayesian-mcmc-regression
cd bayesian-mcmc-regression
pip install -r requirements.txt
python main.py
```

---

## No Probabilistic Libraries

Everything is implemented from scratch using only NumPy:
- Gaussian log-likelihood computed manually
- MH sampler written from first principles  
- ACF, ESS and R-hat implemented directly
- No PyMC, Stan, TensorFlow Probability, or scipy.stats

---

## References

- Gelman et al., *Bayesian Data Analysis* (3rd ed.)
- Roberts & Rosenthal (1997) — optimal acceptance rate 0.234 for Gaussian targets
- Evans, *An Introduction to Stochastic Differential Equations* — mathematical foundations

Built with AI assistance as a learning project. Every mathematical decision and algorithm step is understood and can be explained in detail.
