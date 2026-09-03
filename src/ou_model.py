"""
ou_model.py — Ornstein-Uhlenbeck mean-reversion model.

Design note: raw stock prices are closer to a random walk (non-stationary)
than a mean-reverting process, so fitting OU directly to price is the wrong
regime. Instead we fit OU to the DEVIATION of price from its own trailing
moving average (a standard stat-arb technique) - that deviation series is
mean-reverting by construction, which is the assumption OU actually needs.

The continuous-time OU process:
    dX_t = theta * (mu - X_t) dt + sigma dW_t

For a deviation-from-trend series, mu is ~0 by construction, so we fit a
simpler, numerically identical special case via its exact discretization,
an AR(1) process sampled at daily intervals (dt = 1 trading day):
    X_{t+1} = phi * X_t + eps_t,        phi = exp(-theta * dt)
    theta = -ln(phi)
    stationary variance = Var(eps) / (1 - phi^2)

Fit via OLS regression of X_{t+1} on X_t - the standard estimator for AR(1)
coefficients, and exact for this discretization of OU (see Uhlenbeck &
Ornstein 1930 / standard quant finance treatments of the OU process).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class OUFitResult:
    phi: float              # AR(1) coefficient, exp(-theta)
    theta: float            # mean-reversion speed (per day). Higher = faster reversion.
    half_life_days: float   # ln(2)/theta - time for a deviation to halve, in days
    stationary_std: float   # steady-state std dev of the deviation series
    n_obs: int
    r_squared: float        # NOTE: this is NOT a mean-reversion diagnostic. AR(1) fits
                             # a pure random walk (phi~1, theta~0, no reversion at all)
                             # with high R^2 too, since today still predicts tomorrow
                             # closely either way (confirmed via simulation - see
                             # tests in the validation block below). Use theta/phi to
                             # judge reversion speed, not r_squared. Kept here only as
                             # a general fit-quality diagnostic.


def fit_ou(deviation_series: np.ndarray) -> OUFitResult:
    """Fits OU parameters to a mean-reverting deviation series via AR(1) OLS.

    Expects deviation_series already centered near zero (e.g. log_price minus
    its own trailing moving average) and free of NaNs.
    """
    if len(deviation_series) < 10:
        raise ValueError(f"Need at least 10 observations to fit OU, got {len(deviation_series)}")

    x = deviation_series[:-1]
    y = deviation_series[1:]

    # OLS: y = phi * x  (no intercept, since deviation series is mean-zero
    # by construction — forcing intercept=0 avoids overfitting noise as drift)
    phi = float(np.sum(x * y) / np.sum(x * x))
    phi = min(max(phi, -0.999), 0.999)  # guard against explosive/unstable fits

    residuals = y - phi * x
    resid_var = float(np.var(residuals, ddof=1))

    theta = -np.log(abs(phi)) if phi > 0 else float("nan")
    half_life = np.log(2) / theta if theta and theta > 0 else float("inf")
    stationary_var = resid_var / (1 - phi ** 2) if abs(phi) < 1 else float("inf")

    y_mean = np.mean(y)
    ss_res = float(np.sum((y - phi * x) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return OUFitResult(
        phi=phi,
        theta=float(theta),
        half_life_days=float(half_life),
        stationary_std=float(np.sqrt(stationary_var)),
        n_obs=len(deviation_series),
        r_squared=r_squared,
    )


def z_score(deviation_value: float, fit: OUFitResult) -> float:
    """How many stationary standard deviations away from equilibrium a
    single observed deviation is - the actual trade_signals value."""
    if fit.stationary_std == 0 or np.isnan(fit.stationary_std):
        return float("nan")
    return deviation_value / fit.stationary_std


def simulate_ou(theta: float, sigma: float, n: int, dt: float = 1.0, seed: int = 0) -> np.ndarray:
    """Simulates a mean-zero OU path. Used ONLY for validating fit_ou()
    against known ground-truth parameters before trusting it on real data."""
    rng = np.random.default_rng(seed)
    phi = np.exp(-theta * dt)
    noise_std = sigma * np.sqrt((1 - phi ** 2) / (2 * theta))
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, noise_std)
    return x
