"""
test_ou_model.py — validates fit_ou() recovers known parameters from
simulated data BEFORE it's trusted on real price deviations.

This is not a unit test in the strict sense (no CI runner wired up yet) -
it's a deliberate, runnable validation script, because for a stochastic
estimator, "does it type-check and not crash" is a much weaker bar than
"does it actually recover the true theta from data where we know the
answer." Run manually with: python src/test_ou_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from ou_model import fit_ou, simulate_ou, z_score

TOLERANCE = 0.15  # allow 15% relative error from finite-sample noise


def assert_close(actual, expected, tol=TOLERANCE, label=""):
    rel_err = abs(actual - expected) / abs(expected)
    status = "PASS" if rel_err <= tol else "FAIL"
    print(f"  [{status}] {label}: true={expected:.4f} fitted={actual:.4f} (rel err {rel_err:.1%})")
    assert rel_err <= tol, f"{label} exceeded tolerance"


def test_fast_reversion():
    print("Test: fast mean reversion (theta=0.3)")
    theta_true, sigma_true = 0.3, 1.0
    path = simulate_ou(theta=theta_true, sigma=sigma_true, n=2000, seed=1)
    fit = fit_ou(path)
    assert_close(fit.theta, theta_true, label="theta")
    assert_close(fit.stationary_std, sigma_true / np.sqrt(2 * theta_true), label="stationary_std")


def test_slow_reversion():
    print("Test: slow mean reversion (theta=0.02)")
    theta_true = 0.02
    path = simulate_ou(theta=theta_true, sigma=1.0, n=5000, seed=2)
    fit = fit_ou(path)
    assert_close(fit.theta, theta_true, label="theta")


def test_zscore_arithmetic():
    print("Test: z-score arithmetic")
    path = simulate_ou(theta=0.3, sigma=1.0, n=2000, seed=1)
    fit = fit_ou(path)
    z = z_score(3 * fit.stationary_std, fit)
    assert_close(z, 3.0, tol=0.01, label="z_score of 3-std deviation")


def test_random_walk_shows_no_reversion():
    print("Test: pure random walk should show phi near 1.0 (theta near 0)")
    rw = np.cumsum(np.random.default_rng(3).normal(0, 1, 1000))
    fit = fit_ou(rw)
    print(f"  phi={fit.phi:.4f} (expect close to 1.0 — this is the correct diagnostic,"
          f" NOT r_squared, which stays high ({fit.r_squared:.4f}) for a random walk too)")
    assert fit.phi > 0.95, "random walk should show phi close to 1.0 (near-zero reversion)"
    print("  [PASS] phi correctly indicates near-zero mean reversion")


if __name__ == "__main__":
    test_fast_reversion()
    test_slow_reversion()
    test_zscore_arithmetic()
    test_random_walk_shows_no_reversion()
    print("\nAll OU model validation tests passed.")
