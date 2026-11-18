"""Numerical falsification checks; the analytic proofs are in the TeX fragment."""
import json
import math
from pathlib import Path

import numpy as np

L = 2 * math.log1p(math.sqrt(2))
K = math.sqrt(2) * L
s = np.linspace(1e-8, 1 / math.sqrt(2), 100001)
ell = 2 * np.arctanh(s)
assert np.all(ell / s >= 2 - 1e-12)
assert np.all(ell / s <= K + 1e-12)
assert abs(ell[-1] / s[-1] - K) < 1e-12
assert np.all(np.diff(ell / s) >= -1e-12)

# Exact analytic counterexample, choosing its integer boundaries explicitly.
ell_true = math.log(3)
robust_n = 4 / 0.5 * math.tanh(4 * ell_true / 2)
tuned_n = 2 / 0.5 * math.tanh(2 * ell_true / 2)
assert abs(robust_n - 320 / 41) < 1e-12
assert abs(tuned_n - 16 / 5) < 1e-12
assert abs(robust_n / tuned_n - 100 / 41) < 1e-12

rng = np.random.default_rng(5257)
robust_checks = 0
for _ in range(50000):
    delta = float(np.exp(rng.uniform(math.log(1e-8), math.log(0.25))))
    true_s = float(rng.uniform(0.001, 1 / math.sqrt(2)))
    low_s = float(true_s * rng.uniform(0.001, 1))
    A = math.log1p(-delta) - math.log(delta)
    ep, el = 2 * math.atanh(true_s), 2 * math.atanh(low_s)
    b, bt = math.ceil(A / el), math.ceil(A / ep)
    ratio = b / bt * math.tanh(b * ep / 2) / math.tanh(bt * ep / 2)
    bound = ep / el * (1 + el / A) / (1 - 2 * delta)
    assert ratio <= bound * (1 + 1e-12)
    assert ep / el <= (K / 2) * true_s / low_s * (1 + 1e-12)
    assert b * ep >= A - 1e-12
    robust_checks += 1

# Independent absorbing-chain linear solves check the hitting formulas.
chain_checks = 0
for signal in [0.01, 0.1, 0.5, 1 / math.sqrt(2)]:
    p = (1 + signal) / 2
    for boundary in range(1, 21):
        size = 2 * boundary - 1
        mat = np.eye(size)
        for i in range(size - 1):
            mat[i, i + 1] = -p
            mat[i + 1, i] = -(1 - p)
        dur = np.linalg.solve(mat, np.ones(size))[boundary - 1]
        rhs = np.zeros(size)
        rhs[0] = 1 - p
        err = np.linalg.solve(mat, rhs)[boundary - 1]
        logodds = 2 * math.atanh(signal)
        exact_n = boundary / signal * math.tanh(boundary * logodds / 2)
        exact_err = 1 / (1 + math.exp(boundary * logodds))
        assert math.isclose(dur, exact_n, rel_tol=2e-11, abs_tol=1e-11)
        assert math.isclose(err, exact_err, rel_tol=2e-11, abs_tol=1e-12)
        chain_checks += 1

certified = 0
for _ in range(10000):
    mu = float(rng.uniform(0.0001, 0.125))
    depth_cap = min(int(rng.integers(1, 101)), math.floor(math.pi / (8 * mu)))
    ts = np.arange(1, depth_cap + 1)
    gamma = float(rng.uniform(0, 0.2))
    c0 = float(rng.uniform(0, 2))
    c1 = float(rng.uniform(0.05, 2))
    cC = float(rng.uniform(0.05, 10))
    delta = float(rng.uniform(0.00001, 0.2))
    A = math.log((1 - delta) / delta)
    k = (1 - 2 * delta) * A
    signals = np.exp(-gamma * ts) * np.sin(2 * mu * ts)
    es = 2 * np.arctanh(signals)
    costs = c0 + c1 * ts
    ds = signals * es
    M = float(np.min(costs / signals**2))
    invI = 1 / float(np.max(ds / costs))
    assert M / K <= invI * (1 + 1e-12)
    assert invI <= M / 2 * (1 + 1e-12)
    bs = np.ceil(A / es)
    uq = float(np.min(costs * bs / signals * np.tanh(bs * es / 2)))
    assert uq <= (A + L) / 2 * M * (1 + 1e-12)
    sc = math.sin(2 * mu)
    ic = sc * 2 * math.atanh(sc) / cC
    lc = k / ic
    lhs = cC * ts**2 * np.exp(-2 * gamma * ts) / costs
    if np.any(lhs > math.pi**2 / 4 * (A + L) / k):
        assert uq < lc
        certified += 1

result = {
    'status': 'all checks passed',
    'envelope_grid_points': len(s),
    'robustness_parameter_cases': robust_checks,
    'independent_absorbing_chain_cases': chain_checks,
    'cost_model_cases': 10000,
    'strict_certificates_checked': certified,
    'K': K,
    'limiting_sandwich_factor': K / 2,
    'noise_threshold_constant': 1 / (math.pi**2 * math.e),
    'counterexample_ratio': robust_n / tuned_n,
}
Path(__file__).resolve().with_name('boost_verification_results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
