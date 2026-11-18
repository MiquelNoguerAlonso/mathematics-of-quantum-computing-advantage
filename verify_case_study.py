#!/usr/bin/env python3
"""Deterministic checks for the calibrated amplitude-oracle case study.

Checks the actual two-qubit controlled circuit and dephasing channel,
both Bernoulli KL directions, SPRT overshoot bounds, every allowed integer
depth, and the constructive rounding step. These are model calculations.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent


def bernoulli_kl(p, q):
    return p * np.log(p / q) + (1-p) * np.log((1-p) / (1-q))


def frontier(mu, gamma, c0=20., c1=10., depth_limit=256, delta=.01):
    ts = np.arange(1, min(depth_limit, int(np.floor(np.pi/(8*mu))))+1)
    signals = np.exp(-gamma*ts)*np.sin(2*mu*ts)
    ell = 2*np.arctanh(signals)
    costs = c0+c1*ts
    rate = signals*ell/costs
    boundary_log = np.log((1-delta)/delta)
    barriers = np.ceil(boundary_log/ell).astype(int)
    stopping_costs = costs*barriers/signals*np.tanh(barriers*ell/2)
    i_star, i_upper = int(np.argmax(rate)), int(np.argmin(stopping_costs))
    visibility = np.exp(-gamma*ts)
    quantum_rates = 2*visibility*np.arctanh(visibility)*np.sin(2*mu*ts)**2/costs
    i_quantum = int(np.argmax(quantum_rates))
    return dict(gamma=gamma, depth_cap=int(ts[-1]), t_star=int(ts[i_star]),
                t_U=int(ts[i_upper]), L_Q=float((1-2*delta)*boundary_log/rate[i_star]),
                U_Q=float(stopping_costs[i_upper]),
                I_Q=float(quantum_rates[i_quantum]), t_Q=int(ts[i_quantum]),
                L_Q_all_readouts=float((1-2*delta)*boundary_log/quantum_rates[i_quantum]),
                barrier=int(barriers[i_upper]),
                stopping_error=float(1/(1+np.exp(barriers[i_upper]*ell[i_upper]))))


def main():
    a_minus, a_plus, delta = .036, .044, .01
    theta_minus, theta_plus = np.arcsin(np.sqrt([a_minus, a_plus]))
    theta_mid = (theta_plus+theta_minus)/2
    mu = (theta_plus-theta_minus)/2
    k_delta = (1-2*delta)*np.log((1-delta)/delta)
    kl_plus, kl_minus = bernoulli_kl(a_plus, a_minus), bernoulli_kl(a_minus, a_plus)
    log_boundary = np.log(1/delta)
    positive_step = np.log(a_plus/a_minus)
    negative_step = np.log((1-a_minus)/(1-a_plus))
    L_R = max(k_delta/kl_plus, k_delta/kl_minus)
    U_R = max((log_boundary+positive_step)/kl_plus,
              (log_boundary+negative_step)/kl_minus)
    assert L_R < U_R
    assert np.isclose(a_plus*positive_step-(1-a_plus)*negative_step, kl_plus)
    assert np.isclose(a_minus*positive_step-(1-a_minus)*negative_step, -kl_minus)

    # Independent state-vector/density-matrix construction, not the signal formula.
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.diag([1., -1.])
    I = np.eye(2)
    ket_y_minus = np.array([1, -1j])/np.sqrt(2)
    ket_plus = np.ones(2)/np.sqrt(2)
    psi0 = np.kron(ket_plus, ket_y_minus)
    ancilla_Z = np.kron(Z, I)
    ancilla_Y = np.kron(Y, I)
    rows = [frontier(mu, gamma) for gamma in (.004, .015, .020)]
    circuit_checks = 0
    for h, theta, a in [(-1, theta_minus, a_minus), (1, theta_plus, a_plus)]:
        preparation = expm(-1j*theta*Y)
        assert np.isclose(abs(preparation[1, 0])**2, a)
        Q = preparation @ Z @ preparation.conj().T @ Z
        assert np.allclose(Q, expm(-2j*theta*Y))
        assert np.allclose(Q@ket_y_minus, np.exp(2j*theta)*ket_y_minus)
        controlled_Q = np.zeros((4, 4), dtype=complex)
        controlled_Q[:2, :2], controlled_Q[2:, 2:] = I, Q
        centering = np.kron(np.diag([1, np.exp(-2j*theta_mid)]), I)
        step = centering @ controlled_Q
        for row in rows:
            gamma = row['gamma']
            rho = np.outer(psi0, psi0.conj())
            for t in range(1, row['depth_cap']+1):
                rho = step@rho@step.conj().T
                visibility = np.exp(-gamma)
                rho = (1+visibility)/2*rho + (1-visibility)/2*ancilla_Z@rho@ancilla_Z
                observed = float(np.trace(ancilla_Y@rho).real)
                target = h*np.exp(-gamma*t)*np.sin(2*mu*t)
                assert np.isclose(observed, target, atol=3e-13, rtol=0)
                circuit_checks += 1

    # Independent absorbing-chain solution at each reported finite-cost depth.
    from scipy.linalg import solve
    for row in rows:
        t, barrier = row['t_U'], row['barrier']
        signal = np.exp(-row['gamma']*t)*np.sin(2*mu*t)
        p, q = (1+signal)/2, (1-signal)/2
        size = 2*barrier-1
        transition = np.diag(np.full(size-1, p), 1) + np.diag(np.full(size-1, q), -1)
        rhs = np.column_stack([np.ones(size), np.r_[q, np.zeros(size-1)]])
        solution = solve(np.eye(size)-transition, rhs)[barrier-1]
        assert np.isclose((20+10*t)*solution[0], row['U_Q'], rtol=1e-12)
        assert np.isclose(solution[1], row['stopping_error'], rtol=1e-12)
        assert row['stopping_error'] <= delta
        row['certificate'] = 'Y cost gain' if row['U_Q'] < L_R else 'Every readout loses' if row['L_Q_all_readouts'] > U_R else 'Y cost loss' if row['L_Q'] > U_R else 'Unresolved'
        row['sampling_cost_over_coherent_cost_lower_bound'] = float(L_R/row['U_Q'])
    assert [r['certificate'] for r in rows] == ['Y cost gain', 'Y cost gain', 'Every readout loses']
    assert rows[1]['t_U'] > rows[1]['t_star']  # Rounding need not favor smaller depths.

    rng = np.random.default_rng(20261115)
    rounding_checks = 0
    for mu0, gamma0, depth in [(1/8, 0., 256), (1/8, 1., 1)] + [
            (10**rng.uniform(-7, np.log10(1/8)), 0. if i%10==0 else 10**rng.uniform(-7, 0),
             int(rng.integers(1, 10000))) for i in range(10000)]:
        r_eff = min(depth, 1/mu0, np.inf if gamma0==0 else 1/gamma0)
        t0 = max(1, int(np.floor(np.pi*r_eff/8)))
        assert 1 <= t0 <= min(depth, int(np.floor(np.pi/(8*mu0))))
        assert t0 >= np.pi*r_eff/16-1e-10
        assert gamma0*t0 <= 1+1e-14
        rounding_checks += 1
    # Explicit counterexample to the unscaled floor(r_eff) proposed in the attachment.
    assert int(min(256, 8, np.inf)) > int(np.floor(np.pi/(8*(1/8))))

    result = dict(a_minus=a_minus, a_plus=a_plus, delta=delta,
                  theta_minus=float(theta_minus), theta_plus=float(theta_plus), mu=float(mu),
                  D_plus=float(kl_plus), D_minus=float(kl_minus),
                  L_R=float(L_R), U_R=float(U_R),
                  coherent_rows=rows, controlled_circuit_checks=circuit_checks,
                  independent_absorbing_chain_checks=len(rows), rounding_checks=rounding_checks)
    (ROOT/'case_study_results.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = [r'\begin{table}[htbp]', r'\centering\small', r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{rrrrrrl}', r'\toprule',
             r'$\gamma$ & $t_\star$ & $t_U$ & $L_Q$ & $L_Y$ & $U_Y$ & Certificate\\', r'\midrule']
    for row in rows:
        lines.append(f"{row['gamma']:.3f} & {row['t_star']} & {row['t_U']} & {row['L_Q_all_readouts']:,.1f} & {row['L_Q']:,.1f} & {row['U_Q']:,.1f} & {row['certificate']}" + r'\\')
    lines += [r'\bottomrule\end{tabular}',
              r'\caption{Calibrated amplitude-oracle threshold example at $\delta=0.01$,',
              r'$D=256$, $c_0=20$, and $c_1=10$. The direct Bernoulli sampling bounds',
              r'are $L_R\approx5564.8$ and $U_R\approx5701.1$ at unit sample cost.',
              r'$t_\star,t_U$ optimize the fixed-$Y$ rate and finite-confidence cost.',
              r'$L_Q=k_\delta/I_Q$ optimizes the quantum bound over all allowed depths',
              r'and lower-bounds every adaptive single-block readout at the same price',
              r'(Theorem~\ref{thm:measured}). Gain means $U_Y<L_R$;',
              r'the last-row loss means $L_Q>U_R$. Displayed values are rounded.}',
              r'\label{tab:amplitude-case}', r'\end{table}']
    (ROOT/'case_study_results_table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
