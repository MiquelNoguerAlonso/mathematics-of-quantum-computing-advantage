"""Outward interval certificates for the three noisy research-cost rows.

This final table-writing suite must follow research_checks.py. It retains
the original fixed-Y and sampler calculations, replacing only the weaker
quantum-information cost lower bounds by measured-information bounds.
"""
from pathlib import Path
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
import json
import mpmath as mp
from verify_operational_frontier import dm

ROOT = Path(__file__).resolve().parent
mp.mp.dps = 90
mp.iv.dps = 65
iv = mp.iv

def ends(x):
    return mp.mpf(x.a), mp.mpf(x.b)

def display_interval(x):
    lo, hi = ends(x)
    with localcontext() as ctx:
        ctx.prec = 110
        unit = Decimal('1e-48')
        a = Decimal(mp.nstr(lo, 90)).quantize(unit, rounding=ROUND_FLOOR)-unit
        b = Decimal(mp.nstr(hi, 90)).quantize(unit, rounding=ROUND_CEILING)+unit
    return [format(a, 'f'), format(b, 'f')]

def kl(p, q):
    return p*iv.log(p/q)+(1-p)*iv.log((1-p)/(1-q))

cases = [('Noisy', '.005', '.02', 256, 0, 1),
         ('Costly', '.02', '.04', 256, 50, 5),
         ('Restarts', '.01', '.01', 256, 100, 1)]
k = iv.mpf('.98')*iv.log(99)
rows = []
for name, mu_s, gamma_s, D, c0, c1 in cases:
    mu, gamma = iv.mpf(mu_s), iv.mpf(gamma_s)
    cap = min(D, int(mp.floor(mp.pi/(8*mp.mpf(mu_s)))))
    if cap < D:
        assert ends(iv.pi/(8*mu))[0] > cap
        assert ends(iv.pi/(8*mu))[1] < cap+1
    best_lo = best_hi = mp.mpf(0)
    best_y_lo = best_y_hi = mp.mpf(0)
    details = []
    for t in range(1, cap+1):
        _, phi, _ = dm(mp.exp(-mp.mpf(gamma_s)*t), 2*mp.mpf(mu_s)*t)
        ph = iv.mpf(mp.nstr(phi, 85))
        r, theta = iv.exp(-gamma*t), 2*mu*t
        p = (1+r*iv.cos(ph-theta))/2
        q = (1+r*iv.cos(ph+theta))/2
        l1, l2 = p/q, (1-p)/(1-q)
        divided = iv.log(l1/l2)/(l1-l2)
        grad = r/2*(iv.sin(theta-ph)*divided+iv.sin(theta+ph))
        eta = (r*iv.sin(theta))**2/(1-r*r)
        gap = 4*iv.sqrt(eta*(1+eta))*abs(grad)
        cost = c0+c1*t
        lo, hi = ends(kl(p, q)/cost)[0], ends((kl(p, q)+gap)/cost)[1]
        best_lo, best_hi = max(best_lo, lo), max(best_hi, hi)
        signal = r*iv.sin(theta)
        dy = signal*iv.log((1+signal)/(1-signal))/cost
        best_y_lo = max(best_y_lo, ends(dy)[0])
        best_y_hi = max(best_y_hi, ends(dy)[1])
        rate = display_interval(iv.mpf([lo, hi]))
        details.append(dict(depth=t, angle=mp.nstr(phi, 85),
                            rate_lower=rate[0], rate_upper=rate[1]))
    rate = iv.mpf([best_lo, best_hi])
    yrate = iv.mpf([best_y_lo, best_y_hi])
    lower_cost = k/rate
    improvement = 100*(rate/yrate-1)
    candidates = [d['depth'] for d in details if mp.mpf(d['rate_upper']) >= best_lo]
    assert len(candidates) == 1
    assert best_hi-best_lo < mp.mpf('1e-55')
    rows.append(dict(name=name, mu=mu_s, gamma=gamma_s, D=D, c0=c0, c1=c1,
                     depth_cap=cap, t_M=candidates[0], I_M=display_interval(rate),
                     I_Y=display_interval(yrate), k_delta_over_I_M=display_interval(lower_cost),
                     excess_rate_percent=display_interval(improvement), depths=details))

base = json.loads((ROOT/'research_verification_results.json').read_text())
byname = {row['name']: row for row in rows}
for old in base['cases']:
    if old['name'] in byname:
        new = byname[old['name']]
        assert mp.mpf(new['k_delta_over_I_M'][0]) > old['readout_bound']
        assert abs(float(new['I_Y'][0])/old['Istar']-1) < 1e-12
        if old['name'] == 'Costly':
            assert mp.mpf(new['k_delta_over_I_M'][0]) > old['UC']

grid = json.loads((ROOT/'joint_frontier_results.json').read_text())
lo = min(row['lower_over_scale'] for row in grid['parameter_grid'])
hi = max(row['upper_over_scale'] for row in grid['parameter_grid'])
assert len(grid['parameter_grid']) == 246
assert lo > .05839 and hi < 4.312
assert abs(lo-grid['grid_range']['minimum_lower_over_scale']) < 1e-15
assert abs(hi-grid['grid_range']['maximum_upper_over_scale']) < 1e-15

result = dict(status='All research-table interval certificates passed',
              interval_decimal_precision=65, optimized_depths=sum(r['depth_cap'] for r in rows),
              delta='.01', rows=rows,
              numerical_grid_band=dict(cases=246, lower_over_scale=lo, upper_over_scale=hi,
                    qualification='Finite-grid high-precision evaluations, not universal or interval-certified constants'),
              caveat='Outward interval evaluation of proved inequalities, not a proof-assistant formalization.')
(ROOT/'table_rate_results.json').write_text(json.dumps(result, indent=2)+'\n')

# Parameters and legacy columns are unchanged. The measured bounds are rounded
# down, including cases where ordinary nearest rounding would overstate them.
tex = [r'\begin{table}[htbp]', r'\centering\small', r'\begin{tabular}{lrrrrr}',
       r'\toprule', r'Setting & $\mu$ & $\gamma$ & $D$ & $c_0$ & $c_1$\\', r'\midrule']
for r in base['cases']:
    tex.append(f"{r['name']} & {r['mu']:g} & {r['gamma']:g} & {r['D']} & {r['c0']:g} & {r['c1']:g}"+r'\\')
tex += [r'\bottomrule\end{tabular}', r'\caption{Parameters of the reproducible decision experiments.}',
        r'\end{table}', r'\begin{table}[htbp]', r'\centering\small',
        r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lrrrrrrl}', r'\toprule',
        r'Setting & $t_\star$ & $t_U$ & POVM cost & $L_Y$ & $U_Y$ & $I_Y/I_C$ & Certificate\\', r'\midrule']
for r in base['cases']:
    if r['name'] in byname:
        v = Decimal(byname[r['name']]['k_delta_over_I_M'][0]).quantize(Decimal('.1'), rounding=ROUND_FLOOR)
        bound = r'$\geq '+f'{v:,.1f}'.replace(',', '{,}')+r'$'
    else:
        # A generous upward rounding preserves the existing noiseless upper.
        v = Decimal(str(r['readout_bound'])).quantize(Decimal('.1'), rounding=ROUND_CEILING)
        bound = r'$\leq '+f'{v:,.1f}'.replace(',', '{,}')+r'$'
    tex.append(f"{r['name']} & {r['rate_depth']} & {r['finite_depth']} & {bound} & "
               f"{r['LQ']:,.1f} & {r['UQ']:,.1f} & {r['ratio']:.3f} & {r['verdict']}"+r'\\')
tex += [r'\bottomrule\end{tabular}',
        r'\caption{Bounds on optimal expected cost at $\delta=0.01$.',
        r'$t_\star,t_U,L_Y,U_Y$ concern fixed $Y$. The POVM-cost column bounds',
        r'$C^\star_{M,\delta}$ at the same block price. For $\gamma=0$ it is the',
        r'zero-error upper bound of Proposition~\ref{prop:usd}, minimized over depth',
        r'and rounded up. For $\gamma>0$ it is $k_\delta/I_M$ from',
        r'Theorem~\ref{thm:measured}, evaluated using outward interval upper bounds',
        r'on $I_M$ at every allowed depth and rounded down.',
        r'Other numerical columns are rounded approximations.',
        r'The sampler costs are $U_C\approx1422.7$ (Ideal, Costly),',
        r'$140{,}889.3$ (Shallow), $22{,}542.6$ (Noisy), and $5636.0$ (Restarts).',
        r'Depth gain means $U_Y<L_C$; in the Costly row,',
        r'$k_\delta/I_M>U_C$ certifies a loss for every single-block readout.}',
        r'\label{tab:researchresults}', r'\end{table}']
(ROOT/'research_results_table.tex').write_text('\n'.join(tex)+'\n')
print(json.dumps({key: val for key, val in result.items() if key != 'rows'}, indent=2))
for row in rows:
    print(row['name'], 'cost', row['k_delta_over_I_M'], 'rate excess %', row['excess_rate_percent'])
