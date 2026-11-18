"""Outward-interval evaluation of a global measured-information bound.

The analytic concavity argument is proved in the manuscript. Each candidate
measurement produces a lower bound, and a Frechet-gradient residual gives
a global upper bound over all POVMs. Intervals include the exact decimal
amplitudes and all elementary operations used by the certificate.
"""
from pathlib import Path
import json
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
import mpmath as mp
from verify_operational_frontier import dm
mp.mp.dps=90;mp.iv.dps=65
iv=mp.iv
ROOT=Path(__file__).resolve().parent

def endpoints(x):
    return mp.mpf(x.a),mp.mpf(x.b)
def output(x):
    # Round displayed endpoints outward, with one extra decimal unit to
    # dominate the much smaller binary-to-decimal conversion error.
    lo,hi=endpoints(x)
    with localcontext() as ctx:
        ctx.prec=110;unit=Decimal('1e-48')
        lower=Decimal(mp.nstr(lo,90)).quantize(unit,rounding=ROUND_FLOOR)-unit
        upper=Decimal(mp.nstr(hi,90)).quantize(unit,rounding=ROUND_CEILING)+unit
        return [format(lower,'f'),format(upper,'f')]
def kl(p,q):return p*iv.log(p/q)+(1-p)*iv.log((1-p)/(1-q))

am=iv.mpf('.036');ap=iv.mpf('.044')
mu=(iv.atan2(iv.sqrt(ap),iv.sqrt(1-ap))-iv.atan2(iv.sqrt(am),iv.sqrt(1-am)))/2
mu0=(mp.asin(mp.sqrt(mp.mpf('.044')))-mp.asin(mp.sqrt(mp.mpf('.036'))))/2
cap=38
assert endpoints(iv.pi/(8*mu))[0]>cap and endpoints(iv.pi/(8*mu))[1]<cap+1
k=iv.mpf('.98')*iv.log(99)
B=iv.log(100)
up=iv.log(ap/am);um=iv.log((1-am)/(1-ap))
u1=(B+up)/kl(ap,am);u2=(B+um)/kl(am,ap)
URlower=max(endpoints(u1)[0],endpoints(u2)[0]);URupper=max(endpoints(u1)[1],endpoints(u2)[1])
rows=[]
for gamma in ['.004','.015','.020']:
    gl=iv.mpf(gamma);best_lower=mp.mpf(0);best_upper=mp.mpf(0);details=[]
    for t in range(1,cap+1):
        _,phi,_=dm(mp.exp(-mp.mpf(gamma)*t),2*mu0*t)
        ph=iv.mpf(mp.nstr(phi,85));r=iv.exp(-gl*t);theta=2*mu*t
        p=(1+r*iv.cos(ph-theta))/2;q=(1+r*iv.cos(ph+theta))/2
        lam1=p/q;lam2=(1-p)/(1-q)
        divided=iv.log(lam1/lam2)/(lam1-lam2)
        grad=r/2*(iv.sin(theta-ph)*divided+iv.sin(theta+ph))
        eta=(r*iv.sin(theta))**2/(1-r*r)
        gap=4*iv.sqrt(eta*(1+eta))*abs(grad)
        value=kl(p,q);c=20+10*t
        lo=endpoints(value/c)[0];hi=endpoints((value+gap)/c)[1]
        best_lower=max(best_lower,lo);best_upper=max(best_upper,hi)
        shown=output(iv.mpf([lo,hi]))
        details.append(dict(depth=t,phi=mp.nstr(phi,85),rate_lower=shown[0],rate_upper=shown[1]))
    rate=iv.mpf([best_lower,best_upper])
    bound=k/rate
    candidates=[d['depth'] for d in details if mp.mpf(d['rate_upper'])>=best_lower]
    assert len(candidates)==1, candidates
    rows.append(dict(gamma=gamma,I_M=output(rate),t_M=candidates[0],
                     k_delta_over_I_M=output(bound),depths=details))
    assert best_upper-best_lower<mp.mpf('1e-55')
last=rows[-1]
assert mp.mpf(last['k_delta_over_I_M'][0])>mp.mpf('6658.96')
assert URupper<mp.mpf('5701.07')
result=dict(status='All interval certificates passed',interval_decimal_precision=65,
            source_values='Exact a_minus=.036, a_plus=.044, delta=.01, c0=20, c1=10',
            classical_upper_bound=output(iv.mpf([URlower,URupper])),
            rows=rows,
            conservative_certificate='C_M,0.01 >= 6658.96 > 5701.07 >= C_R,0.01',
            caveat='Analytic inequalities evaluated with outward interval arithmetic; not a proof-assistant formalization.')
(ROOT/'interval_certificate_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
for r in rows: print(r['gamma'],r['k_delta_over_I_M'])

# Regenerate the final tables after the original verification scripts.
case=json.loads((ROOT/'case_study_results.json').read_text())
base={f"{row['gamma']:.3f}":row for row in case['coherent_rows']}
tex=[r'\begin{table}[htbp]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',
     r'\begin{tabular}{rrrrrrl}',r'\toprule',
     r'$\gamma$ & $t_M$ & $t_U$ & $L_Q$ & $L_M$ & $U_Y$ & Certificate\\',r'\midrule']
for row in rows:
    original=base[f"{float(row['gamma']):.3f}"]
    lower=Decimal(row['k_delta_over_I_M'][0]).quantize(Decimal('.01'),rounding=ROUND_FLOOR)
    tex.append(f"{float(row['gamma']):.3f} & {row['t_M']} & {original['t_U']} & "
               f"{original['L_Q_all_readouts']:,.2f} & {lower:,.2f} & {original['U_Q']:,.2f} & "
               +('Y cost gain' if original['U_Q']<case['L_R'] else r'\shortstack{Every readout\\loses}')+r'\\')
tex += [r'\bottomrule\end{tabular}',
        r'\caption{Calibrated amplitude-oracle example at $\delta=.01$, $D=256$,',
        r'$c_0=20$, $c_1=10$, and unit direct-sampling price.',
        r'$t_M$ maximizes measured information; $t_U$ minimizes the fixed-$Y$ counting cost.',
        r'$L_Q=k_\delta/I_Q$ and $L_M=k_\delta/I_M$ lower-bound every allowed single-block readout.',
        r'The $L_M$ entries use interval upper bounds on $I_M$ and are rounded down;',
        r'other entries are rounded approximations. Direct-sampling bounds are',
        r'$L_R\approx5564.8$ and $U_R\approx5701.1$. Gain means $U_Y<L_R$;',
        r'the final loss is certified with outward rounding in~\eqref{eq:strong-loss}.}',
        r'\label{tab:amplitude-case}',r'\end{table}']
(ROOT/'case_study_results_table.tex').write_text('\n'.join(tex)+'\n')
audit=json.loads((ROOT/'operational_frontier_results.json').read_text())
tex=[r'\begin{table}[htbp]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',
     r'\begin{tabular}{lrrrrr}',r'\toprule',
     r'Setting & $t_M$ & $I_Y$ & $I_M$ & $I_Q$ & $I_M/I_Y$\\',r'\midrule']
for row in audit['measured_information_rows']:
    label=row['name'].replace('Amplitude .',r'Amplitude, $\gamma=0.')
    if label.startswith('Amplitude'):label+='$'
    tex.append(f"{label} & {row['t_M']} & {row['I_Y']:.8g} & {row['I_M_estimate']:.8g} & "
               f"{row['I_Q']:.8g} & {row['I_M_over_I_Y']:.6f}"+r'\\')
tex += [r'\bottomrule\end{tabular}',
        r'\caption{Operational rates at their separately optimized depths; values are rounded.',
        r'The first three rows use $(\mu,\gamma)=(.02,.0001),(.02,.01),(.002,.02)$,',
        r'$D=256$, $c_0=0$, $c_1=1$. Amplitude rows use the parameters of',
        r'Table~\ref{tab:amplitude-case}. Residual bounds are evaluated at high precision;',
        r'outward interval arithmetic certifies the three amplitude rows.',
        r'$I_M$ describes high-confidence information per cost, not an exact finite-confidence cost.}',
        r'\label{tab:operational-rates}',r'\end{table}']
(ROOT/'operational_results_table.tex').write_text('\n'.join(tex)+'\n')
