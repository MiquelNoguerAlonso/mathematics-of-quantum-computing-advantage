"""Exact rational certificates for the complete likelihood-lattice frontier.

The analytic proofs cover all parameters. Fraction arithmetic certifies each
reported primal/dual pair and its infinite-barrier cutoff. A separate interval
Bellman recursion permits all adaptive action choices at a finite horizon.
"""
from fractions import Fraction as F
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from pathlib import Path
import json, math, itertools
import numpy as np
import mpmath as mp
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
mp.mp.dps=90;mp.iv.dps=65;iv=mp.iv

def fraction_text(x):return f'{x.numerator}/{x.denominator}'
def point(x):
    x=F(x);return iv.mpf(x.numerator)/x.denominator
def ends(x):return mp.mpf(x.a),mp.mpf(x.b)
def enclosure(x,places=10):
    with localcontext() as ctx:
        ctx.prec=110;d=Decimal(x.numerator)/Decimal(x.denominator)
        unit=Decimal(10)**(-places)
        return [format(d.quantize(unit,rounding=ROUND_FLOOR),'f'),
                format(d.quantize(unit,rounding=ROUND_CEILING),'f')]
def ivout(x):
    with localcontext() as ctx:
        ctx.prec=110;unit=Decimal('1e-55')
        lo,hi=ends(x)
        return [str(Decimal(mp.nstr(lo,95)).quantize(unit,rounding=ROUND_FLOOR)-unit),
                str(Decimal(mp.nstr(hi,95)).quantize(unit,rounding=ROUND_CEILING)+unit)]

def barrier(s,alpha,kappa,b):
    R=(1+s)/(1-s);e=1/(1+R**b);g=b*(1-2*e)
    cost=(kappa*g/(2*alpha*s)+(1-kappa/(2*alpha))*(b%2)
          if kappa<=2*alpha else g/s)
    return e,cost

def exact_frontier(s,alpha,kappa,delta):
    """Find and certify a supporting line; every final check is rational."""
    s,alpha,kappa,delta=map(F,(s,alpha,kappa,delta))
    assert 0<s<1 and 0<alpha<=1 and kappa>0 and 0<delta<F(1,2)
    R=(1+s)/(1-s)
    B=max(5,int(math.ceil(math.log(1/float(delta))/math.log(float(R))))+3)
    for attempt in range(8):
        pts=[barrier(s,alpha,kappa,b) for b in range(B+1)]
        candidates=[]
        for i in range(B):
            ei,ci=pts[i]
            if ei<delta:continue
            for j in range(i+1,B+1):
                ej,cj=pts[j]
                if ej>delta:continue
                w=(delta-ej)/(ei-ej)
                lam=(cj-ci)/(ei-ej)
                if lam<0:continue
                value=w*ci+(1-w)*cj;H=ci+lam*ei
                candidates.append((value,i,j,w,lam,H))
        for value,i,j,w,lam,H in sorted(candidates):
            if any(c+lam*e<H for e,c in pts):continue
            slope=min(kappa/(2*alpha),F(1))
            cutoff=int(H//slope)+1
            allpts=[barrier(s,alpha,kappa,b) for b in range(cutoff+1)]
            if any(c+lam*e<H for e,c in allpts):continue
            assert cutoff*slope>H
            assert w*pts[i][0]+(1-w)*pts[j][0]==delta
            assert value==H-lam*delta
            return dict(value=value,barriers=(i,j),weight_first=w,
                        multiplier=lam,intercept=H,cutoff=cutoff,
                        dual_slacks=[c+lam*e-H for e,c in allpts])
        B*=2
    raise AssertionError('No certified support found')

def solve_fraction(A,b):
    M=[list(row)+[rhs] for row,rhs in zip(A,b)];n=len(M)
    for k in range(n):
        pivot=next(i for i in range(k,n) if M[i][k])
        M[k],M[pivot]=M[pivot],M[k];v=M[k][k]
        M[k]=[x/v for x in M[k]]
        for i in range(n):
            if i==k:continue
            v=M[i][k]
            if v:M[i]=[x-v*y for x,y in zip(M[i],M[k])]
    return [row[-1] for row in M]

def absorbing_barrier(s,alpha,kappa,b):
    """Independent exact linear-system evaluation under the positive label."""
    if b==0:return F(1,2),F(0)
    p=(1+s)/2;q=1-p
    use_E=kappa<=2*alpha
    step=2 if use_E else 1
    plus,minus,hold=(alpha*p*p,alpha*q*q,1-alpha*(p*p+q*q)) if use_E else (p,q,F(0))
    states=list(range(-b+step,b,step));index={z:i for i,z in enumerate(states)}
    A=[[F(int(i==j)) for j in range(len(states))] for i in range(len(states))]
    rhs=[F(0) for _ in states]
    for z,i in index.items():
        for dest,prob in [(z+step,plus),(z-step,minus),(z,hold)]:
            if dest in index:A[i][index[dest]]-=prob
            elif dest==-b:rhs[i]+=prob
    er=solve_fraction(A,rhs) if states else []
    en=solve_fraction(A,[F(1) for _ in states]) if states else []
    def at(z,arr,error=False):
        if z==-b:return F(1) if error else F(0)
        if z==b:return F(0)
        return arr[index[z]]
    if use_E and b%2:
        error=p*at(1,er,True)+q*at(-1,er,True)
        cost=1+kappa*(p*at(1,en)+q*at(-1,en))
    else:
        error=at(0,er,True);cost=(kappa if use_E else 1)*at(0,en)
    return error,cost

def interval_bellman(lam,kappa,N=96):
    """All actions and stopping, exact likelihood lattice, no interpolation."""
    lam=point(lam);kappa=point(kappa)
    powers=[9**z for z in range(2*N+1)]
    err=[point(F(1,1+t)) for t in powers]
    yp=[point(F(9*t+1,10*(1+t))) for t in powers]
    ym=[point(F(t+9,10*(1+t))) for t in powers]
    ep=[point(F(243*t+3,500*(1+t))) for t in powers]
    em=[point(F(3*t+243,500*(1+t))) for t in powers]
    eh=point(F(127,250));V=[lam*x for x in err]
    for n in range(1,N+1):
        nxt=[]
        for z in range(2*(N-n)+1):
            vals=[lam*err[z],1+yp[z]*V[z+1]+ym[z]*V[abs(z-1)],
                  kappa+ep[z]*V[z+2]+em[z]*V[abs(z-2)]+eh*V[z]]
            nxt.append(iv.mpf([min(ends(v)[0] for v in vals),
                               min(ends(v)[1] for v in vals)]))
        V=nxt
    return V[0]

def exhaustive_trees(N,s=F(4,5),alpha=F(3,5),kappa=F(1)):
    p=(1+s)/2;q=1-p
    laws=[((p,q),(q,p),F(1)),
          ((alpha*p*p,alpha*q*q,1-alpha*(p*p+q*q)),
           (alpha*q*q,alpha*p*p,1-alpha*(p*p+q*q)),kappa)]
    # cost under +, cost under -, error under +, error under -
    trees=[(F(0),F(0),F(0),F(1)),(F(0),F(0),F(1),F(0))]
    for n in range(N):
        out=trees[:2]
        for pp,pm,c in laws:
            for children in itertools.product(trees,repeat=len(pp)):
                out.append((c+sum(pp[y]*ch[0] for y,ch in enumerate(children)),
                            c+sum(pm[y]*ch[1] for y,ch in enumerate(children)),
                            sum(pp[y]*ch[2] for y,ch in enumerate(children)),
                            sum(pm[y]*ch[3] for y,ch in enumerate(children))))
        trees=out
    return trees

def main():
    counts={};s=F(4,5);alpha=F(3,5)
    # Positivity, normalization and probabilities checked exactly and by matrices.
    x=F(59,100);a=F(-470,1003);bb=F(15,17);w=F(17,25)
    assert x*x+s*s==F(9881,10000)<1 and a*a+bb*bb<1
    assert F(8,25)>F(94,295)>0
    assert w*a+F(94,295)==0 and 1-w==F(8,25)
    up=w*(1+a*x+bb*s)/2;down=w*(1+a*x-bb*s)/2
    assert (up,down,1-up-down)==(F(243,500),F(3,500),F(127,250))
    X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);I=np.eye(2)
    states=[(I+float(x)*X+sign*float(s)*Y)/2 for sign in [1,-1]]
    effects=[float(w)/2*(I+float(a)*X+sign*float(bb)*Y) for sign in [1,-1]]
    effects.append(float(F(8,25))*I+float(F(94,295))*X)
    assert np.linalg.norm(states[0]@states[1]-states[1]@states[0])>.1
    for rho in states:assert min(np.linalg.eigvalsh(rho))>0
    for effect in effects:assert min(np.linalg.eigvalsh(effect))>0
    assert np.linalg.norm(sum(effects)-I)<1e-14
    assert np.max(abs(np.array([np.trace(states[0]@e).real for e in effects])-
                         np.array([float(up),float(down),float(1-up-down)])))<1e-14
    p=(1+s)/2;q=1-p
    assert alpha*p*p==up and alpha*q*q==down
    assert 2*alpha==F(6,5)
    assert 1-alpha*s*s==F(77,125)
    counts['exact_quantum_realization_and_simulation']=1

    # The analytic feasibility region is checked against its independent
    # quadratic positivity problem using rational candidate effects.
    region=[]
    for ss in [F(k,10) for k in range(2,10)]:
        xx=F(int(10000*math.sqrt(float(1-ss*ss)))-2,10000)
        tt=(1+ss*ss)/2;disc=tt*tt-(1-xx*xx)
        if disc<=0:continue
        maximum=(float(tt)+math.sqrt(float(disc)))/(1+float(xx))
        for factor in [.25,.8,1.01]:
            al=F(int(maximum*factor*10**7),10**7)
            q0=al*tt;A=min(q0*xx/(1-xx*xx),(1-q0)/(1+xx));ww=q0+xx*A
            good=ww*ww>=A*A+al*al and 1-ww>=abs(A)
            assert good==(factor<1)
            region.append([fraction_text(ss),fraction_text(xx),fraction_text(al),good])
    counts['quantum_realizability_region_checks']=len(region)

    # Exact absorbing systems do not use the stopped-drift formula.
    for ss,al,kk in [(s,alpha,F(1)),(s,alpha,F(3,2)),
                      (F(2,5),F(2,3),F(1,2)),(F(3,5),F(3,4),F(7,5))]:
        for b in range(11):
            assert absorbing_barrier(ss,al,kk,b)==barrier(ss,al,kk,b)
    counts['exact_absorbing_chain_checks']=44

    rows=[]
    for kk in [F(1,2),F(1),F(6,5)]:
        for dd in [F(1,20),F(1,100),F(1,10000)]:
            cert=exact_frontier(s,alpha,kk,dd)
            V=interval_bellman(cert['multiplier'],kk)
            H=point(cert['intercept']);tail=point(cert['multiplier'])*point(F(77,125))**96/2
            diff=V-H
            assert ends(diff)[0]>-mp.mpf('1e-50')
            assert ends(diff)[1]<ends(tail)[1]+mp.mpf('1e-50')
            lower=V-point(cert['multiplier'])*point(dd)-tail
            assert ends(lower)[0]<=ends(point(cert['value']))[1]
            assert ends(point(cert['value']))[1]-ends(lower)[0]<mp.mpf('1e-12')
            display=enclosure(cert['value'])
            row=dict(price=fraction_text(kk),delta=fraction_text(dd),
                     optimum=fraction_text(cert['value']),decimal_enclosure=display,
                     barriers=list(cert['barriers']),weight_first=fraction_text(cert['weight_first']),
                     dual_multiplier=fraction_text(cert['multiplier']),
                     dual_intercept=fraction_text(cert['intercept']),tail_cutoff=cert['cutoff'],
                     exact_dual_slacks=[fraction_text(v) for v in cert['dual_slacks']],
                     bellman_horizon=96,bellman_enclosure=ivout(V),
                     horizon_penalty=ivout(tail),independent_cost_lower=ivout(lower))
            rows.append(row)
            print('Certified',kk,dd,display,cert['barriers'],flush=True)
    counts['exact_displayed_optima']=len(rows)
    counts['independent_interval_bellman_checks']=len(rows)
    assert rows[0]['barriers']==[0,2]  # adjacency is not a valid shortcut

    # General s, alpha family, with exact global support checks.
    for ss in [F(1,3),F(1,2),F(2,3),F(4,5)]:
        for al in [F(1,4),F(3,5),F(1)]:
            for kk in [al,2*al,3*al]:
                for dd in [F(1,5),F(1,50)]:exact_frontier(ss,al,kk,dd)
    counts['general_family_exact_certificates']=72

    # An independent finite-tree LP verifies conditional minimax constraints.
    trees=exhaustive_trees(2);assert len(trees)==2942
    vals=np.array(trees,float);cost=(vals[:,0]+vals[:,1])/2;error=(vals[:,2]+vals[:,3])/2
    for dd in [.2,.3]:
        obj=np.r_[np.zeros(len(trees)),1.]
        A=np.zeros((4,len(trees)+1));A[0,:-1]=vals[:,0];A[1,:-1]=vals[:,1]
        A[0,-1]=A[1,-1]=-1;A[2,:-1]=vals[:,2];A[3,:-1]=vals[:,3]
        ae=np.r_[np.ones(len(trees)),0.][None,:]
        sol=linprog(obj,A_ub=A,b_ub=[0,0,dd,dd],A_eq=ae,b_eq=[1],bounds=(0,None),method='highs')
        bayes=linprog(cost,A_ub=error[None,:],b_ub=[dd],A_eq=np.ones((1,len(trees))),b_eq=[1],bounds=(0,None),method='highs')
        assert sol.success and bayes.success and abs(sol.fun-bayes.fun)<1e-10
    for lam in [F(1),F(10),F(100)]:
        exact=min((t[0]+t[1]+lam*(t[2]+t[3]))/2 for t in trees)
        val=interval_bellman(lam,F(1),N=2)
        assert ends(val)[0]<=mp.mpf(exact.numerator)/exact.denominator<=ends(val)[1]
    counts['enumerated_two_stage_trees']=len(trees)
    counts['independent_conditional_minimax_LPs']=2
    counts['exact_tree_Bellman_comparisons']=3

    # Three labels: finite-tree primal versus likelihood-vector dual recursion.
    P=[[[F(8,10) if h==y else F(1,10) for y in range(3)] for h in range(3)],
       [[F(7,10),F(2,10),F(1,10)], [F(2,10),F(6,10),F(2,10)], [F(1,10),F(2,10),F(7,10)]]]
    costs=[F(1),F(1,2)];trees3=[]
    for d in range(3):trees3.append(([F(0)]*3,[F(int(h!=d)) for h in range(3)]))
    for a in range(2):
        for decisions in itertools.product(range(3),repeat=3):
            trees3.append(([costs[a]]*3,[sum(P[a][h][y] for y in range(3) if decisions[y]!=h) for h in range(3)]))
    for dd in [.25,.35]:
        C=np.array([t[0] for t in trees3],float).T;E=np.array([t[1] for t in trees3],float).T
        A=np.zeros((6,len(trees3)+1));A[:3,:-1]=C;A[:3,-1]=-1;A[3:,:-1]=E
        sol=linprog(np.r_[np.zeros(len(trees3)),1],A_ub=A,b_ub=[0]*3+[dd]*3,
                    A_eq=np.r_[np.ones(len(trees3)),0][None,:],b_eq=[1],bounds=(0,None),method='highs')
        assert sol.success
        u=-sol.ineqlin.marginals[:3];v=-sol.ineqlin.marginals[3:]
        assert abs(sum(u)-1)<1e-10 and min(v)>-1e-10
        def stop(l):return np.dot(v,l)-max(v*l)
        W=min([stop(np.ones(3))]+[float(costs[a])*sum(u)+sum(stop(np.array([float(P[a][h][y]) for h in range(3)])) for y in range(3)) for a in range(2)])
        assert abs(W-dd*sum(v)-sol.fun)<1e-10
    counts['three_hypothesis_tree_LP_dual_checks']=2

    # Price curves: every plotted ordinate has an exact global certificate.
    plot_rows=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.2,3.1))
    colors=['#711c2d','#1c4a70','#268073']
    for dd,color in zip([F(1,20),F(1,100),F(1,10000)],colors):
        base=exact_frontier(s,alpha,F(6,5),dd)['value'];prices=[F(k,80) for k in range(20,121)]
        values=[]
        for kk in prices:
            cert=exact_frontier(s,alpha,kk,dd);ratio=cert['value']/base;values.append(float(ratio))
            plot_rows.append({'price':fraction_text(kk),'delta':fraction_text(dd),'cost_ratio':fraction_text(ratio),
                              'barriers':list(cert['barriers']),'cutoff':cert['cutoff']})
        ax.plot([float(k) for k in prices],values,color=color,lw=1.8,label=fr'$\delta={float(dd):g}$')
    ax.axvline(1.2,color='#555555',ls='--',lw=1);ax.axhline(1,color='#aaaaaa',lw=.6)
    ax.text(1.215,.48,r'Exact dominance threshold $6/5$',rotation=90,fontsize=8,color='#444444')
    ax.set(xlabel=r'Readout price $\kappa=c_E/c_Y$',ylabel='Optimal cost / optimal Y-only cost',xlim=(.25,1.5),ylim=(.2,1.04))
    ax.legend(loc='lower right',frameon=False);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'sharp_price_frontier.pdf');plt.close(fig)
    counts['exact_price_curve_certificates']=len(plot_rows)

    tex=[r'\begin{table}[htbp]',r'\centering\small',r'\begin{tabular}{rrrrr}',r'\toprule',
         r'$\kappa$ & $\delta$ & Barriers & Lower display & Upper display\\',r'\midrule']
    for row in rows:
        kk=F(row['price']);dd=F(row['delta']);i,j=row['barriers'];lo,hi=row['decimal_enclosure']
        tex.append(f'{float(kk):g} & {float(dd):g} & $({i},{j})$ & {lo} & {hi}'+r'\\')
    tex += [r'\bottomrule\end{tabular}',r'\caption{Exact catalogue minimax costs for the noncommuting states in',
            r'\eqref{eq:sharp-states}. Each displayed interval encloses an exactly certified rational optimum;',
            r'its width is at most $10^{-10}$, solely from decimal display rounding. Barrier policies attain',
            r'the exact value. The independent interval Bellman check at horizon 96 agrees within $10^{-12}$.}',
            r'\label{tab:sharp-costs}',r'\end{table}']
    (ROOT/'sharp_cost_table.tex').write_text('\n'.join(tex)+'\n')
    result={'status':'All sharp-characterization checks passed','counts':counts,'exact_costs':rows,
            'quantum_realizability_checks':region,'exact_price_curve':plot_rows,
            'scope':'Analytic proofs establish the full parameter family. Fraction arithmetic certifies displayed optima, barrier policies, dual constraints and infinite tails. Interval Bellman values provide independent checks. SciPy finite-tree LP checks are floating-point diagnostics, not certificates.'}
    (ROOT/'sharp_characterization_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],json.dumps(counts),flush=True)

if __name__=='__main__':main()
