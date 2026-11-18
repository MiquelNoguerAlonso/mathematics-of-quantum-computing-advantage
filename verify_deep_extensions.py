"""Independent numerical and interval checks for the deeper resource theory.

Analytical proofs establish all universal claims. Intervals certify the named
finite examples and arithmetic constants. LP feasibility, matrix identities,
finite trees and exact finite-horizon policy envelopes are checked separately.
"""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
import json, itertools, math
import numpy as np
from scipy.optimize import linprog
import mpmath as mp
from verify_operational_frontier import dm

ROOT=Path(__file__).resolve().parent
mp.mp.dps=90; mp.iv.dps=65; iv=mp.iv
rng=np.random.default_rng(20261117)
counts={}
def ends(x): return mp.mpf(x.a),mp.mpf(x.b)
def point(x):
    if isinstance(x,F): return iv.mpf(x.numerator)/x.denominator
    return iv.mpf(str(x))
def out(x):
    a,b=ends(x)
    with localcontext() as ctx:
        ctx.prec=110;unit=Decimal('1e-45')
        return [str(Decimal(mp.nstr(a,95)).quantize(unit,rounding=ROUND_FLOOR)-unit),
                str(Decimal(mp.nstr(b,95)).quantize(unit,rounding=ROUND_CEILING)+unit)]
def ceil40(x):
    with localcontext() as ctx:
        ctx.prec=110;unit=Decimal('1e-40')
        return Decimal(mp.nstr(x,95)).quantize(unit,rounding=ROUND_CEILING)+unit
def shown(x,up=False,places=2):
    return format(Decimal(str(x)).quantize(Decimal(10)**(-places),
                  rounding=ROUND_CEILING if up else ROUND_FLOOR),f',.{places}f')
def kl(p,q):return p*iv.log(p/q)+(1-p)*iv.log((1-p)/(1-q))

# Proved constants are enclosed independently with directed interval arithmetic.
al=3*iv.pi/32;aa=32/iv.pi**2*al**2*iv.exp(-2*al)
b1=32/iv.pi**2*al**2/(iv.exp(2*al)-1);b2=32/iv.pi**2/(iv.exp(2)-1)
assert ends(b1)[1]<ends(b2)[0];bb=b1
kap=iv.log(3)/(2*iv.log(4));h0=1-iv.sqrt(3)/2;q0=1+1/iv.sqrt(2)
a=h0*q0;b=kap/3;local_lower=a*b/(a+b)
a=h0*q0/4;b=kap/(12*(1+iv.log(2)));global_lower=a*b/(a+b)
global_upper=9/(4*aa*bb)
assert ends(local_lower)[0]>mp.mpf(1)/12
assert ends(global_lower)[0]>mp.mpf(1)/70
assert ends(global_upper)[1]<42
constant_results=dict(fixed_depth_lower=out(local_lower),global_lower=out(global_lower),
                      global_upper=out(global_upper),a_sharp=out(aa),b_sharp=out(bb))
counts['interval_constant_inequalities']=5

# Candidate measured information comes from the independent variational solver.
for _ in range(90):
    r=mp.mpf(str(rng.uniform(.03,.999999)))
    theta=mp.mpf(str(rng.uniform(.003,np.pi/4)))
    s=r*mp.sin(theta);eta=s*s/(1-r*r);G=1+mp.log1p(eta)
    measured,phi,gap=dm(r,theta)
    assert measured>=mp.mpf(2)/5*s*s*G
    assert measured+gap<=3*s*s*G
counts['independent_measured_information_cases']=90

# Check full resource bounds at random inputs, phase-cap discontinuities, and
# endpoints. Stable scalar expressions avoid cancellation near the pure limit.
def resources(mu,gamma,D,c0,c1,delta):
    ts=np.arange(1,min(D,int(np.floor(np.pi/(8*mu))))+1,dtype=float)
    theta=2*mu*ts;r=np.exp(-gamma*ts);s=r*np.sin(theta);ss=s*s
    eta=np.sin(theta)**2/np.expm1(2*gamma*ts);G=1+np.log1p(eta)
    costs=c0+c1*ts;j=ss/(1+np.sqrt(1-ss))
    B=np.log(1/delta);Lambda=np.log((1-delta)/delta)
    ly=2*np.arctanh(s);by=np.ceil(Lambda/ly)
    uy=costs*by/s*np.tanh(by*ly/2)
    le=np.log1p(2*r*np.sin(theta)**2/(-np.expm1(-gamma*ts)))
    be=np.ceil(Lambda/le)
    ue=costs*be*(1+np.cos(theta))/(r*np.sin(theta)**2)*np.tanh(be*le/2)
    assert np.all(np.minimum(uy,ue)<=2.25*costs/ss*(1+B/G)*(1+1e-12))
    # The sharp proven upper on measured divergence suffices for the converse.
    k=(1-2*delta)*Lambda;h=1-2*np.sqrt(delta*(1-delta))
    lower=max(h/max(j/costs),k/max(3*ss*G/costs))
    R=min(D,1/mu,1/gamma);A=(c0/R**2+c1/R)/mu**2;GG=1+np.log1p(mu*mu*R/gamma)
    scale=A*(1+B/GG);tsharp=min(len(ts),int(np.ceil(R/2)))
    upper=uy[tsharp-1] if eta[tsharp-1]<1 else ue[tsharp-1]
    assert lower>=scale/70*(1-1e-12)
    assert upper<=42*scale*(1+1e-12)
    assert lower<=min(np.minimum(uy,ue))*(1+1e-12)
    return lower/scale,upper/scale
inputs=[]
for _ in range(600):
    inputs.append((10**rng.uniform(-4,np.log10(.125)),10**rng.uniform(-25,0),
                   int(rng.integers(1,151)),0 if rng.random()<.3 else 10**rng.uniform(-2,4),
                   10**rng.uniform(-1,1),10**rng.uniform(-30,np.log10(.25))))
for D in [1,2,3,4,8,16,64]:
    for g in [1.,.125,1e-15]:inputs.append((.125,g,D,0.,1.,.25))
for n in [3,4,5,10,20,50]:
    for sign in [-1,1]:
        mu=np.pi/(8*(n+1+sign*1e-8));inputs.append((mu,mu,100,100.,1.,.25))
ratios=[resources(*args) for args in inputs]
counts['global_constant_parameter_cases']=len(inputs)

# Interval-certified finite readout gains and overhead tolerance.
gain_rows=[]
for gs,ds in [('1e-8','1e-6'),('1e-12','1e-9'),('1e-20','1e-12')]:
    mu=point('.02');ga=point(gs);delta=point(ds)
    Lambda=iv.log((1-delta)/delta);k=(1-2*delta)*Lambda
    cap=19;assert ends(iv.pi/(8*mu))[0]>cap and ends(iv.pi/(8*mu))[1]<cap+1
    iylo=mp.mpf(0);iyhi=mp.mpf(0);best=None
    for t in range(1,cap+1):
        r=iv.exp(-ga*t);theta=2*mu*t;s=r*iv.sin(theta)
        iy=s*iv.log((1+s)/(1-s))/t
        iylo=max(iylo,ends(iy)[0]);iyhi=max(iyhi,ends(iy)[1])
        le=iv.log(1+2*r*iv.sin(theta)**2/(1-r))
        b_lo,b_hi=ends(Lambda/le)
        b=int(mp.ceil(b_hi));assert b==int(mp.ceil(b_lo))
        ez=iv.exp(b*le)
        ue=t*b*(1+iv.cos(theta))/(r*iv.sin(theta)**2)*(ez-1)/(ez+1)
        err=1/(1+ez);assert ends(err)[1]<=ends(delta)[0]
        if best is None or ends(ue)[1]<ends(best[0])[1]:best=(ue,t,b,err)
    ly=k/iv.mpf([iylo,iyhi]);ue,t,b,err=best
    ratio=ly/ue
    gain_rows.append(dict(gamma=gs,delta=ds,L_Y=out(ly),U_E=out(ue),
                          certified_overhead_threshold=out(ratio),depth=t,boundary=b,error=out(err)))
counts['gain_interval_depths']=57
tex=[r'\begin{table}[htbp]',r'\centering\small',r'\begin{tabular}{rrrrrr}',r'\toprule',
     r'$\gamma$ & $\delta$ & $t_E$ & $L_Y$ & $U_E$ & Certified $\kappa$ threshold\\',r'\midrule']
for row in gain_rows:
    gs=row['gamma'].replace('1e-',r'10^{-')+'}';ds=row['delta'].replace('1e-',r'10^{-')+'}'
    tex.append(f"${gs}$ & ${ds}$ & {row['depth']} & {shown(row['L_Y'][0])} & {shown(row['U_E'][1],True)} & {shown(row['certified_overhead_threshold'][0])}"+r'\\')
tex += [r'\bottomrule\end{tabular}',r'\caption{Finite readout-gain certificates at $\mu=.02$, $D=256$, $c_0=0$, $c_1=1$.',
        r'$L_Y$ and the overhead threshold are rounded down; the attainable $U_E$ is rounded up.',
        r'All 19 allowed depths are included using outward interval arithmetic.',
        r'Any price multiplier strictly below the displayed threshold retains the certified gain over fixed $Y$.}',
        r'\label{tab:deep-gains}',r'\end{table}']
(ROOT/'deep_gain_table.tex').write_text('\n'.join(tex)+'\n')

# Exact rational three-state experiment. Matrix checks use separate float traces.
def outer(v):
    w=np.array([F(int(x)) for x in v],dtype=object)
    return np.outer(w,w)
nu=F(3,1000);eye=np.eye(3,dtype=int).astype(object)
states=[]
for h in range(3):
    v=[1,1,1];v[h]=2;states.append((1-nu)*outer(v)/6+nu*eye/3)
bases=[[outer(np.eye(3,dtype=int)[i]) for i in range(3)]]
base=[outer([2,1,1])/6,outer([0,1,-1])/2,outer([-1,1,1])/3]
for h in range(3):
    perm=list(range(3));perm[0],perm[h]=perm[h],perm[0]
    bases.append([e[np.ix_(perm,perm)] for e in base])
P=np.empty((3,4,3),dtype=object)
for h in range(3):
    rho=np.array(states[h],float);assert np.min(np.linalg.eigvalsh(rho))>0
    for a in range(4):
        assert np.all(sum(bases[a])==eye)
        for y in range(3):
            P[h,a,y]=sum((states[h]@bases[a][y]).diagonal())
            assert P[h,a,y]>0
        assert sum(P[h,a])==1
for h in range(3):
    for k0 in range(h):
        assert np.linalg.norm(np.array(states[h]@states[k0]-states[k0]@states[h],float))>.01
probs=np.array(P,float)
di={};ji={}
for h in range(3):
    for k0 in range(3):
        if h==k0:continue
        for a in range(4):
            pp=[point(x) for x in P[h,a]];qq=[point(x) for x in P[k0,a]]
            di[h,k0,a]=sum(p*iv.log(p/q) for p,q in zip(pp,qq))
            ji[h,k0,a]=1-sum(iv.sqrt(p*q) for p,q in zip(pp,qq))
counts['qutrit_probability_vectors']=12

catalogue_rows=[];ordered=[(h,k) for h in range(3) for k in range(3) if h!=k]
pairs=[(0,1),(0,2),(1,2)]
for delta_s in ['.01','.000001']:
    for overhead in [.25,1,4]:
        costs=np.array([1,overhead,overhead,overhead],float)
        delta=float(delta_s);kdel=(1-2*delta)*np.log((1-delta)/delta);hdel=1-2*np.sqrt(delta*(1-delta))
        A=[];rhs=[]
        for h in range(3):
            row=np.zeros(13);row[h*4:h*4+4]=costs;row[-1]=-1;A.append(row);rhs.append(0)
        for h,k0 in ordered:
            row=np.zeros(13)
            row[h*4:h*4+4]=[-float(ends(di[h,k0,a])[0]) for a in range(4)]
            A.append(row);rhs.append(-kdel)
        for h,k0 in pairs:
            row=np.zeros(13)
            for a in range(4):row[h*4+a]=row[k0*4+a]=-float(ends(ji[h,k0,a])[0])/2
            A.append(row);rhs.append(-hdel)
        obj=np.zeros(13);obj[-1]=1
        sol=linprog(obj,A_ub=A,b_ub=rhs,bounds=(0,None),method='highs');assert sol.success
        dual=-sol.ineqlin.marginals
        ar={pair:point(max(0.,dual[3+i])) for i,pair in enumerate(ordered)}
        br={pair:point(max(0.,dual[9+i])) for i,pair in enumerate(pairs)}
        # Inflate the cost multipliers, then normalize; interval residuals
        # certify all dual inequalities without trusting solver tolerances.
        rawu=[]
        for h in range(3):
            needs=[]
            for a in range(4):
                term=sum(ar[h,k0]*di[h,k0,a]+br[tuple(sorted((h,k0)))]*ji[h,k0,a]/2 for k0 in range(3) if k0!=h)
                needs.append(ends(term/point(costs[a]))[1])
            rawu.append(ceil40(max(needs)))
        with localcontext() as ctx:
            ctx.prec=110;norm=sum(rawu)+Decimal('1e-38')
        normiv=point(norm);uu=[point(u)/normiv for u in rawu]
        ar={x:y/normiv for x,y in ar.items()};br={x:y/normiv for x,y in br.items()}
        assert ends(sum(uu))[1]<1
        for h in range(3):
            for a in range(4):
                term=sum(ar[h,k0]*di[h,k0,a]+br[tuple(sorted((h,k0)))]*ji[h,k0,a]/2 for k0 in range(3) if k0!=h)
                assert ends(term)[1]<=ends(uu[h]*point(costs[a]))[0]
        dd=point(delta_s);kk=(1-2*dd)*iv.log((1-dd)/dd);hh=1-2*iv.sqrt(dd*(1-dd))
        lower=kk*sum(ar.values())+hh*sum(br.values())
        assert abs(float(ends(lower)[0])-sol.fun)<1e-8
        best=None
        for zz in range(401):
            q=np.array([zz/400]+[(1-zz/400)/3]*3)
            aff=max(sum(q[a]*(1-float(ends(ji[h,k0,a])[0])) for a in range(4)) for h,k0 in pairs)
            nn=int(np.ceil(np.log(2/delta)/(-np.log(aff))))
            value=nn*np.dot(q,costs)
            if best is None or value<best[0]-1e-12 or (abs(value-best[0])<=1e-12 and zz>best[1]):best=(value,zz)
        zz=best[1];q=[F(zz,400)]+[F(400-zz,1200)]*3
        affs=[sum(point(q[a])*(1-ji[h,k0,a]) for a in range(4)) for h,k0 in pairs]
        am=max(ends(af)[1] for af in affs)
        amiv=iv.mpf([am,am]);nn=int(mp.ceil(ends(iv.log(2/dd)/(-iv.log(amiv)))[1]))
        assert ends(2*amiv**nn)[1]<=ends(dd)[0]
        upper=nn*sum(point(q[a])*point(costs[a]) for a in range(4))
        upper_exact=nn*sum(q[a]*F(str(costs[a])) for a in range(4))
        with localcontext() as ctx:
            ctx.prec=110;upper_decimal=str(Decimal(upper_exact.numerator)/Decimal(upper_exact.denominator))
        assert ends(lower)[1]<ends(upper)[0]
        catalogue_rows.append(dict(delta=delta_s,measurement_price=overhead,lower=out(lower),upper=out(upper),
             computational_weight=str(q[0]),upper_exact_decimal=upper_decimal,observations=nn,dual_alpha={f'{h},{k}':out(v) for (h,k),v in ar.items()},
             dual_beta={f'{h},{k}':out(v) for (h,k),v in br.items()},dual_cost_weights=[out(u) for u in uu]))
counts['interval_feasible_catalogue_duals']=len(catalogue_rows)
counts['interval_certified_catalogue_designs']=len(catalogue_rows)
tex=[r'\begin{table}[htbp]',r'\centering\small',r'\begin{tabular}{rrrrrr}',r'\toprule',
     r'$\delta$ & $\kappa$ & Lower certificate & Attainable upper & $q_{\rm comp}$ & $n$\\',r'\midrule']
for row in catalogue_rows:
    ds=r'10^{-2}' if row['delta']=='.01' else r'10^{-6}'
    tex.append(f"${ds}$ & {row['measurement_price']} & {shown(row['lower'][0])} & {shown(row['upper_exact_decimal'],True)} & {float(F(row['computational_weight'])):.3f} & {row['observations']}"+r'\\')
tex += [r'\bottomrule\end{tabular}',r'\caption{Three-state catalogue at noise weight $\nu=.003$. Lower values use feasible interval-verified LP duals and are rounded down;',
        r'upper values use a recorded randomized design and are rounded up. The measurement-price multiplier changes the selected design.',
        r'The reported interval is a bound on catalogue minimax cost, not its exact value.}',r'\label{tab:multihypothesis}',r'\end{table}']
(ROOT/'multihypothesis_results_table.tex').write_text('\n'.join(tex)+'\n')

# Independent finite-tree identities for every pair of the three hypotheses.
for rep in range(40):
    active=[np.ones(3)];leaves=[];occ=np.zeros((3,4));loss={pair:0. for pair in pairs}
    for depth in range(5):
        nxt=[]
        for masses in active:
            if depth and rng.random()<.2:leaves.append(masses);continue
            a=int(rng.integers(0,4));occ[:,a]+=masses
            for h,k0 in pairs:loss[h,k0]+=np.sqrt(masses[h]*masses[k0])*float(ends(ji[h,k0,a])[0])
            for y in range(3):nxt.append(masses*probs[:,a,y])
        active=nxt
    leaves.extend(active);leaves=np.array(leaves)
    for h,k0 in pairs:
        af=np.sqrt(leaves[:,h]*leaves[:,k0]).sum()
        assert abs(1-af-loss[h,k0])<1e-12
        budget=sum((occ[h,a]+occ[k0,a])/2*float(ends(ji[h,k0,a])[0]) for a in range(4))
        assert 1-af<=budget+1e-12
    for h,k0 in ordered:
        exact=sum(leaves[:,h]*np.log(leaves[:,h]/leaves[:,k0]))
        budget=sum(occ[h,a]*float(ends(di[h,k0,a])[0]) for a in range(4))
        assert abs(exact-budget)<1e-11
counts['three_hypothesis_adaptive_trees']=40

# The directional policy needs no reflection or equal-affinity assumption.
for rep in range(20):
    pp=rng.uniform(.55,.95,2);qq=rng.uniform(.05,.45,2)
    affinities=np.sqrt(pp*qq)+np.sqrt((1-pp)*(1-qq));am=max(affinities)
    zz=np.array([0.]);hp=np.array([1.]);hm=np.array([1.])
    for n in range(13):
        assert hp[zz<=0].sum()<=am**n+1e-12
        assert hm[zz>=0].sum()<=am**n+1e-12
        assert np.dot(hp,np.exp(-zz/2))<=am**n+1e-12
        which=(zz<0).astype(int);pnow=pp[which];qnow=qq[which]
        zz=np.concatenate([zz+np.log(pnow/qnow),zz+np.log((1-pnow)/(1-qnow))])
        hp=np.concatenate([hp*pnow,hp*(1-pnow)]);hm=np.concatenate([hm*qnow,hm*(1-qnow)])
counts['asymmetric_directional_policy_trees']=20

# Binary Bellman: exact finite policy envelopes versus interpolated recursion.
catalog=[(np.array([.7,.3]),np.array([.3,.7]),1.),
         (np.array([.88,.12]),np.array([.12,.88]),2.3)]
def prune(vectors):
    lines=sorted([(v[0]-v[1],v[1]) for v in vectors],reverse=True)
    uniq=[]
    for a,b in lines:
        if uniq and abs(a-uniq[-1][0])<1e-12:uniq[-1]=(a,min(b,uniq[-1][1]))
        else:uniq.append((a,b))
    hull=[];starts=[]
    for a,b in uniq:
        x=-np.inf
        while hull:
            ap,bp=hull[-1];x=(b-bp)/(ap-a)
            if x>starts[-1]+1e-13:break
            hull.pop();starts.pop()
        if not hull:x=-np.inf
        hull.append((a,b));starts.append(x)
    keep=[]
    for i,(a,b) in enumerate(hull):
        right=starts[i+1] if i+1<len(starts) else np.inf
        if starts[i]<1+1e-13 and right>0-1e-13:keep.append((a+b,b))
    return np.array(keep)
grid=np.linspace(0,1,2001);lam=20.;v=lam*np.minimum(grid,1-grid)
exact=np.array([[0.,lam],[lam,0.]])
for n in range(1,7):
    candidates=[[0.,lam],[lam,0.]]
    for pp,pm,c in catalog:
        for left,right in itertools.product(exact,repeat=2):
            candidates.append([c+pp[0]*left[0]+pp[1]*right[0],c+pm[0]*left[1]+pm[1]*right[1]])
    exact=prune(candidates)
    vals=[lam*np.minimum(grid,1-grid)]
    for pp,pm,c in catalog:
        new=np.full(grid.shape,c)
        for y in range(2):
            mix=grid*pp[y]+(1-grid)*pm[y];post=grid*pp[y]/mix
            new+=mix*np.interp(post,grid,v)
        vals.append(new)
    v=np.min(vals,axis=0)
    ex=np.min(np.array([grid*a+(1-grid)*b for a,b in exact]),axis=0)
    assert np.max(v-ex)<1e-10
    assert np.max(ex-v)<.02
counts['exact_policy_envelope_horizons']=6

bellman_rows=[];grid=np.linspace(0,1,1001);delta=.05;s=.4
ly=np.log((1+s)/(1-s));boundary=np.ceil(np.log((1-delta)/delta)/ly)
attainable=boundary/s*np.tanh(boundary*ly/2)
for lam in [5.,20.,80.]:
    v=lam*np.minimum(grid,1-grid);n=6000
    for _ in range(n):
        vals=[lam*np.minimum(grid,1-grid)]
        for pp,pm,c in catalog:
            new=np.full(grid.shape,c)
            for y in range(2):
                mix=grid*pp[y]+(1-grid)*pm[y];post=grid*pp[y]/mix
                new+=mix*np.interp(post,grid,v)
            vals.append(new)
        v=np.min(vals,axis=0)
    lower=v[500]-lam*lam/(4*n)-lam*delta
    assert lower<=attainable+1e-10
    bellman_rows.append(dict(lambda_penalty=lam,horizon=n,grid_points=len(grid),
                            lower_approximation=float(lower),known_attainable_cost=float(attainable)))
counts['bellman_lower_certificate_checks']=3

result=dict(status='All deep-extension checks passed',counts=counts,constants=constant_results,
    global_grid=dict(minimum_lower_over_scale=min(x[0] for x in ratios),maximum_constructive_over_scale=max(x[1] for x in ratios)),
    readout_gain_certificates=gain_rows,multihypothesis_certificates=catalogue_rows,bellman_checks=bellman_rows,
    scope='Universal inequalities are proved analytically; intervals certify displayed constants, cost examples and LP dual feasibility. Bellman and random-tree checks use floating-point arithmetic.')
(ROOT/'deep_extension_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['multihypothesis_certificates']},indent=2))
for row in catalogue_rows:print('Catalogue',row['delta'],row['measurement_price'],row['lower'],row['upper'],row['computational_weight'])
