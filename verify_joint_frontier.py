"""Independent checks of the uniform noise-depth-confidence theorem.

Analytical proofs, not numerical grids, establish uniformity. This suite checks
scalar inequalities, arbitrary POVMs, mixed-state adaptive trees, exact lazy
random-walk stopping equations, and explicit constants over a seeded grid.
The figure evaluates proved bounds in 70-digit arithmetic; it is not an
outward-interval certificate. No API or market data is required.
"""
from pathlib import Path
import json, math
import mpmath as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
mp.mp.dps=70
rng=np.random.default_rng(20261115)
M=lambda x:mp.mpf(str(x))
counts={}
K0=16*mp.pi*mp.exp(2)*(mp.exp(2)-1)

def bernkl(p,q):
    return p*mp.log(p/q)+(1-p)*mp.log((1-p)/(1-q))

def local(mu,gamma,t,c0,c1,delta):
    r=mp.exp(-gamma*t);theta=2*mu*t;c=c0+c1*t
    sn=mp.sin(theta);q=mp.cos(theta);s=r*sn;s2=s*s
    # Rationalized fidelity loss avoids cancellation at small s.
    j=s2/(1+mp.sqrt(1-s2))
    eta=sn*sn/mp.expm1(2*gamma*t)
    G=1+mp.log1p(eta);logR=2*mp.asinh(mp.sqrt(eta))
    dm_upper=2*(2+logR)*j
    dy=2*s*mp.atanh(s)
    Lambda=mp.log((1-delta)/delta);B=mp.log(1/delta)
    ly=2*mp.atanh(s);by=int(mp.ceil(Lambda/ly))
    uy=c*by/s*mp.tanh(by*ly/2)
    le=mp.log1p(2*r*sn*sn/(-mp.expm1(-gamma*t)))
    be=int(mp.ceil(Lambda/le))
    drift=r*sn*sn/(1+q)
    ue=c*be/drift*mp.tanh(be*le/2)
    assert min(uy,ue)<=4*c/s2*(1+B/G)*(1+mp.mpf('1e-55'))
    assert dm_upper<=7*s2*G*(1+mp.mpf('1e-55'))
    return dict(t=t,c=c,r=r,theta=theta,s2=s2,j=j,eta=eta,G=G,
                dm_upper=dm_upper,dy=dy,uy=uy,ue=ue,le=le,be=be,drift=drift)

def resources(mu,gamma,D,c0,c1,delta):
    cap=min(D,int(mp.floor(mp.pi/(8*mu))))
    rows=[local(mu,gamma,t,c0,c1,delta) for t in range(1,cap+1)]
    R=min(M(D),1/mu,1/gamma)
    A=(c0/(R*R)+c1/R)/(mu*mu)
    G=1+mp.log1p(mu*mu*R/gamma);B=mp.log(1/delta)
    J=max(d['j']/d['c'] for d in rows)
    IMbar=max(d['dm_upper']/d['c'] for d in rows)
    k=(1-2*delta)*mp.log((1-delta)/delta)
    h=1-2*mp.sqrt(delta*(1-delta))
    lower=max(h/J,k/IMbar)
    upper=min(min(d['uy'],d['ue']) for d in rows)
    t0=max(1,int(mp.floor(mp.pi*R/8)));z=rows[t0-1]
    t0upper=z['uy'] if z['eta']<1 else z['ue']
    scale=A*(1+B/G)
    assert lower<=upper*(1+mp.mpf('1e-45'))
    assert lower>=scale/448*(1-mp.mpf('1e-45'))
    assert t0upper<=K0*scale*(1+mp.mpf('1e-45'))
    assert J<=4/A and IMbar<=56*G/A
    a0=mp.exp(-2)/8;b0=2/(mp.pi*(mp.exp(2)-1))
    assert z['s2']/z['c']>=a0/A
    assert z['G']>=b0*G
    return dict(mu=str(mu),gamma=str(gamma),D=D,c0=str(c0),c1=str(c1),
                delta=str(delta),cap=cap,t0=t0,A=float(A),G=float(G),
                scale=float(scale),lower=float(lower),upper=float(upper),
                t0_upper=float(t0upper),lower_over_scale=float(lower/scale),
                upper_over_scale=float(upper/scale))

# 1. Functional domination, with likelihood ratios across 240 log units.
for x in np.concatenate([np.linspace(-120,-.001,240),np.linspace(.001,120,240)]):
    v=mp.exp(M(x)/2);R=max(1,v*v)*mp.mpf('1.01')
    f=v*v*mp.log(v*v)-v*v+1
    assert f<=(2+mp.log(R))*(v-1)**2*(1+mp.mpf('1e-55'))
counts['scalar_divergence']=480

# 2. Arbitrary finite rank-one POVMs; no restriction to planar projectors.
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]],complex)
def rho(r,theta,h):return (I+r*np.cos(theta)*X+h*r*np.sin(theta)*Y)/2
for _ in range(320):
    r=float(rng.uniform(.05,.9999));theta=float(10**rng.uniform(-3,math.log10(math.pi/4)))
    rp,rm=rho(r,theta,1),rho(r,theta,-1)
    n=int(rng.integers(3,9));w=rng.normal(size=(n,2))+1j*rng.normal(size=(n,2))
    raw=[np.outer(v,v.conj()) for v in w];vals,vecs=np.linalg.eigh(sum(raw))
    inv=(vecs*(1/np.sqrt(vals)))@vecs.conj().T;E=[inv@e@inv for e in raw]
    p=np.array([np.trace(rp@e).real for e in E]);q=np.array([np.trace(rm@e).real for e in E])
    s=r*np.sin(theta);eta=s*s/(1-r*r);R=(np.sqrt(1+eta)+np.sqrt(eta))**2
    aff=float(np.sqrt(p*q).sum());F=np.sqrt(1-s*s)
    assert np.min(p/q)>=1/R-1e-10 and np.max(p/q)<=R+1e-10
    assert aff>=F-1e-12
    # Independent spectral formula for quantum fidelity.
    ev,uv=np.linalg.eigh(rp);sq=(uv*np.sqrt(ev))@uv.conj().T
    fspec=float(np.sqrt(np.maximum(0,np.linalg.eigvalsh(sq@rm@sq))).sum())
    assert abs(fspec-F)<1e-11
    kl=float((p*np.log(p/q)).sum())
    assert kl<=2*(2+np.log(R))*(1-aff)+1e-12
counts['arbitrary_povm']=320

# 3. Pointwise constants over tiny angles and tiny positive noise; sampled
# measurements give achievable KL values, not estimates of a global maximum.
for u0,theta0 in zip(10**rng.uniform(-12,.3,180),10**rng.uniform(-8,math.log10(math.pi/4),180)):
    u=M(u0);theta=M(theta0);r=mp.exp(-u);s=r*mp.sin(theta);eta=s*s/(1-r*r)
    G=1+mp.log1p(eta);logR=2*mp.asinh(mp.sqrt(eta));j=s*s/(1+mp.sqrt(1-s*s))
    upper=2*(2+logR)*j
    dy=2*s*mp.atanh(s)
    pm=(1-r)/2;qm=pm+r*mp.sin(theta)**2
    da=r*mp.sin(theta)**2/2*(mp.log((1-pm)/(1-qm))+mp.log(qm/pm))
    assert max(dy,da)>=s*s*G/4
    assert max(dy,da)<=upper*(1+mp.mpf('1e-45'))
    for phi in [mp.pi*M(jj)/32 for jj in range(33)]+[mp.pi-theta,theta]:
        p=pm+r*mp.sin((phi-theta)/2)**2
        q=pm+r*mp.sin((phi+theta)/2)**2
        d=bernkl(p,q)
        assert d<=upper*(1+mp.mpf('1e-35'))+mp.mpf('1e-60')
    assert upper<=7*s*s*G
counts['pointwise_cases']=180;counts['projective_candidates']=180*35

# 4. Finite absorbing Markov systems independently recover exact lazy-walk
# mean block count and error, including b=1 and higher-confidence b>1.
markov=0;max_time_rel=0.;max_error_abs=0.
for _ in range(160):
    r=float(rng.uniform(.75,.999));theta=float(rng.uniform(.18,.76))
    delta=float(10**rng.uniform(-5,math.log10(.25)))
    q=np.cos(theta);v=(1-r)/(2*(1+q));u=r*(1-q)+v
    ell=np.log(u/v);b=int(np.ceil(np.log((1-delta)/delta)/ell))
    if b>35:continue
    states=np.arange(-b+1,b);n=len(states);Q=np.zeros((n,n));wrong=np.zeros(n)
    for j,k in enumerate(states):
        Q[j,j]=1-u-v
        if k+1<b:Q[j,j+1]=u
        if k-1>-b:Q[j,j-1]=v
        else:wrong[j]=v
    system=np.eye(n)-Q;mean=np.linalg.solve(system,np.ones(n))[b-1]
    err=np.linalg.solve(system,wrong)[b-1]
    exact=b/(u-v)*np.tanh(b*ell/2);exerr=1/(1+np.exp(b*ell))
    rel=abs(mean/exact-1);ae=abs(err-exerr)
    max_time_rel=max(max_time_rel,rel);max_error_abs=max(max_error_abs,ae)
    assert rel<1e-9 and ae<1e-10 and err<=delta+1e-10
    markov+=1
counts['absorbing_markov_systems']=markov

# 5. Affinity telescoping on finite mixed-state adaptive history trees.
for rep in range(60):
    mu=float(rng.uniform(.02,.12));gamma=float(10**rng.uniform(-4,-.2))
    cap=min(7,int(np.floor(np.pi/(8*mu))));c0=float(rng.uniform(0,4));c1=float(rng.uniform(.2,2))
    J=max((np.exp(-2*gamma*t)*np.sin(2*mu*t)**2)/(1+np.sqrt(1-np.exp(-2*gamma*t)*np.sin(2*mu*t)**2))/(c0+c1*t) for t in range(1,cap+1))
    active=[(1.,1.)];leaves=[];loss=weighted=cp=cm=0.
    for level in range(5):
        nxt=[]
        for pp,pm in active:
            if level and rng.random()<.12:leaves.append((pp,pm));continue
            t=int(rng.integers(1,cap+1));theta=2*mu*t;r=np.exp(-gamma*t);cost=c0+c1*t
            rp,rm=rho(r,theta,1),rho(r,theta,-1)
            if rng.random()<.5:
                q=np.cos(theta);ep=(I-rho(1,theta,-1))/(1+q);em=(I-rho(1,theta,1))/(1+q)
                effects=[ep,em,I-ep-em]
            else:
                phi=float(rng.uniform(0,np.pi));e=(I+np.cos(phi)*X+np.sin(phi)*Y)/2;effects=[e,I-e]
            p=np.array([np.trace(rp@e).real for e in effects]);q=np.array([np.trace(rm@e).real for e in effects])
            af=np.sqrt(p*q).sum();w=np.sqrt(pp*pm)
            loss+=w*(1-af);weighted+=w*cost;cp+=pp*cost;cm+=pm*cost
            for py,qy in zip(p,q):
                pair=(pp*py,pm*qy)
                if level==4:leaves.append(pair)
                else:nxt.append(pair)
        active=nxt
    terminal=sum(np.sqrt(p*q) for p,q in leaves)
    assert abs(1-terminal-loss)<1e-11
    assert loss<=J*weighted+1e-10 and weighted<=(cp+cm)/2+1e-10
counts['mixed_adaptive_trees']=60

# 6. Full parameter sweeps, with every depth evaluated and explicit constants.
grid=[]
for _ in range(240):
    mu=M(10**rng.uniform(-4,math.log10(.125)))
    gamma=M(10**rng.uniform(-13,0));D=int(rng.integers(1,65))
    c0=M(0 if rng.random()<.25 else 10**rng.uniform(-2,4));c1=M(10**rng.uniform(-1,1))
    delta=M(10**rng.uniform(-12,math.log10(.25)))
    grid.append(resources(mu,gamma,D,c0,c1,delta))
# Edges of the allowed parameter ranges and near-pure joint limits.
for mu,gamma,D,c0,c1,delta in [('.125','1',1,0,1,'.25'),('.125','1',100,100,1,'1e-20'),
    ('.001','1e-30',1,0,1,'1e-4'),('.001','1e-30',16,100,1,'1e-40'),
    ('.001','1e-8',16,100,1,'1e-8'),('.001','.1',16,100,1,'.25')]:
    grid.append(resources(M(mu),M(gamma),D,M(c0),M(c1),M(delta)))
counts['full_parameter_cases']=len(grid)

# 7. The separate pure-state boundary is approached by the exact noisy
# counting rule at fixed error; the integer boundary is one for tiny noise.
for mu0 in ['.001','.02','.125']:
    mu=mp.mpf(mu0);cap=min(16,int(mp.floor(mp.pi/(8*mu))))
    for t in [1,cap]:
        z=local(mu,mp.mpf('1e-45'),t,mp.mpf(20),mp.mpf(1),mp.mpf('.01'))
        pure=(20+t)*(1+mp.cos(2*mu*t))/mp.sin(2*mu*t)**2
        assert z['be']==1
        assert abs(z['ue']/pure-1)<mp.mpf('1e-35')
counts['noiseless_boundary_limits']=6

# 8. Reproducible scientific figure, based on exact attainable costs and
# analytic converse formulas. Displayed values are approximate, not intervals.
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.labelsize':9})
fig,axes=plt.subplots(1,2,figsize=(10.2,3.8),sharey=True)
figure_data=[]
for ax,delta in zip(axes,[mp.mpf('.01'),mp.mpf('1e-6')]):
    vals=[resources(mp.mpf('.005'),M(g),16,mp.mpf(20),mp.mpf(1),delta) for g in np.logspace(-12,-1,70)]
    figure_data.extend(vals)
    x=np.array([float(z['gamma']) for z in vals]);lo=np.array([z['lower'] for z in vals]);hi=np.array([z['upper'] for z in vals]);sc=np.array([z['scale'] for z in vals])
    ax.fill_between(x,lo,hi,color='#a7bac9',alpha=.45,label='Converse to attainable cost')
    ax.loglog(x,lo,color='#1c4a70',lw=1.4,label='Analytical lower bound')
    ax.loglog(x,hi,color='#711c2d',lw=1.4,label='Attainable upper cost')
    ax.loglog(x,sc,'--',color='#555555',lw=1.1,label='Uniform scale (coefficient 1)')
    ax.set_title(r'$\delta='+('10^{-2}' if delta>mp.mpf('1e-5') else '10^{-6}')+'$')
    ax.set_xlabel(r'Noise per oracle use $\gamma$');ax.grid(True,which='major',alpha=.22)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Expected total resource cost')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.015),fontsize=8)
fig.tight_layout(rect=(0,.12,1,1));fig.savefig(ROOT/'joint_frontier.pdf',bbox_inches='tight');plt.close(fig)
counts['figure_parameter_cases']=len(figure_data)
result=dict(status='All uniform-frontier checks passed',arithmetic='70 decimal digits for scalar and cost bounds; float64 for independent matrix and Markov checks',
    counts=counts,explicit_constants={'lower':1/448,'upper':str(K0)},max_markov_relative_time_error=max_time_rel,max_markov_absolute_probability_error=max_error_abs,
    grid_range={'minimum_lower_over_scale':min(x['lower_over_scale'] for x in grid),'maximum_upper_over_scale':max(x['upper_over_scale'] for x in grid)},
    parameter_grid=grid,figure_data=figure_data,note='Finite checks supplement the proof. Constants are derived analytically, not fitted to this grid. No claim of formal proof-assistant verification.')
(ROOT/'joint_frontier_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['parameter_grid','figure_data']},indent=2))
