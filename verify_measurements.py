"""Independent matrix checks for the measurement-aware revision.

No quantum hardware is used. Numerical projective optimization supplies
achievable rates, not a certified optimum over all POVMs. All random seeds
are fixed. Rebuilds the paper's measurement table and figure.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
I=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],dtype=complex)
Y=np.array([[0,-1j],[1j,0]],dtype=complex)
def rho(r,theta,h): return (I+r*(np.cos(theta)*X+h*np.sin(theta)*Y))/2
def kl(p,q):
    p,q=np.asarray(p),np.asarray(q)
    return float(np.sum(p*np.log(p/q)))
def mlog(a):
    w,v=np.linalg.eigh(a)
    return (v*np.log(w))@v.conj().T
def probs(state,effects): return np.array([np.trace(state@e).real for e in effects])
def projective(angle):
    e=(I+np.cos(angle)*X+np.sin(angle)*Y)/2
    return [e,I-e]
def usd(theta):
    q=np.cos(theta)
    pp=rho(1,theta,1);pm=rho(1,theta,-1)
    ep=(I-pm)/(1+q);em=(I-pp)/(1+q)
    return [ep,em,I-ep-em]
def y_info(r,theta):
    s=r*np.sin(theta)
    return 2*s*np.arctanh(s)
def q_info(r,theta): return 2*r*np.arctanh(r)*np.sin(theta)**2
def axis_info(r,theta):
    p=(1+r)/2;q=(1+r*np.cos(2*theta))/2
    return (p-q)/2*np.log(p*(1-q)/(q*(1-p)))

counts=dict(matrix_entropy=0,axis_information=0,usd=0,adaptive_identities=0,
            random_povm_data_processing=0,optimized_gap_grid=0)
max_entropy_error=0.
rng=np.random.default_rng(20261115)
for r in [.01,.1,.4,.8,.95,.999,1-1e-7]:
  for theta in [.004,.03,.1,.4,np.pi/4]:
    rp,rm=rho(r,theta,1),rho(r,theta,-1)
    assert np.allclose(X@rp@X,rm)
    dq=np.trace(rp@(mlog(rp)-mlog(rm))).real
    expected=q_info(r,theta)
    max_entropy_error=max(max_entropy_error,abs(dq-expected))
    assert np.isclose(dq,expected,rtol=2e-8,atol=1e-10)
    assert np.isclose(np.trace(rm@(mlog(rm)-mlog(rp))).real,expected,rtol=2e-8,atol=1e-10)
    counts['matrix_entropy']+=2
    E=[e/2 for e in projective(theta)]+[e/2 for e in projective(-theta)]
    p,q=probs(rp,E),probs(rm,E)
    assert np.isclose(kl(p,q),axis_info(r,theta),rtol=1e-8,atol=1e-12)
    assert np.isclose(kl(q,p),kl(p,q),rtol=1e-8,atol=1e-12)
    counts['axis_information']+=1
    e=projective(theta+.12)
    p,q=probs(rp,e),probs(rm,e)
    llr=np.log(p/q);affinity=np.sqrt(p*q).sum()
    assert np.isclose(p@llr,kl(p,q)) and np.isclose(-q@llr,kl(q,p))
    assert np.isclose(p@np.exp(-llr/2),affinity)
    assert np.isclose(q@np.exp(llr/2),affinity)
    mirrored=[X@ee@X for ee in e]
    assert np.allclose(probs(rp,mirrored),q)
    counts['adaptive_identities']+=1
    for _ in range(5):
      raw=[]
      for j in range(4):
        a=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
        raw.append(a.conj().T@a)
      w,v=np.linalg.eigh(sum(raw));inv=(v*(1/np.sqrt(w)))@v.conj().T
      effects=[inv@a@inv for a in raw]
      assert np.allclose(sum(effects),I)
      d=kl(probs(rp,effects),probs(rm,effects))
      assert d<=dq+1e-9
      counts['random_povm_data_processing']+=1

for theta in [.004,.03,.1,.4,np.pi/4]:
  effects=usd(theta);q=np.cos(theta)
  assert np.allclose(sum(effects),I)
  assert min(np.linalg.eigvalsh(e).min() for e in effects)>-1e-14
  for r in [0.,.5,.95,.999,1.]:
    for h in [1,-1]:
      p=probs(rho(r,theta,h),effects)
      correct=0 if h==1 else 1;wrong=1-correct
      con=p[0]+p[1]
      assert np.isclose(con,(1-r*q*q)/(1+q),rtol=1e-9,atol=1e-14)
      assert np.isclose(p[wrong]/con,(1-r)/(2*(1-r*q*q)),rtol=1e-8,atol=1e-10)
      if r==1: assert abs(p[wrong])<1e-14
      counts['usd']+=1
  delta=.01
  r_boundary=(1-2*delta)/(1-2*delta*q*q)
  probs0=probs(rho(r_boundary,theta,1),effects)
  assert np.isclose(probs0[1]/probs0[:2].sum(),delta,atol=2e-10)

# Deterministic propagation of adaptive-testing outcome mass.
# Tracks the full log-likelihood history counts and absorbs at either boundary.
# Outcome counts determine the LLR; policy depends only on its sign.
adaptive_runs=[]
r,theta=.8,.4
p=probs(rho(r,theta,1),projective(theta))
q=probs(rho(r,theta,-1),projective(theta))
steps=np.log(p/q);d=kl(p,q);dr=kl(q,p)
aff=np.sqrt(p*q).sum();L=np.max(abs(steps))
for delta in [.1,.01,.001]:
  B=np.log(1/delta)
  active={(0,0):1.};wrong=0.;expected=0.;wrong_visits=0.
  for n in range(2000):
    mass=sum(active.values());expected+=mass
    if mass<1e-13: break
    nxt={}
    for (k0,k1),w in active.items():
      z=k0*steps[0]+k1*steps[1]
      selections=[(1,1.)] if z>1e-12 else [(-1,1.)] if z<-1e-12 else [(1,.5),(-1,.5)]
      for sign,choose in selections:
        if sign<0: wrong_visits+=w*choose
        pr=p if sign==1 else q
        for outcome in (0,1):
          key=(k0+sign*(outcome==0),k1+sign*(outcome==1))
          zp=key[0]*steps[0]+key[1]*steps[1];weight=w*choose*pr[outcome]
          if zp<=-B:wrong+=weight
          elif zp<B:nxt[key]=nxt.get(key,0.)+weight
    active=nxt
  else: raise AssertionError('adaptive propagation failed to absorb')
  upper=(B+L+abs(d-dr)/(1-aff))/d
  assert wrong<=delta+1e-12 and expected<=upper+1e-10
  assert wrong_visits<=1/(1-aff)+1e-10
  adaptive_runs.append(dict(delta=delta,error=wrong,expected_blocks=expected,
    analytic_cost_bound=upper,wrong_direction_visits=wrong_visits,
    residual_probability=sum(active.values()),propagation_steps=n))

# Actual matrix-measurement costs at t=19, mu=.02, unit query cost.
rows=[]
mu,t,cost,delta=.02,19,19.,.01
theta=2*mu*t
q0=np.cos(theta)
for gamma in [0.,1e-6,1e-4,1e-3,.01]:
  r=np.exp(-gamma*t);e=usd(theta)
  probs_=probs(rho(r,theta,1),e);con=probs_[:2].sum()
  err=probs_[1]/con
  rows.append(dict(gamma=gamma,error=max(0.,float(err)),expected_cost=float(cost/con),
                   meets_target=bool(err<=delta)))

# Achievable projective rate optimized numerically (not a POVM optimum).
rates=[]
for mu,gamma,D,c0,c1 in [(.02,.0001,256,0.,1.),(.02,.01,256,0.,1.),(.002,.02,256,0.,1.)]:
  cap=min(D,int(np.floor(np.pi/(8*mu))))
  best=[(0.,0)]*4
  for t in range(1,cap+1):
    r=np.exp(-gamma*t);theta=2*mu*t;c=c0+c1*t
    def f(phi):
      p=(1+r*np.cos(phi-theta))/2;q=(1+r*np.cos(phi+theta))/2
      return kl([p,1-p],[q,1-q])
    phis=np.linspace(0,np.pi,257)
    fs=np.array([f(phi) for phi in phis]);j=int(fs.argmax())
    opt=minimize_scalar(lambda phi:-f(phi),bounds=(phis[max(0,j-1)],phis[min(256,j+1)]),method='bounded')
    proj=max(fs[j],-opt.fun)
    vals=[y_info(r,theta),axis_info(r,theta),proj,q_info(r,theta)]
    assert vals[0]<=proj+1e-8 and proj<=vals[3]+1e-8
    for j,val in enumerate(vals):
      if val/c>best[j][0]:best[j]=(val/c,t)
  rates.append(dict(mu=mu,gamma=gamma,D=D,c0=c0,c1=c1,
                    Y=best[0],randomized_axis=best[1],numerically_optimized_projective=best[2],quantum_bound=best[3]))

normalized=[]
for mu in [.001,.01,.1]:
 for gamma in [1e-8,1e-5,.001,.1,1.]:
  for D in [1,8,256]:
   for c0,c1 in [(0.,1.),(100.,1.)]:
    ts=np.arange(1,min(D,int(np.floor(np.pi/(8*mu))))+1)
    rr=np.exp(-gamma*ts);theta=2*mu*ts;costs=c0+c1*ts
    iy=np.max(y_info(rr,theta)/costs);iq=np.max(q_info(rr,theta)/costs)
    reff=min(D,1/mu,1/gamma);factor=1+np.log(1/(gamma*reff))
    normalized.append(float(iq/iy/factor));counts['optimized_gap_grid']+=1
assert min(normalized)>.1 and max(normalized)<3

lines=[r'\begin{table}[htbp]',r'\centering\small',r'\begin{tabular}{rrrrl}',r'\toprule',
 r'$\gamma$ & First-conclusive error & Expected cost & Pure-state cost & Meets $\delta=.01$\\',r'\midrule']
for row in rows:
 lines.append(f"{row['gamma']:g} & {row['error']:.6f} & {row['expected_cost']:.3f} & {19/(1-q0):.3f} & "+('Yes' if row['meets_target'] else 'No')+r'\\')
lines += [r'\bottomrule\end{tabular}',r'\caption{Exact first-conclusive readout at $\mu=.02$, $t=19$, $c_0=0$, $c_1=1$.',
 r'The decreasing mean cost at higher noise reflects more conclusive outcomes, including errors;',
 r'rows failing the target are not admissible upper bounds at that confidence.}',r'\label{tab:crossover}',r'\end{table}']
(ROOT/'measurement_results_table.tex').write_text('\n'.join(lines)+'\n')

plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
u=np.geomspace(1e-7,1,400);rr=np.exp(-u)
fig,axs=plt.subplots(1,2,figsize=(10.3,3.6),layout='constrained')
for values,name,color in [(y_info(rr,np.pi/4),'Fixed Y','#1c4a70'),(axis_info(rr,np.pi/4),'Randomized axes','#711c2d'),(q_info(rr,np.pi/4),'Quantum bound','#b17b28')]:
 axs[0].semilogx(u,values,label=name,color=color,lw=2)
axs[0].set(xlabel='Noise per block (gamma t)',ylabel='Information (nats / block)',title='Readout information at phase pi/4');axs[0].legend()
for angle in [.1,.4,np.pi/4]:
 qq=np.cos(angle);err=(1-rr)/(2*(1-rr*qq*qq))
 axs[1].loglog(u,err,label=f'Phase = {angle:.3f}',lw=2)
axs[1].axhline(.01,color='black',ls='--',lw=1,label='Error target = 0.01')
axs[1].set(xlabel='Noise per block (gamma t)',ylabel='First-conclusive error',title='Finite-confidence crossover');axs[1].legend(fontsize=8)
for ax in axs:ax.grid(alpha=.15,which='both')
fig.savefig(ROOT/'measurement_frontier.pdf');plt.close(fig)
result=dict(status='All measurement checks passed',counts=counts,max_matrix_entropy_error=max_entropy_error,
 adaptive_stopping_checks=adaptive_runs,crossover_rows=rows,achievable_rates=rates,
 normalized_gap_ratio_range=[min(normalized),max(normalized)],
 caveat='Projective rates are numerical achievable constructions, not certified global POVM optima.')
(ROOT/'measurement_verification_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
