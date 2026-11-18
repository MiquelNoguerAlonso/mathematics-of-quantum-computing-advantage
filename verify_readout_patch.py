"""Independent matrix and integer-depth checks for the readout refinements.

Checks the SLD/Bures chi-square formula, random and spectral measurements,
noise-dominance bounds, the confidence-independent noiseless construction,
and the new every-readout loss certificates. No hardware data are used.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import rel_entr

ROOT=Path(__file__).resolve().parent
RNG=np.random.default_rng(202611151)
I=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]],complex)
K=2*np.sqrt(2)*np.log1p(np.sqrt(2))
COUNTS=dict(block_sld=0,random_povm=0,spectral_chi_saturation=0,
            general_density_pairs=0,readout_gap=0,factor_three=0,
            noiseless_resource=0,global_noise_condition=0,
            independent_table_depths=0)
max_relative_sld_error=0.

def state(r,theta,h):
    return (I+r*(np.cos(theta)*X+h*np.sin(theta)*Y))/2

def sld(rho,sigma):
    w,v=np.linalg.eigh(sigma)
    diff=v.conj().T@(rho-sigma)@v
    L=v@(2*diff/(w[:,None]+w[None,:]))@v.conj().T
    assert np.allclose((sigma@L+L@sigma)/2,rho-sigma,rtol=1e-7,atol=2e-9)
    return L,float(np.trace((rho-sigma)@L).real)

def probs(rho,effects):
    return np.array([np.trace(rho@e).real for e in effects])

def random_povm(n):
    raw=[]
    for _ in range(n+2):
        a=RNG.normal(size=(n,n))+1j*RNG.normal(size=(n,n))
        raw.append(a.conj().T@a)
    w,v=np.linalg.eigh(sum(raw))
    inv=(v/np.sqrt(w))@v.conj().T
    return [inv@a@inv for a in raw]

def check_measurement(rho,sigma,effects,chi_bound):
    p,q=probs(rho,effects),probs(sigma,effects)
    chi=float(np.sum((p-q)**2/q));d=float(np.sum(rel_entr(p,q)))
    assert chi<=chi_bound+2e-8*max(1.,chi_bound)
    assert d<=np.log1p(chi)+1e-10
    return chi,d

for i in range(4000):
    r=float(RNG.uniform(.002,.98) if i%2 else 1-10**RNG.uniform(-7,-.02))
    theta=float(10**RNG.uniform(-3,np.log10(np.pi/4)))
    rp,rm=state(r,theta,1),state(r,theta,-1)
    L,chi=sld(rp,rm)
    signal=r*np.sin(theta)
    predicted=4*signal**2*(1+signal**2/(1-r*r))
    error=abs(chi-predicted)/max(predicted,1e-30)
    max_relative_sld_error=max(max_relative_sld_error,error)
    assert np.isclose(chi,predicted,rtol=2e-8,atol=1e-11)
    COUNTS['block_sld']+=1
    check_measurement(rp,rm,random_povm(2),chi)
    COUNTS['random_povm']+=1
    _,vectors=np.linalg.eigh(L)
    spectral=[np.outer(vectors[:,j],vectors[:,j].conj()) for j in range(2)]
    measured_chi,_=check_measurement(rp,rm,spectral,chi)
    assert np.isclose(measured_chi,chi,rtol=2e-8,atol=1e-10)
    COUNTS['spectral_chi_saturation']+=1
    dy=2*signal*np.arctanh(signal)
    effects=[(I+np.cos(a)*X+np.sin(a)*Y)/4 for a in [theta,theta+np.pi,-theta,-theta+np.pi]]
    p,q=probs(rp,effects),probs(rm,effects)
    da=float(np.sum(rel_entr(p,q)))
    lower=np.log1p(2*r*np.sin(theta)**2/(1-r))/(2*K*r)
    assert lower<=da/dy+1e-6
    dq=2*r*np.arctanh(r)*np.sin(theta)**2
    assert max(dy,da)<=dq+1e-8
    assert dq/dy<=np.arctanh(r)/r+1e-9
    COUNTS['readout_gap']+=1
    if np.sin(theta)**2<=1-r:
        assert np.log1p(predicted)<=3*dy+1e-12
        COUNTS['factor_three']+=1

# The lemma is not limited to qubits: test unrelated full-rank pairs.
for i in range(250):
    n=2+i%5
    matrices=[]
    for _ in range(2):
        a=RNG.normal(size=(n,n))+1j*RNG.normal(size=(n,n))
        a=a@a.conj().T+.05*np.eye(n);matrices.append(a/np.trace(a))
    rho,sigma=matrices
    L,chi=sld(rho,sigma)
    check_measurement(rho,sigma,random_povm(n),chi)
    COUNTS['general_density_pairs']+=1

max_constructive_to_14_bound=0.
for i in range(4000):
    mu=float(10**RNG.uniform(-6,np.log10(.125)))
    D=int(RNG.integers(1,4097));c0=float(0 if i%4==0 else 10**RNG.uniform(-3,4))
    c1=float(10**RNG.uniform(-3,2))
    reff=min(D,1/mu);t0=max(1,int(np.floor(np.pi*reff/8)))
    cap=min(D,int(np.floor(np.pi/(8*mu))))
    assert 1<=t0<=cap and t0>=np.pi*reff/16-1e-12
    cost=(c0+c1*t0)/(2*np.sin(mu*t0)**2)
    upper=14/mu**2*(c0/reff**2+c1/reff)
    assert cost<=upper*(1+1e-12)
    max_constructive_to_14_bound=max(max_constructive_to_14_bound,cost/upper)
    COUNTS['noiseless_resource']+=1
    # Sample noise rates at and above the global sufficient condition.
    threshold=8*mu*mu*cap
    if threshold<=1:
        gamma=float(threshold+(1-threshold)*RNG.uniform(0,1))
        ts=np.arange(1,cap+1)
        assert np.all(np.sin(2*mu*ts)**2<=-np.expm1(-gamma*ts)+1e-14)
        COUNTS['global_noise_condition']+=1

# Recompute reported quantum rates through direct matrix logarithms at
# every allowed depth, rather than the closed-form entropy expression.
def matrix_entropy(rho,sigma):
    def logm(a):
        w,v=np.linalg.eigh(a);return (v*np.log(w))@v.conj().T
    return float(np.trace(rho@(logm(rho)-logm(sigma))).real)

research=json.loads((ROOT/'research_verification_results.json').read_text())
case=json.loads((ROOT/'case_study_results.json').read_text())
check_rows=[]
for row in research['cases']:
    check_rows.append((row['name'],row['mu'],row['gamma'],row['D'],row['c0'],row['c1'],row['readout_bound']))
for row in case['coherent_rows']:
    check_rows.append((f"amplitude_{row['gamma']}",case['mu'],row['gamma'],256,20,10,row['L_Q_all_readouts']))
checked_bounds={}
k=.98*np.log(99)
for name,mu,gamma,D,c0,c1,expected in check_rows:
    cap=min(D,int(np.floor(np.pi/(8*mu))))
    values=[]
    for t in range(1,cap+1):
        theta=2*mu*t;r=np.exp(-gamma*t)
        if gamma>0:
            values.append(matrix_entropy(state(r,theta,1),state(r,theta,-1))/(c0+c1*t))
        else:
            # POVM probability, directly traced against a pure density matrix.
            ep=(I-state(1,theta,-1))/(1+np.cos(theta))
            success=float(np.trace(state(1,theta,1)@ep).real)
            values.append((c0+c1*t)/success)
        COUNTS['independent_table_depths']+=1
    bound=k/max(values) if gamma>0 else min(values)
    assert np.isclose(bound,expected,rtol=1e-9)
    checked_bounds[name]=float(bound)
assert checked_bounds['Costly']>next(r['UC'] for r in research['cases'] if r['name']=='Costly')
assert checked_bounds['amplitude_0.02']>case['U_R']
assert np.ceil(10*checked_bounds['Ideal'])/10==69.1
assert np.floor(10*checked_bounds['Restarts'])/10==1134.6
result=dict(status='All readout-patch checks passed',counts=COUNTS,
            max_relative_sld_error=max_relative_sld_error,
            maximum_constructive_to_14_bound=max_constructive_to_14_bound,
            independent_cost_bounds=checked_bounds,
            amplitude_every_readout_loss_margin=(checked_bounds['amplitude_0.02']/case['U_R']-1),
            finite_confidence_ratio_uniform_bound=6*(1+2*np.log1p(np.sqrt(2))/np.log(3)))
(ROOT/'readout_patch_verification_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
