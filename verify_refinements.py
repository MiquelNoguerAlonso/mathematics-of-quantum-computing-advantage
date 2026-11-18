"""Checks for matrix identities and conditional-signal robustness.

The dynamic programs optimize finite-horizon error and duration over all
history-dependent choices in the stated signal interval. The proofs, rather
than these finite grids, establish the infinite-horizon guarantees.
"""
from pathlib import Path
import json
import numpy as np
import scipy.linalg as la
import mpmath as mp
from verify_operational_frontier import dm

ROOT=Path(__file__).resolve().parent
rng=np.random.default_rng(20261116)
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1.,-1.])
counts={}
omega=np.array([1,0,0,1])/np.sqrt(2)
bell=np.outer(omega,omega)
partial=bell.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4)
assert np.allclose(la.eigvalsh(partial),[-.5,.5,.5,.5],atol=1e-14)
counts['partial_transpose']=1

def dephase(v,rho):return ((1+v)*rho+(1-v)*(Z@rho@Z))/2
max_spectrum_error=0.
for _ in range(320):
    a=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2));rho=a@a.conj().T;rho/=np.trace(rho)
    v,w=rng.uniform(0,1,2)
    assert np.allclose(dephase(v,dephase(w,rho)),dephase(v*w,rho),atol=2e-14)
    r=float(rng.uniform(.02,.999));theta=float(rng.uniform(.001,np.pi/4))
    rp=(I+r*(np.cos(theta)*X+np.sin(theta)*Y))/2
    rm=(I+r*(np.cos(theta)*X-np.sin(theta)*Y))/2
    eta=r*r*np.sin(theta)**2/(1-r*r)
    actual=la.eigvalsh(rp,rm)
    high=1+2*eta+2*np.sqrt(eta*(1+eta));exact=np.array([1/high,high])
    er=float(np.max(np.abs(actual/exact-1)));max_spectrum_error=max(max_spectrum_error,er)
    assert er<1e-9
    assert np.isclose(np.trace(la.solve(rm,rp)),2+4*eta,rtol=1e-10)
counts['dephasing_semigroup']=320;counts['generalized_eigenvalue_pairs']=320

# The supplied review's 2,000-point grid does not guarantee a 1e-5 residual.
# Record its actual error indicator, then refine with the existing independent
# 80-digit stationary-point solver used by the operational certificate suite.
r=.85;theta=.35;phis=np.linspace(0,np.pi,2000)
def pieces(phi):
    p=(1+r*np.cos(phi-theta))/2;q=(1+r*np.cos(phi+theta))/2
    l1=p/q;l2=(1-p)/(1-q)
    div=(np.log(l1)-np.log(l2))/(l1-l2) if abs(l1-l2)>1e-12 else 1/l1
    d=p*np.log(l1)+(1-p)*np.log(l2)
    g=r/2*(np.sin(theta-phi)*div+np.sin(theta+phi))
    eta=r*r*np.sin(theta)**2/(1-r*r)
    return d,4*np.sqrt(eta*(1+eta))*abs(g)
coarse_phi=phis[np.argmax([pieces(phi)[0] for phi in phis])]
coarse_value,coarse_residual=pieces(coarse_phi)
refined_value,refined_phi,refined_residual=dm(mp.mpf('.85'),mp.mpf('.35'))
assert refined_residual<mp.mpf('1e-50')
assert coarse_value<=float(refined_value)+1e-13
assert float(refined_value)<=coarse_value+coarse_residual+1e-13
counts['coarse_and_refined_residual']=1

# Exact finite-state Bellman recursions: the extremum over an interval of
# Bernoulli probabilities is attained at one of its endpoints. Each horizon
# permits a different optimizing action at every history/state.
rows=[];H=4000
for floor in [.08,.15,.3,.55]:
    for delta in [.2,.1,.01,.001]:
        ell=np.log((1+floor)/(1-floor));b=int(np.ceil(np.log((1-delta)/delta)/ell))
        n=2*b+1;mid=b;states=np.arange(-b,b+1)
        pmin=(1+floor)/2;pmax=(1+1/np.sqrt(2))/2
        err=np.zeros(n);err[0]=1
        duration=np.zeros(n)
        bound=1/(1+np.exp(b*ell))
        for horizon in range(1,H+1):
            lo=pmin*err[2:]+(1-pmin)*err[:-2]
            hi=pmax*err[2:]+(1-pmax)*err[:-2]
            newer=np.zeros(n);newer[0]=1;newer[1:-1]=np.maximum(lo,hi)
            lo=pmin*duration[2:]+(1-pmin)*duration[:-2]
            hi=pmax*duration[2:]+(1-pmax)*duration[:-2]
            newduration=np.zeros(n);newduration[1:-1]=1+np.maximum(lo,hi)
            err,duration=newer,newduration
        assert err[mid]<=bound+2e-13 and bound<=delta+2e-13
        assert duration[mid]<=b/floor+1e-8
        assert np.all(duration[1:-1]<=(b-states[1:-1])/floor+1e-8)
        for signal in np.linspace(floor,1/np.sqrt(2),51):
            assert np.cosh(ell)-signal*np.sinh(ell)<=1+1e-13
        rows.append(dict(signal_floor=floor,delta=delta,boundary=b,horizon=H,
             worst_error_before_horizon=float(err[mid]),proved_error_bound=float(bound),
             worst_truncated_mean=float(duration[mid]),proved_mean_bound=float(b/floor)))
counts['adversarial_drift_dynamic_programs']=len(rows)
counts['conditional_supermartingale_checks']=len(rows)*51
result=dict(status='All matrix and drift refinement checks passed',counts=counts,
    max_generalized_eigenvalue_relative_error=max_spectrum_error,
    angular_grid={'grid_points':2000,'coarse_residual':coarse_residual,
                  'coarse_residual_below_1e_minus_5':bool(coarse_residual<1e-5),
                  'refined_residual':str(refined_residual),'refined_angle':str(refined_phi),
                  'note':'The grid assertion is not adopted; the residual is controlled by refinement.'},
    drift_cases=rows,note='Finite deterministic checks support the supplied analytical proofs; no formal proof-assistant claim.')
(ROOT/'refinement_verification_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='drift_cases'},indent=2))
