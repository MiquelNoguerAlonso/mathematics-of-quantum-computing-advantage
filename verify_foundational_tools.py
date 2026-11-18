"""Independent small-matrix checks for the integrated foundational sections.

These checks compare constructions with direct spectral calculations.
The accompanying manuscript supplies the mathematical proofs.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import rel_entr

ROOT=Path(__file__).resolve().parent
rng=np.random.default_rng(1547829)
counts={k:0 for k in ['channel_reconstruction','trace_contraction',
    'helstrom_random_effects','pure_copy_error','holevo_measurements',
    'one_copy_Y_optimal','noiseless_depth_optimum','noiseless_two_sided_scale']}

def randmat(d):
    return rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))
def density(d):
    a=randmat(d);r=a@a.conj().T
    return r/np.trace(r)
def invsqrt(a):
    w,v=np.linalg.eigh(a)
    return (v/np.sqrt(w))@v.conj().T
def trace_distance(a,b):
    return float(np.abs(np.linalg.eigvalsh(a-b)).sum()/2)
def entropy(r):
    w=np.maximum(0,np.linalg.eigvalsh(r));w=w[w>0]
    return float(-np.sum(w*np.log(w)))
def povm(d,n):
    mats=[randmat(d) for _ in range(n)]
    mats=[a@a.conj().T for a in mats];s=invsqrt(sum(mats))
    return [s@a@s for a in mats]

for d in [2,3,4]:
    for _ in range(70):
        raw=[randmat(d) for j in range(3)]
        norm=invsqrt(sum(k.conj().T@k for k in raw));ks=[k@norm for k in raw]
        def channel(a):return sum(k@a@k.conj().T for k in ks)
        choi=np.block([[channel(np.outer(np.eye(d)[i],np.eye(d)[j]))
                        for j in range(d)] for i in range(d)])
        w,v=np.linalg.eigh(choi)
        assert w.min()>-1e-11
        reconstructed=[(np.sqrt(ev)*vec).reshape(d,d).T
                       for ev,vec in zip(w,v.T) if ev>1e-12]
        rho,sigma=density(d),density(d)
        restored=sum(k@rho@k.conj().T for k in reconstructed)
        assert np.allclose(restored,channel(rho),atol=2e-12)
        assert np.isclose(np.trace(channel(rho)),1)
        counts['channel_reconstruction']+=1
        assert trace_distance(channel(rho),channel(sigma))<=trace_distance(rho,sigma)+2e-12
        counts['trace_contraction']+=1
        prior=float(rng.uniform(.05,.95));delta=prior*rho-(1-prior)*sigma
        w,v=np.linalg.eigh(delta);e=v[:,w>0]@v[:,w>0].conj().T
        exact=(1-np.abs(w).sum())/2
        direct=prior*np.trace((np.eye(d)-e)@rho).real+(1-prior)*np.trace(e@sigma).real
        assert np.isclose(exact,direct,atol=1e-12)
        for j in range(12):
            effect=povm(d,2)[0]
            risk=prior*np.trace((np.eye(d)-effect)@rho).real+(1-prior)*np.trace(effect@sigma).real
            assert risk>=exact-2e-12
            counts['helstrom_random_effects']+=1
        ensemble=[density(d) for j in range(3)];probs=rng.dirichlet(np.ones(3))
        avg=sum(p*r for p,r in zip(probs,ensemble));effects=povm(d,5)
        cond=np.array([[np.trace(r@effect).real for effect in effects] for r in ensemble])
        marginal=probs@cond
        mutual=float(np.sum(probs[:,None]*rel_entr(cond,marginal[None,:])))
        chi=entropy(avg)-sum(p*entropy(r) for p,r in zip(probs,ensemble))
        assert mutual<=chi+2e-12 and chi<=np.log(d)+2e-12
        counts['holevo_measurements']+=1

for q in np.linspace(.03,.98,50):
    psi=np.array([1.,0.]);phi=np.array([q,np.sqrt(1-q*q)])
    pp=psi;qq=phi
    for n in range(1,5):
        if n>1:pp=np.kron(pp,psi);qq=np.kron(qq,phi)
        error=(1-trace_distance(np.outer(pp,pp),np.outer(qq,qq)))/2
        assert np.isclose(error,(1-np.sqrt(1-q**(2*n)))/2,atol=1e-12)
        counts['pure_copy_error']+=1

I=np.eye(2);X=np.array([[0,1],[1,0]]);Y=np.array([[0,-1j],[1j,0]])
for _ in range(1000):
    theta=float(rng.uniform(.005,np.pi/4));r=float(rng.uniform(.02,.999))
    a=(I+r*(np.cos(theta)*X+np.sin(theta)*Y))/2
    b=(I+r*(np.cos(theta)*X-np.sin(theta)*Y))/2
    assert np.isclose((1-trace_distance(a,b))/2,(1-r*np.sin(theta))/2,atol=1e-13)
    counts['one_copy_Y_optimal']+=1
    mu=float(rng.uniform(.0005,.125));D=int(rng.integers(1,257))
    c0=float(rng.uniform(0,100));c1=float(rng.uniform(.05,10))
    cap=min(D,int(np.floor(np.pi/(8*mu))));ts=np.arange(1,cap+1)
    u=(c0+c1*ts)/(2*np.sin(mu*ts)**2)
    assert np.all(np.diff(u)<0) and np.argmin(u)==cap-1
    counts['noiseless_depth_optimum']+=1
    effective=min(D,1/mu);scale=(c0/effective**2+c1/effective)/mu**2
    assert u[-1]>=.5*scale-1e-8 and u[-1]<=14*scale+1e-8
    confidence=float(rng.uniform(0,.25))
    lower=(1-2*np.sqrt(confidence*(1-confidence)))*u[-1]
    assert lower>=(1-np.sqrt(3)/2)/2*scale-1e-8
    counts['noiseless_two_sided_scale']+=1

result=dict(status='All foundational checks passed',counts=counts,
            note='Matrix and finite-parameter checks supplement the proofs; no formal proof-assistant claim.')
(ROOT/'foundational_verification_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
