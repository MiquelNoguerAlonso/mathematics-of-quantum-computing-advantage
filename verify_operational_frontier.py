"""Operational frontier verification: measured information and a stopped-affinity converse.

The mathematical proofs are in the accompanying manuscript. Numeric stationary
points use 80-digit arithmetic and an analytic concavity-gap diagnostic;
this code does not claim formally rounded interval certification.
"""
from pathlib import Path
import json
import numpy as np
import mpmath as mp
from scipy.optimize import minimize_scalar,brentq
from scipy.special import rel_entr
mp.mp.dps=80
ROOT=Path(__file__).resolve().parent

def dm(r,theta):
    rf,tf=float(r),float(theta)
    def f(phi):
        p=(1+rf*np.cos(phi-tf))/2;q=(1+rf*np.cos(phi+tf))/2
        return float(rel_entr(p,q)+rel_entr(1-p,1-q))
    phis=np.linspace(0,np.pi,513);vals=np.array([f(a) for a in phis])
    j=int(vals.argmax())
    opt=minimize_scalar(lambda a:-f(a),bounds=(phis[max(0,j-1)],phis[min(512,j+1)]),method='bounded',options={'xatol':1e-14})
    def pieces(phi):
        p=(1+r*mp.cos(phi-theta))/2;q=(1+r*mp.cos(phi+theta))/2
        l1=p/q;l2=(1-p)/(1-q)
        divided=mp.log(l1/l2)/(l1-l2) if l1!=l2 else 1/l1
        g=r/2*(mp.sin(theta-phi)*divided+mp.sin(theta+phi))
        D=p*mp.log(l1)+(1-p)*mp.log(l2)
        return D,g,l1,l2
    phi=mp.findroot(lambda a:pieces(a)[1],(mp.mpf(str(opt.x-.00001)),mp.mpf(str(opt.x+.00001))),tol=mp.mpf('1e-65'))
    D,g,l1,l2=pieces(phi)
    k=(r*mp.sin(theta))**2/(1-r*r)
    gap=4*mp.sqrt(k*(1+k))*abs(g)
    dy=2*r*mp.sin(theta)*mp.atanh(r*mp.sin(theta))
    dq=2*r*mp.atanh(r)*mp.sin(theta)**2
    assert D>dy and D<dq
    assert gap<mp.mpf('1e-50')
    return D,phi,gap

def main():
    cases=[('Low noise',mp.mpf('.02'),mp.mpf('.0001'),256,0,1),
           ('Moderate noise',mp.mpf('.02'),mp.mpf('.01'),256,0,1),
           ('Noise limited',mp.mpf('.002'),mp.mpf('.02'),256,0,1)]
    a0=mp.mpf('.04');Delta=mp.mpf('.004')
    mu=(mp.asin(mp.sqrt(a0+Delta))-mp.asin(mp.sqrt(a0-Delta)))/2
    for g in ['.004','.015','.020']:cases.append(('Amplitude '+g,mu,mp.mpf(g),256,20,10))
    rows=[];maxgap=mp.mpf(0);depths=0
    for name,mu0,gamma,D,c0,c1 in cases:
        cap=min(D,int(mp.floor(mp.pi/(8*mu0))))
        bestY=(mp.mpf(0),0);bestM=(mp.mpf(0),0);bestQ=(mp.mpf(0),0);bestangle=None
        for t in range(1,cap+1):
            r=mp.exp(-gamma*t);theta=2*mu0*t;c=c0+c1*t
            d,phi,gap=dm(r,theta);maxgap=max(maxgap,gap);depths+=1
            dy=2*r*mp.sin(theta)*mp.atanh(r*mp.sin(theta))
            dq=2*r*mp.atanh(r)*mp.sin(theta)**2
            if dy/c>bestY[0]:bestY=(dy/c,t)
            if d/c>bestM[0]:bestM=(d/c,t);bestangle=phi
            if dq/c>bestQ[0]:bestQ=(dq/c,t)
        rows.append(dict(name=name,mu=str(mu0),gamma=str(gamma),depth_cap=cap,
            I_Y=float(bestY[0]),t_Y=bestY[1],I_M_estimate=float(bestM[0]),t_M=bestM[1],
            I_Q=float(bestQ[0]),t_Q=bestQ[1],measurement_angle=float(bestangle),
            I_M_over_I_Y=float(bestM[0]/bestY[0]),I_Q_over_I_M=float(bestQ[0]/bestM[0]),
            general_readout_KL_lower=float(mp.mpf('.98')*mp.log(99)/bestM[0])))

    # Enumerate finite adaptive pure-state trees; independently check the
    # stopped Hellinger telescoping identity and its expected-cost bound.
    rng=np.random.default_rng(654981)
    I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex)
    def rho(theta,h):return (I+np.cos(theta)*X+h*np.sin(theta)*Y)/2
    tree_count=0;max_telescope_error=0.
    for rep in range(80):
        muv=float(rng.uniform(.015,.12));D=int(rng.integers(1,9));D=min(D,int(np.floor(np.pi/(8*muv))))
        c0=float(rng.uniform(0,3));c1=float(rng.uniform(.2,2))
        J=max((1-np.cos(2*muv*t))/(c0+c1*t) for t in range(1,D+1))
        active=[(1.,1.,0)];leaves=[];affinity_loss=0.;weighted_cost=0.;Cplus=Cminus=0.
        for n in range(6):
            nxt=[]
            for pp,pm,key in active:
                if n and rng.uniform()<.15:
                    leaves.append((pp,pm));continue
                t=int(rng.integers(1,D+1));theta=2*muv*t;c=c0+c1*t
                rp,rm=rho(theta,1),rho(theta,-1)
                if (key+n)%3==0:
                    q=np.cos(theta);ep=(I-rm)/(1+q);em=(I-rp)/(1+q)
                    effects=[ep,em,I-ep-em]
                else:
                    phi=float(rng.uniform(0,np.pi));e=(I+np.cos(phi)*X+np.sin(phi)*Y)/2
                    effects=[e,I-e]
                p=np.maximum(0.,[np.trace(rp@e).real for e in effects]);q=np.maximum(0.,[np.trace(rm@e).real for e in effects])
                p/=p.sum();q/=q.sum()
                a=float(np.sqrt(p*q).sum());w=np.sqrt(pp*pm)
                assert a>=np.cos(theta)-1e-7
                affinity_loss+=w*(1-a);weighted_cost+=w*c
                Cplus+=pp*c;Cminus+=pm*c
                for j,(pj,qj) in enumerate(zip(p,q)):
                    pair=(pp*pj,pm*qj)
                    if min(pair)<1e-17 or n==5:leaves.append(pair)
                    else:nxt.append((*pair,key*3+j+1))
            active=nxt
        fidelity=sum(np.sqrt(pp*pm) for pp,pm in leaves)
        err=abs(1-fidelity-affinity_loss);max_telescope_error=max(max_telescope_error,err)
        assert err<1e-11
        assert 1-fidelity<=J*weighted_cost+1e-7
        assert weighted_cost<=(Cplus+Cminus)/2+1e-10
        tree_count+=1

    # Sensitivity of the manuscript's narrow gamma=.020 loss certificate.
    case=json.loads((ROOT/'case_study_results.json').read_text())
    mu_float=float(mu);UR=case['U_R'];k=.98*np.log(99)
    def quantum_lower(gamma):
        ts=np.arange(1,int(np.floor(np.pi/(8*mu_float)))+1);r=np.exp(-gamma*ts)
        rates=2*r*np.arctanh(r)*np.sin(2*mu_float*ts)**2/(20+10*ts)
        return float(k/max(rates))
    gamma_crit=brentq(lambda g:quantum_lower(g)-UR,.015,.025,xtol=1e-15)
    LQ=quantum_lower(.02)
    result=dict(status='All independent audit checks passed',measured_information_rows=rows,
                optimized_depths=depths,max_concavity_gap_diagnostic=str(maxgap),
                arithmetic='80 decimal digits; no interval rounding certification',
                stopped_affinity_tree_cases=tree_count,max_affinity_telescoping_error=max_telescope_error,
                noiseless_uniform_lower_constant=float((1-mp.sqrt(3)/2)/2),
                amplitude_loss=dict(gamma=.02,L_Q=LQ,U_R=UR,relative_bound_margin=LQ/UR-1,
                                   coherent_cost_scale_threshold=UR/LQ,gamma_threshold=gamma_crit,
                                   relative_noise_reduction_to_threshold=1-gamma_crit/.02))
    (ROOT/'operational_frontier_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
