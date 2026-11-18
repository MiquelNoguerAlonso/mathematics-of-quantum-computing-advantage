"""Reproduce the decision-cost frontier and certificates.
All costs are exact calculations in the stated synthetic model, not hardware data.
Dependencies: numpy, scipy, matplotlib.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DELTA=0.01

def frontier(mu,gamma,D,c0,c1,delta=DELTA,cC=1.0):
    cap=min(int(D),int(np.floor(np.pi/(8*mu))))
    assert cap>=1 and 0<mu<=.125 and 0<=gamma<=1 and c0>=0 and c1>0
    ts=np.arange(1,cap+1,dtype=float)
    s=np.exp(-gamma*ts)*np.sin(2*mu*ts)
    ell=np.log1p(s)-np.log1p(-s)
    div=s*ell
    costs=c0+c1*ts
    rates=div/costs
    A=np.log((1-delta)/delta);k=(1-2*delta)*A
    h=np.ceil(A/ell)
    upper=costs*h/s*np.tanh(h*ell/2)
    ir=int(np.argmax(rates));iu=int(np.argmin(upper))
    sC=np.sin(2*mu);ellC=np.log1p(sC)-np.log1p(-sC)
    IC=sC*ellC/cC;hC=np.ceil(A/ellC)
    UC=cC*hC/sC*np.tanh(hC*ellC/2)
    LQ=k/rates[ir];LC=k/IC;UQ=upper[iu]
    assert LQ<=UQ*(1+1e-12)
    assert upper[ir]<=(A+ell[ir])/rates[ir]*(1+1e-12)
    verdict="Depth gain" if UQ<LC else ("Depth loss" if LQ>UC else "Unresolved")
    return dict(mu=mu,gamma=gamma,D=int(D),c0=c0,c1=c1,
                rate_depth=int(ts[ir]),finite_depth=int(ts[iu]),
                Istar=float(rates[ir]),IC=float(IC),ratio=float(rates[ir]/IC),
                LQ=float(LQ),UQ=float(UQ),LC=float(LC),UC=float(UC),
                verdict=verdict,s=float(s[iu]),ell=float(ell[iu]),
                h=int(h[iu]),expected_blocks=float(upper[iu]/costs[iu]))

cases=[
    ("Ideal",.02,0.,256,0.,1.),
    ("Shallow",.002,0.,16,0.,1.),
    ("Noisy",.005,.02,256,0.,1.),
    ("Costly",.02,.04,256,50.,5.),
    ("Restarts",.01,.01,256,100.,1.)
]
rows=[]
for name,mu,gamma,D,c0,c1 in cases:
    z=frontier(mu,gamma,D,c0,c1);z["name"]=name
    ts=np.arange(1,min(D,int(np.floor(np.pi/(8*mu))))+1,dtype=float)
    costs=c0+c1*ts;angles=2*mu*ts
    if gamma==0:
        usd_costs=costs/(2*np.sin(angles/2)**2)
        idx=int(np.argmin(usd_costs))
        z.update(readout_bound_kind="upper",readout_bound=float(usd_costs[idx]),
                 readout_bound_depth=int(ts[idx]))
    else:
        vis=np.exp(-gamma*ts)
        qrates=2*vis*np.arctanh(vis)*np.sin(angles)**2/costs
        idx=int(np.argmax(qrates));iq=float(qrates[idx])
        z.update(readout_bound_kind="lower",I_Q=iq,
                 readout_bound=float((1-2*DELTA)*np.log((1-DELTA)/DELTA)/iq),
                 readout_bound_depth=int(ts[idx]),I_Q_over_I_Y=iq/z['Istar'])
        r=np.exp(-gamma*z['rate_depth']);theta=2*mu*z['rate_depth']
        pp=(1+r)/2;qq=(1+r*np.cos(2*theta))/2
        da=(pp-qq)/2*np.log(pp*(1-qq)/(qq*(1-pp)))
        dy=2*r*np.sin(theta)*np.arctanh(r*np.sin(theta))
        z['axis_over_Y_at_Y_optimum']=float(da/dy)
        assert da<=dy
        if z['readout_bound']>z['UC']:z['verdict']='Every readout loses'
    rows.append(z)

# Independent absorbing Markov-chain calculation of both stopping quantities.
walk_checks=[]
for s,h in [(.2,5),(.5,4),(.7,3),(.08,9)]:
    p=(1+s)/2;q=1-p
    states=np.arange(-h+1,h)
    Q=np.zeros((len(states),len(states)));exitminus=np.zeros(len(states))
    for i,j in enumerate(states):
        if j+1<h:Q[i,i+1]=p
        if j-1>-h:Q[i,i-1]=q
        else:exitminus[i]=q
    expected=np.linalg.solve(np.eye(len(states))-Q,np.ones(len(states)))[h-1]
    error=np.linalg.solve(np.eye(len(states))-Q,exitminus)[h-1]
    ell=np.log(p/q)
    predicted_E=h/s*np.tanh(h*ell/2)
    predicted_err=1/(1+np.exp(h*ell))
    assert np.isclose(expected,predicted_E,rtol=1e-11,atol=1e-11)
    assert np.isclose(error,predicted_err,rtol=1e-11,atol=1e-11)
    walk_checks.append(dict(s=s,h=h,expected=float(expected),error=float(error)))

# Parameter grid checks of finite-confidence bounds.
grid_count=0
for mu in np.geomspace(.001,.1,9):
    for gamma in [0.,.001,.01,.05,.2,1.]:
        for c0,c1 in [(0.,1.),(50.,5.),(200.,.5)]:
            frontier(float(mu),gamma,256,c0,c1);grid_count+=1

rng=np.random.default_rng(5257)
perturb_ratios=[]
for _ in range(1000):
    n=5
    G=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    G=(G+G.conj().T)/2;G/=np.linalg.norm(G,2)
    x=rng.normal(size=n)+1j*rng.normal(size=n);x/=np.linalg.norm(x)
    h=rng.normal(size=n)+1j*rng.normal(size=n)
    eta=float(rng.uniform(1e-5,.8));h*=eta/np.linalg.norm(h)
    u=float(np.vdot(x,G@x).real);xp=(x+h)/np.linalg.norm(x+h)
    up=float(np.vdot(xp,G@xp).real)
    R=np.linalg.norm((G-u*np.eye(n))@x);B=np.linalg.norm(G-u*np.eye(n),2)
    bound=(2*R*eta+B*eta**2)/(1-eta)**2
    assert abs(up-u)<=bound+1e-12
    perturb_ratios.append(float(abs(up-u)/bound))

# Finite-difference test of the derivative with matrix and RHS perturbations.
derivative_errors=[]
for _ in range(50):
    n=4
    A=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    A=A.conj().T@A+np.eye(n)
    b=rng.normal(size=n)+1j*rng.normal(size=n)
    E=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));E/=np.linalg.norm(E)
    e=rng.normal(size=n)+1j*rng.normal(size=n);e/=np.linalg.norm(e)
    G=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    G=(G+G.conj().T)/2;G/=np.linalg.norm(G,2)
    y=np.linalg.solve(A,b);r=np.linalg.norm(y);x=y/r
    u=np.vdot(x,G@x).real
    analytic=2/r*np.vdot(x,(G-u*np.eye(n))@np.linalg.solve(A,e-E@y)).real
    step=1e-6
    def util(h):
        yy=np.linalg.solve(A+h*E,b+h*e);xx=yy/np.linalg.norm(yy)
        return np.vdot(xx,G@xx).real
    numerical=(util(step)-util(-step))/(2*step)
    derivative_errors.append(float(abs(numerical-analytic)))
assert max(derivative_errors)<1e-7

# Envelope optimum checked against continuous optimization.
envelope_errors=[]
for _ in range(60):
    gamma=float(10**rng.uniform(-3,0))
    c0=float(rng.uniform(0,200));c1=float(10**rng.uniform(-1,1))
    opt=(c1-2*gamma*c0+np.sqrt((2*gamma*c0-c1)**2+16*gamma*c1*c0))/(4*gamma*c1)
    opt=np.clip(opt,1.,256.)
    def lf(t):return np.log(c0+c1*t)+2*gamma*t-2*np.log(t)
    res=minimize_scalar(lf,bounds=(1,256),method="bounded",options={"xatol":1e-10})
    numerical=min([(res.fun,res.x),(lf(1.),1.),(lf(256.),256.)])[0]
    envelope_errors.append(float(abs(lf(opt)-numerical)))
assert max(envelope_errors)<1e-7

# Confidence allocation: strict convex program on a simplex.
a=np.array([1.,4.,20.,100.]);delta=.01;weights=a/a.sum()
def cost(w):return np.dot(a,np.log(1/(delta*w)))
res=minimize(cost,np.ones(4)/4,method="SLSQP",
             bounds=[(1e-8,1.)]*4,
             constraints={"type":"eq","fun":lambda w:np.sum(w)-1},
             options={"ftol":1e-10,"maxiter":2000})
assert res.success and np.allclose(res.x,weights,atol=2e-6)
entropy=-np.dot(weights,np.log(weights))
optimal=a.sum()*(np.log(1/delta)+entropy)
uniform=a.sum()*np.log(4/delta)
assert np.isclose(optimal,cost(weights))
assert np.isclose(uniform-optimal,a.sum()*(np.log(4)-entropy))

# Task-sensitive example checks both a harmless and a harmful direction.
eps=1e-8;A=np.diag([1.,eps,eps]);b=np.array([1.,0.,0.])
G=np.diag([1.,1.,-1.])
direction_checks=[]
for z in [.01,.2,.8,1.,2.]:
    yp=np.linalg.solve(A,b+np.array([0.,eps*z,0.]))
    xp=yp/np.linalg.norm(yp)
    assert np.isclose(xp@G@xp,1)
    yq=np.linalg.solve(A,b+np.array([0.,0.,eps*z]))
    xq=yq/np.linalg.norm(yq)
    observed=xq@G@xq;exact=(1-z*z)/(1+z*z)
    assert np.isclose(observed,exact)
    direction_checks.append({"z":z,"observable":float(observed)})

# Export exact table values for LaTeX.
tex=[r"\begin{table}[htbp]",r"\centering\small",
     r"\begin{tabular}{lrrrrr}",r"\toprule",
     r"Setting & $\mu$ & $\gamma$ & $D$ & $c_0$ & $c_1$\\",r"\midrule"]
for z in rows:
    tex.append(f"{z['name']} & {z['mu']:g} & {z['gamma']:g} & {z['D']} & {z['c0']:g} & {z['c1']:g}"+r"\\")
tex += [r"\bottomrule\end{tabular}",
        r"\caption{Parameters of the reproducible decision experiments.}",
        r"\end{table}",
        r"\begin{table}[htbp]",r"\centering\small",r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrrl}",r"\toprule",
        r"Setting & $t_\star$ & $t_U$ & POVM cost & $L_Y$ & $U_Y$ & $I_Y/I_C$ & Certificate\\",r"\midrule"]
for z in rows:
    upper=z['readout_bound_kind']=='upper'
    rounded=(np.ceil if upper else np.floor)(10*z['readout_bound'])/10
    sign=r"\leq" if upper else r"\geq"
    display=f"{rounded:,.1f}".replace(",","{,}")
    bound=f"${sign} {display}$"
    verdict=z['verdict']
    tex.append(f"{z['name']} & {z['rate_depth']} & {z['finite_depth']} & {bound} & {z['LQ']:,.1f} & {z['UQ']:,.1f} & {z['ratio']:.3f} & {verdict}"+r"\\")
tex += [r"\bottomrule\end{tabular}",
        r"\caption{Bounds on optimal expected cost at $\delta=0.01$.",
        r"$t_\star,t_U,L_Y,U_Y$ concern fixed $Y$. The POVM-cost column bounds",
        r"$C^\star_{M,\delta}$ at the same block price. For $\gamma=0$ it is the",
        r"zero-error upper bound of Proposition~\ref{prop:usd}, minimized over depth;",
        r"for $\gamma>0$ it is the lower bound $k_\delta/I_Q$ from",
        r"Theorem~\ref{thm:measured}. These inequalities are rounded conservatively.",
        r"The sampler's achievable costs are $U_C=1422.7$ (Ideal, Costly),",
        r"$140{,}889.3$ (Shallow), $22{,}542.6$ (Noisy), and $5636.0$ (Restarts),",
        r"rounded to one decimal. Depth gain means $U_Y<L_C$; in the Costly row,",
        r"$k_\delta/I_Q>U_C$ certifies a loss for every single-block readout.}",
        r"\label{tab:researchresults}",r"\end{table}"]
(ROOT/"research_results_table.tex").write_text("\n".join(tex)+"\n")

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
                     "axes.spines.top":False,"axes.spines.right":False})
mus=np.geomspace(.001,.1,70);gammas=np.geomspace(1e-4,.3,70)
ratios=np.array([[frontier(float(m),float(g),256,50.,5.)["ratio"] for m in mus] for g in gammas])
fig,axs=plt.subplots(1,2,figsize=(10.3,3.8),constrained_layout=True)
mesh=axs[0].pcolormesh(mus,gammas,np.log10(ratios),shading="auto",
                     cmap="RdBu",vmin=-1.5,vmax=1.5)
ct=axs[0].contour(mus,gammas,ratios,levels=[1],colors="black",linewidths=1.2)
axs[0].clabel(ct,fmt={1:"Equal cost rate"},fontsize=8)
axs[0].set(xscale="log",yscale="log",xlabel="Angular half-separation (mu)",
           ylabel="Dephasing per query (gamma)",title="Fixed-Y rate versus depth-one sampling")
fig.colorbar(mesh,ax=axs[0],label="log10(sampling / fixed-Y coherent cost)")
gs=np.geomspace(.001,.3,120)
for c0,color in [(0.,"#1c4a70"),(50.,"#711c2d"),(200.,"#c08628")]:
    depths=[frontier(.005,float(g),256,c0,1.)["rate_depth"] for g in gs]
    axs[1].loglog(gs,depths,color=color,lw=2,label=f"Restart cost = {c0:g}")
axs[1].set(xlabel="Dephasing per query (gamma)",ylabel="Rate-optimal integer depth",
           title="Restart cost shifts the chosen depth")
axs[1].grid(which="both",alpha=.2);axs[1].legend(fontsize=8)
fig.savefig(ROOT/"quantum_decision_frontier.pdf")
fig.savefig(ROOT/"quantum_decision_frontier.png",dpi=180)
plt.close(fig)

result={
 "model":"Calibrated binary restarted phase experiment; no hardware data.",
 "delta":DELTA,"cases":rows,"grid_checks":grid_count,
 "absorbing_chain_checks":walk_checks,
 "random_perturbation_cases":len(perturb_ratios),
 "maximum_actual_to_bound_ratio":max(perturb_ratios),
 "maximum_derivative_error":max(derivative_errors),
 "maximum_envelope_objective_error":max(envelope_errors),
 "confidence_allocation":{"a":a.tolist(),"delta_by_task":(delta*weights).tolist(),
                         "optimal_cost":float(optimal),"uniform_cost":float(uniform),
                         "saving":float(uniform-optimal)},
 "directional_perturbation":direction_checks,
 "status":"All research-model checks passed."
}
(ROOT/"research_verification_results.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))

