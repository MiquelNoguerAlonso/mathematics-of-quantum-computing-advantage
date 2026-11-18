"""Reproduce the paper's small examples and analytic figure.
These classical simulations check formulas; they are not speedup benchmarks.
Run: python3 verify_examples.py
Requires numpy, scipy, matplotlib.
"""
from pathlib import Path
from fractions import Fraction
from math import gcd
import json
import numpy as np
from scipy.linalg import hadamard, expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
checks = {}
tol = 1e-11
H = np.array([[1, 1], [1, -1]], dtype=float) / np.sqrt(2)
assert np.allclose(H @ H, np.eye(2), atol=tol)
checks["hadamard_squared_max_error"] = float(np.max(np.abs(H @ H - np.eye(2))))

# Full Boolean Fourier transform for s=101.
n, s = 3, 0b101
N = 2**n
Hn = hadamard(N) / np.sqrt(N)
phase = np.array([(-1)**((x & s).bit_count() % 2) for x in range(N)])
bv = Hn @ (phase / np.sqrt(N))
assert np.allclose(abs(bv)**2, np.eye(N)[s], atol=tol)
checks["bernstein_vazirani_probabilities"] = (abs(bv)**2).round(12).tolist()

# Build Simon's reversible oracle as an explicit permutation on both registers.
# f(x)=min(x,x xor s) gives exactly the promised paired collisions.
f = [min(x, x ^ s) for x in range(N)]
oracle = np.zeros((N*N, N*N))
for x in range(N):
    for z in range(N):
        oracle[x*N + (z ^ f[x]), x*N + z] = 1
assert np.allclose(oracle.T @ oracle, np.eye(N*N))
init = np.zeros(N*N)
init[0] = 1
first_h = np.kron(Hn, np.eye(N))
out = first_h @ oracle @ first_h @ init
probs = (abs(out.reshape(N,N))**2).sum(axis=1)
expected = np.array([0.25 if (y&s).bit_count()%2 == 0 else 0 for y in range(N)])
assert np.allclose(probs, expected, atol=tol)
checks["simon_probabilities"] = {format(i,"03b"):float(round(p,12)) for i,p in enumerate(probs)}

# Full Grover operator, compared with the closed-form rotation.
N = 8
uniform = np.ones(N)/np.sqrt(N)
oracle = np.eye(N)
oracle[0,0] = -1
diffusion = 2*np.outer(uniform,uniform)-np.eye(N)
Q = diffusion @ oracle
psi = uniform.copy()
grover = []
theta = np.arcsin(1/np.sqrt(N))
for j in range(4):
    p = float(abs(psi[0])**2)
    assert np.isclose(p, np.sin((2*j+1)*theta)**2, atol=tol)
    grover.append(p)
    psi = Q @ psi
assert np.allclose(grover,[1/8,25/32,121/128,169/512],atol=tol)
checks["grover_probabilities"] = grover

def phase_distribution(phi, M):
    k = np.arange(M)
    y = np.arange(M)
    amp = np.exp(2j*np.pi*np.outer(phi-y/M,k)).mean(axis=1)
    return abs(amp)**2

qpe = phase_distribution(3/8,8)
assert np.allclose(qpe,np.eye(8)[3],atol=tol)
checks["qpe_exact_phase"] = qpe.round(12).tolist()

# Actual modular exponentiation state followed by inverse QFT.
N, a, M = 15, 2, 256
r = next(k for k in range(1,N+1) if pow(a,k,N)==1)
state = np.zeros((M,N),dtype=complex)
for k in range(M):
    state[k,pow(a,k,N)] = 1/np.sqrt(M)
# numpy.fft implements the negative-exponent transform.
out = np.fft.fft(state,axis=0)/np.sqrt(M)
p = (abs(out)**2).sum(axis=1)
expected = np.zeros(M)
expected[[0,64,128,192]] = 0.25
assert np.allclose(p,expected,atol=tol)
factors = [gcd(pow(a,r//2)-1,N),gcd(pow(a,r//2)+1,N)]
assert r==4 and factors==[3,5]
assert Fraction(64,256).denominator==4
checks["shor_order_factors"]={"order":r,"factors":factors,
                             "phase_peaks":np.flatnonzero(p>1e-10).tolist()}

# Ancilla success branch of the HHL example.
A = np.diag([1.,0.5])
b = np.ones(2)/np.sqrt(2)
raw = np.linalg.solve(A,b)
success = 0.5*raw
ps = float(np.vdot(success,success).real)
x = raw/np.linalg.norm(raw)
z_expectation = float(np.vdot(x,np.diag([1.,-1.])@x).real)
assert np.isclose(ps,5/8) and np.isclose(z_expectation,-3/5)
checks["linear_system"]={"normalized_solution":x.tolist(),
                         "success_probability":ps,"Z_expectation":z_expectation}

# First-order Lie product formula against matrix exponentiation.
X = np.array([[0.,1.],[1.,0.]])
Z = np.diag([1.,-1.])
t = 0.7
exact = expm(-1j*t*(X+Z))
trotter = []
for steps in [1,2,4,8,16,32,64]:
    approx = np.linalg.matrix_power(expm(-1j*t*X/steps)@expm(-1j*t*Z/steps),steps)
    err = float(np.linalg.norm(exact-approx,2))
    bound = t*t/steps
    assert err <= bound + tol
    trotter.append({"steps":steps,"operator_error":err,"bound":bound})
checks["trotter"] = trotter

# Canonical QAE: equal mixture of the two conjugate eigenphases.
a, M = 0.2, 64
theta = np.arcsin(np.sqrt(a))
p = (phase_distribution(theta/np.pi,M)+phase_distribution(1-theta/np.pi,M))/2
ahat = np.sin(np.pi*np.arange(M)/M)**2
bound = 2*np.pi*np.sqrt(a*(1-a))/M+np.pi**2/M**2
within = float(p[np.abs(ahat-a)<=bound+tol].sum())
assert np.isclose(p.sum(),1,atol=tol) and within >= 8/np.pi**2-tol
checks["amplitude_estimation"]={"a":a,"M":M,"error_bound":bound,
                              "probability_within_bound":within,
                              "guaranteed_probability":8/np.pi**2}

# Spectral dilation and Chebyshev identity in the manuscript.
spectral_errors = []
for eig in [-1.,-0.7,0.,0.3,1.]:
    off = np.sqrt(1-eig**2)
    dilation = np.array([[eig,off],[off,-eig]])
    W = np.diag([1.,-1.]) @ dilation
    assert np.allclose(dilation @ dilation,np.eye(2),atol=tol)
    for degree in range(9):
        err = abs(np.linalg.matrix_power(W,degree)[0,0]
                  -np.cos(degree*np.arccos(eig)))
        spectral_errors.append(float(err))
        assert err < tol
checks["spectral_chebyshev_max_error"] = max(spectral_errors)

# Analytic plot, with no hardware timing interpretation.
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False})
fig,axs = plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
js=np.arange(0,17)
theta=np.arcsin(1/8)
axs[0].plot(js,np.sin((2*js+1)*theta)**2,"o-",color="#711c2d",lw=1.8,ms=4)
axs[0].axhline(1/64,color="#888888",ls="--",lw=1,label="Initial probability")
axs[0].set(xlabel="Grover iterations",ylabel="Success probability",
           title="Coherent rotation: N = 64, K = 1",ylim=(0,1.04),xticks=[0,4,8,12,16])
axs[0].grid(alpha=.2)
axs[0].legend(fontsize=8,loc="upper right")
eps=np.logspace(-1,-4,200)
axs[1].loglog(1/eps,eps**-2,color="#1c4a70",lw=2,label=r"Classical: $\varepsilon^{-2}$")
axs[1].loglog(1/eps,eps**-1,color="#711c2d",lw=2,label=r"Quantum: $\varepsilon^{-1}$")
axs[1].set(xlabel=r"Inverse target error $1/\varepsilon$",
           ylabel="Idealized call count",title="Precision scaling (constants omitted)")
axs[1].grid(alpha=.2,which="both")
axs[1].legend(fontsize=9)
fig.savefig(ROOT/"quantum_advantage_figures.pdf")
fig.savefig(ROOT/"quantum_advantage_figures.png",dpi=160)
plt.close(fig)
checks["status"]="All mathematical example checks passed."
(ROOT/"verification_results.json").write_text(json.dumps(checks,indent=2)+"\n")
print(json.dumps(checks,indent=2))

