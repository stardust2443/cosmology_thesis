#!/usr/bin/env python3
"""
compare_samplers_gaussian.py

Concrete demonstration that emcee and CBL's `sample_stretch_move` produce
statistically equivalent posteriors on the same problem.

The CBL sampler is driven through a small C++ shim (`cbl_sampler_shim.so`)
that bridges a Python `ctypes.CFUNCTYPE` callback into the
`std::function<double(std::vector<double>&)>` that CBL requires. This
documents that the SWIG-Python ABI gap can be closed with about 50 lines
of C++ + a compile step, but at the cost of a separate build artefact —
clarifying why we keep the production v5 pipeline on emcee.

Test problem: 2D Gaussian log-likelihood with truth=(0.0, 0.0) and
diagonal covariance with σ=(0.3, 0.4). 32 walkers × 2000 steps each.

Outputs:
  plots/cbl_mcmc/sampler_comparison_emcee_vs_cbl.png
"""
import os, sys, ctypes
import numpy as np
import matplotlib.pyplot as plt
import emcee

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'plots', 'cbl_mcmc')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Load the C++ shim ────────────────────────────────────────────────
SHIM_PATH = os.path.join(HERE, 'cbl_sampler_shim.so')
if not os.path.exists(SHIM_PATH):
    sys.exit(f"Build the shim first:\n"
             f"  see compile incantation at top of cbl_sampler_shim.cpp\n"
             f"  expected at: {SHIM_PATH}")

shim = ctypes.CDLL(SHIM_PATH)
# int run_stretch_move(int npar, int nwalkers, int nsteps,
#                      const double* start_flat,
#                      double (*loglik)(const double*, int),
#                      unsigned int seed, double aa,
#                      double* out_chain_flat)
PY_LOGLIK_T = ctypes.CFUNCTYPE(
    ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int)
shim.run_stretch_move.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    PY_LOGLIK_T,
    ctypes.c_uint, ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
shim.run_stretch_move.restype = ctypes.c_int


# ─── Target: 2D Gaussian log-likelihood ───────────────────────────────
MU = np.array([0.0, 0.0])
SIGMA = np.array([0.30, 0.40])

def loglik_np(x):
    return -0.5 * np.sum(((x - MU) / SIGMA)**2)

# C-ABI callback wrapper for CBL shim
@PY_LOGLIK_T
def loglik_cb(x_ptr, n):
    x = np.frombuffer((ctypes.c_double * n).from_address(
        ctypes.addressof(x_ptr.contents)), dtype=np.float64, count=n)
    return float(loglik_np(x))


# ─── Run config ───────────────────────────────────────────────────────
NPAR = 2
NWALKERS = 32
NSTEPS = 2000
SEED = 42
AA = 2.0

# Common starting positions: small ball at origin
rng = np.random.default_rng(SEED)
start = 1e-3 * rng.standard_normal((NWALKERS, NPAR))

# ─── 1. emcee chain ───────────────────────────────────────────────────
print("Running emcee...")
sampler_e = emcee.EnsembleSampler(NWALKERS, NPAR, loglik_np)
sampler_e.run_mcmc(start.copy(), NSTEPS, progress=False)
chain_e = sampler_e.get_chain()           # (nsteps, nwalkers, npar)
flat_e = sampler_e.get_chain(flat=True, discard=NSTEPS // 4)
acc_e = sampler_e.acceptance_fraction.mean()
print(f"  emcee acceptance = {acc_e:.3f}")

# ─── 2. CBL chain (via shim) ──────────────────────────────────────────
print("Running CBL Sampler via ctypes shim...")
start_flat = np.ascontiguousarray(start.flatten(), dtype=np.float64)
out_chain = np.zeros(NSTEPS * NWALKERS * NPAR, dtype=np.float64)
mean_acc = ctypes.c_double(0.0)

rc = shim.run_stretch_move(
    NPAR, NWALKERS, NSTEPS,
    start_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    loglik_cb,
    SEED, AA,
    out_chain.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.byref(mean_acc),
)
if rc != 0:
    sys.exit(f"CBL shim returned error code {rc}")

chain_c = out_chain.reshape(NSTEPS, NWALKERS, NPAR)
flat_c = chain_c[NSTEPS // 4:].reshape(-1, NPAR)
print(f"  CBL chain shape: {chain_c.shape}")
print(f"  CBL acceptance = {mean_acc.value:.3f}")

# ─── 3. Statistical comparison ────────────────────────────────────────
from scipy.stats import ks_2samp
print("\n=== Statistical comparison ===")
for i, name in enumerate(['x0', 'x1']):
    m_e, s_e = flat_e[:, i].mean(), flat_e[:, i].std()
    m_c, s_c = flat_c[:, i].mean(), flat_c[:, i].std()
    ks, pval = ks_2samp(flat_e[:, i], flat_c[:, i])
    print(f"  {name}: emcee mean={m_e:+.4f} std={s_e:.4f}   "
          f"CBL mean={m_c:+.4f} std={s_c:.4f}   "
          f"KS p={pval:.3f}")
print(f"  Truth: mean=(0.000, 0.000), std=({SIGMA[0]}, {SIGMA[1]})")

# ─── 4. Plot ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'figure.facecolor': 'white',
})
fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                          gridspec_kw=dict(hspace=0.32, wspace=0.28))

# Top: 2D scatter overlay
ax = axes[0, 0]
ax.scatter(flat_e[:, 0], flat_e[:, 1], s=2, alpha=0.18,
           color='#2980b9', rasterized=True, label='emcee')
ax.scatter(flat_c[:, 0], flat_c[:, 1], s=2, alpha=0.18,
           color='#e67e22', rasterized=True, label='CBL (via shim)')
# Truth ellipse
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(SIGMA[0]*np.cos(theta), SIGMA[1]*np.sin(theta),
        'r--', lw=1.5, alpha=0.8, label='target 1σ')
ax.plot(2*SIGMA[0]*np.cos(theta), 2*SIGMA[1]*np.sin(theta),
        'r--', lw=1.0, alpha=0.5, label='target 2σ')
ax.set_xlabel(r'$x_0$'); ax.set_ylabel(r'$x_1$')
ax.set_title('2D posterior overlay')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, ls='--', alpha=0.2)
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)

# 1D marginals
for i, name in enumerate(['$x_0$', '$x_1$']):
    ax = axes[0, 1] if i == 0 else axes[1, 0]
    bins = np.linspace(-1.5, 1.5, 60)
    ax.hist(flat_e[:, i], bins=bins, density=True, histtype='step',
            color='#2980b9', lw=2, label='emcee')
    ax.hist(flat_c[:, i], bins=bins, density=True, histtype='step',
            color='#e67e22', lw=2, label='CBL')
    # Target Gaussian
    xs = np.linspace(-1.5, 1.5, 300)
    target = np.exp(-0.5 * ((xs - MU[i]) / SIGMA[i])**2) / (SIGMA[i] * np.sqrt(2*np.pi))
    ax.plot(xs, target, 'r--', lw=1.5, alpha=0.8, label='target')
    ax.set_xlabel(name); ax.set_ylabel('density')
    ax.set_title(f'1D marginal: {name}')
    ax.legend(fontsize=10)
    ax.grid(True, ls='--', alpha=0.2)

# Bottom-right: walker trace comparison (param x0)
ax = axes[1, 1]
for w in range(0, NWALKERS, 4):
    ax.plot(chain_e[:, w, 0], color='#2980b9', alpha=0.25, lw=0.5)
    ax.plot(chain_c[:, w, 0], color='#e67e22', alpha=0.25, lw=0.5)
ax.plot(chain_e[:, :, 0].mean(axis=1), color='#2980b9', lw=1.5, label='emcee walker mean')
ax.plot(chain_c[:, :, 0].mean(axis=1), color='#e67e22', lw=1.5, label='CBL walker mean')
ax.axhline(MU[0], color='r', ls='--', alpha=0.6, label='truth')
ax.set_xlabel('step'); ax.set_ylabel(r'$x_0$')
ax.set_title(r'Walker traces ($x_0$, 8 of 32 shown)')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, ls='--', alpha=0.2)

fig.suptitle(
    f'emcee vs CBL `sample_stretch_move` — 2D Gaussian'
    f'  ({NWALKERS} walkers × {NSTEPS} steps, seed={SEED})',
    fontsize=14, y=0.995)

out = os.path.join(OUT_DIR, 'sampler_comparison_emcee_vs_cbl.png')
fig.savefig(out, dpi=180, bbox_inches='tight')
print(f"\nSaved: {out}")
