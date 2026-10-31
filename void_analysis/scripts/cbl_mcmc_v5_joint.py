#!/usr/bin/env python3
"""
cbl_mcmc_v5_joint.py

Joint MCMC across all 4 NREC50 (or BIT) subsamples with SHARED cosmology
and SHARED nuisance parameters. Tests whether requiring a single
(Omega_m, sigma8, B_slope, B_offset) to explain all four data vectors
simultaneously breaks the degeneracy that hits the per-subsample fits.

Free parameters: 4 total
  - Omega_m, sigma8  : cosmology, shared
  - B_slope, B_offset: Vdn extended-bias nuisance, shared
Total data: ~26 bins (4 subsamples × ~6-7 bins at R > 7 h^-1 Mpc)
Degrees of freedom: ~22

Designed to run on the cluster. Supports:
  - --nwalkers, --nsteps, --thin, --nthreads
  - Checkpoint saving every 100 steps (resume-friendly)
  - --field NREC50|BIT, --rmin <value>
  - Posterior summary + per-subsample chi^2 / chi^2-per-bin output

Local smoke test:
    /Users/arun/miniforge3/bin/python3.12 scripts/cbl_mcmc_v5_joint.py \
        --nwalkers 16 --nsteps 200 --nthreads 4

Cluster (typical):
    python cbl_mcmc_v5_joint.py --nwalkers 64 --nsteps 8000 \
        --thin 1 --nthreads 16 --field NREC50

Outputs:
    data/mcmc_chains/cbl_v5_joint_R{rmin}_{field}.npy        (chain)
    data/mcmc_chains/cbl_v5_joint_R{rmin}_{field}_meta.json  (metadata)
    plots/cbl_mcmc/corner_joint_R{rmin}_{field}{tag}.png     (corner plot)
"""
import os, sys, time, json, argparse
import numpy as np
import pandas as pd

CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

import emcee

HERE = os.path.dirname(os.path.abspath(__file__))
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
OUT_CHAINS = os.path.join(HERE, '..', 'data', 'mcmc_chains')
OUT_PLOTS  = os.path.join(HERE, '..', 'plots', 'cbl_mcmc')
os.makedirs(OUT_CHAINS, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

# ─── Aletheia truth + constants ──────────────────────────────────────
OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05
DELTA_NL, B_EFF = -0.7, 1.0

FA_MAP = {'2_5p':'025','5p':'05','10p':'10','20p':'20'}
SUBSAMPLES = ['20p', '10p', '5p', '2_5p']

# 4 free parameters with flat priors
PRIORS = [
    ('Omega_m',  0.10, 0.50),
    ('sigma8',   0.50, 1.20),
    ('B_slope',  0.30, 1.50),
    ('B_offset', 0.0,  1.0),
]
NDIM = len(PRIORS)
PARAM_NAMES = [p[0] for p in PRIORS]


# ─── Theory helpers ──────────────────────────────────────────────────
def make_cosmo(om, s8):
    c = cbl.cosmology.LCDM(om, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", s8)
    return c

# Size-function model — settable via --model CLI flag. Read by theory_at_R.
# Accepted values: "Vdn" (Jennings 2013, default), "SvdW" (Sheth & van de Weygaert 2004).
# "Verza" requires the lucap-branch cherry-pick (see Experiment E).
_MODEL = "Vdn"


def theory_at_R(R_data, om, s8, slope, offset):
    """One call per R bin. Returns NaN-safe array."""
    try:
        c = make_cosmo(om, s8)
        sf = cbl.cosmology.SizeFunction(c)
        dc = c.deltac(0.0)
        return np.array([
            sf.size_function(float(R), 0.0, _MODEL,
                             B_EFF, slope, offset, DELTA_NL, dc, "EisensteinHu")
            for R in R_data
        ])
    except Exception:
        return None


# ─── Data loading ────────────────────────────────────────────────────
def load_subsample(frac, field, r_min, overlap='OV00', data_dir=None):
    base = data_dir if data_dir else FA
    fpath = os.path.join(base, f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_{field}_'
                                f'SM0p00_CS0p79_TH-0p80_{overlap}.dat')
    df = pd.read_csv(fpath, sep=r'\s+', comment='#', names=['r','vsf','err'])
    m = (df['vsf'] > 0) & (df['r'] >= r_min)
    return df.loc[m, 'r'].values, df.loc[m, 'vsf'].values, df.loc[m, 'err'].values


def assemble_dataset(field, r_min, overlap='OV00', data_dir=None):
    """Load all four subsamples, return as a list of dicts."""
    data = []
    for frac in SUBSAMPLES:
        R, v, e = load_subsample(frac, field, r_min, overlap=overlap, data_dir=data_dir)
        log_v = np.log10(v)
        log_e = e / (v * np.log(10))
        data.append(dict(frac=frac, R=R, vsf=v, err=e,
                         log_vsf=log_v, log_err=log_e))
        print(f"  {field} {frac}: {len(R)} bins from R={R[0]:.2f} to R={R[-1]:.2f}")
    return data


# ─── Joint likelihood ────────────────────────────────────────────────
def log_prior(theta):
    for v, (_, lo, hi) in zip(theta, PRIORS):
        if not (lo < v < hi):
            return -np.inf
    return 0.0

def log_likelihood_joint(theta, data):
    om, s8, slope, offset = theta
    total = 0.0
    for d in data:
        th = theory_at_R(d['R'], om, s8, slope, offset)
        if th is None or np.any(~np.isfinite(th)) or np.any(th <= 0):
            return -np.inf
        total += np.sum(((d['log_vsf'] - np.log10(th)) / d['log_err'])**2)
    return -0.5 * total

def log_posterior(theta, data):
    lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    ll = log_likelihood_joint(theta, data)
    if not np.isfinite(ll): return -np.inf
    return lp + ll


# ─── MAP via scipy ───────────────────────────────────────────────────
def find_map(data):
    from scipy.optimize import minimize
    x0 = np.array([OMEGA_M_TRUE, SIGMA8_TRUE, 0.857, 0.42])
    print("\n[MAP] Nelder-Mead minimization of -log_posterior ...")
    t0 = time.time()
    res = minimize(lambda th: -log_posterior(th, data), x0,
                   method='Nelder-Mead',
                   options=dict(xatol=1e-3, fatol=1e-2, maxiter=600, disp=True))
    elapsed = time.time() - t0
    print(f"[MAP] wall: {elapsed:.1f}s")
    x_map = res.x
    chi2 = -2 * log_likelihood_joint(x_map, data)
    ntot = sum(len(d['R']) for d in data)
    dof = ntot - NDIM
    print(f"[MAP] Ω_m={x_map[0]:.4f}  σ_8={x_map[1]:.4f}  "
          f"B_slope={x_map[2]:.4f}  B_offset={x_map[3]:.4f}")
    print(f"[MAP] χ² total = {chi2:.2f}  Ntot = {ntot}  DoF = {dof}  χ²/ν = {chi2/dof:.3f}")
    return x_map, chi2, dof


# ─── MCMC driver ─────────────────────────────────────────────────────
def run_mcmc(data, x_map, nwalkers, nsteps, nthreads, chain_path, sampler_type='cbl'):
    pos = x_map + 1e-3 * np.random.randn(nwalkers, NDIM)
    for i, (_, lo, hi) in enumerate(PRIORS):
        pos[:, i] = np.clip(pos[:, i], lo + 1e-3, hi - 1e-3)

    print(f"\n[MCMC] {nwalkers} walkers × {nsteps} steps, sampler={sampler_type}, nthreads={nthreads}")

    if sampler_type == 'emcee':
        if nthreads > 1:
            from multiprocessing import Pool
            pool = Pool(nthreads)
        else:
            pool = None

        sampler = emcee.EnsembleSampler(
            nwalkers, NDIM, log_posterior, args=(data,), pool=pool)

        # Checkpoint every 100 steps
        t0 = time.time()
        chunk = 100
        for i in range(0, nsteps, chunk):
            step_end = min(i + chunk, nsteps)
            sampler.run_mcmc(pos if i == 0 else None, step_end - i, progress=True)
            # Save intermediate full chain
            np.save(chain_path, sampler.get_chain())
            elapsed = time.time() - t0
            rate = step_end / elapsed
            eta = (nsteps - step_end) / rate if rate > 0 else 0
            print(f"  [chkpt] step {step_end}/{nsteps}, "
                  f"elapsed={elapsed:.1f}s, ETA={eta/60:.1f} min")

        if pool: pool.close()
        return sampler

    elif sampler_type == 'cbl':
        import ctypes
        shim_path = os.path.join(HERE, 'cbl_sampler_shim.so')
        if not os.path.exists(shim_path):
            raise FileNotFoundError(f"CBL sampler shim not found at {shim_path}. Please compile it first.")

        shim = ctypes.CDLL(shim_path)
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

        @PY_LOGLIK_T
        def loglik_cb(x_ptr, n):
            x = np.frombuffer((ctypes.c_double * n).from_address(
                ctypes.addressof(x_ptr.contents)), dtype=np.float64, count=n)
            return float(log_posterior(x, data))

        # Allocate full chain
        full_chain = np.zeros((nsteps, nwalkers, NDIM), dtype=np.float64)
        chunk = 100
        t0 = time.time()
        current_start = pos.copy()
        rng_seed = int(time.time()) & 0xFFFFFFFF
        mean_acceptances = []

        for i in range(0, nsteps, chunk):
            step_end = min(i + chunk, nsteps)
            chunk_steps = step_end - i

            out_chunk = np.zeros(chunk_steps * nwalkers * NDIM, dtype=np.float64)
            start_flat = np.ascontiguousarray(current_start.flatten(), dtype=np.float64)
            chunk_mean_acc = ctypes.c_double(0.0)

            rc = shim.run_stretch_move(
                NDIM, nwalkers, chunk_steps,
                start_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                loglik_cb,
                rng_seed + i, 2.0,
                out_chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.byref(chunk_mean_acc),
            )
            if rc != 0:
                raise RuntimeError(f"CBL sampler shim returned error code {rc}")

            chunk_chain = out_chunk.reshape(chunk_steps, nwalkers, NDIM)
            full_chain[i:step_end] = chunk_chain
            current_start = chunk_chain[-1].copy()

            np.save(chain_path, full_chain[:step_end])
            mean_acceptances.append(chunk_mean_acc.value)

            elapsed = time.time() - t0
            rate = step_end / elapsed
            eta = (nsteps - step_end) / rate if rate > 0 else 0
            print(f"  [chkpt] step {step_end}/{nsteps}, "
                  f"elapsed={elapsed:.1f}s, ETA={eta/60:.1f} min, acc={chunk_mean_acc.value:.3f}")

        class CBLSamplerWrapper:
            def __init__(self, chain, mean_acc):
                self._chain = chain
                self.acceptance_fraction = np.array([mean_acc] * nwalkers)

            def get_chain(self, discard=0, thin=1, flat=False):
                c = self._chain[discard::thin]
                if flat:
                    return c.reshape(-1, c.shape[-1])
                return c

            def get_autocorr_time(self, quiet=True):
                try:
                    return emcee.autocorr.integrated_time(self._chain, quiet=quiet)
                except Exception:
                    return np.array([self._chain.shape[0] / 50.] * self._chain.shape[-1])

        return CBLSamplerWrapper(full_chain, np.mean(mean_acceptances))


# ─── CLI ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--field', default='NREC50', choices=['NREC50','BIT'])
    ap.add_argument('--rmin',  default=7.0, type=float)
    ap.add_argument('--overlap', default='OV00', choices=['OV00','OV05'],
                    help='Overlap criterion: OV00 (TOUCH) or OV05 (CENTER)')
    ap.add_argument('--nwalkers', default=32, type=int)
    ap.add_argument('--nsteps',   default=2000, type=int)
    ap.add_argument('--nthreads', default=1, type=int)
    ap.add_argument('--tag', default='', help='Optional suffix on output filenames')
    ap.add_argument('--data-dir', default=None,
                    help='Override data dir (default: data/ForArun). '
                         'Use data/ForArun_v4cleaned to run on our cleaned catalogs.')
    ap.add_argument('--model', default='Vdn', choices=['Vdn', 'SvdW', 'Verza'],
                    help='Size-function model passed to cbl.cosmology.SizeFunction. '
                         '"Verza" requires the lucap-branch cherry-pick.')
    ap.add_argument('--sampler', default='cbl', choices=['emcee', 'cbl'],
                    help='Sampler to use: "cbl" uses CBL native sampler (sequential), "emcee" uses emcee.')
    args = ap.parse_args()

    # Set the module-level model so theory_at_R picks it up
    global _MODEL
    _MODEL = args.model

    print(f"Joint MCMC: field={args.field}, overlap={args.overlap}, R>{args.rmin}, "
          f"{args.nwalkers}x{args.nsteps}, threads={args.nthreads}, model={args.model}, sampler={args.sampler}")
    if args.data_dir:
        print(f"           data_dir = {args.data_dir}")

    print("\n[Data] Loading subsamples ...")
    data = assemble_dataset(args.field, args.rmin, overlap=args.overlap, data_dir=args.data_dir)
    ntot = sum(len(d['R']) for d in data)
    print(f"\n[Data] Total bins across all subsamples: {ntot}")

    # MAP first
    x_map, chi2_map, dof_map = find_map(data)

    # MCMC
    tag = f"_{args.tag}" if args.tag else ""
    fn_suffix = f"R{args.rmin:.0f}_{args.field}_{args.overlap}{tag}"
    chain_path = os.path.join(OUT_CHAINS, f'cbl_v5_joint_{fn_suffix}_chain.npy')
    meta_path = os.path.join(OUT_CHAINS, f'cbl_v5_joint_{fn_suffix}_meta.json')

    sampler = run_mcmc(data, x_map, args.nwalkers, args.nsteps,
                       args.nthreads, chain_path, sampler_type=args.sampler)

    # Diagnostics
    print(f"\n[Diagnostics]")
    print(f"  mean acceptance fraction: {sampler.acceptance_fraction.mean():.3f}")
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"  tau: {tau}")
    except emcee.autocorr.AutocorrError as e:
        print(f"  tau warning: {e}")
        tau = np.array([args.nsteps/50.] * NDIM)

    burn = int(0.25 * args.nsteps)
    thin = max(1, int(np.nanmin(tau)/2))
    flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
    print(f"  burn={burn}, thin={thin}, eff samples={len(flat)}")

    # Save flat chain too
    flat_path = chain_path.replace('_chain.npy', '_flat.npy')
    np.save(flat_path, flat)
    print(f"  flat chain: {flat_path}")

    # Posterior summary
    print("\n[Posterior]")
    summary = {}
    for i, name in enumerate(PARAM_NAMES):
        q = np.percentile(flat[:, i], [16, 50, 84])
        truth = OMEGA_M_TRUE if name=='Omega_m' else (SIGMA8_TRUE if name=='sigma8' else None)
        dev = ((q[1] - truth) / max((q[2]-q[0])/2, 1e-6)) if truth is not None else None
        print(f"  {name:<10s} = {q[1]:.4f} (+{q[2]-q[1]:.4f}/-{q[1]-q[0]:.4f})"
              + (f"   truth={truth}  dev={dev:+.2f}σ" if truth is not None else ""))
        summary[name] = dict(median=float(q[1]),
                             p16=float(q[0]), p84=float(q[2]),
                             truth=truth, sigma_dev=dev)

    # Save metadata
    meta = dict(field=args.field, rmin=args.rmin, model=args.model, sampler=args.sampler,
                nwalkers=args.nwalkers, nsteps=args.nsteps,
                nthreads=args.nthreads, ntot_bins=ntot,
                dof=dof_map, chi2_map=float(chi2_map),
                chi2_per_dof_map=float(chi2_map/dof_map),
                x_map=dict(zip(PARAM_NAMES, x_map.tolist())),
                tau=tau.tolist(), burn=burn, thin=thin,
                neff=len(flat), summary=summary)
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n[Meta] {meta_path}")

    # Quick corner
    try:
        import corner
        fig = corner.corner(flat, labels=[r'$\Omega_m$', r'$\sigma_8$',
                                          r'$B_{\rm slope}$', r'$B_{\rm offset}$'],
                            truths=[OMEGA_M_TRUE, SIGMA8_TRUE, None, None],
                            quantiles=[0.16, 0.5, 0.84], show_titles=True,
                            levels=(1-np.exp(-0.5), 1-np.exp(-2)),
                            color='#2196F3', truth_color='#E53935')
        fig.suptitle(f'Joint MCMC ({args.field}, overlap={args.overlap}, R>{args.rmin}, all 4 subsamples)',
                     fontsize=12, y=1.02)
        out_corner = os.path.join(
            OUT_PLOTS, f'corner_joint_{fn_suffix}.png')
        fig.savefig(out_corner, dpi=180, bbox_inches='tight')
        print(f"[Corner] {out_corner}")
    except ImportError:
        print("(corner not installed, skipping corner plot)")

if __name__ == '__main__':
    main()
