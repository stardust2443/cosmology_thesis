#!/usr/bin/env python3
"""
run_mcmc_vsf.py
MCMC parameter estimation of (Ω_m, σ_8) — and optionally δ_v — from the
BitVF Void Size Function, using the Vdn theoretical model.

Supports:
  - Both BitVF (BIT) and OT Reconstructed (NREC50) data vectors
  - Minimum radius cut (--rmin) to avoid the resolution bump
  - Optional 3rd free parameter δ_v (--fit-delta-v)

Usage:
    # Stage A: Restricted-range 2D fit on Simone data
    python run_mcmc_vsf.py --field BIT --subsample 20p --rmin 6.0

    # Stage B: Full-range 3D fit with δ_v free, on Simone data
    python run_mcmc_vsf.py --field BIT --subsample 20p --fit-delta-v

    # Cluster production run on Simone data
    python run_mcmc_vsf.py --field BIT --subsample all --fit-delta-v --nsteps 3000

    # Run on OUR v4 locally-cleaned NREC50 catalog instead of Simone .dat
    python run_mcmc_vsf.py --field NREC50 --subsample 20p --fit-delta-v --use-local
"""
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from vsf_model import get_theoretical_vsf, DELTA_V_LINEAR

# ─── Paths ────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'ForArun')
LOCAL_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned')
PLOT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'plots')
CHAIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'mcmc_chains')

# ─── v4 calibration: maps Eulerian-rescaled radii onto Simone's binning ──
MULT = {'20p': 1.13, '10p': 1.27, '5p': 1.43, '2_5p': 1.27}

# ─── ForArun binning, used for local v4 VSF computation ─────────────────
FORARUN_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])
_log_c   = np.log(FORARUN_BIN_CENTERS)
_DLNR    = float(np.median(np.diff(_log_c)))
_EDGES   = np.exp(np.append(_log_c - _DLNR/2, _log_c[-1] + _DLNR/2))
_V_EFF   = 250.0**3

# ─── True Aletheia cosmology ──────────────────────────────────────────
OMEGA_M_TRUE = 0.315
SIGMA8_TRUE = 0.811

# ─── Subsample file-key mapping ──────────────────────────────────────
SUBSAMPLE_MAP = {
    '20p': '20', '10p': '10', '5p': '05', '2_5p': '025'
}

# ─── Plotting style ───────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 14,
    'axes.labelsize': 16,
})


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data_vector(field_type, subsample, r_min=0.0, use_local=False):
    """
    Load the empirical VSF.
      use_local=False (default): read from data/ForArun/*.dat (Simone's data)
      use_local=True : read v4 cleaned catalog, bin into ForArun's bin scheme
                       with calibrated radii R_resc * MULT[subsample]
    Returns (r_bins, vsf, err). Bins with zero counts or r < r_min are removed.
    """
    if use_local:
        if field_type != 'NREC50':
            raise ValueError("--use-local currently only supports field=NREC50 "
                             "(only NREC50 v4 catalogs exist).")
        fpath = os.path.join(LOCAL_DIR, f'cleaned_v4_recon_cat_{subsample}.csv')
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Local v4 catalog not found: {fpath}")
        cat = pd.read_csv(fpath)
        radii = cat['R_resc'].values * MULT[subsample]
        counts, _ = np.histogram(radii, bins=_EDGES)
        vsf  = counts / (_V_EFF * _DLNR)
        err  = np.sqrt(counts) / (_V_EFF * _DLNR)
        # Mask empty bins and bins below r_min
        valid = (counts > 0) & (FORARUN_BIN_CENTERS >= r_min)
        return FORARUN_BIN_CENTERS[valid], vsf[valid], err[valid]

    # ── default: Simone's .dat files ─────────────────────────────────
    fk = SUBSAMPLE_MAP[subsample]
    fname = f"vsf_dm_250cube_ss0p{fk}_{field_type}_SM0p00_CS0p79_TH-0p80_OV00.dat"
    fpath = os.path.join(DATA_DIR, fname)

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data file not found: {fpath}")

    df = pd.read_csv(fpath, sep=r'\s+', comment='#', names=['r', 'vsf', 'err'])

    # Mask empty bins and bins below r_min
    valid = (df['vsf'] > 0) & (df['r'] >= r_min)
    df = df[valid].copy()

    return df['r'].values, df['vsf'].values, df['err'].values


# ═══════════════════════════════════════════════════════════════════════
# BAYESIAN FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════

def log_prior_2d(theta):
    """Flat priors: Ω_m ∈ (0.1, 0.5), σ_8 ∈ (0.5, 1.2)."""
    omega_m, sigma8 = theta
    if 0.1 < omega_m < 0.5 and 0.5 < sigma8 < 1.2:
        return 0.0
    return -np.inf


def log_prior_3d(theta):
    """Flat priors: Ω_m ∈ (0.1, 0.5), σ_8 ∈ (0.5, 1.2), δ_v ∈ (-3.5, -1.5)."""
    omega_m, sigma8, delta_v = theta
    if 0.1 < omega_m < 0.5 and 0.5 < sigma8 < 1.2 and -3.5 < delta_v < -1.5:
        return 0.0
    return -np.inf


def log_likelihood(theta, r_bins, vsf_data, inv_var, fit_delta_v):
    """Gaussian log-likelihood with diagonal Poisson covariance."""
    if fit_delta_v:
        omega_m, sigma8, delta_v = theta
    else:
        omega_m, sigma8 = theta
        delta_v = DELTA_V_LINEAR

    try:
        model = get_theoretical_vsf(r_bins, omega_m, sigma8, delta_v=delta_v)
    except Exception:
        return -np.inf

    if np.any(~np.isfinite(model)) or np.any(model <= 0):
        return -np.inf

    chi2 = np.sum((vsf_data - model)**2 * inv_var)
    return -0.5 * chi2


def log_probability(theta, r_bins, vsf_data, inv_var, fit_delta_v):
    """Prior + Likelihood."""
    lp = log_prior_3d(theta) if fit_delta_v else log_prior_2d(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, r_bins, vsf_data, inv_var, fit_delta_v)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ═══════════════════════════════════════════════════════════════════════
# MCMC RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_mcmc(field_type, subsample, r_min=0.0, fit_delta_v=False,
             nwalkers=32, nsteps=2000, burn_frac=0.25, use_local=False):
    """Full MCMC pipeline for a single (field_type, subsample) run."""

    mode = "3D" if fit_delta_v else "2D"
    src  = "local_v4" if use_local else "Simone"
    run_label = f"{field_type}_{subsample}_{mode}_{src}"
    if r_min > 0 and not fit_delta_v:
        run_label += f"_rmin{r_min:.0f}"

    print(f"\n{'='*60}")
    print(f"  MCMC Run: {run_label}")
    print(f"  Mode: {'Ω_m, σ_8, δ_v' if fit_delta_v else 'Ω_m, σ_8'}")
    print(f"  R_min cut: {r_min:.1f} Mpc/h")
    print(f"{'='*60}")

    # 1. Load data
    print("\n[Step 1] Loading data vector...")
    print(f"  Source: {'local v4 catalog' if use_local else 'Simone .dat'}")
    r_bins, vsf_data, vsf_err = load_data_vector(field_type, subsample,
                                                 r_min=r_min, use_local=use_local)
    inv_var = 1.0 / vsf_err**2
    print(f"  → {len(r_bins)} bins, R ∈ [{r_bins.min():.2f}, {r_bins.max():.2f}] Mpc/h")

    # 2. Initialize walkers
    print("\n[Step 2] Initializing walkers...")
    ndim = 3 if fit_delta_v else 2
    if fit_delta_v:
        initial = np.array([OMEGA_M_TRUE, SIGMA8_TRUE, DELTA_V_LINEAR])
    else:
        initial = np.array([OMEGA_M_TRUE, SIGMA8_TRUE])
    pos = initial + 1e-3 * np.random.randn(nwalkers, ndim)
    # Clip to prior bounds
    pos[:, 0] = np.clip(pos[:, 0], 0.11, 0.49)
    pos[:, 1] = np.clip(pos[:, 1], 0.51, 1.19)
    if fit_delta_v:
        pos[:, 2] = np.clip(pos[:, 2], -3.49, -1.51)

    # 3. Run sampler
    print(f"\n[Step 3] Running emcee ({nwalkers} walkers × {nsteps} steps)...")
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(r_bins, vsf_data, inv_var, fit_delta_v)
    )
    t0 = time.time()
    sampler.run_mcmc(pos, nsteps, progress=True)
    elapsed = time.time() - t0
    print(f"  → Completed in {elapsed:.1f}s")

    # 4. Diagnostics
    print("\n[Step 4] Chain diagnostics...")
    acceptance = np.mean(sampler.acceptance_fraction)
    print(f"  → Mean acceptance fraction: {acceptance:.3f}")
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        tau_str = ", ".join([f"{t:.1f}" for t in tau])
        print(f"  → Autocorrelation times: [{tau_str}]")
    except emcee.autocorr.AutocorrError:
        tau = np.array([np.nan] * ndim)
        print("  → Autocorrelation: could not estimate (chain too short)")

    # 5. Extract samples
    burn_in = int(burn_frac * nsteps)
    thin = max(1, int(np.nanmin(tau) / 2)) if np.all(np.isfinite(tau)) else 15
    flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    print(f"  → Burn-in: {burn_in}, thin: {thin}, effective samples: {len(flat_samples)}")

    # 6. Parameter estimates
    print(f"\n  ┌{'─'*50}┐")
    param_names = ['Ω_m', 'σ_8', 'δ_v'] if fit_delta_v else ['Ω_m', 'σ_8']
    truths = [OMEGA_M_TRUE, SIGMA8_TRUE, DELTA_V_LINEAR] if fit_delta_v else [OMEGA_M_TRUE, SIGMA8_TRUE]
    for i, (name, truth) in enumerate(zip(param_names, truths)):
        q = np.percentile(flat_samples[:, i], [16, 50, 84])
        print(f"  │  {name:>4s} = {q[1]:.4f}  (+{q[2]-q[1]:.4f} / -{q[1]-q[0]:.4f})  [true: {truth}]")
    print(f"  └{'─'*50}┘")

    # 7. Save chain
    os.makedirs(CHAIN_DIR, exist_ok=True)
    chain_path = os.path.join(CHAIN_DIR, f"chain_{run_label}.npy")
    np.save(chain_path, flat_samples)
    print(f"\n  Chain saved: {chain_path}")

    # 8. Plots
    os.makedirs(PLOT_DIR, exist_ok=True)
    _plot_corner(flat_samples, run_label, fit_delta_v)
    _plot_traces(sampler, burn_in, run_label, fit_delta_v)
    _plot_bestfit(r_bins, vsf_data, vsf_err, flat_samples, run_label, fit_delta_v)

    return flat_samples


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def _plot_corner(flat_samples, run_label, fit_delta_v):
    if fit_delta_v:
        labels = [r"$\Omega_m$", r"$\sigma_8$", r"$\delta_v$"]
        truths = [OMEGA_M_TRUE, SIGMA8_TRUE, DELTA_V_LINEAR]
    else:
        labels = [r"$\Omega_m$", r"$\sigma_8$"]
        truths = [OMEGA_M_TRUE, SIGMA8_TRUE]

    fig = corner.corner(
        flat_samples, labels=labels, truths=truths,
        quantiles=[0.16, 0.5, 0.84], show_titles=True,
        title_kwargs={"fontsize": 13},
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),
        color='#2196F3', truth_color='#E53935',
    )
    fig.suptitle(f"VSF MCMC: {run_label}", fontsize=14, y=1.02)
    out = os.path.join(PLOT_DIR, f"corner_{run_label}.png")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Corner plot: {out}")
    plt.close()


def _plot_traces(sampler, burn_in, run_label, fit_delta_v):
    chain = sampler.get_chain()
    if fit_delta_v:
        param_labels = [r"$\Omega_m$", r"$\sigma_8$", r"$\delta_v$"]
        truth_vals = [OMEGA_M_TRUE, SIGMA8_TRUE, DELTA_V_LINEAR]
    else:
        param_labels = [r"$\Omega_m$", r"$\sigma_8$"]
        truth_vals = [OMEGA_M_TRUE, SIGMA8_TRUE]

    ndim = len(param_labels)
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2.5 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    for i, (ax, label, truth) in enumerate(zip(axes, param_labels, truth_vals)):
        for w in range(chain.shape[1]):
            ax.plot(chain[:, w, i], alpha=0.2, color='#2196F3', lw=0.5)
        ax.axvline(burn_in, color='red', ls='--', lw=1, label='burn-in')
        ax.axhline(truth, color='#E53935', ls='-', lw=1.5, alpha=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"traces_{run_label}.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Trace plot: {out}")
    plt.close()


def _plot_bestfit(r_bins, vsf_data, vsf_err, flat_samples, run_label, fit_delta_v):
    omega_m_best = np.median(flat_samples[:, 0])
    sigma8_best = np.median(flat_samples[:, 1])
    delta_v_best = np.median(flat_samples[:, 2]) if fit_delta_v else DELTA_V_LINEAR

    R_fine = np.linspace(max(2.0, r_bins.min() * 0.8), r_bins.max() * 1.2, 200)
    vsf_best = get_theoretical_vsf(R_fine, omega_m_best, sigma8_best, delta_v=delta_v_best)
    vsf_true = get_theoretical_vsf(R_fine, OMEGA_M_TRUE, SIGMA8_TRUE)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(r_bins, vsf_data, yerr=vsf_err, fmt='o', color='#333',
                ms=5, capsize=3, zorder=5, label='Data (OV00)')
    ax.plot(R_fine, vsf_true, 'k--', lw=1.5, alpha=0.5,
            label=f'True cosmo ($\\Omega_m$={OMEGA_M_TRUE}, $\\sigma_8$={SIGMA8_TRUE})')

    dv_label = f', $\\delta_v$={delta_v_best:.2f}' if fit_delta_v else ''
    ax.plot(R_fine, vsf_best, '-', color='#2196F3', lw=2.5,
            label=f'MCMC best-fit ($\\Omega_m$={omega_m_best:.3f}, $\\sigma_8$={sigma8_best:.3f}{dv_label})')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$R_v \; [h^{-1}\mathrm{Mpc}]$')
    ax.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v \; [h^{3}\mathrm{Mpc}^{-3}]$')
    ax.set_title(f'Best-Fit VSF: {run_label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, which='both', ls='--', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"bestfit_{run_label}.png")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Best-fit plot: {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MCMC parameter estimation from the BitVF Void Size Function")
    parser.add_argument('--field', type=str, default='BIT',
                        choices=['BIT', 'NREC50'],
                        help='BIT = BitVF true displacement, NREC50 = OT reconstructed')
    parser.add_argument('--subsample', type=str, default='20p',
                        help='Subsample: 20p, 10p, 5p, 2_5p, or "all"')
    parser.add_argument('--rmin', type=float, default=0.0,
                        help='Minimum void radius for fitting [Mpc/h] (Stage A fix)')
    parser.add_argument('--fit-delta-v', action='store_true',
                        help='Float δ_v as a 3rd free MCMC parameter (Stage B)')
    parser.add_argument('--use-local', action='store_true',
                        help='Use the local v4 cleaned catalog '
                             '(cleaned_v4_recon_cat_{frac}.csv with calibrated R_resc) '
                             'instead of Simone .dat. NREC50 only.')
    parser.add_argument('--nwalkers', type=int, default=32)
    parser.add_argument('--nsteps', type=int, default=2000)
    args = parser.parse_args()

    subsamples = list(SUBSAMPLE_MAP.keys()) if args.subsample == 'all' else [args.subsample]

    for ss in subsamples:
        run_mcmc(args.field, ss,
                 r_min=args.rmin,
                 fit_delta_v=args.fit_delta_v,
                 nwalkers=args.nwalkers,
                 nsteps=args.nsteps,
                 use_local=args.use_local)


if __name__ == "__main__":
    main()
