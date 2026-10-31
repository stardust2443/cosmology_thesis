#!/usr/bin/env python3
"""
run_mcmc_local.py

Run the same Vdn-MCMC pipeline used for Simone's data, but on OUR
locally-cleaned NREC50 VSFs (from data/cleaned/vsf_jk/), with jackknife
errors. Then compare posteriors to the Simone-data runs.

This is the Phase-1 -> Phase-2 self-consistency check: if the cleaning
is correct (Phase 1 says yes for 20p/10p), the MCMC posteriors derived
from our locally-cleaned data should agree with those derived from
Simone's data within statistical error.

Run:
    python run_mcmc_local.py --subsample 20p --fit-delta-v
    python run_mcmc_local.py --subsample 10p --fit-delta-v
"""
import os, sys, argparse, time, numpy as np, pandas as pd
import emcee, corner, matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from vsf_model import get_theoretical_vsf, DELTA_V_LINEAR

# Load helpers from the production runner so we share log-prior/likelihood
from run_mcmc_vsf import (
    log_prior_2d, log_prior_3d, log_likelihood, log_probability,
    OMEGA_M_TRUE, SIGMA8_TRUE
)

JK_DIR    = os.path.join(HERE, '..', 'data', 'cleaned', 'vsf_jk')
FA_DIR    = os.path.join(HERE, '..', 'data', 'ForArun')
PLOT_DIR  = os.path.join(HERE, '..', 'plots', 'mcmc_local')
CHAIN_DIR = os.path.join(HERE, '..', 'data', 'mcmc_chains')
os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(CHAIN_DIR, exist_ok=True)

FA_MAP = {'20p':'20','10p':'10','5p':'05','2_5p':'025'}

def load_local(subsample, r_min=0.0):
    f = os.path.join(JK_DIR, f'vsf_NREC50_{subsample}_jk.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])
    m = (df['vsf'] > 0) & (df['err'] > 0) & (df['r'] >= r_min)
    df = df[m]
    return df['r'].values, df['vsf'].values, df['err'].values

def load_simone(subsample, r_min=0.0):
    f = os.path.join(FA_DIR,
        f'vsf_dm_250cube_ss0p{FA_MAP[subsample]}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])
    m = (df['vsf'] > 0) & (df['err'] > 0) & (df['r'] >= r_min)
    df = df[m]
    return df['r'].values, df['vsf'].values, df['err'].values

def run_one(loader, label, subsample, r_min, fit_delta_v, nwalkers, nsteps):
    print(f"\n{'='*64}\n  RUN: {label}\n{'='*64}")
    r, vsf, err = loader(subsample, r_min=r_min)
    inv_var = 1.0 / err**2
    print(f"  data: {len(r)} bins, R ∈ [{r.min():.2f}, {r.max():.2f}]")

    ndim = 3 if fit_delta_v else 2
    initial = np.array([OMEGA_M_TRUE, SIGMA8_TRUE, DELTA_V_LINEAR][:ndim])
    pos = initial + 1e-3 * np.random.randn(nwalkers, ndim)
    pos[:, 0] = np.clip(pos[:, 0], 0.11, 0.49)
    pos[:, 1] = np.clip(pos[:, 1], 0.51, 1.19)
    if fit_delta_v:
        pos[:, 2] = np.clip(pos[:, 2], -3.49, -1.51)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(r, vsf, inv_var, fit_delta_v))

    t0 = time.time()
    sampler.run_mcmc(pos, nsteps, progress=True)
    print(f"  → {time.time()-t0:.1f}s, acceptance {np.mean(sampler.acceptance_fraction):.3f}")

    burn = int(0.25 * nsteps)
    flat = sampler.get_chain(discard=burn, thin=15, flat=True)

    names = ['Ω_m','σ_8','δ_v'][:ndim]
    truths = [OMEGA_M_TRUE, SIGMA8_TRUE, DELTA_V_LINEAR][:ndim]
    summary = {}
    for i, (n, t) in enumerate(zip(names, truths)):
        q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
        summary[n] = (q50, q84-q50, q50-q16, t)
        print(f"   {n} = {q50:.4f}  +{q84-q50:.4f} / -{q50-q16:.4f}   [true {t}]")

    return flat, summary, (r, vsf, err)

def overlay_corner(samples_local, samples_simone, run_label, fit_delta_v):
    if fit_delta_v:
        labels = [r"$\Omega_m$", r"$\sigma_8$", r"$\delta_v$"]
        truths = [OMEGA_M_TRUE, SIGMA8_TRUE, DELTA_V_LINEAR]
    else:
        labels = [r"$\Omega_m$", r"$\sigma_8$"]
        truths = [OMEGA_M_TRUE, SIGMA8_TRUE]

    fig = corner.corner(
        samples_simone, labels=labels, truths=truths,
        color='#7f8c8d', hist_kwargs={'density':True, 'alpha':0.6},
        plot_datapoints=False, plot_density=False,
        fill_contours=True, contour_kwargs={'alpha':0.6},
        levels=(1-np.exp(-0.5), 1-np.exp(-2)),
    )
    corner.corner(
        samples_local, labels=labels, truths=truths, fig=fig,
        color='#2980b9', hist_kwargs={'density':True, 'alpha':0.7},
        plot_datapoints=False, plot_density=False,
        fill_contours=True, contour_kwargs={'alpha':0.6},
        levels=(1-np.exp(-0.5), 1-np.exp(-2)),
    )
    # legend
    import matplotlib.lines as mlines
    h_loc = mlines.Line2D([], [], color='#2980b9', lw=4, label='Local cleaned + JK errors')
    h_sim = mlines.Line2D([], [], color='#7f8c8d', lw=4, label='Simone data + Poisson)')
    fig.legend(handles=[h_loc, h_sim], loc='upper right',
               bbox_to_anchor=(0.98, 0.98), fontsize=11)
    fig.suptitle(f'NREC50 {run_label}: posterior overlay  (local vs Simone)',
                 fontsize=14, y=1.02)
    out = os.path.join(PLOT_DIR, f'corner_overlay_NREC50_{run_label}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out

def overlay_data(triplet_local, triplet_simone, run_label):
    rl, vl, el = triplet_local
    rs, vs, es = triplet_simone
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(rl, vl, yerr=el, fmt='o', color='#2980b9', ms=5, capsize=3,
                label='My pipeline + JK errors', zorder=5)
    ax.errorbar(rs, vs, yerr=es, fmt='s', color='#7f8c8d', ms=5, capsize=3,
                label='Simone data + Poisson errors', zorder=4)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$')
    ax.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v\ [h^{3}\mathrm{Mpc}^{-3}]$')
    ax.set_title(f'NREC50 {run_label}: data vectors fed to MCMC', fontsize=13)
    ax.grid(True, which='both', ls='--', alpha=0.4); ax.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'data_overlay_NREC50_{run_label}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subsample', default='20p', choices=['20p','10p','5p','2_5p'])
    p.add_argument('--fit-delta-v', action='store_true')
    p.add_argument('--rmin', type=float, default=0.0)
    p.add_argument('--nwalkers', type=int, default=32)
    p.add_argument('--nsteps', type=int, default=2000)
    args = p.parse_args()

    np.random.seed(42)
    flat_local,  s_loc, t_loc = run_one(load_local,  f'LOCAL  NREC50 {args.subsample}',
                                        args.subsample, args.rmin, args.fit_delta_v,
                                        args.nwalkers, args.nsteps)
    flat_simone, s_sim, t_sim = run_one(load_simone, f'SIMONE NREC50 {args.subsample}',
                                        args.subsample, args.rmin, args.fit_delta_v,
                                        args.nwalkers, args.nsteps)

    # Save chains
    suffix = f'NREC50_{args.subsample}_{"3D" if args.fit_delta_v else "2D"}'
    np.save(os.path.join(CHAIN_DIR, f'chain_LOCAL_{suffix}.npy'), flat_local)

    # Comparison plots
    p1 = overlay_corner(flat_local, flat_simone, args.subsample, args.fit_delta_v)
    p2 = overlay_data(t_loc, t_sim, args.subsample)

    print("\n" + "="*64)
    print(f"  Posterior comparison (NREC50 {args.subsample})")
    print("="*64)
    print(f"  {'param':<6}{'local (μ ±σ)':>22}{'Simone (μ ±σ)':>24}{'shift/σ':>11}")
    print("  " + "-"*60)
    for n in s_loc:
        l_mu, l_sp, l_sm, t = s_loc[n]
        s_mu, s_sp, s_sm, _ = s_sim[n]
        # symmetric sigma proxy
        l_sig = 0.5*(l_sp + l_sm); s_sig = 0.5*(s_sp + s_sm)
        sigma_comb = np.sqrt(l_sig**2 + s_sig**2)
        nshift = (l_mu - s_mu) / sigma_comb if sigma_comb > 0 else float('nan')
        print(f"  {n:<6}{l_mu:>10.3f} ±{l_sig:.3f}{s_mu:>13.3f} ±{s_sig:.3f}{nshift:>11.2f}")
    print("\n  Saved:")
    print(f"    {p1}\n    {p2}")
    print(f"    chain: data/mcmc_chains/chain_LOCAL_{suffix}.npy")

if __name__ == '__main__':
    main()
