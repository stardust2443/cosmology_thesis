#!/usr/bin/env python3
"""
figure_chain_diagnostics.py

Visual convergence diagnostics for the v5 joint MCMC chain. Generates a
single-page summary figure with:
  - Walker trace plot for each parameter (top, 4 panels)
  - Running posterior mean ± 1σ over chain progression (middle)
  - Integrated autocorrelation time vs chain length (bottom-left)
  - Split-R-hat vs chain length (bottom-right)

These are standard MCMC diagnostics (e.g. Gelman et al. 2013, BDA3 §11.4;
Foreman-Mackey 2013) and provide visual evidence that the production chain
has converged.

Run:
    python scripts/figure_chain_diagnostics.py [--field NREC50|BIT]
"""
import os, sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import emcee

HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN_DIR = os.path.join(HERE, '..', 'data', 'mcmc_chains')
OUT_DIR   = os.path.join(HERE, '..', 'plots', 'cbl_mcmc')
os.makedirs(OUT_DIR, exist_ok=True)

PARAM_NAMES = ['$\\Omega_m$', '$\\sigma_8$', '$B_{\\rm slope}$', '$B_{\\rm offset}$']
PARAM_KEYS  = ['Omega_m', 'sigma8', 'B_slope', 'B_offset']
PARAM_COLORS = ['#9b59b6', '#3498db', '#27ae60', '#e67e22']
TRUTH = {'Omega_m': 0.315, 'sigma8': 0.811}


def split_rhat_at(chain_3d, upto):
    """Split-R-hat using first `upto` post-burn steps."""
    c = chain_3d[:upto]
    nsteps, nwalkers, ndim = c.shape
    half = nsteps // 2
    chains = np.concatenate([c[:half], c[half:2*half]], axis=1)
    n, m = half, 2 * nwalkers
    chain_means = chains.mean(axis=0)
    overall = chain_means.mean(axis=0)
    B = n * ((chain_means - overall)**2).sum(axis=0) / (m - 1)
    W = chains.var(axis=0, ddof=1).mean(axis=0)
    var_plus = (n - 1) / n * W + B / n
    return np.sqrt(var_plus / W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--field', default='NREC50', choices=['NREC50', 'BIT'])
    args = ap.parse_args()

    chain_path = os.path.join(
        CHAIN_DIR, f'cbl_v5_joint_R7_{args.field}_cluster_chain.npy')
    meta_path = os.path.join(
        CHAIN_DIR, f'cbl_v5_joint_R7_{args.field}_cluster_meta.json')

    chain_3d = np.load(chain_path)
    with open(meta_path) as f:
        meta = json.load(f)

    nsteps, nwalkers, ndim = chain_3d.shape
    burn = meta.get('burn', 2000)
    print(f"Loaded {args.field}: {nsteps} × {nwalkers} × {ndim}, burn={burn}")

    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 12, 'figure.facecolor': 'white',
    })

    fig = plt.figure(figsize=(15, 11))
    gs = GridSpec(4, 2, figure=fig,
                  hspace=0.45, wspace=0.22,
                  left=0.07, right=0.97, top=0.94, bottom=0.06,
                  height_ratios=[1.1, 1.1, 1.1, 1.4])

    # ── Trace plots: 4 rows × 1 col (4 parameters) ──
    for i in range(ndim):
        ax = fig.add_subplot(gs[i, 0]) if i < 3 else None
        if ax is None:
            continue
        # Plot subsample of walkers for clarity
        walker_idx = np.arange(0, nwalkers, max(1, nwalkers // 16))
        for w in walker_idx:
            ax.plot(chain_3d[:, w, i], color=PARAM_COLORS[i],
                    alpha=0.15, lw=0.5)
        # Walker mean per step
        walker_mean = chain_3d[:, :, i].mean(axis=1)
        ax.plot(walker_mean, color='k', lw=1.2, alpha=0.7,
                label='walker mean')
        # Burn-in marker
        ax.axvspan(0, burn, color='#fadbd8', alpha=0.4, label='burn-in')
        # Truth line for cosmology params
        if PARAM_KEYS[i] in TRUTH:
            ax.axhline(TRUTH[PARAM_KEYS[i]], color='#c0392b',
                       ls='--', lw=1.2, label='truth')
        ax.set_ylabel(PARAM_NAMES[i], fontsize=12)
        ax.grid(True, ls='--', alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8, loc='upper right', ncol=3)
        if i == 2:
            ax.set_xlabel('MCMC step')

    # 4th parameter on the right column, top
    i = 3
    ax = fig.add_subplot(gs[0, 1])
    walker_idx = np.arange(0, nwalkers, max(1, nwalkers // 16))
    for w in walker_idx:
        ax.plot(chain_3d[:, w, i], color=PARAM_COLORS[i],
                alpha=0.15, lw=0.5)
    walker_mean = chain_3d[:, :, i].mean(axis=1)
    ax.plot(walker_mean, color='k', lw=1.2, alpha=0.7)
    ax.axvspan(0, burn, color='#fadbd8', alpha=0.4)
    ax.set_ylabel(PARAM_NAMES[i], fontsize=12)
    ax.set_xlabel('MCMC step')
    ax.grid(True, ls='--', alpha=0.2)
    ax.set_title('Walker traces (16 of 64 shown)', fontsize=11)

    # ── Right column rows 1 and 2: running mean and stddev ──
    ax_mean = fig.add_subplot(gs[1, 1])
    ax_std  = fig.add_subplot(gs[2, 1])

    step_grid = np.arange(burn + 200, nsteps, 200)
    for i in range(ndim):
        means = np.array([chain_3d[burn:upto, :, i].flatten().mean()
                          for upto in step_grid])
        stds = np.array([chain_3d[burn:upto, :, i].flatten().std()
                         for upto in step_grid])
        ax_mean.plot(step_grid, means, '-', color=PARAM_COLORS[i],
                     lw=1.5, label=PARAM_NAMES[i])
        ax_std.plot(step_grid, stds, '-', color=PARAM_COLORS[i],
                    lw=1.5, label=PARAM_NAMES[i])

    ax_mean.set_xlabel('chain length used (post-burn)')
    ax_mean.set_ylabel('posterior mean')
    ax_mean.grid(True, ls='--', alpha=0.2)
    ax_mean.set_title('Running posterior mean (stability check)', fontsize=11)
    ax_mean.legend(fontsize=8, ncol=4, loc='lower center')

    ax_std.set_xlabel('chain length used (post-burn)')
    ax_std.set_ylabel('posterior std')
    ax_std.grid(True, ls='--', alpha=0.2)
    ax_std.set_title('Running posterior std', fontsize=11)

    # ── Bottom row: tau and R-hat vs chain length ──
    ax_tau = fig.add_subplot(gs[3, 0])
    ax_rhat = fig.add_subplot(gs[3, 1])

    post = chain_3d[burn:]
    L = post.shape[0]
    n_points = 12
    upto_grid = np.linspace(L // 6, L, n_points).astype(int)

    tau_arr = np.zeros((n_points, ndim))
    rhat_arr = np.zeros((n_points, ndim))
    for k, upto in enumerate(upto_grid):
        sub = post[:upto]
        try:
            tau_arr[k] = emcee.autocorr.integrated_time(sub, quiet=True)
        except Exception:
            tau_arr[k] = np.nan
        rhat_arr[k] = split_rhat_at(post, upto)

    for i in range(ndim):
        ax_tau.plot(upto_grid, tau_arr[:, i], 'o-', color=PARAM_COLORS[i],
                    lw=1.5, ms=5, label=PARAM_NAMES[i])
        ax_rhat.plot(upto_grid, rhat_arr[:, i], 'o-', color=PARAM_COLORS[i],
                     lw=1.5, ms=5, label=PARAM_NAMES[i])

    # Reference: tau = N/50 (rule of thumb for convergence)
    ax_tau.plot(upto_grid, upto_grid / 50, 'k--', lw=1, alpha=0.5,
                label='$N/50$ rule')
    ax_tau.set_xlabel('post-burn chain length $N$')
    ax_tau.set_ylabel(r'integrated $\tau$')
    ax_tau.grid(True, ls='--', alpha=0.2)
    ax_tau.legend(fontsize=8, ncol=2, loc='upper left')
    ax_tau.set_title(r'$\tau(N)$ — should plateau as $N$ grows', fontsize=11)

    ax_rhat.axhline(1.01, color='#27ae60', ls=':', lw=1, alpha=0.7,
                    label='strict ($<1.01$)')
    ax_rhat.axhline(1.05, color='#f39c12', ls=':', lw=1, alpha=0.7,
                    label='lenient ($<1.05$)')
    ax_rhat.set_xlabel('post-burn chain length $N$')
    ax_rhat.set_ylabel(r'split-$\hat R$')
    ax_rhat.set_ylim(0.999, max(1.10, rhat_arr.max() * 1.05))
    ax_rhat.grid(True, ls='--', alpha=0.2)
    ax_rhat.legend(fontsize=8, ncol=2, loc='upper right')
    ax_rhat.set_title(r'Split-$\hat R(N)$ — should approach 1', fontsize=11)

    fig.suptitle(
        f'v5 joint MCMC convergence diagnostics — {args.field} '
        f'($64$ walkers $\\times\\ {nsteps}$ steps)',
        fontsize=14, y=0.985)

    outpath = os.path.join(OUT_DIR, f'chain_diagnostics_{args.field}.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close(fig)


if __name__ == '__main__':
    main()
