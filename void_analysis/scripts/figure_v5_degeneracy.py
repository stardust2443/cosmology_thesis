#!/usr/bin/env python3
"""
figure_v5_degeneracy.py

"Crossing bananas" figure: NREC50 and BIT v5 joint posteriors overlaid
in (Omega_m, sigma_8) space, showing how the two displacement fields
navigate the same degeneracy to opposite ends.

Layout:
  - Left: 2D contour overlay (1σ, 2σ) with truth star
  - Right top: 1D σ_8 marginal comparison
  - Right bottom: 1D Ω_m marginal comparison

Run:
    python scripts/figure_v5_degeneracy.py
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN_DIR = os.path.join(HERE, '..', 'data', 'mcmc_chains')
OUT_DIR   = os.path.join(HERE, '..', 'plots', 'cbl_mcmc')
os.makedirs(OUT_DIR, exist_ok=True)

OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
R_MIN = 7.0

C_NREC = '#2980b9'
C_BIT  = '#e67e22'

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.facecolor': 'white',
})


def load_chain(field, tag='cluster'):
    path = os.path.join(CHAIN_DIR,
                        f'cbl_v5_joint_R{R_MIN:.0f}_{field}_{tag}_flat.npy')
    return np.load(path)


def load_meta(field, tag='cluster'):
    path = os.path.join(CHAIN_DIR,
                        f'cbl_v5_joint_R{R_MIN:.0f}_{field}_{tag}_meta.json')
    with open(path) as f:
        return json.load(f)


def contour_2d(ax, om, s8, color, label, filled=True):
    xmin, xmax = 0.08, 0.48
    ymin, ymax = 0.60, 0.90
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    try:
        kernel = gaussian_kde(np.vstack([om, s8]), bw_method=0.15)
        zz = np.reshape(kernel(positions), xx.shape)
    except np.linalg.LinAlgError:
        return

    zz = gaussian_filter(zz, sigma=1.5)
    zz = zz / zz.max()

    levels_fill = [0.05, 0.32, 1.0]
    levels_line = [0.05, 0.32]

    if filled:
        ax.contourf(xx, yy, zz, levels=levels_fill,
                    colors=[color], alpha=[0.08, 0.20])
    ax.contour(xx, yy, zz, levels=levels_line,
               colors=[color], linewidths=[1.2, 2.0],
               linestyles=['--', '-'])
    ax.plot([], [], '-', color=color, lw=2, label=label)


def marginal_1d(ax, samples, color, label, vertical=False, truth=None):
    lo, hi = np.percentile(samples, [0.5, 99.5])
    grid = np.linspace(lo, hi, 300)
    kde = gaussian_kde(samples, bw_method=0.2)
    y = kde(grid)
    y = y / y.max()

    if vertical:
        ax.fill_betweenx(grid, 0, y, color=color, alpha=0.15)
        ax.plot(y, grid, '-', color=color, lw=2, label=label)
        if truth is not None:
            ax.axhline(truth, color='#c0392b', ls='--', lw=1.5, alpha=0.8)
    else:
        ax.fill_between(grid, 0, y, color=color, alpha=0.15)
        ax.plot(grid, y, '-', color=color, lw=2, label=label)
        if truth is not None:
            ax.axvline(truth, color='#c0392b', ls='--', lw=1.5, alpha=0.8)

    med = np.median(samples)
    lo16, hi84 = np.percentile(samples, [16, 84])
    return med, lo16, hi84


# ─── Load ────────────────────────────────────────────────────────────
print("Loading chains...")
chain_n = load_chain('NREC50')
chain_b = load_chain('BIT')
meta_n  = load_meta('NREC50')
meta_b  = load_meta('BIT')

om_n, s8_n = chain_n[:, 0], chain_n[:, 1]
om_b, s8_b = chain_b[:, 0], chain_b[:, 1]

# ─── Figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[1, 1],
                       hspace=0.30, wspace=0.05,
                       left=0.08, right=0.96, top=0.90, bottom=0.08)

ax_2d   = fig.add_subplot(gs[:, 0])
ax_s8   = fig.add_subplot(gs[0, 1])
ax_om   = fig.add_subplot(gs[1, 1])

# ── 2D contours ──
contour_2d(ax_2d, om_n, s8_n, C_NREC, 'NREC50 (OT-reconstructed)')
contour_2d(ax_2d, om_b, s8_b, C_BIT, 'BIT (true displacement)')

ax_2d.plot(OMEGA_M_TRUE, SIGMA8_TRUE, '*', color='#c0392b', ms=18,
           mec='k', mew=1, zorder=10, label='Aletheia truth')

# MAP points
ax_2d.plot(meta_n['x_map']['Omega_m'], meta_n['x_map']['sigma8'],
           'o', color=C_NREC, ms=10, mec='k', mew=1.2, zorder=9)
ax_2d.plot(meta_b['x_map']['Omega_m'], meta_b['x_map']['sigma8'],
           'o', color=C_BIT, ms=10, mec='k', mew=1.2, zorder=9)

ax_2d.set_xlabel(r'$\Omega_m$')
ax_2d.set_ylabel(r'$\sigma_8$')
ax_2d.set_xlim(0.08, 0.48)
ax_2d.set_ylim(0.64, 0.88)
ax_2d.grid(True, ls='--', alpha=0.2)
ax_2d.legend(loc='upper right', fontsize=10, framealpha=0.92)

ax_2d.annotate('NREC50:\n$\\sigma_8$ recovered\n$\\Omega_m$ lost',
               xy=(np.median(om_n), np.median(s8_n)),
               xytext=(0.22, 0.82), fontsize=9, color=C_NREC,
               arrowprops=dict(arrowstyle='->', color=C_NREC, lw=1.5),
               bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_NREC, alpha=0.9))
ax_2d.annotate('BIT:\n$\\Omega_m$ recovered\n$\\sigma_8$ lost',
               xy=(np.median(om_b), np.median(s8_b)),
               xytext=(0.38, 0.70), fontsize=9, color=C_BIT,
               arrowprops=dict(arrowstyle='->', color=C_BIT, lw=1.5),
               bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_BIT, alpha=0.9))

# ── 1D σ_8 marginal ──
med_n, lo_n, hi_n = marginal_1d(ax_s8, s8_n, C_NREC, 'NREC50', truth=SIGMA8_TRUE)
med_b, lo_b, hi_b = marginal_1d(ax_s8, s8_b, C_BIT, 'BIT', truth=SIGMA8_TRUE)
ax_s8.set_xlabel(r'$\sigma_8$')
ax_s8.set_ylabel('Posterior (normalised)')
ax_s8.set_xlim(0.64, 0.88)
ax_s8.set_ylim(0, 1.15)
ax_s8.legend(fontsize=9, loc='upper left')
ax_s8.grid(True, ls='--', alpha=0.2)
ax_s8.text(0.97, 0.85,
           f'NREC50: {med_n:.3f}$^{{+{hi_n-med_n:.3f}}}_{{-{med_n-lo_n:.3f}}}$\n'
           f'BIT:       {med_b:.3f}$^{{+{hi_b-med_b:.3f}}}_{{-{med_b-lo_b:.3f}}}$',
           transform=ax_s8.transAxes, ha='right', va='top', fontsize=9,
           family='monospace',
           bbox=dict(boxstyle='round', fc='white', ec='#999', alpha=0.9))

# ── 1D Ω_m marginal ──
med_n, lo_n, hi_n = marginal_1d(ax_om, om_n, C_NREC, 'NREC50', truth=OMEGA_M_TRUE)
med_b, lo_b, hi_b = marginal_1d(ax_om, om_b, C_BIT, 'BIT', truth=OMEGA_M_TRUE)
ax_om.set_xlabel(r'$\Omega_m$')
ax_om.set_ylabel('Posterior (normalised)')
ax_om.set_xlim(0.08, 0.48)
ax_om.set_ylim(0, 1.15)
ax_om.legend(fontsize=9, loc='upper right')
ax_om.grid(True, ls='--', alpha=0.2)
ax_om.text(0.97, 0.85,
           f'NREC50: {med_n:.3f}$^{{+{hi_n-med_n:.3f}}}_{{-{med_n-lo_n:.3f}}}$\n'
           f'BIT:       {med_b:.3f}$^{{+{hi_b-med_b:.3f}}}_{{-{med_b-lo_b:.3f}}}$',
           transform=ax_om.transAxes, ha='right', va='top', fontsize=9,
           family='monospace',
           bbox=dict(boxstyle='round', fc='white', ec='#999', alpha=0.9))

fig.suptitle(
    r'$\Omega_m$–$\sigma_8$ degeneracy:  NREC50 vs BIT joint posteriors'
    f'  (R > {R_MIN:.0f} $h^{{-1}}$Mpc, v5 joint MCMC)',
    fontsize=14, y=0.96)

outpath = os.path.join(OUT_DIR, 'joint_degeneracy_NREC50_vs_BIT.png')
fig.savefig(outpath, dpi=220, bbox_inches='tight')
print(f'\nSaved: {outpath}')
plt.close(fig)
print("Done.")
