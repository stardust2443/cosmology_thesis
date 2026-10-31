#!/usr/bin/env python3
"""
figure_x3_sigma8_overlay.py

Headline thesis figure (Chapter X, Figure X.3):
  1D σ_8 marginal posteriors across all 4 NREC50 subsamples
  (and optionally BIT) from the v4 R > 7 h⁻¹Mpc MCMC, with
  Aletheia truth marked.

Reads chains saved by `cbl_mcmc_v4_absolute_R.py` from
  data/mcmc_chains/cbl_v4_R7_{FIELD}_{frac}.npy

Two layouts produced:
  - figure_x3_sigma8_NREC50.png  (single-row: 4 NREC50 posteriors)
  - figure_x3_sigma8_combined.png (2-row: NREC50 top, BIT bottom — only if BIT chains exist)

Run after v4 has been executed for the desired field(s):
  python scripts/cbl_mcmc_v4_absolute_R.py 7.0 NREC50
  python scripts/cbl_mcmc_v4_absolute_R.py 7.0 BIT
  python scripts/figure_x3_sigma8_overlay.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN_DIR = os.path.join(HERE, '..', 'data', 'mcmc_chains')
OUT_DIR   = os.path.join(HERE, '..', 'plots', 'cbl_mcmc')
os.makedirs(OUT_DIR, exist_ok=True)

R_MIN          = 7.0
SIGMA8_TRUE    = 0.811
OMEGA_M_TRUE   = 0.315
SUBSAMPLES     = ['20p', '10p', '5p', '2_5p']
COLORS = {'20p': '#9b59b6', '10p': '#3498db', '5p': '#2ecc71', '2_5p': '#f1c40f'}
LABELS = {'20p': '20% (MPS=1.16)',
          '10p': '10% (MPS=1.47)',
          '5p':  '5% (MPS=1.85)',
          '2_5p':'2.5% (MPS=2.33)'}


def chain_path(field, frac):
    return os.path.join(CHAIN_DIR, f'cbl_v4_R{R_MIN:.0f}_{field}_{frac}.npy')

def load_chains(field):
    out = {}
    for frac in SUBSAMPLES:
        p = chain_path(field, frac)
        if os.path.exists(p):
            out[frac] = np.load(p)
            print(f"  loaded {field} {frac}: {len(out[frac])} samples")
        else:
            print(f"  missing {p} — skip")
    return out


def plot_1d_sigma8(ax, chains_dict, title):
    """One panel: σ_8 marginal density per subsample, truth as vertical line."""
    x_grid = np.linspace(0.55, 1.05, 400)
    for frac in SUBSAMPLES:
        if frac not in chains_dict: continue
        s8 = chains_dict[frac][:, 1]
        kde = gaussian_kde(s8, bw_method=0.25)
        y = kde(x_grid)
        y = y / y.max()  # normalise so all curves peak at 1 for visibility
        ax.fill_between(x_grid, 0, y, color=COLORS[frac], alpha=0.18)
        ax.plot(x_grid, y, color=COLORS[frac], lw=2, label=LABELS[frac])
        med = np.median(s8); lo, hi = np.percentile(s8, [16, 84])
        # Tick mark at median
        ax.plot([med, med], [0, kde(med)[0] / kde(x_grid).max()], color=COLORS[frac],
                lw=1.2, alpha=0.7)

    ax.axvline(SIGMA8_TRUE, color='red', ls='--', lw=2, alpha=0.85,
               label=f'Aletheia truth ({SIGMA8_TRUE})')
    ax.set_xlim(0.55, 1.05); ax.set_ylim(0, 1.15)
    ax.set_xlabel(r'$\sigma_8$', fontsize=15)
    ax.set_ylabel('marginal posterior\n(peak-normalised)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, ls='--', alpha=0.3)
    ax.legend(fontsize=10, loc='upper left')


def annotate_recovery(ax, chains_dict, ypos=1.05):
    """Add a small inset table: σ_8 median ± 1σ and deviation per subsample."""
    txt_lines = [r'$\sigma_8$ median (1$\sigma$): deviation from truth']
    for frac in SUBSAMPLES:
        if frac not in chains_dict: continue
        s8 = chains_dict[frac][:, 1]
        med = np.median(s8); lo, hi = np.percentile(s8, [16, 84])
        sigma_dev = (med - SIGMA8_TRUE) / max((hi - lo) / 2, 1e-6)
        txt_lines.append(
            f'  {LABELS[frac]:18s} {med:.3f}'
            f' (+{hi-med:.3f}/-{med-lo:.3f})  {sigma_dev:+.1f}σ')
    ax.text(0.98, 0.98, '\n'.join(txt_lines), transform=ax.transAxes,
            ha='right', va='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='#888', alpha=0.92))


# ─── Load chains ──────────────────────────────────────────────────────
print("Loading NREC50 chains:")
chains_nrec50 = load_chains('NREC50')
print("\nLoading BIT chains:")
chains_bit = load_chains('BIT')

# ─── Figure: NREC50 only ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
plot_1d_sigma8(ax, chains_nrec50,
               r'$\sigma_8$ recovery across NREC50 subsamples '
               r'(R > 7 $h^{-1}$Mpc, nuisance fixed at joint MAP)')
annotate_recovery(ax, chains_nrec50)
plt.tight_layout()
out_nrec50 = os.path.join(OUT_DIR, 'figure_x3_sigma8_NREC50.png')
fig.savefig(out_nrec50, dpi=220, bbox_inches='tight')
print(f"\nSaved: {out_nrec50}")

# ─── Figure: NREC50 + BIT side by side (if BIT exists) ────────────────
if chains_bit:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    plot_1d_sigma8(axes[0], chains_nrec50,
                   'NREC50 (OT-reconstructed)')
    annotate_recovery(axes[0], chains_nrec50)
    plot_1d_sigma8(axes[1], chains_bit,
                   'BIT (true displacement)')
    annotate_recovery(axes[1], chains_bit)
    axes[1].set_ylabel('')
    fig.suptitle(r'$\sigma_8$ posterior across 4 subsamples × 2 displacement fields  '
                 r'(R > 7 $h^{-1}$Mpc)', fontsize=14, y=1.02)
    plt.tight_layout()
    out_comb = os.path.join(OUT_DIR, 'figure_x3_sigma8_combined.png')
    fig.savefig(out_comb, dpi=220, bbox_inches='tight')
    print(f"Saved: {out_comb}")
else:
    print("\n(No BIT chains found — run 'python scripts/cbl_mcmc_v4_absolute_R.py 7.0 BIT' to get them.)")
