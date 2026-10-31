#!/usr/bin/env python3
"""
mcmc_diagnostic_plots.py

Two diagnostic plots that explain the NREC50 20p MCMC result:

  1. theory_overlay_NREC50_20p.png
     -- Both data vectors (local + Simone) overlaid with the Vdn theory
        evaluated at (truth, local-mode, Simone-mode). Shows that no
        point in the prior interior fits either dataset well -- the
        chains ran to prior walls because that's where the chi-square
        minimum lies.

  2. chi2_per_bin_NREC50_20p.png
     -- Per-bin (data-model)^2/sigma^2 contribution at each posterior
        mode, for each data vector. Reveals which R bins are driving
        each chain to its corner.
"""
import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from vsf_model import get_theoretical_vsf, DELTA_V_LINEAR

CLN  = os.path.join(HERE, '..', 'data', 'cleaned')
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
JK   = os.path.join(CLN, 'vsf_jk')
OUT  = os.path.join(HERE, '..', 'plots', 'mcmc_local')
os.makedirs(OUT, exist_ok=True)

# ----- truth and posterior modes (NREC50 20p, 3D run) -----
TRUTH      = dict(om=0.315, s8=0.811, dv=-2.71)
LOCAL_MODE = dict(om=0.500, s8=1.200, dv=-2.30)   # from chain_LOCAL_NREC50_20p_3D
SIMONE_MODE= dict(om=0.160, s8=0.955, dv=-1.99)   # from this session's Simone re-run

def load_local():
    f = os.path.join(JK, 'vsf_NREC50_20p_jk.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])
    return df[(df['vsf']>0) & (df['err']>0)]

def load_simone():
    f = os.path.join(FA, 'vsf_dm_250cube_ss0p20_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])
    return df[(df['vsf']>0) & (df['err']>0)]

dl = load_local().reset_index(drop=True)
ds = load_simone().reset_index(drop=True)

# fine grid for theory curves
R_fine = np.logspace(np.log10(1.5), np.log10(16), 200)

def theory(R, m): return get_theoretical_vsf(R, m['om'], m['s8'], delta_v=m['dv'])

t_truth_fine = theory(R_fine, TRUTH)
t_local_fine = theory(R_fine, LOCAL_MODE)
t_simone_fine= theory(R_fine, SIMONE_MODE)

# === Plot 1: theory overlay ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, df, title in [
        (axes[0], dl, 'Local cleaned + jackknife errors'),
        (axes[1], ds, 'Simone Simone data + Poisson errors')]:
    ax.errorbar(df['r'], df['vsf'], yerr=df['err'], fmt='o',
                color='#34495e', ms=5, capsize=3, lw=1.2,
                label='data', zorder=10)
    ax.plot(R_fine, t_truth_fine,  'k--',  lw=2,
            label=f'truth  Ω={TRUTH["om"]}, σ₈={TRUTH["s8"]}, δᵥ={TRUTH["dv"]}')
    ax.plot(R_fine, t_local_fine,  '-',  color='#2980b9', lw=2,
            label=f'LOCAL mode  Ω={LOCAL_MODE["om"]}, σ₈={LOCAL_MODE["s8"]}, δᵥ={LOCAL_MODE["dv"]}')
    ax.plot(R_fine, t_simone_fine, '-',  color='#7f8c8d', lw=2,
            label=f'SIMONE mode  Ω={SIMONE_MODE["om"]}, σ₈={SIMONE_MODE["s8"]}, δᵥ={SIMONE_MODE["dv"]}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$')
    ax.set_xlim(1.5, 16); ax.set_ylim(1e-7, 3e-3)
    ax.grid(True, which='both', ls='--', alpha=0.35)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc='lower left')
axes[0].set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v\ [h^{3}\mathrm{Mpc}^{-3}]$')
fig.suptitle('NREC50 20p — Vdn theory at three posterior modes vs both data vectors',
             fontsize=14, y=1.02)
plt.tight_layout()
out1 = os.path.join(OUT, 'theory_overlay_NREC50_20p.png')
fig.savefig(out1, dpi=200, bbox_inches='tight'); plt.close(fig)

# === Plot 2: per-bin chi^2 contribution ===
def bin_chi2(df, mode):
    t = theory(df['r'].values, mode)
    return (df['vsf'].values - t)**2 / df['err'].values**2

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, df, title in [
        (axes[0], dl, 'Local cleaned + jackknife errors'),
        (axes[1], ds, 'Simone Simone data + Poisson errors')]:
    R = df['r'].values
    chi2_truth  = bin_chi2(df, TRUTH)
    chi2_local  = bin_chi2(df, LOCAL_MODE)
    chi2_simone = bin_chi2(df, SIMONE_MODE)

    width = 0.27
    x = np.arange(len(R))
    ax.bar(x - width, chi2_truth,  width, color='black',    alpha=0.55,
           label=f'at truth (Σ={chi2_truth.sum():.0f})')
    ax.bar(x,         chi2_local,  width, color='#2980b9', alpha=0.85,
           label=f'at LOCAL mode (Σ={chi2_local.sum():.0f})')
    ax.bar(x + width, chi2_simone, width, color='#7f8c8d', alpha=0.85,
           label=f'at SIMONE mode (Σ={chi2_simone.sum():.0f})')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:.1f}' for r in R], rotation=45, fontsize=8)
    ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$')
    ax.set_ylabel(r'$(d-m)^2/\sigma^2$  per bin')
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis='y', ls='--', alpha=0.35)
    ax.legend(fontsize=9, loc='upper left')
fig.suptitle('NREC50 20p — per-bin χ² contribution at three modes',
             fontsize=14, y=1.02)
plt.tight_layout()
out2 = os.path.join(OUT, 'chi2_per_bin_NREC50_20p.png')
fig.savefig(out2, dpi=200, bbox_inches='tight'); plt.close(fig)

# small numerical summary
print(f"{'mode':<8}{'Σχ²_local':>14}{'Σχ²_simone':>14}")
print("-"*40)
for name, m in [('truth', TRUTH), ('LOCAL', LOCAL_MODE), ('SIMONE', SIMONE_MODE)]:
    cl = bin_chi2(dl, m).sum()
    cs = bin_chi2(ds, m).sum()
    print(f"{name:<8}{cl:>14.1f}{cs:>14.1f}")

print(f"\nSaved:\n  {out1}\n  {out2}")
