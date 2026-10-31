#!/usr/bin/env python3
"""
mcmc_landscape.py

Explanatory figure for the 20p MCMC result. Two diagnostics:

  (1) Chi-square landscape on a (Omega_m, sigma_8) grid at delta_v = truth,
      computed separately for the local-cleaned data vector (with JK errors)
      and for Simone's data vector (with Poisson errors).
      The minimum is marked. The chain medians (from the 2000-step runs)
      are overlaid as white stars. The prior box is drawn as a dashed
      rectangle.

  (2) Best-fit VSF curves: the model evaluated at the chain medians for
      both runs, plotted alongside the theoretical-cosmology curve and both
      data vectors. Shows that even at the prior wall, the model still
      cannot fit the data shape -- which is the model-limitation result.

Outputs
-------
plots/mcmc_local/landscape_NREC50_20p.png
plots/mcmc_local/bestfit_compare_NREC50_20p.png
"""
import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from vsf_model import get_theoretical_vsf, DELTA_V_LINEAR

JK_DIR    = os.path.join(HERE, '..', 'data', 'cleaned', 'vsf_jk')
FA_DIR    = os.path.join(HERE, '..', 'data', 'ForArun')
PLOT_DIR  = os.path.join(HERE, '..', 'plots', 'mcmc_local')
CHAIN_DIR = os.path.join(HERE, '..', 'data', 'mcmc_chains')

OMEGA_M_TRUE, SIGMA8_TRUE, DELTA_V_TRUE = 0.315, 0.811, -2.71

# Prior box
OM_LO, OM_HI = 0.10, 0.50
S8_LO, S8_HI = 0.50, 1.20

def load_local(frac='20p'):
    f = os.path.join(JK_DIR, f'vsf_NREC50_{frac}_jk.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])
    df = df[(df['vsf']>0) & (df['err']>0)]
    return df['r'].values, df['vsf'].values, df['err'].values

def load_simone(frac='20p'):
    fa_map = {'20p':'20','10p':'10','5p':'05','2_5p':'025'}
    f = os.path.join(FA_DIR,
        f'vsf_dm_250cube_ss0p{fa_map[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])
    df = df[(df['vsf']>0) & (df['err']>0)]
    return df['r'].values, df['vsf'].values, df['err'].values

def chi2_at(omega_m, sigma8, delta_v, r, vsf_data, err):
    try:
        m = get_theoretical_vsf(r, omega_m, sigma8, delta_v=delta_v)
    except Exception:
        return np.nan
    if np.any(~np.isfinite(m)) or np.any(m <= 0):
        return np.nan
    return np.sum((vsf_data - m)**2 / err**2)

def chi2_grid(r, vsf_data, err, n=25, delta_v=DELTA_V_TRUE):
    om = np.linspace(OM_LO, OM_HI, n)
    s8 = np.linspace(S8_LO, S8_HI, n)
    g = np.full((n, n), np.nan)
    for i, o in enumerate(om):
        for j, s in enumerate(s8):
            g[j, i] = chi2_at(o, s, delta_v, r, vsf_data, err)
    return om, s8, g

# ------------- compute -------------

print("Loading data vectors and chains...")
r_loc, v_loc, e_loc = load_local('20p')
r_sim, v_sim, e_sim = load_simone('20p')

ch_loc = np.load(os.path.join(CHAIN_DIR, 'chain_LOCAL_NREC50_20p_3D.npy'))
# Chain for the Simone-data run from this comparison was computed but not
# saved in chain_LOCAL_*; use the 50-quantile we recorded (0.160, 0.955, -1.985)
sim_med = np.array([0.160, 0.955, -1.985])
loc_med = np.median(ch_loc, axis=0)

print(f"  Local 20p: {len(r_loc)} bins, chain median = "
      f"({loc_med[0]:.3f}, {loc_med[1]:.3f}, {loc_med[2]:.2f})")
print(f"  Simone 20p: {len(r_sim)} bins, chain median = "
      f"({sim_med[0]:.3f}, {sim_med[1]:.3f}, {sim_med[2]:.2f})")

print("Computing chi^2 landscapes at each chain's delta_v (~30s)...")
# Use the chain's own delta_v to make the landscape consistent with where
# the chain actually landed -- otherwise a 2D slice at delta_v=truth
# misrepresents the 3D chi^2 minimum each chain found.
om_g, s8_g, chi_loc = chi2_grid(r_loc, v_loc, e_loc, n=30, delta_v=loc_med[2])
om_g, s8_g, chi_sim = chi2_grid(r_sim, v_sim, e_sim, n=30, delta_v=sim_med[2])

# Normalize: chi^2 / dof
dof_loc = max(1, len(r_loc) - 3)
dof_sim = max(1, len(r_sim) - 3)

# ------------- figure 1: landscape -------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, (om, s8, chi, dof, label, med) in zip(
        axes,
        [(om_g, s8_g, chi_loc, dof_loc, 'my pipeline + JK errors', loc_med),
         (om_g, s8_g, chi_sim, dof_sim, 'SIMONE data + Poisson errors', sim_med)]):

    # log-scale colormap because chi^2 spans orders of magnitude near walls
    z = chi / dof
    z_clip = np.log10(np.clip(z, 1.0, 1e6))
    pcm = ax.pcolormesh(om, s8, z_clip, shading='auto', cmap='viridis')
    cbar = plt.colorbar(pcm, ax=ax, label=r'$\log_{10}(\chi^2/\mathrm{dof})$')

    # min on grid
    j_min, i_min = np.unravel_index(np.nanargmin(z), z.shape)
    ax.plot(om[i_min], s8[j_min], 'o', mec='white', mfc='red', ms=10, mew=2,
            label=f'min on grid ({om[i_min]:.2f}, {s8[j_min]:.2f}), χ²/dof={z[j_min,i_min]:.1e}')

    # theoretical truth
    ax.plot(OMEGA_M_TRUE, SIGMA8_TRUE, '+', color='cyan', ms=10, mew=3,
            label=f'theoretical ({OMEGA_M_TRUE}, {SIGMA8_TRUE})')

    # chain median
    ax.plot(med[0], med[1], '*', color='gold', ms=10, mec='black', mew=1,
            label=f'chain median ({med[0]:.2f}, {med[1]:.2f})')

    # prior box
    ax.add_patch(Rectangle((OM_LO, S8_LO), OM_HI-OM_LO, S8_HI-S8_LO,
                           fill=False, edgecolor='white', lw=2, ls='--', label='prior'))

    ax.set_title(label, fontsize=12)
    ax.set_xlabel(r'$\Omega_m$')
    ax.set_xlim(OM_LO, OM_HI); ax.set_ylim(S8_LO, S8_HI)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.85)

axes[0].set_ylabel(r'$\sigma_8$')
fig.suptitle(rf'$\chi^2/\mathrm{{dof}}$ landscape at each chain\'s $\delta_v$ (local: {loc_med[2]:.2f}, Simone: {sim_med[2]:.2f}) NREC50 20p',
             fontsize=12, y=1.02)
plt.tight_layout()
out1 = os.path.join(PLOT_DIR, 'landscape_NREC50_20p_v1.png')
fig.savefig(out1, dpi=200, bbox_inches='tight'); plt.close()
print(f"Saved: {out1}")

# ------------- figure 2: best-fit overlay -------------

R_fine = np.linspace(1.5, 16, 200)
v_truth = get_theoretical_vsf(R_fine, OMEGA_M_TRUE, SIGMA8_TRUE, delta_v=DELTA_V_TRUE)
v_loc_fit = get_theoretical_vsf(R_fine, loc_med[0], loc_med[1], delta_v=loc_med[2])
v_sim_fit = get_theoretical_vsf(R_fine, sim_med[0], sim_med[1], delta_v=sim_med[2])

fig, ax = plt.subplots(figsize=(10, 7))
ax.errorbar(r_loc, v_loc, yerr=e_loc, fmt='o', color='#2980b9', ms=5, capsize=3,
            label='Local cleaned (JK)', zorder=5)
ax.errorbar(r_sim, v_sim, yerr=e_sim, fmt='s', color='#7f8c8d', ms=5, capsize=3,
            label='Simone data (Poisson)', zorder=5)
ax.plot(R_fine, v_truth, 'k--', lw=2,
        label=fr'theoretical ($\Omega_m$={OMEGA_M_TRUE}, $\sigma_8$={SIGMA8_TRUE})')
ax.plot(R_fine, v_loc_fit, '-', color='#2980b9', lw=2,
        label=fr'local best-fit ($\Omega_m$={loc_med[0]:.2f}, $\sigma_8$={loc_med[1]:.2f}, $\delta_v$={loc_med[2]:.2f})')
ax.plot(R_fine, v_sim_fit, '-', color='#7f8c8d', lw=2,
        label=fr'Simone best-fit ($\Omega_m$={sim_med[0]:.2f}, $\sigma_8$={sim_med[1]:.2f}, $\delta_v$={sim_med[2]:.2f})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$'); ax.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v$')
ax.set_xlim(1.5, 16); ax.set_ylim(1e-7, 5e-3)
ax.set_title('NREC50 20p: best-fit VSF curves at chain medians (both pegged to prior walls)',
             fontsize=12)
ax.grid(True, which='both', ls='--', alpha=0.4)
ax.legend(fontsize=9, loc='lower left')
plt.tight_layout()
out2 = os.path.join(PLOT_DIR, 'bestfit_compare_NREC50_20p_v1.png')
fig.savefig(out2, dpi=200, bbox_inches='tight'); plt.close()
print(f"Saved: {out2}")

# Numerical summary
print("\nχ²/dof at key points:")
print(f"  Local data:")
print(f"    theoretical (0.315, 0.811):           {chi2_at(0.315, 0.811, DELTA_V_TRUE, r_loc, v_loc, e_loc)/dof_loc:.2f}")
print(f"    chain median ({loc_med[0]:.2f}, {loc_med[1]:.2f}):  "
      f"{chi2_at(loc_med[0], loc_med[1], loc_med[2], r_loc, v_loc, e_loc)/dof_loc:.2f}")
print(f"  Simone data:")
print(f"    theoretical (0.315, 0.811):           {chi2_at(0.315, 0.811, DELTA_V_TRUE, r_sim, v_sim, e_sim)/dof_sim:.2f}")
print(f"    chain median ({sim_med[0]:.2f}, {sim_med[1]:.2f}):  "
      f"{chi2_at(sim_med[0], sim_med[1], sim_med[2], r_sim, v_sim, e_sim)/dof_sim:.2f}")
