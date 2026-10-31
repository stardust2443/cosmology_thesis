#!/usr/bin/env python3
"""
figure_cbl_cleaned_5p_check.py

Inspect the CBL-cleaned 5% NREC50 catalog: bin it, plot vs Simone's
reference 5% NREC50 VSF, and overlay Vdn at Aletheia truth.

Also prints per-bin counts and the v_cbl/v_simone amplitude ratio in the
fit range R >= 7 h^-1 Mpc.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

HERE = os.path.dirname(os.path.abspath(__file__))
CBL_CAT = os.path.join(HERE, '..', 'data', 'cleaned_cbl',
                       'cleaned_cbl_nrec50_cat_5p.csv')
SIMONE  = os.path.join(HERE, '..', 'data', 'ForArun',
                       'vsf_dm_250cube_ss0p05_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
OUT_DIR = os.path.join(HERE, '..', 'plots', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

# ForArun bin centers — match Simone exactly
BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])
LOG_C = np.log(BIN_CENTERS)
DLNR  = float(np.median(np.diff(LOG_C)))
EDGES = np.exp(np.append(LOG_C - DLNR / 2, LOG_C[-1] + DLNR / 2))
V_EFF = 250.0 ** 3

OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05
DELTA_NL, B_EFF = -0.7, 1.0
R_MIN = 7.0


def vdn_truth(R_grid):
    c = cbl.cosmology.LCDM(OMEGA_M_TRUE, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", SIGMA8_TRUE)
    sf = cbl.cosmology.SizeFunction(c)
    dc = c.deltac(0.0)
    # slope=0 disables the bias rescaling — pure DM Vdn at truth
    return np.array([sf.size_function(float(R), 0.0, "Vdn",
                     B_EFF, 0.0, 1.0, DELTA_NL, dc, "EisensteinHu")
                     for R in R_grid])


# ─── Load data ────────────────────────────────────────────────────────
cbl_df = pd.read_csv(CBL_CAT)
print(f"CBL-cleaned voids: {len(cbl_df)}")
print(f"  R range:  [{cbl_df['R'].min():.2f}, {cbl_df['R'].max():.2f}]")
print(f"  R median: {cbl_df['R'].median():.2f}")
print(f"  N at R>=7: {(cbl_df['R'] >= 7).sum()}")

# Bin CBL-cleaned voids
radii = cbl_df['R'].values
counts, _ = np.histogram(radii, bins=EDGES)
dn   = counts / (V_EFF * DLNR)
derr = np.sqrt(counts) / (V_EFF * DLNR)

print(f"\nCBL-binned VSF (R, count, vsf):")
for r, c, v in zip(BIN_CENTERS, counts, dn):
    flag = " ← fit" if r >= R_MIN else ""
    print(f"  R={r:6.2f}  N={c:4d}  vsf={v:.3e}{flag}")

# Load Simone reference
sim = pd.read_csv(SIMONE, sep=r'\s+', comment='#', names=['r','vsf','err'])
sim = sim[sim['vsf'] > 0]
print(f"\nSimone reference: {len(sim)} occupied bins")

# Vdn at truth (smooth curve)
R_grid = np.logspace(np.log10(2.5), np.log10(20), 200)
vdn_curve_truth = vdn_truth(R_grid)

# ─── Ratio in fit range ───────────────────────────────────────────────
print(f"\n=== Ratio v_cbl / v_simone in fit range R >= {R_MIN} ===")
sim_dict = dict(zip(sim['r'].round(3), sim['vsf']))
for r, v_cbl in zip(BIN_CENTERS, dn):
    if r < R_MIN: continue
    v_sim = sim_dict.get(round(r, 3))
    if v_sim and v_sim > 0 and v_cbl > 0:
        ratio = v_cbl / v_sim
        print(f"  R={r:6.2f}  cbl={v_cbl:.3e}  simone={v_sim:.3e}  "
              f"ratio={ratio:.3f}x")
    elif v_sim and v_sim > 0:
        print(f"  R={r:6.2f}  cbl=ZERO    simone={v_sim:.3e}  (cbl has no voids in this bin)")

# ─── Plot ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'xtick.labelsize': 11,
    'ytick.labelsize': 11, 'legend.fontsize': 10, 'figure.facecolor': 'white',
})

fig, ax = plt.subplots(figsize=(10, 7))

# Exclusion zone
ax.axvspan(1.0, R_MIN, color='#f0f0f0', zorder=0)
ax.axvline(R_MIN, color='#aaa', ls='-', lw=1.2, zorder=1)

# Vdn theory at truth
ax.plot(R_grid, vdn_curve_truth, '--', color='black', lw=2.2, zorder=3,
        label=f'Vdn theory (truth: $\\Omega_m$={OMEGA_M_TRUE}, $\\sigma_8$={SIGMA8_TRUE})')

# Simone reference
ax.errorbar(sim['r'], sim['vsf'], yerr=sim['err'],
            fmt='o-', ms=7, lw=1.8, color='#2c3e50', mec='k', mew=0.4,
            elinewidth=1.5, capsize=3, zorder=5,
            label='Simone reference 5% NREC50  (CBL-cleaned by Simone)')

# CBL-cleaned (this work)
ok = counts > 0
ax.errorbar(BIN_CENTERS[ok], dn[ok], yerr=derr[ok],
            fmt='s--', ms=9, lw=1.8, color='#27ae60',
            mec='k', mew=0.5, elinewidth=1.5, capsize=3, zorder=6,
            label=f'CBL-cleaned 5% NREC50 (this work, {len(cbl_df)} voids)')

# Annotate per-bin counts for fit range
for r, c in zip(BIN_CENTERS, counts):
    if r >= R_MIN:
        ymin = ax.get_ylim()[0]
        ax.annotate(f'N={c}', xy=(r, 2e-6), fontsize=9, ha='center',
                    color='#27ae60', fontweight='bold')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(2.5, 20); ax.set_ylim(1e-6, 4e-3)
ax.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
ax.set_ylabel(r'$dn/d\ln R_v$  [$h^3$Mpc$^{-3}$]')
ax.set_title(
    'CBL-cleaned 5% NREC50 vs Simone reference vs Vdn theory at truth\n'
    'Grey zone: $R < 7$ $h^{-1}$Mpc (excluded from MCMC fit)',
    fontsize=12, fontweight='bold', pad=10)
ax.grid(True, ls='--', alpha=0.25, which='both')
ax.legend(fontsize=10, loc='lower left', framealpha=0.95)

out = os.path.join(OUT_DIR, 'cbl_cleaned_vs_simone_5p.png')
fig.savefig(out, dpi=220, bbox_inches='tight')
print(f"\nSaved: {out}")
plt.close(fig)
