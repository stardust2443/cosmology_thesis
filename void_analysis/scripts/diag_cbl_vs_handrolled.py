#!/usr/bin/env python3
"""
diag_cbl_vs_handrolled.py

The decisive theory-disagreement diagnostic.

Question: does CBL's Vdn — with the bias parameters from the canonical
example (b_eff=1.122, slope=0.857, offset=0.422) — fit our 20% NREC50 v4
VSF when evaluated at the *true* Aletheia cosmology?

Three theory curves are evaluated at the Aletheia truth (no fitting) and
overlaid on two data sources (Simone reference and our v4 cleaned catalog),
for all four subsamples. Reduced χ² is reported per (curve, data, subsample).

If the "Vdn + CBL bias" curve lands within Poisson errors of the data
across the resolved R range, the supervisor's critique resolves to:
"the previous Vdn implementation was missing bias parameters; CBL has them".

Outputs:
  plots/diagnostics/theory_disagreement_audit.png   (2 rows × 4 subsamples)
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CBL bootstrap
CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

# ─── Paths ────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
CLN  = os.path.join(HERE, '..', 'data', 'cleaned')
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
OUT  = os.path.join(HERE, '..', 'plots', 'diagnostics')
os.makedirs(OUT, exist_ok=True)

# ─── Aletheia cosmology and canonical CBL bias parameters ────────────
OMEGA_M, OMEGA_B, H0 = 0.315, 0.0493, 0.6736
N_S, SIGMA8, TAU     = 0.9649, 0.811, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05

# From Examples/cosmicVoids/codes/modelling_VoidAbundances.cpp:
CBL_BIAS = dict(b_eff=1.1218389, slope=0.85676, offset=0.42153, delta_v_NL=-0.7)
# Bias-less variant (what our hand-rolled vsf_model.py implicitly assumes)
NO_BIAS  = dict(b_eff=1.0,        slope=0.0,    offset=0.0,    delta_v_NL=-0.7)

MULT   = {'20p': 1.13, '10p': 1.27, '5p': 1.43, '2_5p': 1.27}
COLORS = {'20p': '#9b59b6', '10p': '#3498db', '5p': '#2ecc71', '2_5p': '#f1c40f'}
LABELS = {'20p': 'ss=20%', '10p': 'ss=10%', '5p': 'ss=5%', '2_5p': 'ss=2.5%'}
FA_MAP = {'2_5p': '025', '5p': '05', '10p': '10', '20p': '20'}

# ─── ForArun binning (for histogramming our v4 catalog) ──────────────
FORARUN_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])
log_c = np.log(FORARUN_BIN_CENTERS)
DLNR  = float(np.median(np.diff(log_c)))
EDGES = np.exp(np.append(log_c - DLNR/2, log_c[-1] + DLNR/2))
V_EFF = 250.0**3

# ─── Build cosmology + size function ─────────────────────────────────
print("Building Aletheia cosmology and SizeFunction object...")
cosmo = cbl.cosmology.LCDM(OMEGA_M, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
cosmo.set_parameter("sigma8", SIGMA8)
sf = cbl.cosmology.SizeFunction(cosmo)

DEL_C = cosmo.deltac(0.0)
print(f"  delta_c(z=0) = {DEL_C:.4f}")

# Evaluate theory once per (model, bias) on a fine R grid
R_fine = np.logspace(np.log10(1.5), np.log10(20.), 80)
def theory(model, biasdict):
    return np.array([
        sf.size_function(float(R), 0.0, model,
                         biasdict['b_eff'], biasdict['slope'], biasdict['offset'],
                         biasdict['delta_v_NL'], DEL_C, "EisensteinHu")
        for R in R_fine
    ])

print("Evaluating theory curves at Aletheia truth...")
vdn_bias  = theory("Vdn",  CBL_BIAS)
vdn_noBias = theory("Vdn",  NO_BIAS)
svdw_bias = theory("SvdW", CBL_BIAS)

# CBL's size_function returns dn/dlnRv (per log radius bin), same as our histogram convention
def chi2_per_dof(R_data, vsf_data, err_data, theory_curve_fn):
    """theory_curve_fn(R) -> predicted vsf at R."""
    mask = (vsf_data > 0) & (err_data > 0)
    R_d = R_data[mask]; v_d = vsf_data[mask]; e_d = err_data[mask]
    # Interpolate theory onto data R values
    theory_at_data = np.interp(R_d, R_fine, theory_curve_fn)
    chi2 = np.sum(((np.log10(v_d) - np.log10(theory_at_data)) /
                   (e_d / (v_d * np.log(10))))**2)
    return chi2 / max(len(R_d), 1)

# ─── Load data ───────────────────────────────────────────────────────
def load_simone(frac):
    f = os.path.join(FA, f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])
    return df[df['vsf'] > 0].reset_index(drop=True)

def load_v4(frac):
    df = pd.read_csv(os.path.join(CLN, f'cleaned_v4_recon_cat_{frac}.csv'))
    R = df['R_resc'].values * MULT[frac]
    counts, _ = np.histogram(R, bins=EDGES)
    vsf = counts / (V_EFF * DLNR)
    err = np.sqrt(counts) / (V_EFF * DLNR)
    m = counts > 0
    return pd.DataFrame({
        'r':   FORARUN_BIN_CENTERS[m],
        'vsf': vsf[m],
        'err': err[m]
    })

# ─── Plot 2 × 4 ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=True, sharey=True)

print(f"\n{'subsample':<8} {'source':<8} {'Vdn+bias':>12} {'Vdn no-bias':>14} {'SvdW+bias':>12}")
print('-' * 60)

for col, frac in enumerate(['20p','10p','5p','2_5p']):
    fa = load_simone(frac)
    v4 = load_v4(frac)

    for row, (src, df) in enumerate([('Simone', fa), ('v4 (this work)', v4)]):
        ax = axes[row, col]

        # Theory at truth (no fit)
        ax.loglog(R_fine, vdn_bias,   '-',  lw=2.2, color='#c0392b',
                  label='CBL Vdn + bias (truth)')
        ax.loglog(R_fine, vdn_noBias, '--', lw=1.8, color='#7f8c8d',
                  label='CBL Vdn no-bias (our impl)')
        ax.loglog(R_fine, svdw_bias,  ':',  lw=1.5, color='#2c3e50',
                  label='CBL SvdW + bias')

        # Data
        ax.errorbar(df['r'], df['vsf'], yerr=df['err'], fmt='o',
                    color=COLORS[frac], ms=4, capsize=2,
                    label=f'{src} {LABELS[frac]}')

        ax.set_xlim(1.5, 22); ax.set_ylim(1e-7, 1e-2)
        ax.grid(True, which='both', ls='--', alpha=0.3)
        if row == 1:
            ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$', fontsize=12)
        if col == 0:
            ax.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v\ [h^3 \mathrm{Mpc}^{-3}]$', fontsize=12)
        ax.set_title(f'{src} | {LABELS[frac]}', fontsize=12)
        if col == 0 and row == 0:
            ax.legend(fontsize=8, loc='lower left')

        # Reduced chi-squared per (model, source, subsample)
        c1 = chi2_per_dof(df['r'].values, df['vsf'].values, df['err'].values, vdn_bias)
        c2 = chi2_per_dof(df['r'].values, df['vsf'].values, df['err'].values, vdn_noBias)
        c3 = chi2_per_dof(df['r'].values, df['vsf'].values, df['err'].values, svdw_bias)
        ax.text(0.95, 0.95,
                f'χ²/ν:\nVdn+b={c1:.2f}\nVdn nob={c2:.2f}\nSvdW+b={c3:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        print(f"{frac:<8} {src:<8} {c1:>12.3f} {c2:>14.3f} {c3:>12.3f}")

fig.suptitle('Theory at Aletheia truth (no fit) vs data — '
             'is "Vdn + bias" the missing ingredient?',
             fontsize=14, y=1.00)
plt.tight_layout()
out = os.path.join(OUT, 'theory_disagreement_audit.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"\nSaved: {out}")
print("\nInterpretation key:")
print("  - if Vdn+bias (red solid) lands within errors → bias parameters were the issue")
print("  - if Vdn no-bias (grey dashed) is far below data → confirms what was wrong with our impl")
print("  - if NEITHER fits, the model needs more (Verza / moving barrier / restricted R range)")
