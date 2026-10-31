#!/usr/bin/env python3
"""
diag_cbl_fixed.py

Corrected diagnostic following Contarini et al. 2023 (BOSS DR12 VSF paper).

The previous diagnostic had two bugs traceable to a misread of CBL's API:

  1. Our "no-bias" call used (b_eff=1, slope=0, offset=0). In CBL the bias
     function is F = slope * b_eff + offset, so this gives F = 0 — a
     divide-by-zero inside the threshold conversion. The correct "identity"
     (no bias rescaling, appropriate for DM tracers) is
     (b_eff=1, slope=0, offset=1)  →  F = 1.

  2. We fit over R = 1.6 - 15 h^-1 Mpc. Contarini fits R > 30 h^-1 Mpc,
     which is ~3 × mean galaxy separation. We should mimic that ratio
     with our MPS values per subsample.

Three theory curves at Aletheia truth, no fitting:

  - "DM correct"    : b_eff=1, slope=0, offset=1, deltav_NL=-0.7 (identity)
  - "CBL example"   : b_eff=1.122, slope=0.857, offset=0.42 (BOSS galaxy bias)
  - "buggy no-bias" : b_eff=1, slope=0, offset=0  (kept to document the bug)

For each (subsample, theory) we compute:
  - chi^2/dof on the full R range
  - chi^2/dof on R > MPS
  - chi^2/dof on R > 3*MPS (matches Contarini's R/MGS ratio)
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
CLN  = os.path.join(HERE, '..', 'data', 'cleaned')
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
OUT  = os.path.join(HERE, '..', 'plots', 'diagnostics')
os.makedirs(OUT, exist_ok=True)

# Aletheia cosmology
cosmo = cbl.cosmology.LCDM(0.315, 0.0493, 0.0, 0.6736,
                           2.194e-9, 0.05, 0.9649, 0.0925, True)
cosmo.set_parameter("sigma8", 0.811)
sf = cbl.cosmology.SizeFunction(cosmo)
DEL_C = cosmo.deltac(0.0)

# Three theory configurations
CFG = {
    'DM correct': dict(b_eff=1.0,        slope=0.0,    offset=1.0,    dvNL=-0.7,
                       color='#27ae60', ls='-',  lw=2.4),
    'CBL example (galaxy)': dict(b_eff=1.122,      slope=0.857, offset=0.42,
                                  dvNL=-0.7, color='#c0392b', ls='--', lw=1.8),
    'buggy no-bias':       dict(b_eff=1.0, slope=0.0, offset=0.0, dvNL=-0.7,
                                color='#7f8c8d', ls=':', lw=1.5),
}

MPS_OF = {'20p': 1.16, '10p': 1.47, '5p': 1.85, '2_5p': 2.33}
FA_MAP = {'2_5p': '025', '5p': '05', '10p': '10', '20p': '20'}

# ForArun binning (for our v4 cleaned histograms)
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
MULT  = {'20p': 1.13, '10p': 1.27, '5p': 1.43, '2_5p': 1.27}

def theory_curve(R_arr, cfg):
    """Evaluate CBL Vdn over R_arr with given config; NaN-safe."""
    out = np.full(len(R_arr), np.nan)
    for i, R in enumerate(R_arr):
        try:
            out[i] = sf.size_function(float(R), 0.0, "Vdn",
                                      cfg['b_eff'], cfg['slope'], cfg['offset'],
                                      cfg['dvNL'], DEL_C, "EisensteinHu")
        except Exception:
            pass
    return out

def chi2(R_d, v_d, e_d, R_th, th):
    if len(R_d) == 0: return np.nan, 0
    interp = np.interp(R_d, R_th, th)
    valid = np.isfinite(interp) & (interp > 0)
    if valid.sum() == 0: return np.nan, 0
    err_log = e_d[valid] / (v_d[valid] * np.log(10))
    c = np.sum(((np.log10(v_d[valid]) - np.log10(interp[valid])) / err_log)**2)
    return c / valid.sum(), valid.sum()

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
    return pd.DataFrame({'r': FORARUN_BIN_CENTERS[m], 'vsf': vsf[m], 'err': err[m]})

# ─── Evaluate theory once on a fine R grid ────────────────────────────
print("Evaluating theory curves at Aletheia truth...")
R_fine = np.logspace(np.log10(1.5), np.log10(25.), 80)
theories = {name: theory_curve(R_fine, cfg) for name, cfg in CFG.items()}

# ─── Plot 2 × 4 ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=True, sharey=True)
COLORS = {'20p': '#9b59b6', '10p': '#3498db', '5p': '#2ecc71', '2_5p': '#f1c40f'}
LABELS = {'20p': 'ss=20%', '10p': 'ss=10%', '5p': 'ss=5%', '2_5p': 'ss=2.5%'}

print(f"\n{'subsample':<6} {'source':<8} {'cfg':<22} {'R cut':<10} {'Nbins':>6} {'χ²/ν':>8}")
print('-'*68)

for col, frac in enumerate(['20p','10p','5p','2_5p']):
    mps = MPS_OF[frac]
    for row, (src, df) in enumerate([('Simone', load_simone(frac)),
                                      ('v4',     load_v4(frac))]):
        ax = axes[row, col]

        # plot theory
        for name, cfg in CFG.items():
            ax.loglog(R_fine, theories[name], cfg['ls'], color=cfg['color'],
                      lw=cfg['lw'], label=name)
        # data
        ax.errorbar(df['r'], df['vsf'], yerr=df['err'], fmt='o',
                    color=COLORS[frac], ms=4, label=f'{src} {LABELS[frac]}')
        # mark R = MPS and 3*MPS
        ax.axvline(mps,    color='#bdc3c7', ls='-',  lw=1, alpha=0.7)
        ax.axvline(3*mps,  color='#bdc3c7', ls='--', lw=1, alpha=0.7)

        ax.set_xlim(1.5, 25); ax.set_ylim(1e-7, 1e-2)
        ax.grid(True, which='both', ls='--', alpha=0.3)
        if row == 1: ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$', fontsize=12)
        if col == 0: ax.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v$', fontsize=12)
        ax.set_title(f'{src} | {LABELS[frac]}  (MPS={mps:.2f}, 3MPS={3*mps:.1f})',
                     fontsize=10)
        if col == 0 and row == 0:
            ax.legend(fontsize=8, loc='lower left')

        # chi2 at three cuts × three theories
        text_lines = ['χ²/ν (DM-corr | example | bug):']
        for cut_name, R_cut in [('all', 0), (f'>MPS', mps), (f'>3MPS', 3*mps)]:
            m = df['r'].values >= R_cut
            Rd, vd, ed = df['r'].values[m], df['vsf'].values[m], df['err'].values[m]
            c1, n = chi2(Rd, vd, ed, R_fine, theories['DM correct'])
            c2, _ = chi2(Rd, vd, ed, R_fine, theories['CBL example (galaxy)'])
            c3, _ = chi2(Rd, vd, ed, R_fine, theories['buggy no-bias'])
            text_lines.append(f' {cut_name:>5} (n={n:2d}): {c1:6.2f} | {c2:6.2f} | {c3:6.2f}')
            print(f"{frac:<6} {src:<8} {'all configs':<22} {cut_name:<10} {n:>6} "
                  f"{c1:>8.2f} {c2:>8.2f} {c3:>8.2f}")
        ax.text(0.98, 0.98, '\n'.join(text_lines), transform=ax.transAxes,
                ha='right', va='top', fontsize=7, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

fig.suptitle('Vdn at Aletheia truth (no fit) — corrected DM identity vs '
             'BOSS-galaxy bias vs buggy no-bias.   '
             'Vertical lines: MPS (solid) and 3×MPS (dashed, Contarini cut).',
             fontsize=13, y=1.00)
plt.tight_layout()
out = os.path.join(OUT, 'theory_disagreement_corrected.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"\nSaved: {out}")
