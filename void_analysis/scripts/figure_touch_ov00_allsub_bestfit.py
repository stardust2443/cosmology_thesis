#!/usr/bin/env python3
"""
figure_touch_ov00_allsub_bestfit.py

Cross-subsample validation: TOUCH ↔ OV00 with Vdn-with-free-δ_v fits
at all 4 NREC50 subsamples (20%, 10%, 5%, 2.5%).

Produces:
  - 2×2 grid (per subsample) of TOUCH/OV00/Vdn overlay + best-fit
  - Trend plot: best-fit δ_v vs mean particle separation (MPS)
  - Summary table with v/v_sim, best-fit δ_v, χ²_red per subsample

Output:
  plots/diagnostics/bestfit_touch_ov00_vdn_allsubs.png
  plots/diagnostics/bestfit_deltav_across_subsamples.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

from bin_v4_cleaned_to_vsf import (
    FORARUN_BIN_CENTERS, EDGES, DLNR, V_EFF, FRAC_MAP,
)
from _vsf_plot_utils import (
    UNIFIED_RCPARAMS, XLIM, YLIM, R_MIN,
    SIMONE_COLOR, SIMONE_MARKER, SIMONE_MS,
    SUBSAMPLE_COLORS, MPS,
)

DATA = os.path.join(HERE, '..', 'data')
OUT_DIR = os.path.join(HERE, '..', 'plots', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

SUBSAMPLES = ['20p', '10p', '5p', '2_5p']
LABELS     = {'20p': '20%', '10p': '10%', '5p': '5%', '2_5p': '2.5%'}

OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05

DELTA_V_FIDUCIAL = -0.80
DELTA_V_SCAN = np.linspace(-0.95, -0.55, 41)
TOUCH_COLOR = '#c0392b'

# ─── CBL Vdn setup (once) ──────────────────────────────────────────
print("Setting up CBL Vdn evaluator…")
_cosmo = cbl.cosmology.LCDM(OMEGA_M_TRUE, OMEGA_B, 0.0, H0,
                            SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
_cosmo.set_parameter("sigma8", SIGMA8_TRUE)
_sf = cbl.cosmology.SizeFunction(_cosmo)
_dc = _cosmo.deltac(0.0)


def vdn_at(R_array, delta_v):
    return np.array([_sf.size_function(float(R), 0.0, "Vdn",
                     1.0, 0.0, 1.0, float(delta_v), _dc, "EisensteinHu")
                     for R in R_array])


def bin_vsf(radii):
    counts, _ = np.histogram(radii, bins=EDGES)
    return counts, counts / (V_EFF * DLNR), np.sqrt(counts) / (V_EFF * DLNR)


def chi2_at_deltav(R, vsf, err, delta_v, r_min=R_MIN):
    vdn = vdn_at(R, delta_v)
    ok = (vsf > 0) & (vdn > 0) & (err > 0) & (R >= r_min)
    if not ok.any():
        return np.inf, 0
    return float(np.sum(((vsf[ok] - vdn[ok]) / err[ok]) ** 2)), int(ok.sum())


def scan_deltav(R, vsf, err, deltavs):
    chi2s = np.zeros(len(deltavs))
    ndof = 0
    for i, dv in enumerate(deltavs):
        c, n = chi2_at_deltav(R, vsf, err, dv); chi2s[i] = c; ndof = n
    imin = int(np.argmin(chi2s))
    if 0 < imin < len(deltavs) - 1:
        x0, x1, x2 = deltavs[imin-1], deltavs[imin], deltavs[imin+1]
        y0, y1, y2 = chi2s[imin-1], chi2s[imin], chi2s[imin+1]
        denom = (y0 - 2*y1 + y2)
        best = x1 - 0.5 * (y2 - y0) / denom * (x1 - x0) if abs(denom) > 1e-30 else x1
    else:
        best = deltavs[imin]
    best_chi2, _ = chi2_at_deltav(R, vsf, err, best)
    # Errorbar from Δχ²=1
    err_dv = 0.0
    if 0 < imin < len(deltavs) - 1 and abs(y0 - 2*y1 + y2) > 1e-30:
        # curvature → 1-sigma
        curvature = (y0 - 2*y1 + y2) / (deltavs[1] - deltavs[0]) ** 2
        if curvature > 0:
            err_dv = float(np.sqrt(2.0 / curvature))
    return chi2s, ndof, float(best), float(best_chi2), float(err_dv)


# ─── Loop over subsamples ──────────────────────────────────────────
results = {}
for frac in SUBSAMPLES:
    print(f"\n── {LABELS[frac]} NREC50 ──")
    touch_csv = os.path.join(DATA, 'cleaned_simple0',
        f'cleaned_simple0_nrec50_cat_{frac}_touch.csv')
    ov00_ref  = os.path.join(DATA, 'ForArun',
        f'vsf_dm_250cube_ss0p{FRAC_MAP[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')

    if not os.path.exists(touch_csv):
        print(f"  SKIP: {touch_csv} missing")
        continue
    if not os.path.exists(ov00_ref):
        print(f"  SKIP: {ov00_ref} missing")
        continue

    df = pd.read_csv(touch_csv)
    touch_R_arr = df['R_resc'].values
    _, touch_vsf, touch_err = bin_vsf(touch_R_arr)
    print(f"  TOUCH N = {len(df)}")

    ov00_df = pd.read_csv(ov00_ref, sep=r'\s+', comment='#',
                          names=['r', 'vsf', 'err'])
    ov00_df = ov00_df[ov00_df['vsf'] > 0].reset_index(drop=True)
    ov00_R   = ov00_df['r'].values
    ov00_vsf = ov00_df['vsf'].values
    ov00_err = ov00_df['err'].values

    print(f"  scanning δ_v for TOUCH…")
    t_chi2s, t_ndof, t_best, t_chi2, t_err = scan_deltav(
        FORARUN_BIN_CENTERS, touch_vsf, touch_err, DELTA_V_SCAN)
    print(f"  scanning δ_v for OV00…")
    o_chi2s, o_ndof, o_best, o_chi2, o_err = scan_deltav(
        ov00_R, ov00_vsf, ov00_err, DELTA_V_SCAN)

    # χ² at fiducial −0.80
    t_chi2_fid, _ = chi2_at_deltav(FORARUN_BIN_CENTERS, touch_vsf, touch_err,
                                    DELTA_V_FIDUCIAL)
    o_chi2_fid, _ = chi2_at_deltav(ov00_R, ov00_vsf, ov00_err,
                                    DELTA_V_FIDUCIAL)

    # v/v_sim per bin (R≥7 median)
    sim_by_r = dict(zip(ov00_df['r'].round(3), ov00_df['vsf']))
    rs = []
    for r, v in zip(FORARUN_BIN_CENTERS, touch_vsf):
        if r < R_MIN: continue
        vs = sim_by_r.get(round(r, 3))
        if vs and vs > 0 and v > 0: rs.append(v / vs)
    med_ratio = float(np.median(rs)) if rs else float('nan')

    results[frac] = dict(
        N=len(df),
        touch_vsf=touch_vsf, touch_err=touch_err,
        ov00_R=ov00_R, ov00_vsf=ov00_vsf, ov00_err=ov00_err,
        t_best=t_best, t_chi2=t_chi2, t_chi2_fid=t_chi2_fid,
        t_err=t_err, t_ndof=t_ndof,
        o_best=o_best, o_chi2=o_chi2, o_chi2_fid=o_chi2_fid,
        o_err=o_err, o_ndof=o_ndof,
        med_ratio=med_ratio,
    )

if not results:
    print("\n❌ No subsamples processed — cleaning hasn't completed yet?")
    sys.exit(1)

# ─── Console summary ───────────────────────────────────────────────
print(f"\n{'='*100}")
print('CROSS-SUBSAMPLE BEST-FIT  (TOUCH ↔ OV00, Vdn-free-δ_v at R ≥ 7)')
print('='*100)
print(f"{'sub':<6}{'MPS':>6}{'N_TOUCH':>10}"
      f"{'v/v_sim':>10}"
      f"{'TOUCH δ_v':>14}{'TOUCH χ²_r':>14}"
      f"{'OV00 δ_v':>14}{'OV00 χ²_r':>14}")
print('-'*100)
for frac in SUBSAMPLES:
    if frac not in results: continue
    r = results[frac]
    print(f"{LABELS[frac]:<6}{MPS[frac]:>6.2f}{r['N']:>10}"
          f"{r['med_ratio']:>10.3f}"
          f"{r['t_best']:>14.4f}"
          f"{r['t_chi2']/max(r['t_ndof']-1,1):>14.2f}"
          f"{r['o_best']:>14.4f}"
          f"{r['o_chi2']/max(r['o_ndof']-1,1):>14.2f}")


# ═══════════════════════════════════════════════════════════════════
# Figure A — 2×2 panel of TOUCH/OV00/Vdn per subsample
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update(UNIFIED_RCPARAMS)
fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                          sharex=True, sharey=True, squeeze=False)
fig.subplots_adjust(top=0.90, wspace=0.10, hspace=0.16)
fig.suptitle('TOUCH ↔ Simone OV00 ↔ Vdn — all NREC50 subsamples\n'
             'V$_{eff}$ = 210³ h$^{-3}$Mpc³,  Vdn-free-δ_v fit at R ≥ 7',
             fontsize=14, fontweight='bold', y=0.97)

R_grid = np.logspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 200)
vdn_fid_smooth = vdn_at(R_grid, DELTA_V_FIDUCIAL)

for idx, frac in enumerate(SUBSAMPLES):
    ax = axes.flat[idx]
    if frac not in results:
        ax.text(0.5, 0.5, f"{LABELS[frac]}: data missing",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='#888')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(XLIM); ax.set_ylim(YLIM)
        continue
    r = results[frac]

    # Vdn fiducial
    ax.plot(R_grid, vdn_fid_smooth, ls='--', color='#888', lw=1.6, zorder=3,
            label='Vdn fiducial (δ_v=−0.80)' if idx == 0 else None)

    # TOUCH
    ok_t = r['touch_vsf'] > 0
    ax.errorbar(FORARUN_BIN_CENTERS[ok_t], r['touch_vsf'][ok_t],
                yerr=r['touch_err'][ok_t],
                fmt='v', ls='none', ms=7, color=TOUCH_COLOR,
                ecolor=TOUCH_COLOR, elinewidth=1.0, capsize=2,
                mec='k', mew=0.3, zorder=5,
                label='TOUCH (this work)' if idx == 0 else None)
    ax.plot(R_grid, vdn_at(R_grid, r['t_best']),
            ls=':', color=TOUCH_COLOR, lw=1.8, zorder=4)

    # OV00
    ax.errorbar(r['ov00_R'], r['ov00_vsf'], yerr=r['ov00_err'],
                fmt=SIMONE_MARKER, ls='none', ms=SIMONE_MS - 1,
                color=SIMONE_COLOR, ecolor=SIMONE_COLOR,
                elinewidth=1.0, capsize=2, mec='k', mew=0.3,
                alpha=0.85, zorder=10,
                label='Simone OV00' if idx == 0 else None)
    ax.plot(R_grid, vdn_at(R_grid, r['o_best']),
            ls=':', color=SIMONE_COLOR, lw=1.8, zorder=9)

    ax.axvline(R_MIN, color='#aaa', ls='-', lw=1.0, zorder=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.grid(True, ls='--', alpha=0.2, which='both')
    ax.set_title(
        f"{LABELS[frac]} NREC50  (MPS = {MPS[frac]} h⁻¹Mpc, N={r['N']})\n"
        f"best δ_v:  TOUCH={r['t_best']:.3f},  OV00={r['o_best']:.3f}   "
        f"v/v$_{{sim}}$(R≥7) = {r['med_ratio']:.2f}",
        fontsize=10, pad=6)
    if idx >= 2:
        ax.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
    if idx % 2 == 0:
        ax.set_ylabel(r'$dn/d\ln R_v$  [$h^3$Mpc$^{-3}$]')

# Single legend on first panel
axes[0, 0].legend(loc='lower left', fontsize=9, framealpha=0.95)

outA = os.path.join(OUT_DIR, 'bestfit_touch_ov00_vdn_allsubs.png')
fig.savefig(outA, dpi=220, bbox_inches='tight')
print(f"\nSaved: {outA}")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Figure B — best-fit δ_v vs MPS (trend line)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 7))
fig.subplots_adjust(top=0.90)
fig.suptitle('Best-fit δ_v vs mean particle separation  (Vdn fit at R ≥ 7)\n'
             'Flat trend ⇒ algorithm scale-invariant.  Slope ⇒ resolution-dependent δ_v shift.',
             fontsize=13, fontweight='bold', y=0.99)

ax.axhline(DELTA_V_FIDUCIAL, color='#888', ls='--', lw=1.5, zorder=1,
           label=f'cleaning threshold (δ_th = {DELTA_V_FIDUCIAL:.2f})')

mps_arr = np.array([MPS[f] for f in SUBSAMPLES if f in results])
t_bests = np.array([results[f]['t_best'] for f in SUBSAMPLES if f in results])
t_errs  = np.array([results[f]['t_err']  for f in SUBSAMPLES if f in results])
o_bests = np.array([results[f]['o_best'] for f in SUBSAMPLES if f in results])
o_errs  = np.array([results[f]['o_err']  for f in SUBSAMPLES if f in results])
labels  = [LABELS[f] for f in SUBSAMPLES if f in results]

ax.errorbar(mps_arr, t_bests, yerr=t_errs,
            fmt='v', ls='-', ms=11, color=TOUCH_COLOR,
            ecolor=TOUCH_COLOR, elinewidth=1.5, capsize=3,
            mec='k', mew=0.5, lw=1.5, zorder=5,
            label='TOUCH (this work)')
ax.errorbar(mps_arr, o_bests, yerr=o_errs,
            fmt=SIMONE_MARKER, ls='-', ms=11, color=SIMONE_COLOR,
            ecolor=SIMONE_COLOR, elinewidth=1.5, capsize=3,
            mec='k', mew=0.5, lw=1.5, alpha=0.85, zorder=4,
            label='Simone OV00')

for i, lbl in enumerate(labels):
    ax.text(mps_arr[i], min(t_bests[i], o_bests[i]) - 0.02, lbl,
            ha='center', va='top', fontsize=10, color='#444',
            fontweight='bold')

ax.set_xlabel(r'Mean particle separation  [$h^{-1}$Mpc]')
ax.set_ylabel(r'best-fit $\delta_v$  (Vdn at R $\geq$ 7)')
ax.grid(True, ls='--', alpha=0.25, which='both')
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

outB = os.path.join(OUT_DIR, 'bestfit_deltav_across_subsamples.png')
fig.savefig(outB, dpi=220, bbox_inches='tight')
print(f"Saved: {outB}")
plt.close(fig)
