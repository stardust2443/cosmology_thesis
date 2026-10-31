#!/usr/bin/env python3
"""
figure_simone_corrected.py

Post-process fix applying Simone's prescription:
  1. Border cut: void centers must be ≥ 20 h⁻¹Mpc from any face
  2. V_eff = (250 - 2*20)³ = 210³
  3. Vdn theory uses δ_v = -0.80 (matching cleaning threshold)
  4. OV00 = TOUCH (every overlap rejected — strong condition)

Compares pre-fix (V=250³, pad=0, δ_Vdn=-0.7) vs post-fix on the same data.

Output:
  plots/diagnostics/simone_corrected_vsf.png
  plots/diagnostics/simone_corrected_comparison.png
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

from bin_v4_cleaned_to_vsf import FORARUN_BIN_CENTERS, EDGES, DLNR
from _vsf_plot_utils import (
    UNIFIED_RCPARAMS, XLIM, YLIM, R_MIN,
    VDN_TRUTH_COLOR, SIMONE_COLOR, SIMONE_MARKER, SIMONE_MS,
)

DATA = os.path.join(HERE, '..', 'data')
OUT_DIR = os.path.join(HERE, '..', 'plots', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

SIMONE_REF = os.path.join(DATA, 'ForArun',
    'vsf_dm_250cube_ss0p05_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')

# Simone's exact prescription
BOX        = 250.0                  # Aletheia box side
BORDER_PAD = 20.0                   # void center must be ≥ 20 h⁻¹Mpc from each face
V_EFF_OLD  = BOX ** 3                              # 15,625,000  (what we used before)
V_EFF_NEW  = (BOX - 2 * BORDER_PAD) ** 3           # 9,261,000   (Simone's)
DELTA_V    = -0.80                  # density threshold — consistent everywhere

OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05

print(f"V_eff (old, 250³)              = {V_EFF_OLD:>13,.0f}  h⁻³Mpc³")
print(f"V_eff (Simone, (250-40)³=210³) = {V_EFF_NEW:>13,.0f}  h⁻³Mpc³")
print(f"Amplification factor for VSF   = {V_EFF_OLD / V_EFF_NEW:.4f}")


def vdn_curve(R_grid, delta_v=DELTA_V):
    """Vdn theory with the SAME δ_v as the cleaning threshold."""
    c = cbl.cosmology.LCDM(OMEGA_M_TRUE, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", SIGMA8_TRUE)
    sf = cbl.cosmology.SizeFunction(c)
    dc = c.deltac(0.0)
    return np.array([sf.size_function(float(R), 0.0, "Vdn",
                     1.0, 0.0, 1.0, delta_v, dc, "EisensteinHu")
                     for R in R_grid])


def bin_vsf(radii, v_eff):
    counts, _ = np.histogram(radii, bins=EDGES)
    vsf = counts / (v_eff * DLNR)
    err = np.sqrt(counts) / (v_eff * DLNR)
    return counts, vsf, err


def apply_border_cut(df, pad=BORDER_PAD):
    """Keep voids whose CENTER is at least `pad` from every face."""
    x, y, z = df['x'].values, df['y'].values, df['z'].values
    lo = min(x.min(), y.min(), z.min())
    hi = lo + BOX
    keep = (
        (x - lo >= pad) & (hi - x >= pad) &
        (y - lo >= pad) & (hi - y >= pad) &
        (z - lo >= pad) & (hi - z >= pad)
    )
    return df[keep].reset_index(drop=True), lo, hi


# ─── Load data ──────────────────────────────────────────────────────
sim = pd.read_csv(SIMONE_REF, sep=r'\s+', comment='#',
                  names=['r', 'vsf', 'err'])
sim = sim[sim['vsf'] > 0].reset_index(drop=True)
sim_by_r = dict(zip(sim['r'].round(3), sim['vsf']))

FILES = {
    'TOUCH (OV00 = strong)':
        os.path.join(DATA, 'cleaned_simple0',
                     'cleaned_simple0_nrec50_cat_5p_touch.csv'),
    'CENTER':
        os.path.join(DATA, 'cleaned_simple0',
                     'cleaned_simple0_nrec50_cat_5p_center.csv'),
}
COLORS  = {'TOUCH (OV00 = strong)': '#c0392b', 'CENTER': '#2980b9'}
MARKERS = {'TOUCH (OV00 = strong)': 'v',       'CENTER': '^'}

# Compute both pre-fix and post-fix VSF for each
results = {}
for label, path in FILES.items():
    df = pd.read_csv(path)
    radii_full = df['R_resc'].values

    # POST-FIX: apply 20 h⁻¹Mpc center border cut
    df_cut, lo, hi = apply_border_cut(df, pad=BORDER_PAD)
    radii_cut = df_cut['R_resc'].values

    counts_old, vsf_old, err_old = bin_vsf(radii_full, V_EFF_OLD)
    counts_new, vsf_new, err_new = bin_vsf(radii_cut, V_EFF_NEW)

    # Ratio to Simone in R≥7
    def med_ratio(vsf):
        rs = []
        for r, v in zip(FORARUN_BIN_CENTERS, vsf):
            if r < R_MIN: continue
            vs = sim_by_r.get(round(r, 3))
            if vs and vs > 0 and v > 0:
                rs.append(v / vs)
        return float(np.median(rs)) if rs else float('nan')
    def med_ratio_all(vsf):
        rs = []
        for r, v in zip(FORARUN_BIN_CENTERS, vsf):
            vs = sim_by_r.get(round(r, 3))
            if vs and vs > 0 and v > 0:
                rs.append(v / vs)
        return float(np.median(rs)) if rs else float('nan')

    results[label] = dict(
        n_full=len(df), n_cut=len(df_cut),
        radii_full=radii_full, radii_cut=radii_cut,
        vsf_old=vsf_old, err_old=err_old,
        vsf_new=vsf_new, err_new=err_new,
        ratio_old_R7=med_ratio(vsf_old),
        ratio_new_R7=med_ratio(vsf_new),
        ratio_old_all=med_ratio_all(vsf_old),
        ratio_new_all=med_ratio_all(vsf_new),
        box_lo=lo, box_hi=hi,
    )

# ─── Vdn at the correct threshold (−0.80) and at wrong (−0.70) ─────
R_grid = np.logspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 200)
vdn_correct = vdn_curve(R_grid, delta_v=-0.80)
vdn_wrong   = vdn_curve(R_grid, delta_v=-0.70)

# ─── Console summary ───────────────────────────────────────────────
print(f"\n{'='*90}")
print('PRE-FIX (V=250³, pad=0)  vs  POST-FIX (V=210³, pad=20 h⁻¹Mpc)')
print('='*90)
print(f"{'variant':<30}{'N_full':>8}{'N_cut':>8}"
      f"{'old R≥7':>10}{'new R≥7':>10}{'old all':>10}{'new all':>10}")
print('-'*90)
for label, r in results.items():
    print(f"{label:<30}{r['n_full']:>8}{r['n_cut']:>8}"
          f"{r['ratio_old_R7']:>10.3f}{r['ratio_new_R7']:>10.3f}"
          f"{r['ratio_old_all']:>10.3f}{r['ratio_new_all']:>10.3f}")

# ═══════════════════════════════════════════════════════════════════
# PLOT 1: Pre-fix vs post-fix VSF overlay
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update(UNIFIED_RCPARAMS)
fig, (ax_old, ax_new) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
fig.subplots_adjust(top=0.85, wspace=0.05)
fig.suptitle("Simone's prescription applied: before vs after fixes\n"
             "ss5%, 50 OTs — Vdn(δ_v = −0.80, correct), "
             "V_eff = (250−40)³, center ≥ 20 h⁻¹Mpc from border",
             fontsize=13, fontweight='bold', y=0.97)

# LEFT: old (V=250³, pad=0, Vdn at −0.70)
ax_old.plot(R_grid, vdn_wrong, ls=':', color='#888', lw=2, zorder=3,
            label='Vdn at $δ_v = −0.70$ (old, INCONSISTENT)')
ax_old.plot(R_grid, vdn_correct, ls='--', color=VDN_TRUTH_COLOR, lw=2.2,
            zorder=4, label='Vdn at $δ_v = −0.80$ (correct)')
ax_old.errorbar(sim['r'], sim['vsf'], yerr=sim['err'],
                fmt=SIMONE_MARKER, ls='none', ms=SIMONE_MS,
                color=SIMONE_COLOR, alpha=0.8, zorder=10,
                label='Simone reference (OV00)')
for label, r in results.items():
    ok = r['vsf_old'] > 0
    ax_old.errorbar(FORARUN_BIN_CENTERS[ok], r['vsf_old'][ok],
                    yerr=r['err_old'][ok],
                    fmt=MARKERS[label], ls='none', ms=8,
                    color=COLORS[label], mec='k', mew=0.4,
                    label=f"{label}  N={r['n_full']}  v/v_sim(R≥7)={r['ratio_old_R7']:.2f}",
                    zorder=5)
ax_old.set_xscale('log'); ax_old.set_yscale('log')
ax_old.set_xlim(XLIM); ax_old.set_ylim(YLIM)
ax_old.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
ax_old.set_ylabel(r'$dn/d\ln R_v$  [$h^3$Mpc$^{-3}$]')
ax_old.set_title('BEFORE fix:  V_eff = 250³,  border = sphere fits,  Vdn at δ = −0.70',
                 fontsize=11)
ax_old.grid(True, ls='--', alpha=0.2, which='both')
ax_old.legend(loc='lower left', fontsize=9, framealpha=0.95)

# RIGHT: new (V=210³, pad=20 center cut, Vdn at −0.80)
ax_new.plot(R_grid, vdn_correct, ls='--', color=VDN_TRUTH_COLOR, lw=2.2,
            zorder=4, label='Vdn at $δ_v = −0.80$ (correct)')
ax_new.errorbar(sim['r'], sim['vsf'], yerr=sim['err'],
                fmt=SIMONE_MARKER, ls='none', ms=SIMONE_MS,
                color=SIMONE_COLOR, alpha=0.8, zorder=10,
                label='Simone reference (OV00)')
for label, r in results.items():
    ok = r['vsf_new'] > 0
    ax_new.errorbar(FORARUN_BIN_CENTERS[ok], r['vsf_new'][ok],
                    yerr=r['err_new'][ok],
                    fmt=MARKERS[label], ls='none', ms=8,
                    color=COLORS[label], mec='k', mew=0.4,
                    label=f"{label}  N_cut={r['n_cut']}  v/v_sim(R≥7)={r['ratio_new_R7']:.2f}",
                    zorder=5)
ax_new.set_xscale('log'); ax_new.set_yscale('log')
ax_new.set_xlim(XLIM); ax_new.set_ylim(YLIM)
ax_new.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
ax_new.set_title("AFTER fix:  V_eff = 210³,  center ≥ 20 h⁻¹Mpc,  Vdn at δ = −0.80",
                 fontsize=11)
ax_new.grid(True, ls='--', alpha=0.2, which='both')
ax_new.legend(loc='lower left', fontsize=9, framealpha=0.95)

out1 = os.path.join(OUT_DIR, 'simone_corrected_vsf.png')
fig.savefig(out1, dpi=220, bbox_inches='tight')
print(f"\nSaved: {out1}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
# PLOT 2: Ratio vs Simone — before & after
# ═══════════════════════════════════════════════════════════════════
fig, (ax_old, ax_new) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
fig.subplots_adjust(top=0.85, wspace=0.05)
fig.suptitle('Ratio vs Simone — effect of border cut + V_eff fix\n'
             'green ±20%,  orange ±factor 2',
             fontsize=13, fontweight='bold', y=0.97)

for ax in (ax_old, ax_new):
    ax.axhline(1, color='k', lw=0.8, zorder=2)
    ax.fill_between(XLIM, 0.8, 1.2, color='#27ae60', alpha=0.12, zorder=0)
    ax.fill_between(XLIM, 0.5, 2.0, color='#f39c12', alpha=0.07, zorder=0)
    ax.axvline(R_MIN, color='#aaa', ls='-', lw=1.2, zorder=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(XLIM); ax.set_ylim(0.2, 5)
    ax.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax.set_yticklabels(['0.25', '0.5', '1', '2', '4'])
    ax.grid(True, ls='--', alpha=0.2, which='both')
    ax.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')

ax_old.set_ylabel('simple-cleaned / Simone')
ax_old.set_title('BEFORE fix  (V=250³, pad=0)', fontsize=11)
ax_new.set_title("AFTER fix   (V=210³, center ≥20 h⁻¹Mpc)", fontsize=11)

for label, r in results.items():
    for ax, vsf, ratio in [
        (ax_old, r['vsf_old'], r['ratio_old_R7']),
        (ax_new, r['vsf_new'], r['ratio_new_R7']),
    ]:
        ratio_arr = np.full(len(FORARUN_BIN_CENTERS), np.nan)
        for j, rb in enumerate(FORARUN_BIN_CENTERS):
            vs = sim_by_r.get(round(rb, 3))
            if vs and vs > 0 and vsf[j] > 0:
                ratio_arr[j] = vsf[j] / vs
        fit_m = FORARUN_BIN_CENTERS >= R_MIN
        ok = np.isfinite(ratio_arr)
        ax.plot(FORARUN_BIN_CENTERS[ok & ~fit_m], ratio_arr[ok & ~fit_m],
                MARKERS[label], ls='none', ms=5,
                color=COLORS[label], alpha=0.3, zorder=4)
        ax.plot(FORARUN_BIN_CENTERS[ok & fit_m], ratio_arr[ok & fit_m],
                MARKERS[label], ls='none', ms=8,
                color=COLORS[label], mec='k', mew=0.4, zorder=5,
                label=f"{label}  median(R≥7) = {ratio:.2f}")

for ax in (ax_old, ax_new):
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)

out2 = os.path.join(OUT_DIR, 'simone_corrected_comparison.png')
fig.savefig(out2, dpi=220, bbox_inches='tight')
print(f"Saved: {out2}")
plt.close(fig)
