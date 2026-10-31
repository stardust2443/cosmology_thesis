#!/usr/bin/env python3
"""
figure_touch_ov00_allsub_fixed.py

VSF validation plots for all subsamples:
  - TOUCH (strict overlap) vs Simone OV00
  - CENTER (center-overlap) vs Simone OV05
All curves are computed with delta_v fixed at -0.80 in the Vdn model.
No mention of MPS is included on the figures.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

from bin_v4_cleaned_to_vsf import FORARUN_BIN_CENTERS, EDGES, DLNR, FRAC_MAP
from _vsf_plot_utils import (
    UNIFIED_RCPARAMS, XLIM, YLIM, R_MIN,
    VDN_TRUTH_COLOR, SIMONE_COLOR, SIMONE_MARKER, SIMONE_MS,
    SUBSAMPLE_COLORS, MPS,
    draw_exclusion_zone, apply_axis_style,
)

DATA = os.path.join(HERE, '..', 'data')
OUT_DIR = os.path.join(HERE, '..', 'plots', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

# Aletheia box effective volume after 20 h^-1 Mpc center border cut
V_EFF = (250.0 - 2 * 20.0) ** 3  # 210^3 = 9,261,000 h^-3 Mpc^3

OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05
DELTA_V_NL = -0.80

LABELS = {'20p': '20%', '10p': '10%', '5p': '5%', '2_5p': '2.5%'}
MARKERS = {'touch': 'v', 'center': '^'}

# ─── CBL Vdn setup ───────────────────────────────────────────────────
print("Setting up CBL Vdn LCDM cosmology at truth...")
_cosmo = cbl.cosmology.LCDM(OMEGA_M_TRUE, OMEGA_B, 0.0, H0,
                            SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
_cosmo.set_parameter("sigma8", SIGMA8_TRUE)
_sf = cbl.cosmology.SizeFunction(_cosmo)
_dc = _cosmo.deltac(0.0)

def vdn_truth_curve(R_grid):
    """Vdn theory with delta_v_NL fixed at -0.80, no bias (b=1, slope=0, offset=1)."""
    return np.array([_sf.size_function(float(R), 0.0, "Vdn",
                     1.0, 0.0, 1.0, DELTA_V_NL, _dc, "EisensteinHu")
                     for R in R_grid])

def bin_vsf(radii):
    counts, _ = np.histogram(radii, bins=EDGES)
    vsf = counts / (V_EFF * DLNR)
    err = np.sqrt(counts) / (V_EFF * DLNR)
    return vsf, err

def compute_chi2_R7(R, vsf, err, vdn_at_bins):
    ok = (vsf > 0) & (vdn_at_bins > 0) & (err > 0) & (R >= R_MIN)
    if not ok.any():
        return 0.0, 0
    chi2 = np.sum(((vsf[ok] - vdn_at_bins[ok]) / err[ok]) ** 2)
    return float(chi2), int(ok.sum())

# ─── Plotting Function ────────────────────────────────────────────────
def make_vsf_grid(field, subsamples, overlap_type, simone_ov, outfile):
    print(f"\nGenerating {field} {overlap_type.upper()} vs {simone_ov} plot ({outfile})...")
    
    n_subs = len(subsamples)
    n_cols = min(n_subs, 2)
    n_rows = (n_subs + n_cols - 1) // n_cols
    
    plt.rcParams.update(UNIFIED_RCPARAMS)
    fig = plt.figure(figsize=(7.5 * n_cols, 6 * n_rows))
    
    # GridSpec for main + ratio subplots
    outer = gridspec.GridSpec(n_rows, n_cols, hspace=0.30, wspace=0.24,
                              left=0.07, right=0.97, top=0.91, bottom=0.07)
    
    R_grid = np.logspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 200)
    vdn_smooth = vdn_truth_curve(R_grid)
    vdn_at_bins = vdn_truth_curve(FORARUN_BIN_CENTERS)
    
    summary_data = []
    
    for idx, frac in enumerate(subsamples):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[idx], height_ratios=[3, 1], hspace=0.0)
        
        ax_main = fig.add_subplot(inner[0])
        ax_rat  = fig.add_subplot(inner[1], sharex=ax_main)
        
        # Load our cleaned catalog
        cat_path = os.path.join(DATA, 'cleaned_simple0',
                                f'cleaned_simple0_{field.lower()}_cat_{frac}_{overlap_type}.csv')
        if not os.path.exists(cat_path):
            print(f"  Warning: cleaned catalog not found for {frac} ({cat_path}). Skipping panel.")
            continue
            
        df = pd.read_csv(cat_path)
        radii = df['R_resc'].values
        vsf, err = bin_vsf(radii)
        
        # Load Simone reference (OV00 or OV05)
        ref_path = os.path.join(DATA, 'ForArun',
                                f'vsf_dm_250cube_ss0p{FRAC_MAP[frac]}_{field}_SM0p00_CS0p79_TH-0p80_{simone_ov}.dat')
        if not os.path.exists(ref_path):
            print(f"  Warning: Simone reference not found for {frac} ({ref_path}). Skipping panel.")
            continue
            
        ref_df = pd.read_csv(ref_path, sep=r'\s+', comment='#', names=['r', 'vsf', 'err'])
        ref_df = ref_df[ref_df['vsf'] > 0].reset_index(drop=True)
        ref_R = ref_df['r'].values
        ref_vsf = ref_df['vsf'].values
        ref_err = ref_df['err'].values
        
        sim_by_r = dict(zip(ref_R.round(3), ref_vsf))
        
        # ── Main panel ──
        draw_exclusion_zone(ax_main)
        
        # Vdn theory (dashed line)
        ax_main.plot(R_grid, vdn_smooth, ls='--', color=VDN_TRUTH_COLOR, lw=2.0, zorder=3,
                     label='Vdn theory ($\\delta_v^{NL}=-0.80$)' if idx == 0 else None)
        
        # Simone reference (markers)
        ax_main.errorbar(ref_R, ref_vsf, yerr=ref_err, fmt=SIMONE_MARKER, ls='none', ms=SIMONE_MS,
                         color=SIMONE_COLOR, ecolor=SIMONE_COLOR, elinewidth=1.0, capsize=2.5,
                         mec='k', mew=0.4, alpha=0.8, zorder=10,
                         label=f'Simone reference ({simone_ov})' if idx == 0 else None)
                         
        # Our cleaning (markers)
        ok_vsf = vsf > 0
        ax_main.errorbar(FORARUN_BIN_CENTERS[ok_vsf], vsf[ok_vsf], yerr=err[ok_vsf], fmt=MARKERS[overlap_type], ls='none', ms=7.5,
                         color=SUBSAMPLE_COLORS[frac], ecolor=SUBSAMPLE_COLORS[frac], elinewidth=1.2, capsize=2.5,
                         mec='k', mew=0.4, zorder=5,
                         label=f'{overlap_type.upper()} (this work)' if idx == 0 else None)
                         
        ax_main.set_xscale('log')
        ax_main.set_yscale('log')
        ax_main.set_xlim(XLIM)
        ax_main.set_ylim(YLIM)
        ax_main.set_ylabel(r'$dn/d\ln R_v$  [$h^3$Mpc$^{-3}$]')
        ax_main.tick_params(labelbottom=False)
        ax_main.grid(True, ls='--', alpha=0.25, which='both')
        ax_main.set_title(f"{field}  {LABELS[frac]}", pad=8)
        
        if idx == 0:
            ax_main.legend(loc='lower left', framealpha=0.95, edgecolor='#bbb', fontsize=9.5)
            
        # Stats
        chi2_R7, ndof_R7 = compute_chi2_R7(FORARUN_BIN_CENTERS, vsf, err, vdn_at_bins)
        chi2_red = chi2_R7 / max(ndof_R7 - 1, 1)
        
        # Ratio ours / Simone
        ratio_sim = np.full(len(FORARUN_BIN_CENTERS), np.nan)
        ratio_sim_err = np.full(len(FORARUN_BIN_CENTERS), np.nan)
        for j, r in enumerate(FORARUN_BIN_CENTERS):
            vs = sim_by_r.get(round(r, 3))
            if vs and vs > 0 and vsf[j] > 0:
                ratio_sim[j] = vsf[j] / vs
                ratio_sim_err[j] = ratio_sim[j] * np.sqrt((err[j]/vsf[j])**2 + (ref_err[ref_R.round(3) == round(r, 3)][0]/vs)**2)
                
        fit_mask = FORARUN_BIN_CENTERS >= R_MIN
        med_ratio = float(np.median(ratio_sim[fit_mask & np.isfinite(ratio_sim)]))
        
        summary_data.append((frac, len(df), int((radii >= R_MIN).sum()), chi2_R7, ndof_R7, chi2_red, med_ratio))
        
        # Add badge to main panel
        ax_main.text(0.97, 0.05,
                     f'{overlap_type.upper()} / Simone (R≥7, median): {med_ratio:.2f}$\\times$\n'
                     f'$\\chi^2_\\nu$ vs Vdn (R≥7): {chi2_red:.2f} ({chi2_R7:.1f}/{ndof_R7})',
                     transform=ax_main.transAxes, ha='right', va='bottom', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#bbb', alpha=0.92))
                     
        # ── Ratio sub-panel ──
        draw_exclusion_zone(ax_rat)
        ax_rat.axhline(1, color='k', lw=0.8, zorder=2)
        ax_rat.fill_between(list(XLIM), 0.8, 1.2, color='#27ae60', alpha=0.10, zorder=0)
        ax_rat.fill_between(list(XLIM), 0.5, 2.0, color='#f39c12', alpha=0.07, zorder=0)
        
        ok_ratio = np.isfinite(ratio_sim)
        # Faded below R_MIN
        below = ok_ratio & ~fit_mask
        if below.any():
            ax_rat.errorbar(FORARUN_BIN_CENTERS[below], ratio_sim[below], yerr=ratio_sim_err[below],
                            fmt=MARKERS[overlap_type], ls='none', ms=4.5, color=SUBSAMPLE_COLORS[frac], alpha=0.3,
                            ecolor=SUBSAMPLE_COLORS[frac], elinewidth=0.8, capsize=2, zorder=4)
        # Solid at R >= 7
        above = ok_ratio & fit_mask
        if above.any():
            ax_rat.errorbar(FORARUN_BIN_CENTERS[above], ratio_sim[above], yerr=ratio_sim_err[above],
                            fmt=MARKERS[overlap_type], ls='none', ms=6.5, color=SUBSAMPLE_COLORS[frac],
                            ecolor=SUBSAMPLE_COLORS[frac], elinewidth=1.2, capsize=2.5,
                            mec='k', mew=0.4, zorder=5)
                            
        ax_rat.set_xscale('log')
        ax_rat.set_yscale('log')
        ax_rat.set_xlim(XLIM)
        ax_rat.set_ylim(0.2, 5.0)
        ax_rat.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0])
        ax_rat.set_yticklabels(['0.25', '0.5', '1', '2', '4'])
        ax_rat.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
        ax_rat.set_ylabel(f'{overlap_type.upper()} / Simone', fontsize=8.5)
        ax_rat.grid(True, ls='--', alpha=0.25, which='both')
        
    fig.suptitle(f'{field} VSF: Simone reference ({simone_ov}) vs {overlap_type.upper()} simple-cleaned\n'
                 f'Dashed line = Vdn theory at truth ($\\delta_v^{{NL}}={DELTA_V_NL}$). '
                 f'Grey zone: $R < 7$ $h^{{-1}}$Mpc (excluded from fit).',
                 fontsize=13, y=0.98)
                 
    fig.savefig(outfile, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {outfile}")
    
    # Print summary table
    print(f"\n  Summary Table ({field} {overlap_type.upper()} vs {simone_ov}):")
    print(f"  {'subsample':<10} {'N_total':>8} {'N_R≥7':>7} {'χ²':>10} {'ndof':>5} {'χ²_red':>7} {'ratio_R7':>10}")
    print(f"  {'-'*65}")
    for row in summary_data:
        print(f"  {row[0]:<10} {row[1]:>8d} {row[2]:>7d} {row[3]:>10.1f} {row[4]:>5d} {row[5]:>7.2f} {row[6]:>10.2f}")
    print(f"  {'-'*65}")

def main():
    nrec50_subs = ['20p', '10p', '5p', '2_5p']
    bit_subs = ['5p', '2_5p']
    
    # 1. NREC50 TOUCH vs Simone OV00
    nrec50_touch_out = os.path.join(OUT_DIR, 'touch_vs_simone_ov00_nrec50_fixed.png')
    make_vsf_grid('NREC50', nrec50_subs, 'touch', 'OV00', nrec50_touch_out)
    
    # 2. BIT TOUCH vs Simone OV00
    bit_touch_out = os.path.join(OUT_DIR, 'touch_vs_simone_ov00_bit_fixed.png')
    make_vsf_grid('BIT', bit_subs, 'touch', 'OV00', bit_touch_out)
    
    # 3. NREC50 CENTER vs Simone OV05
    nrec50_center_out = os.path.join(OUT_DIR, 'center_vs_simone_ov05_nrec50_fixed.png')
    make_vsf_grid('NREC50', nrec50_subs, 'center', 'OV05', nrec50_center_out)

if __name__ == '__main__':
    main()
