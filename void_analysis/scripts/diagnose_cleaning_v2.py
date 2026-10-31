#!/usr/bin/env python3
"""
diagnose_cleaning_v2.py
Corrected diagnostic script that tests the full hypothesis space against ForArun data.
Fixes all 6 issues identified in the cleaning validation:
  1. Tests delta(2MPS) <= -0.79 threshold (CS0p79)
  2. Tests deltaAv <= -0.80 threshold (TH-0p80)
  3. Uses strict containment boundary (not center-cut)
  4. Radius-dependent V_eff = (250 - 2*R)^3
  5. Correct coordinate system (no % 250 wrapping)
  6. Uses ForArun's exact binning scheme for comparison
"""
import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/Users/arun/projects/my_cosmo_analysis/void_analysis'

# ---------- ForArun binning (extracted from their .dat files) ----------
FORARUN_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])

def get_forarun_bins():
    """Reconstruct bin edges from the geometric-mean bin centers."""
    centers = FORARUN_BIN_CENTERS
    log_c = np.log(centers)
    # Assume uniform log spacing
    dlnr = np.median(np.diff(log_c))
    edges = np.exp(np.append(log_c - dlnr/2, log_c[-1] + dlnr/2))
    return edges, centers, dlnr

def compute_vsf_forarun_bins(radii):
    """Compute VSF using reference's exact binning and radius-dependent V_eff."""
    edges, centers, dlnr = get_forarun_bins()
    counts, _ = np.histogram(radii, bins=edges)
    v_eff = (250.0 - 2.0 * centers)**3
    v_eff = np.clip(v_eff, 1e-5, None)
    dn_dlnr = counts / (v_eff * dlnr)
    return centers, dn_dlnr, counts

def compute_vsf_fixed_vol(radii, v_eff):
    """Compute VSF using reference bins but fixed volume."""
    edges, centers, dlnr = get_forarun_bins()
    counts, _ = np.histogram(radii, bins=edges)
    dn_dlnr = counts / (v_eff * dlnr)
    return centers, dn_dlnr, counts


def clean_catalog(df_raw, cs_thresh, th_thresh, sort_by, ov_rad, bound_mode):
    """Apply a specific cleaning configuration."""
    df = df_raw.copy()

    # Step 1: Density thresholds
    if cs_thresh is not None:
        df = df[df['delta_2MPS'] <= cs_thresh].copy()
    if th_thresh is not None:
        df = df[df['delta_avg'] <= th_thresh].copy()

    if len(df) == 0:
        return df

    # Detect box origin
    x_min = df['x'].min()
    box_lower = 375.0 if x_min > 300 else 0.0
    box_upper = box_lower + 250.0

    # Step 2: Boundary cut
    if bound_mode == 'containment_r_ext':
        mask = (
            (df['x'] - df['r_ext'] > box_lower) & (df['x'] + df['r_ext'] < box_upper) &
            (df['y'] - df['r_ext'] > box_lower) & (df['y'] + df['r_ext'] < box_upper) &
            (df['z'] - df['r_ext'] > box_lower) & (df['z'] + df['r_ext'] < box_upper)
        )
        df = df[mask].copy()
    elif bound_mode == 'center_20':
        cut = 20.0
        mask = (
            (df['x'] > box_lower + cut) & (df['x'] < box_upper - cut) &
            (df['y'] > box_lower + cut) & (df['y'] < box_upper - cut) &
            (df['z'] > box_lower + cut) & (df['z'] < box_upper - cut)
        )
        df = df[mask].copy()

    if len(df) == 0:
        return df

    # Step 3: Sort
    if sort_by == 'r_ext':
        df = df.sort_values('r_ext', ascending=False).reset_index(drop=True)
    elif sort_by == 'radius':
        df = df.sort_values('radius', ascending=False).reset_index(drop=True)

    # Step 4: Overlap removal (OV00: no overlap allowed)
    coords = df[['x', 'y', 'z']].values
    if ov_rad == 'r_ext':
        radii = df['r_ext'].values
    else:
        radii = df['radius'].values

    keep_mask = np.ones(len(df), dtype=bool)
    tree = cKDTree(coords)
    max_search = radii.max() * 2.1

    for i in range(len(df)):
        if not keep_mask[i]:
            continue
        neighbors = tree.query_ball_point(coords[i], r=max_search)
        for j in neighbors:
            if j <= i or not keep_mask[j]:
                continue
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < (radii[i] + radii[j]):
                keep_mask[j] = False

    return df[keep_mask].copy()


def compute_ssr(r_calc, vsf_calc, df_truth):
    """Compute fractional SSR between our VSF and reference truth."""
    ssr = 0.0
    n = 0
    for _, row in df_truth.iterrows():
        r_t, vsf_t = row['r'], row['vsf']
        if vsf_t <= 0:
            continue
        idx = np.abs(r_calc - r_t).argmin()
        rel_diff = np.abs(r_calc[idx] - r_t) / r_t
        if rel_diff < 0.05:  # strict matching
            ssr += ((vsf_calc[idx] - vsf_t) / vsf_t) ** 2
            n += 1
    return ssr / n if n > 3 else float('inf'), n


def run_ablation(raw_file, truth_file, label):
    """Run full ablation study for one raw catalog vs one reference file."""
    cols = ['x', 'y', 'z', 'radius', 'r_ext', 'delta_2MPS', 'delta_avg']
    df_raw = pd.read_csv(raw_file, sep=r'\s+', comment='#', names=cols)
    df_truth = pd.read_csv(truth_file, sep=r'\s+', comment='#', names=['r', 'vsf', 'err'])

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Raw voids: {len(df_raw)}")
    print(f"  Truth non-zero bins: {(df_truth['vsf'] > 0).sum()}")
    print(f"{'='*70}")

    configs = []

    # Key combinations to test
    cs_thresholds = [None, -0.79]
    th_thresholds = [None, -0.80]
    sort_bys = ['r_ext', 'radius']
    ov_rads = ['r_ext', 'radius']
    bound_modes = ['containment_r_ext', 'center_20']
    v_eff_modes = ['variable', 'fixed_250']

    combos = list(itertools.product(
        cs_thresholds, th_thresholds, sort_bys, ov_rads, bound_modes, v_eff_modes
    ))

    results = []
    for combo in combos:
        cs, th, sort_by, ov_rad, bound, v_mode = combo

        # Skip illogical combos
        if bound == 'center_20' and v_mode == 'variable':
            continue

        df_clean = clean_catalog(df_raw, cs, th, sort_by, ov_rad, bound)
        if len(df_clean) == 0:
            continue

        if v_mode == 'variable':
            r_c, vsf_c, counts = compute_vsf_forarun_bins(df_clean['r_ext'].values)
        else:
            r_c, vsf_c, counts = compute_vsf_fixed_vol(df_clean['r_ext'].values, 250.0**3)

        ssr, n_pts = compute_ssr(r_c, vsf_c, df_truth)
        results.append((ssr, n_pts, len(df_clean), combo))

    # Sort by SSR
    results.sort(key=lambda x: x[0])

    print(f"\n  Top 10 configurations:")
    print(f"  {'SSR':>8}  {'Pts':>4}  {'NVoid':>6}  Config")
    print(f"  {'-'*8}  {'-'*4}  {'-'*6}  {'-'*50}")
    for ssr, n_pts, n_voids, combo in results[:10]:
        cs, th, sort_by, ov_rad, bound, v_mode = combo
        cs_str = f"CS{cs}" if cs else "NoCS"
        th_str = f"TH{th}" if th else "NoTH"
        print(f"  {ssr:8.4f}  {n_pts:4d}  {n_voids:6d}  "
              f"{cs_str:>8} {th_str:>8} sort={sort_by:>5} ov={ov_rad:>5} "
              f"bnd={bound:>20} vol={v_mode:>10}")

    return results


def make_overlay_plot(raw_file, truth_file, config, label, out_path):
    """Generate overlay plot for the best config."""
    cols = ['x', 'y', 'z', 'radius', 'r_ext', 'delta_2MPS', 'delta_avg']
    df_raw = pd.read_csv(raw_file, sep=r'\s+', comment='#', names=cols)
    df_truth = pd.read_csv(truth_file, sep=r'\s+', comment='#', names=['r', 'vsf', 'err'])

    cs, th, sort_by, ov_rad, bound, v_mode = config
    df_clean = clean_catalog(df_raw, cs, th, sort_by, ov_rad, bound)

    if v_mode == 'variable':
        r_c, vsf_c, counts = compute_vsf_forarun_bins(df_clean['r_ext'].values)
    else:
        r_c, vsf_c, counts = compute_vsf_fixed_vol(df_clean['r_ext'].values, 250.0**3)

    # Truth
    mask_t = df_truth['vsf'] > 0
    r_t = df_truth['r'][mask_t].values
    vsf_t = df_truth['vsf'][mask_t].values
    err_t = df_truth['err'][mask_t].values

    # Our data (only non-zero)
    mask_c = vsf_c > 0
    r_c_nz = r_c[mask_c]
    vsf_c_nz = vsf_c[mask_c]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(r_t, vsf_t, yerr=err_t, fmt='o', color='black', ms=5,
                capsize=3, label='reference', zorder=5)
    ax.plot(r_c_nz, vsf_c_nz, 's-', color='#e74c3c', ms=4, lw=1.5,
            label='Our cleaning', zorder=4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$R_v \; [h^{-1}\mathrm{Mpc}]$', fontsize=14)
    ax.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v \; [h^3\mathrm{Mpc}^{-3}]$', fontsize=14)
    ax.set_title(label, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', ls='--', alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    plot_dir = os.path.join(BASE, 'plots', 'cleaning_validation')
    os.makedirs(plot_dir, exist_ok=True)

    # ---- Test 1: NREC50 10p OV00 ----
    results_nrec10 = run_ablation(
        os.path.join(BASE, 'data/raw/voids_recon_50_bitvf_aletheia_10p'),
        os.path.join(BASE, 'data/ForArun/vsf_dm_250cube_ss0p10_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat'),
        "NREC50 10p OV00"
    )

    # ---- Test 2: BIT 2.5p OV00 ----
    bit_2_5p = os.path.join(BASE, 'data/raw/0_voids_bitvf_aletheia_2_5p.txt')
    if os.path.exists(bit_2_5p):
        results_bit25 = run_ablation(
            bit_2_5p,
            os.path.join(BASE, 'data/ForArun/vsf_dm_250cube_ss0p025_BIT_SM0p00_CS0p79_TH-0p80_OV00.dat'),
            "BIT 2.5p OV00"
        )

    # ---- Test 3: NREC50 5p OV00 ----
    results_nrec5 = run_ablation(
        os.path.join(BASE, 'data/raw/voids_recon_50_bitvf_aletheia_5p'),
        os.path.join(BASE, 'data/ForArun/vsf_dm_250cube_ss0p05_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat'),
        "NREC50 5p OV00"
    )

    # ---- Test 4: NREC50 20p OV00 ----
    results_nrec20 = run_ablation(
        os.path.join(BASE, 'data/raw/voids_recon_50_bitvf_aletheia_20p'),
        os.path.join(BASE, 'data/ForArun/vsf_dm_250cube_ss0p20_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat'),
        "NREC50 20p OV00"
    )

    # Generate overlay plots for the best config from NREC50 10p
    if results_nrec10:
        best_config = results_nrec10[0][3]
        print(f"\n\nBest overall config: {best_config}")

        # Plot for each available combination
        test_cases = [
            ('voids_recon_50_bitvf_aletheia_10p',
             'vsf_dm_250cube_ss0p10_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat',
             'NREC50 10p OV00'),
            ('voids_recon_50_bitvf_aletheia_5p',
             'vsf_dm_250cube_ss0p05_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat',
             'NREC50 5p OV00'),
            ('voids_recon_50_bitvf_aletheia_20p',
             'vsf_dm_250cube_ss0p20_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat',
             'NREC50 20p OV00'),
            ('voids_recon_50_bitvf_aletheia_2_5p',
             'vsf_dm_250cube_ss0p025_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat',
             'NREC50 2.5p OV00'),
        ]

        for raw_name, truth_name, label in test_cases:
            raw_f = os.path.join(BASE, 'data/raw', raw_name)
            truth_f = os.path.join(BASE, 'data/ForArun', truth_name)
            if os.path.exists(raw_f) and os.path.exists(truth_f):
                out_f = os.path.join(plot_dir, f"overlay_v2_{label.replace(' ', '_')}.png")
                make_overlay_plot(raw_f, truth_f, best_config, label, out_f)


if __name__ == '__main__':
    main()
