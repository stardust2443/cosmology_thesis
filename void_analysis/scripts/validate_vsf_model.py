#!/usr/bin/env python3
"""
validate_vsf_model.py
Pre-MCMC validation: overlay the Vdn theoretical VSF (at the known Aletheia
cosmology) on the empirical OV00 data from the thesis author.

This is the critical "sanity check" — if the theory doesn't approximately
match the data at the TRUE cosmology, MCMC will not produce sensible posteriors.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add scripts dir to path so we can import vsf_model
sys.path.insert(0, os.path.dirname(__file__))
from vsf_model import get_theoretical_vsf

# ─── Configuration ────────────────────────────────────────────────────
# True Aletheia simulation cosmology
OMEGA_M_TRUE = 0.315
SIGMA8_TRUE = 0.811

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'ForArun')
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')

SUBSAMPLES = {
    '20p': {'file_key': '20', 'color': '#9b59b6', 'label': 'ss = 20%'},
    '10p': {'file_key': '10', 'color': '#3498db', 'label': 'ss = 10%'},
    '5p':  {'file_key': '05', 'color': '#2ecc71', 'label': 'ss = 5%'},
    '2_5p': {'file_key': '025', 'color': '#f1c40f', 'label': 'ss = 2.5%'},
}

# ─── Plotting style ───────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 14,
    'axes.labelsize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'axes.linewidth': 1.2,
})


def load_thesis_vsf(subsample_key, field_type='NREC00'):
    """Load the pre-computed OV00 VSF from the thesis author."""
    fk = SUBSAMPLES[subsample_key]['file_key']
    fname = f"vsf_dm_250cube_ss0p{fk}_{field_type}_SM0p00_CS0p79_TH-0p80_OV00.dat"
    fpath = os.path.join(DATA_DIR, fname)
    
    if not os.path.exists(fpath):
        print(f"  [SKIP] File not found: {fname}")
        return None
    
    df = pd.read_csv(fpath, sep=r'\s+', comment='#', names=['r', 'vsf', 'err'])
    # Keep only bins with nonzero counts
    df = df[df['vsf'] > 0].copy()
    return df


def compute_chi2(r_data, vsf_data, err_data, r_theory, vsf_theory):
    """Compute reduced χ² by interpolating theory onto data bins."""
    theory_interp = np.interp(r_data, r_theory, vsf_theory)
    residuals = (vsf_data - theory_interp) / err_data
    chi2 = np.sum(residuals**2)
    ndof = len(r_data) - 2  # 2 free parameters (Ωm, σ8)
    return chi2, ndof, chi2 / max(ndof, 1)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Evaluate theory on a fine grid
    R_theory = np.linspace(2.0, 20.0, 200)
    vsf_theory = get_theoretical_vsf(R_theory, OMEGA_M_TRUE, SIGMA8_TRUE)
    
    # ─── Figure 1: NREC00 (True Displacement Field) ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, field_type, title in zip(
        axes, ['NREC00', 'NREC50'],
        ['True Displacement Field (NREC00)', 'OT Reconstructed (NREC50)']
    ):
        ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Plot theory curve
        ax.plot(R_theory, vsf_theory, 'k--', lw=2, alpha=0.8,
                label=f'Vdn Theory ($\\Omega_m$={OMEGA_M_TRUE}, $\\sigma_8$={SIGMA8_TRUE})')
        
        for ss_key, ss_info in SUBSAMPLES.items():
            df = load_thesis_vsf(ss_key, field_type=field_type)
            if df is None or df.empty:
                continue
            
            ax.errorbar(df['r'], df['vsf'], yerr=df['err'],
                        fmt='o', color=ss_info['color'], ms=5, capsize=3,
                        label=ss_info['label'])
            
            # Compute χ²
            chi2, ndof, chi2_red = compute_chi2(
                df['r'].values, df['vsf'].values, df['err'].values,
                R_theory, vsf_theory
            )
            print(f"  {field_type} {ss_key}: χ²/dof = {chi2:.1f}/{ndof} = {chi2_red:.2f}")
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$R_v \; [h^{-1}\mathrm{Mpc}]$')
        ax.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v \; [h^{3}\mathrm{Mpc}^{-3}]$')
        ax.set_xlim(2.5, 20.0)
        ax.set_ylim(1e-6, 5e-3)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, which='both', ls='--', alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, 'vsf_theory_vs_data_validation.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nValidation plot saved to: {out_path}")
    plt.close()
    
    # ─── Figure 2: Residual plot ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, field_type, title in zip(
        axes, ['NREC00', 'NREC50'],
        ['NREC00 Residuals', 'NREC50 Residuals']
    ):
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.axhline(0, color='k', ls='--', lw=1)
        
        for ss_key, ss_info in SUBSAMPLES.items():
            df = load_thesis_vsf(ss_key, field_type=field_type)
            if df is None or df.empty:
                continue
            
            theory_at_data = np.interp(df['r'].values, R_theory, vsf_theory)
            residual = (df['vsf'].values - theory_at_data) / df['err'].values
            
            ax.plot(df['r'], residual, 'o-', color=ss_info['color'],
                    ms=5, label=ss_info['label'])
        
        ax.set_xlabel(r'$R_v \; [h^{-1}\mathrm{Mpc}]$')
        ax.set_ylabel(r'$(d_i - m_i) / \sigma_i$')
        ax.set_xscale('log')
        ax.set_xlim(2.5, 20.0)
        ax.set_ylim(-10, 10)
        ax.legend(fontsize=10)
        ax.grid(True, which='both', ls='--', alpha=0.3)
    
    plt.tight_layout()
    out_path2 = os.path.join(PLOT_DIR, 'vsf_residuals_validation.png')
    fig.savefig(out_path2, dpi=200, bbox_inches='tight')
    print(f"Residual plot saved to: {out_path2}")
    plt.close()


if __name__ == "__main__":
    main()
