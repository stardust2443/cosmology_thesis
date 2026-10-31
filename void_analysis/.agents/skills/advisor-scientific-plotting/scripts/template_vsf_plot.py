#!/usr/bin/env python3
"""
template_vsf_plot.py — Advisor-Style Publication VSF Plot Template

Implements:
1. Discrete data points with error bars (NO connecting lines).
2. Continuous smooth theory curve over dense grid.
3. Lower residual pull panel (d - m)/sigma with 1-sigma and 2-sigma shaded bands.
4. Asymmetric log error clipping to prevent 10^-24 auto-scaling.
5. Legend hygiene (excluding internal _child handles).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# Apply standard rcParams
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.titlesize': 12.5,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.1,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

def make_advisor_vsf_plot(R_data, vsf_data, err_data,
                          R_dense, vsf_theory_dense,
                          *,
                          sample_label="BitVF (TOUCH)",
                          data_color="#2980b9",
                          theory_color="#1a1a1a",
                          r_min=7.0,
                          out_path="advisor_vsf_template.png"):
    """
    R_data, vsf_data, err_data: measured discrete points
    R_dense, vsf_theory_dense: smooth theoretical evaluation (N >= 200)
    """
    fig, (ax_main, ax_pull) = plt.subplots(
        2, 1, figsize=(7.5, 6.8),
        sharex=True,
        gridspec_kw={'height_ratios': [3.2, 1.2], 'hspace': 0.05}
    )

    # 1. Exclusion zone (R < r_min)
    for ax in (ax_main, ax_pull):
        ax.axvspan(0, r_min, color='#f0f0f0', zorder=0)
        ax.axvline(r_min, color='#aaaaaa', linestyle='--', linewidth=1.0, zorder=1)

    # 2. Main Panel: Smooth Theoretical Model (Continuous line)
    ax_main.plot(
        R_dense, vsf_theory_dense,
        color=theory_color, linewidth=1.8, label="Vdn Model (Fiducial)",
        zorder=3
    )

    # 3. Main Panel: Discrete Data (Error bars ONLY, NO line)
    # Clip lower error to prevent negative values in log space
    yerr_low = np.minimum(err_data, vsf_data * 0.95)
    yerr_high = err_data
    ax_main.errorbar(
        R_data, vsf_data,
        yerr=[yerr_low, yerr_high],
        fmt='o', color=data_color, ecolor=data_color,
        markersize=6, capsize=2.5, elinewidth=1.2,
        label=sample_label,
        zorder=4
    )

    # Scales and formatting
    ax_main.set_yscale('log')
    ax_main.set_ylabel(r"$\mathrm{d}n / \mathrm{d}\ln R \quad [h^3\,\mathrm{Mpc}^{-3}]$")
    ax_main.set_ylim(2e-9, 2e-5)
    ax_main.grid(True, which='both', linestyle=':', alpha=0.4)

    # Legend hygiene: filter out internal handles
    handles, labels = ax_main.get_legend_handles_labels()
    clean = [(h, l) for h, l in zip(handles, labels) if not l.startswith('_')]
    if clean:
        ax_main.legend(*zip(*clean), frameon=True, framealpha=0.9, loc='upper right')

    # 4. Pull Panel: Calculate pulls at discrete data points
    # Interpolate dense theory to exact discrete data radii
    theory_at_data = np.interp(R_data, R_dense, vsf_theory_dense)
    pull = (vsf_data - theory_at_data) / err_data

    # Reference bands
    ax_pull.axhline(0, color='#1a1a1a', linewidth=1.0, zorder=2)
    ax_pull.axhspan(-1, 1, color='#27ae60', alpha=0.15, zorder=1)
    ax_pull.axhspan(-2, 2, color='#27ae60', alpha=0.07, zorder=0)
    ax_pull.axhline(1, color='#27ae60', linestyle=':', linewidth=0.8, alpha=0.7)
    ax_pull.axhline(-1, color='#27ae60', linestyle=':', linewidth=0.8, alpha=0.7)
    ax_pull.axhline(2, color='#27ae60', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_pull.axhline(-2, color='#27ae60', linestyle='--', linewidth=0.8, alpha=0.5)

    # Discrete pull markers with unit error bars
    ax_pull.errorbar(
        R_data, pull,
        yerr=np.ones_like(pull),
        fmt='o', color=data_color, ecolor=data_color,
        markersize=5, capsize=2, elinewidth=1.0,
        zorder=4
    )

    ax_pull.set_ylabel(r"$(d - m) / \sigma_{\mathrm{JK}}$")
    ax_pull.set_xlabel(r"$R \quad [h^{-1}\mathrm{Mpc}]$")
    ax_pull.set_ylim(-3.5, 3.5)
    ax_pull.set_xlim(np.min(R_dense), np.max(R_dense))
    ax_pull.grid(True, linestyle=':', alpha=0.4)

    # Set explicit layout margins to avoid subplots_adjust / tight_layout warnings
    fig.subplots_adjust(top=0.96, bottom=0.10, left=0.15, right=0.95, hspace=0.06)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {out_path}")

if __name__ == '__main__':
    # Demo synthesis: avoid r < 6 to eliminate negative power warnings
    R_d = np.linspace(8.0, 32.0, 14)
    theory_fn = lambda r: 1.5e-5 * np.exp(-np.maximum(0.0, (r - 6.0) / 7.0)**1.8)
    err_synthetic = theory_fn(R_d) * 0.12
    data_synthetic = theory_fn(R_d) + np.random.normal(0, err_synthetic * 0.7)

    R_fine = np.linspace(6.0, 35.0, 250)
    theory_fine = theory_fn(R_fine)

    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/advisor_vsf_demo.png"
    make_advisor_vsf_plot(
        R_d, data_synthetic, err_synthetic,
        R_fine, theory_fine,
        out_path=out
    )
