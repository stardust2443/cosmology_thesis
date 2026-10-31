"""
_vsf_plot_utils.py

Shared helpers for VSF diagnostic plots (ratio sweep, overlap variants,
v4-vs-Simone). One place for plot style, best-fit fitting, and the
"scatter + smooth curve" rendering used across the three diagnostic
figures.

Public surface:
    UNIFIED_RCPARAMS        — call plt.rcParams.update(UNIFIED_RCPARAMS) once per script
    SUBSAMPLE_COLORS, MPS   — shared palette for the four subsamples
    R_MIN                   — fit-range cutoff used everywhere (= 7.0 h⁻¹Mpc)
    XLIM, YLIM              — unified axis limits
    ZONE_COLOR, RMIN_LINE   — exclusion-zone styling
    fit_loglog_poly         — fit log10(vsf) ~ poly(log10(R)); returns eval callable
    plot_scatter_with_fit   — scatter markers + smooth best-fit curve
    draw_exclusion_zone     — consistent grey R<R_MIN band + vertical line
"""
import numpy as np
import matplotlib.pyplot as plt

# ─── Shared style ────────────────────────────────────────────────────
UNIFIED_RCPARAMS = {
    'font.size': 12,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.titlesize': 12.5,
    'axes.titleweight': 'bold',
}

R_MIN = 7.0
XLIM = (2.5, 20.0)
YLIM = (5e-7, 4e-3)

ZONE_COLOR = '#f0f0f0'
RMIN_LINE_COLOR = '#aaa'

SUBSAMPLE_COLORS = {
    '20p':  '#8e44ad',
    '10p':  '#2980b9',
    '5p':   '#27ae60',
    '2_5p': '#d4ac0d',
}
MPS = {'20p': 1.16, '10p': 1.47, '5p': 1.85, '2_5p': 2.33}

# Reference / theory styling
SIMONE_COLOR = '#1a1a1a'           # near-black
VDN_TRUTH_COLOR = '#1a1a1a'
SIMONE_MARKER = 'o'
SIMONE_MS = 8

# Variant marker palette (used for any non-Simone, non-truth series)
VARIANT_MARKERS = ['s', '^', 'v', 'D', 'P', 'X', '*']
VARIANT_COLORS  = ['#2980b9', '#e67e22', '#27ae60', '#c0392b', '#8e44ad', '#16a085']


# ─── Fitting ─────────────────────────────────────────────────────────
def fit_loglog_poly(R, vsf, *, degree=2, fit_mask=None, min_points=4):
    """
    Fit log10(vsf) = polynomial(log10(R)) of given degree.

    Parameters
    ----------
    R : array_like
        Bin centres (positive).
    vsf : array_like
        Number density per ln R (positive entries only contribute).
    degree : int
        Polynomial degree in log(R). Default 3.
    fit_mask : boolean array_like or None
        Optional boolean mask selecting bins to fit. If None, uses all
        bins where vsf > 0.
    min_points : int
        If fewer than this many valid bins, returns None.

    Returns
    -------
    eval_fn : callable or None
        eval_fn(R_grid) → vsf prediction at R_grid (linear space).
        Returns None if there aren't enough valid bins.
    """
    R = np.asarray(R, dtype=float)
    vsf = np.asarray(vsf, dtype=float)
    mask = (vsf > 0)
    if fit_mask is not None:
        mask = mask & np.asarray(fit_mask, dtype=bool)
    if mask.sum() < min_points:
        return None
    deg = min(degree, mask.sum() - 1)
    coeffs = np.polyfit(np.log10(R[mask]), np.log10(vsf[mask]), deg)
    def eval_fn(R_eval):
        return 10 ** np.polyval(coeffs, np.log10(np.asarray(R_eval, dtype=float)))
    return eval_fn


# ─── Plotting primitives ─────────────────────────────────────────────
def draw_exclusion_zone(ax, *, rmin=R_MIN, xlim_lo=None):
    """Grey R<rmin band + vertical separator line. Returns the ax."""
    lo = xlim_lo if xlim_lo is not None else XLIM[0]
    ax.axvspan(lo, rmin, color=ZONE_COLOR, zorder=0)
    ax.axvline(rmin, color=RMIN_LINE_COLOR, ls='-', lw=1.2, zorder=1)
    return ax


def plot_scatter_with_fit(ax, R, vsf, err, *,
                          fit_fn=None, fit_R_grid=None,
                          color, marker, label, ms=7,
                          scatter_alpha=1.0, fit_alpha=0.9,
                          fit_ls='-', fit_lw=1.8, fit_zorder=4,
                          scatter_zorder=5):
    """
    Render one VSF series as scatter (markers + errorbars, no connecting
    line) overlaid with a smooth best-fit curve.

    Returns the matplotlib artists in a dict for further tweaking.
    """
    R   = np.asarray(R, dtype=float)
    vsf = np.asarray(vsf, dtype=float)
    err = np.asarray(err, dtype=float)
    ok  = vsf > 0

    # Scatter
    ebar = ax.errorbar(
        R[ok], vsf[ok], yerr=err[ok],
        fmt=marker, ls='none',      # ← key: no connecting line
        ms=ms, color=color, ecolor=color,
        elinewidth=1.2, capsize=2.5,
        mec='k', mew=0.4,
        alpha=scatter_alpha, zorder=scatter_zorder,
        label=label,
    )

    # Best-fit. Default grid starts at R_MIN so any drawn fit naturally
    # clips to the MCMC fit range, with no extrapolation into the
    # exclusion zone. Callers can override via `fit_R_grid`.
    line = None
    if fit_fn is not None:
        grid = fit_R_grid if fit_R_grid is not None else \
               np.logspace(np.log10(R_MIN), np.log10(XLIM[1]), 200)
        line = ax.plot(
            grid, fit_fn(grid),
            ls=fit_ls, lw=fit_lw, color=color,
            alpha=fit_alpha, zorder=fit_zorder,
        )[0]

    return dict(errorbar=ebar, fit_line=line)


def apply_axis_style(ax, *, title=None, xlabel=True, ylabel=True,
                     xlim=XLIM, ylim=YLIM, legend_loc='lower left'):
    """Apply the unified log-log axis style."""
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
    if ylabel:
        ax.set_ylabel(r'$dn/d\ln R_v$  [$h^3$Mpc$^{-3}$]')
    if title:
        ax.set_title(title, pad=10)
    ax.grid(True, ls='--', alpha=0.25, which='both')
    if legend_loc is not None:
        ax.legend(loc=legend_loc, framealpha=0.95, edgecolor='#bbb')
    return ax
