---
name: advisor-scientific-plotting
description: >-
  Use this skill when creating or modifying scientific figures, plotting Void Size Functions (VSF),
  generating residual pull panels, formatting multi-sample comparative grids, or styling plots for
  thesis and paper publication according to Prof. Federico Marulli's standards.
---

# Advisor-Style Scientific Plotting Runbook

This skill codifies the exact publication and styling standards required by thesis advisor Prof. Federico Marulli for cosmic void analysis and astrophysical parameter inference.

---

## 1. Golden Rules of Figure Presentation

### Rule A: Discrete Data vs Continuous Theory
- **Measured Data (Simulations / Mock Catalogs)**:
  - Must be plotted as discrete markers with error bars: `ax.errorbar(..., fmt='o', linestyle='none', capsize=2, ...)`
  - **NEVER** connect discrete measured data points with lines. Lines imply a continuous measurement or functional form where none exists.
- **Theoretical Models (e.g., Jennings et al. Vdn)**:
  - Must be plotted as continuous smooth curves over a dense, finely sampled radius grid ($N \ge 200$ points between $R_{\min}$ and $R_{\max}$).
  - Never evaluate theory only at the discrete bin centers and connect with line segments.

### Rule B: Residual Subpanels & Standardized Pulls
Every model-to-data comparison plot must include a lower subpanel showing the standardized residual pull:
$$\text{Pull} = \frac{d_i - m(R_i)}{\sigma_{\mathrm{JK}, i}}$$
- **Zero Line**: Solid black or dark grey horizontal line at $y = 0$.
- **Error Bounds**:
  - Horizontal dashed reference lines at $y = \pm 1$ and $y = \pm 2$.
  - Shaded bands: $\pm 1\sigma$ band (e.g. `alpha=0.15`) and $\pm 2\sigma$ band (e.g. `alpha=0.08`).
- **Data Points in Pull Panel**:
  - Plot normalized pulls as discrete points with unit error bars ($\pm 1$):
    `ax_pull.errorbar(R, pull, yerr=np.ones_like(pull), fmt='o', linestyle='none')`
  - Limit pull y-axis to a symmetric range: typically `ylim(-3.5, 3.5)`.

### Rule C: Log-Scale Asymmetric Error Bar Clipping
On logarithmic $y$-axes (such as $\mathrm{d}n/\mathrm{d}\ln R$):
- If $\text{err}_i \ge y_i$, the lower error bar extends into negative numbers, causing matplotlib to panic and auto-scale the $y$-axis down to $10^{-24}$ or throw log-scale warnings.
- **Mandatory fix**:
  ```python
  yerr_low = np.minimum(err, y * 0.95)
  yerr_high = err
  ax.errorbar(R, y, yerr=[yerr_low, yerr_high], fmt='o', linestyle='none', capsize=2)
  ```
- **Explicit Limits**: Always set explicit $y$-limits matching the physical density domain:
  - For standard void samples: `ax.set_ylim(2e-9, 2e-5)` or check `_vsf_plot_utils.py: YLIM = (5e-7, 4e-3)`.

### Rule D: Legend Cleanliness & No Internal Handles
- Never allow matplotlib internal objects (e.g. `_child0`, `_line0`) to appear in `legend()`.
- Explicitly collect handles and labels, filtering out any starting with `_`:
  ```python
  handles, labels = ax.get_legend_handles_labels()
  clean = [(h, l) for h, l in zip(handles, labels) if not l.startswith('_')]
  if clean:
      ax.legend(*zip(*clean), frameon=True, framealpha=0.9, loc='best')
  ```

---

## 2. Shared Palettes and rcParams

Refer to [style_reference.md](./references/style_reference.md) for full visual configuration:
- Use `UNIFIED_RCPARAMS` from `scripts/_vsf_plot_utils.py`.
- **Halo Subsample Palette**:
  - `20p`: `#8e44ad` (Purple)
  - `10p`: `#2980b9` (Blue)
  - `5p`: `#27ae60` (Green)
  - `2_5p`: `#d4ac0d` (Gold)
- **Cutoff / Exclusion Zone**:
  - For $R < R_{\min}$ (typically $R_{\min} = 7.0$ or $16.0\,h^{-1}\mathrm{Mpc}$): shade region in `#f0f0f0` with a vertical dashed marker at $R_{\min}$.

---

## 3. Reference Implementation

See [template_vsf_plot.py](./scripts/template_vsf_plot.py) for a complete, runnable script implementing these rules.
