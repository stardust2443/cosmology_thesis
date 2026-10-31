# CLAUDE.md — BitVF + Aletheia Thesis & Cosmology Project

Short orientation for any AI agent session opened in this repo. Read this first.

## What this project is

PhD thesis research by Arun Mitra (supervised by Prof. Federico Marulli): validate the **BitVF** back-in-time void finder (Sartori 2025) on the **Aletheia** N-body simulation ($\Omega_m = 0.315$, $\sigma_8 = 0.811$, $L = 1000\,h^{-1}\mathrm{Mpc}$ box, $z = 0$), benchmark against **REVOLVER** (ZOBOV Voronoi watershed), test the universality of the linear void barrier ($B_{\mathrm{slope}} \sim 0.9$), evaluate theoretical model compatibility, and perform cosmological parameter recovery via MCMC.

### The Three Phases:

| Phase | Goal | Status |
|---|---|---|
| **1 — Catalog Construction** | Cleaned BitVF VSF matches ForArun reference | **Complete** (validated within 1%–7% for TOUCH/OV00 and CENTER/OV05) |
| **2 — DM Cosmological Inference** | $(\Omega_m, \sigma_8)$ posteriors via CBL Vdn + MCMC | **Complete & Validated** (recovers truth within $1\sigma$ with informative priors) |
| **3 — Halo Voids & Cross-Finder Benchmark** | BitVF vs. REVOLVER across 4 halo catalogues (100P, 75P, 50P, 30P), barrier universality ($B_{\mathrm{slope}} \approx 0.91$), model compatibility | **Complete & Validated** ($\Delta B_{\mathrm{slope}} = 0.07\sigma$; $\sigma_8 = 0.8208 \pm 0.009$; model-finder compatibility proven) |

---

## Where Things Live

- **Code**: `void_analysis/scripts/`
  - `plot_calibration_vsf_residuals_grid.py`: 4-sample comparative grid of VSF fits and standardized pulls.
  - `audit_bitvf_model_robustness.py`: 5-part BitVF model robustness audit.
  - `compare_barrier_universality.py`: Linear barrier regression comparing BitVF vs. REVOLVER.
  - `plot_mcmc_finder_comparison.py`: 2D $(\Omega_m, \sigma_8)$ posterior contours.
- **Data**: `void_analysis/data/`
  - `voids_halo/`: BitVF halo catalogues and 64-region spatial Jackknife covariance matrices.
  - `voids_halo_revolver/`: REVOLVER halo catalogues and covariance matrices.
  - `mcmc_chains_halo/`: BitVF MCMC chains with 68% calibrated Gaussian priors.
  - `mcmc_chains_revolver/`: REVOLVER MCMC chains.
- **Plots**: `void_analysis/plots/halos/` (and mirrored in `void_analysis/main_thesis_docs/figures/`)
- **Documentation & Context**:
  - `void_analysis/LLM_suggestions/session_handoff.md`: **Live state of the project, latest diagnostics, and to-do list.** Read this at the start of every session!
  - `void_analysis/main_thesis_docs/main.tex`: Current complete thesis LaTeX source.
- **CosmoBolognaLib**: `/Users/arun/projects/CosmoBolognaLib/` (Python bindings + dylibs).

---

## Critical Physics & Methodological Rules

1. **Python Environment**:
   Always execute Python scripts using Homebrew Python 3.12:
   `/opt/homebrew/bin/python3.12 <script.py>`
2. **Model-Finder Compatibility**:
   Void finders define objects algorithmically. The Vdn model (Jennings et al.) is an idealized spherical excursion-set model:
   - **BitVF** (dynamical displacement divergence) is structurally compatible with Vdn ($\chi^2/\nu \approx 0.90\text{--}1.67$).
   - **REVOLVER** (unmerged watershed basins) has a severe shape mismatch with Vdn ($\chi^2/\nu \approx 3.14\text{--}8.12$). Its biased MCMC posteriors ($\Omega_m \sim 0.40\text{--}0.45$) are an artifact of forcing an incompatible model to fit watershed basins.
3. **Volume Conservation Prerequisite (`TOUCH` vs `CENTER`)**:
   Jennings et al. Vdn is strictly volume-conserving. Voids must be cleaned as non-overlapping isolated spheres (`TOUCH`). Overlapping spheres (`CENTER`) double-count volume and blow up $\chi^2/\nu$ to $\sim 25$.
4. **Plotting Standards for Academic Advisors**:
   - Discrete measured VSF data points must be plotted as points with error bars (**never with connecting lines**).
   - Smooth theoretical model curves must be plotted as continuous lines over a dense radius grid.
   - Always include lower residual subpanels showing standardized pulls $(d - m)/\sigma_{\mathrm{JK}}$ with $\pm 1\sigma$ and $\pm 2\sigma$ reference bands.
   - On log-scale VSF axes, clip non-positive lower error bounds (`yerr_low = np.minimum(err, y * 0.95)`) and explicitly set `ylim(2e-9, 2e-5)` to avoid matplotlib auto-scaling down to $10^{-24}$.
5. **CosmoBolognaLib C++ Bias Override**:
   If effective bias $b_{\mathrm{eff}} == 1.0$, CBL silently overrides user bias parameters. Always pass `b_eff = 1.0001` or the measured halo bias (e.g., $1.0211, 0.9583, \dots$).
6. **Degrees of Freedom ($\nu$)**:
   - 1D barrier calibration ($b_{\mathrm{void}}$ fitted at fiducial cosmology): $\nu = p - 1$.
   - 2D barrier calibration ($B_{\mathrm{slope}}, B_{\mathrm{offset}}$): $\nu = p - 2$.
