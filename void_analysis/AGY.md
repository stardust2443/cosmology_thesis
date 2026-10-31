# AGY.md — Antigravity Agent Guidelines for Void Analysis

## Project Mission
PhD research on cosmic void statistics and parameter inference from Void Size Functions (VSF) using **BitVF** and **REVOLVER** on the **Aletheia** simulation suite with **CosmoBolognaLib (CBL)**. Supervised by Prof. Federico Marulli.

## Live Context & Memory Files
- `LLM_suggestions/session_handoff.md`: **PRIMARY LIVE STATE**. Consult this at the start of every session for recent diagnostics, numerical values, and to-do lists. Keep it updated after major runs.
- `CLAUDE.md`: General repository conventions and history across all phases.
- `main_thesis_docs/main.tex`: The master thesis LaTeX document.

## Execution Environment
- **Python**: Always invoke `/opt/homebrew/bin/python3.12` (has numpy, scipy, pandas, emcee, matplotlib, corner, getdist, and CBL bindings).
- **CBL Libraries**:
  ```bash
  export CBL_HOME=/Users/arun/projects/CosmoBolognaLib
  export PYTHONPATH=$CBL_HOME/Python:$PYTHONPATH
  export DYLD_LIBRARY_PATH=$CBL_HOME:$DYLD_LIBRARY_PATH
  ```

## Physics & Modeling Conventions
1. **Model-Finder Compatibility**:
   - Vdn (Jennings et al.) is an excursion-set spherical evolution model.
   - **BitVF** matches Vdn physics ($\chi^2/\nu \approx 0.90\text{--}1.67$).
   - **REVOLVER** (unmerged watershed basins) has an intrinsic shape mismatch ($\chi^2/\nu \approx 3.14\text{--}8.12$). Apparent MCMC bias ($\Omega_m \sim 0.40\text{--}0.45$) is a mathematical artifact of model mismatch.
2. **Volume Conservation**:
   - Vdn is strictly volume-conserving $\implies$ Voids must be cleaned as isolated non-overlapping spheres (`TOUCH`). Overlapping spheres (`CENTER`) double-count volume and blow up $\chi^2/\nu \sim 25$.
3. **Linear Barrier Universality**:
   - $b_{\mathrm{void}} = B_{\mathrm{slope}} b_{\mathrm{eff}} + B_{\mathrm{offset}}$
   - BitVF: $0.914 \pm 0.043$; REVOLVER: $0.910 \pm 0.046$; difference $0.07\sigma$.
   - Scale-invariant above $R_{\mathrm{min}} \ge 16\,h^{-1}\mathrm{Mpc}$ ($< 1\%$ drift).
4. **Advisor Plotting Rules**:
   - Discrete VSF data: points with error bars only (**no connecting lines**).
   - Smooth Vdn model: continuous lines over dense grid.
   - Always include residual pull subpanels: $(d - m)/\sigma_{\mathrm{JK}}$ with $\pm 1\sigma$ and $\pm 2\sigma$ bands.
   - On log axes, prevent non-positive lower error bounds from forcing $y$-axis down to $10^{-24}$. Set `ylim(2e-9, 2e-5)`.
   - Never pass unlabeled lines into `legend()`; avoid matplotlib internal labels like `_child0`.
