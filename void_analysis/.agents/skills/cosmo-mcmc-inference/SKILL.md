---
name: cosmo-mcmc-inference
description: >-
  Use this skill when running, diagnosing, or plotting Markov Chain Monte Carlo (MCMC) parameter
  inference (using emcee, getdist, corner), checking autocorrelation times, evaluating chain convergence,
  detecting prior-wall pinning, or generating posterior contours for cosmological parameters (Omega_m, sigma_8, b_v).
---

# Cosmological MCMC Parameter Inference Runbook

This skill covers the setup, convergence auditing, and posterior analysis for cosmological parameter recovery using `emcee`, `getdist`, and `corner` within the void analysis pipeline.

---

## 1. Environment & Setup

Always run MCMC workflows using `/opt/homebrew/bin/python3.12`:

```bash
export CBL_HOME=/Users/arun/projects/CosmoBolognaLib
export PYTHONPATH=$CBL_HOME/Python:$PYTHONPATH
export DYLD_LIBRARY_PATH=$CBL_HOME:$DYLD_LIBRARY_PATH

/opt/homebrew/bin/python3.12 <mcmc_script>.py
```

Common project data paths:
- Chains storage: `data/mcmc_chains_halo/` (BitVF) and `data/mcmc_chains_revolver/` (REVOLVER)
- Diagnostic plots: `plots/mcmc_local/` and `plots/halos/`

---

## 2. Convergence Standards & Auditing Rules

Before claiming a converged posterior, you **MUST** verify the following four criteria:

### A. Autocorrelation Time ($\tau$)
- Using `emcee`:
  ```python
  tau = sampler.get_autocorr_time(quiet=True)
  ```
- **Rule of Thumb**: The total chain length per walker $N_{\mathrm{steps}}$ must satisfy:
  $$N_{\mathrm{steps}} \ge 50 \times \max(\tau)$$
- If the sampler raises `AutocorrError` or returns `NaN`, the chain has **NOT** converged. Do not publish or use these chains for parameter constraints.

### B. Burn-in Removal
- Discard the first $2\text{--}5 \times \tau$ steps from each walker before computing posterior statistics:
  ```python
  burn_in = int(3 * np.max(tau))
  flat_samples = sampler.get_chain(discard=burn_in, thin=int(np.max(tau) / 2), flat=True)
  ```

### C. Gelman-Rubin Statistic ($\hat{R}$)
- For split chains, compute the potential scale reduction factor:
  $$\hat{R} < 1.05$$
  Values $\hat{R} \ge 1.05$ indicate that different walkers have explored disconnected modes or have not fully mixed.

### D. Prior-Wall Boundary Detection
- **Check**: Inspect 1D histograms of the posteriors. If the peak or 68% credible interval borders the hard prior edge (e.g., $\Omega_m = 0.50$ or $\sigma_8 = 1.20$):
  - **Do NOT** simply widen the prior without diagnosing the physics.
  - In this project, chains pinned to boundaries (e.g. REVOLVER with Vdn) indicate **model-finder incompatibility** (the theoretical model cannot describe the data, forcing the optimizer to runaway limits).

---

## 3. Reporting & Visualization Standards

1. **Posterior Quantiles**:
   Report the median with asymmetric $1\sigma$ (68.3%) uncertainties:
   $$\theta = \theta_{\mathrm{median}} {}_{-\Delta\theta_{-}}^{+\Delta\theta_{+}}$$
   where $\Delta\theta_{-} = \theta_{50} - \theta_{15.87}$ and $\Delta\theta_{+} = \theta_{84.13} - \theta_{50}$.
2. **True Value Benchmarks**:
   Always overlay the Aletheia simulation fiducial truth on 1D/2D posterior contours:
   - $\Omega_m = 0.315$ (dashed vertical/horizontal line)
   - $\sigma_8 = 0.811$ (dashed vertical/horizontal line)
3. **GetDist / Corner Plotting**:
   - Use `getdist.plots.get_subplot_plotter()` or `corner.corner()`.
   - Set 2D contour levels to $68\%$ and $95\%$ credible regions (`levels=[0.683, 0.954]`).
   - Use distinct colors: Blue for BitVF (`#2980b9`), Crimson for REVOLVER (`#c0392b`).

See [convergence_checklist.md](./references/convergence_checklist.md) for a pre-flight and post-run checklist.
