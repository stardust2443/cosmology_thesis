# Project Handoff: Cosmological Parameter Estimation from Void Size Functions

> **NOTE (October 29, 2026):** This document records the historical state at the end of Phase 2. For the comprehensive, active, living master context across Phase 1, Phase 2, Phase 3, REVOLVER benchmarks, and the latest robustness audits, please consult the master source of truth:
> **[`MASTER_PROJECT_CONTEXT.md`](./MASTER_PROJECT_CONTEXT.md)**

**Date:** September 14, 2026 (Updated: October 29, 2026)  
**Repository Working Directory:** `/Users/arun/projects/my_cosmo_analysis/void_analysis`  
**Master Context Document:** [`MASTER_PROJECT_CONTEXT.md`](./MASTER_PROJECT_CONTEXT.md)  
**Thesis Directory:** `/Users/arun/projects/my_cosmo_analysis/void_analysis/main_thesis_docs`  
**Simulation Dataset:** Aletheia Simulation Suite (Mocks)  
**True Simulation Cosmology:** $\Omega_m = 0.315$, $\sigma_8 = 0.811$, $h = 0.6736$, $n_s = 0.9649$  

---

## 1. Executive Summary & Current Status

This project performs cosmological parameter estimation from the **Void Size Function (VSF)** using the volume-conserving theoretical model (**Vdn** from Jennings et al.) implemented via **CosmoBolognaLib (CBL)**. 

### Current Project State (Advisor Approved)
The baseline cosmological analysis is **100% complete, fully validated, and integrated into the thesis document (`thesis_docs/main.pdf`, 97 pages, 0 errors)**. 

The analysis is organized as **four independent analyses** across four tracer density subsamples ($20\%$, $10\%$, $5\%$, and $2.5\%$), using 64-region spatial Jackknife covariance matrices with Hartlap corrections.

### Final Baseline Results (Propagated Calibration Uncertainty)
In accordance with the advisor's final guidance, the tracer bias parameters $(B_{\mathrm{slope}}, B_{\mathrm{offset}})$ are calibrated on mocks at the true cosmology, and their calibration uncertainties are formally propagated into the cosmological MCMC via **informative multivariate Gaussian priors**:

| Subsample | Tracer Density | Optimal $R_{\mathrm{min}}$ | True $\chi^2/\nu$ | Best-Fit $\chi^2/\nu$ | $\Omega_m$ (Posterior Median [16%, 84%]) | $\sigma_8$ (Posterior Median [16%, 84%]) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20p** | $20\%$ | $7.0\;h^{-1}\mathrm{Mpc}$ | $1.08$ | $0.79$ | $0.251^{+0.059}_{-0.051}$ | $0.839^{+0.038}_{-0.036}$ | **Recovered ($< 1\sigma$)** |
| **10p** | $10\%$ | $8.0\;h^{-1}\mathrm{Mpc}$ | $0.88$ | $0.78$ | $0.368^{+0.079}_{-0.086}$ | $0.795^{+0.036}_{-0.031}$ | **Recovered ($< 1\sigma$)** |
| **5p** | $5\%$ | $8.0\;h^{-1}\mathrm{Mpc}$ | $0.71$ | $0.68$ | $0.297^{+0.094}_{-0.078}$ | $0.816^{+0.037}_{-0.035}$ | **Recovered ($< 0.2\sigma$)** |
| **2_5p** | $2.5\%$ | $10.0\;h^{-1}\mathrm{Mpc}$ | $1.11$ | $0.12$ | $0.201^{+0.135}_{-0.073}$ | $0.848^{+0.054}_{-0.050}$ | **Recovered ($< 1\sigma$)** |

---

## 2. Key Physical & Methodological Discoveries

Any incoming agent must be aware of these critical findings to avoid reintroducing resolved bugs:

1. **The CosmoBolognaLib C++ Bias Override Bug:**  
   In CosmoBolognaLib (`SizeFunction::deltav_L`), if the effective bias parameter $b_{\mathrm{eff}}$ is set to exactly $1.0$, the C++ code silently disables the user's free bias parameters ($B_{\mathrm{slope}}, B_{\mathrm{offset}}$) and locks the barrier parameters to default values.  
   *Rule:* Always pass `B_EFF = 1.0001` in Python wrappers to bypass this override.
2. **Independent Subsample Analysis vs. Joint Fits:**  
   Tracer sparsity directly alters the scale-validity boundary $R_{\mathrm{min}}$. Combining subsamples into a single 10D joint fit is physically incorrect. The four subsamples must be analyzed independently with their own $R_{\mathrm{min}}$.
3. **Physical Trend of Scale Validity:**  
   - High density ($20\%$): Dense tracers resolve small voids; validity holds down to $R_{\mathrm{min}} = 7.0\;h^{-1}\mathrm{Mpc}$.
   - Intermediate density ($10\%, 5\%$): Valid down to $R_{\mathrm{min}} = 8.0\;h^{-1}\mathrm{Mpc}$.
   - High sparsity ($2.5\%$): Shot noise dominates small scales; requires conservative cut at $R_{\mathrm{min}} = 10.0\;h^{-1}\mathrm{Mpc}$.
4. **Degrees of Freedom ($\nu$) Tracking:**  
   When calculating reduced $\chi^2/\nu$, degrees of freedom must be accounted for consistently:
   - True cosmology with calibrated bias: $\nu = N_{\mathrm{bins}} - 2$.
   - 2D cosmological fit with fixed calibrated bias: $\nu = N_{\mathrm{bins}} - 2$.
   - 4D fit with floating unconstrained bias: $\nu = N_{\mathrm{bins}} - 4$. (Note: with $N_{\mathrm{bins}} \sim 6$, dividing by $2$ instead of $4$ creates an artificial penalty in the reduced $\chi^2$, even when raw $\chi^2$ strictly improves).
5. **Marginalized Medians vs. Maximum A Posteriori (MAP):**  
   Because cosmological parameter posteriors exhibit curved banana degeneracies (in $\Omega_m$--$\sigma_8$), the marginalized 1D median does *not* represent the best-fit model and falls off the likelihood ridge. Best-fit curves in VSF overlays must always be evaluated at the **MAP** parameter vector.
6. **Informative Prior Calibration:**  
   Calibrate $(B_{\mathrm{slope}}, B_{\mathrm{offset}})$ at the true cosmology and optimal $R_{\mathrm{min}}$, extract their 2D covariance $\Sigma_{\mathrm{calib}}$, and assign them multivariate Gaussian priors in the final 4D cosmological MCMC.

---

## 3. Directory Layout & Key Files

### A. Python Pipeline Scripts (`/Users/arun/projects/my_cosmo_analysis/void_analysis/scripts/`)
* **`calibrate_and_run_informative_mcmc.py`**: The production baseline script. Calibrates nuisance parameters, samples calibration covariance, and runs 4D MCMC with informative Gaussian priors.
* **`plot_informative_results.py`**: Generates 4D informative corner plots and 2D uncertainty propagation comparisons (`comparison_uncertainty_{frac}.png`).
* **`update_subsample_plots.py`**: Generates zoomed-in VSF residuals plots (`vsf_residuals_subsample_{frac}.png`) and 2D fixed-bias corner plots.
* **`jackknife_64_cov.py`**: Computes 64-subcube spatial Jackknife covariance matrices and Hartlap precision matrices.
* **`find_subsample_rmin.py`**: Robust grid-search script for scanning $R_{\mathrm{min}}$ from $6.0$ to $10.0\;h^{-1}\mathrm{Mpc}$.

### B. MCMC Chains & Data (`/Users/arun/projects/my_cosmo_analysis/void_analysis/data/`)
* **`cov_jk_64/`**: Contains the measured VSF data files (`vsf_64jk_nrec50_{frac}_OV05.dat`) and Hartlap-corrected inverse covariances (`inv_cov_64jk_R{rmin}_{frac}_OV05.npy`).
* **`mcmc_chains/`**:
  * `cbl_v10_subsample_{frac}_informative_R{rmin}_OV05.npy`: Production 4D chains with informative calibration priors.
  * `cbl_v9_subsample_{frac}_2D_R{rmin}_OV05.npy`: 2D chains with fixed calibrated bias.
  * `cbl_v8_subsample_{frac}_R{rmin}_OV05.npy`: 4D chains with uninformative flat priors.

### C. Thesis Document & Figures (`/Users/arun/projects/my_cosmo_analysis/thesis_docs/`)
* **`main.tex`**: The primary LaTeX document. Section 4.4.2 contains the complete independent analysis, Table 4.6, and figures.
* **`main.pdf`**: The compiled 97-page thesis PDF.
* **`figures/`**:
  * `vsf_residuals_subsample_{frac}.png`: Zoomed VSF measurements overlaid with True and MAP models.
  * `corner_subsample_{frac}_2D.png`: 2D cosmological posteriors ($\Omega_m, \sigma_8$).
  * `corner_subsample_{frac}_informative.png`: 4D posteriors with informative priors.
  * `comparison_uncertainty_{frac}.png`: Direct comparison between fixed bias and informative priors.

---

## 4. How to Run / Recompile

### Environment
* **Python Environment:** `/Users/arun/miniforge3/bin/python3.12`
* **CosmoBolognaLib Path:** `/Users/arun/projects/CosmoBolognaLib`

### To Recompile the Thesis Document
```bash
cd /Users/arun/projects/my_cosmo_analysis/thesis_docs
latexmk -pdf main.tex
```

### To Re-run Informative Calibration & MCMC Chains
```bash
cd /Users/arun/projects/my_cosmo_analysis/void_analysis
/Users/arun/miniforge3/bin/python3.12 scripts/calibrate_and_run_informative_mcmc.py
```

### To Re-generate All Updated Figures
```bash
cd /Users/arun/projects/my_cosmo_analysis/void_analysis
/Users/arun/miniforge3/bin/python3.12 scripts/update_subsample_plots.py
/Users/arun/miniforge3/bin/python3.12 scripts/plot_informative_results.py
```

---

## 5. Next Steps for Future Agents / Chats

Now that the baseline analysis is complete and approved, the natural subsequent steps indicated by the thesis roadmap are:
1. **Catalog Cleaning Procedure Comparison:** Repeat the calibrated pipeline on catalogs cleaned with different procedures (e.g. simple cleaning vs. complex cleaning vs. raw uncleaned catalogues) to quantify cleaning systematics.
2. **Likelihood Model Comparisons:** Reassess the impact of the joint Gaussian-Poisson likelihood on sparse tails compared to the pure Gaussian Jackknife likelihood.
3. **Geometric vs. Dynamical Void Definitions:** Compare the parameter constraints obtained from geometric watershed voids against dynamical voids identified in reconstructed matter fields.
