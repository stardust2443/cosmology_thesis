# Thesis Update Plan: Comprehensive LaTeX Rewrite

## Context

Arun's thesis on cosmic void size function (VSF) analysis currently covers only Phase 1 (pipeline validation). Phase 2 (MCMC cosmological parameter estimation) is fully complete but absent from the draft. The presentation feedback was: "pipelines are well built, work is nice, but should have calibrated the data" -- the VdN model disagrees with data at true cosmology in pre-MCMC sanity checks. The thesis needs to be expanded into a verbose, content-rich document with ~30+ figures, ~40+ references, and honest discussion of the calibration issue.

**Files to modify:**
- `/Users/arun/Downloads/BITS_F422_ArunMitra_2443/main.tex` (complete rewrite)
- `/Users/arun/Downloads/BITS_F422_ArunMitra_2443/references.bib` (add ~15 new entries)

**Figures to copy into `figures/` directory** (from `/Users/arun/projects/my_cosmo_analysis/void_analysis/plots/`):

| Target name | Source |
|---|---|
| `vsf_comparison_v4.png` | `plots/vsf_comparison_v4.png` |
| `vsf_theory_vs_data.png` | `plots/vsf_theory_vs_data_validation.png` |
| `vsf_residuals.png` | `plots/vsf_residuals_validation.png` |
| `vsf_delta_nl_scan.png` | `plots/vsf_delta_nl_scan.png` |
| `NREC50_validation_4panel.png` | `plots/cleaning_validation/NREC50_validation_4panel.png` |
| `peak_shift_diagnosis.png` | `plots/cleaning_validation/peak_shift_diagnosis_v4.png` |
| `ensemble_hypothesis.png` | `plots/cleaning_validation/thesis_figure_ensemble_hypothesis.png` |
| `jackknife_vs_poisson.png` | `plots/cleaning_validation/jackknife_vs_poisson.png` |
| `variance_octants.png` | `plots/cleaning_validation/variance_octants_NREC50_v4.png` |
| `revolver_crosscheck.png` | `plots/cleaning_validation/revolver_crosscheck_v4.png` |
| `corner_BIT_20p_2D.png` | `presentation_plots/corner_BIT_20p.png` |
| `corner_BIT_20p_2D_rmin6.png` | `presentation_plots/corner_NREC50_20p_2D_local_v4_rmin6.png` |
| `bestfit_BIT_20p_2D.png` | `presentation_plots/bestfit_BIT_20p.png` |
| `bestfit_rmin6.png` | `presentation_plots/bestfit_NREC50_20p_2D_local_v4_rmin6.png` |
| `corner_BIT_20p_3D.png` | `plots/cluster_mcmc/mcmc_plots/corner_BIT_20p_3D.png` |
| `corner_NREC50_20p_3D.png` | `plots/cluster_mcmc/mcmc_plots/corner_NREC50_20p_3D.png` |
| `corner_NREC50_2_5p_3D.png` | `plots/cluster_mcmc/mcmc_plots/corner_NREC50_2_5p_3D.png` |
| `bestfit_NREC50_20p_3D.png` | `plots/cluster_mcmc/mcmc_plots/bestfit_NREC50_20p_3D.png` |
| `bestfit_NREC50_2_5p_3D.png` | `plots/cluster_mcmc/mcmc_plots/bestfit_NREC50_2_5p_3D.png` |
| `traces_NREC50_2_5p_3D.png` | `plots/cluster_mcmc/mcmc_plots/traces_NREC50_2_5p_3D.png` |
| `traces_BIT_20p_3D.png` | `plots/cluster_mcmc/mcmc_plots/traces_BIT_20p_3D.png` |
| `landscape_NREC50.png` | `plots/mcmc_local/landscape_NREC50_20p.png` |
| `bestfit_compare.png` | `plots/mcmc_local/bestfit_compare_NREC50_20p.png` |

---

## Revised Chapter Structure

### Front Matter
- Updated abstract covering both phases + calibration lesson
- Updated acknowledgments (same)

### Chapter 1: Introduction (~8 pages)
- 1.1 Context and Motivation (ΛCDM tensions, voids as probes)
- 1.2 The Void Size Function as a Cosmological Observable
- 1.3 The Algorithmic Challenge: Identifying Voids in Tracer Space
- 1.4 The Role of Post-Processing and Cleaning
- 1.5 From Geometric to Dynamical Void Definitions
- 1.6 **NEW**: Cosmological Parameter Estimation from the VSF (previews Phase 2)
- 1.7 Scope and Objectives (updated to include MCMC + calibration)

### Chapter 2: Theoretical Background and Literature Review (~15 pages)
- 2.1 The Cosmic Web and the Role of Voids
- 2.2 The VSF as a Statistical Observable
- 2.3 **NEW**: The Volume-Conserving (Vdn) Theoretical Model
  - Full SvdW multiplicity derivation with equations
  - Lagrangian-to-Eulerian mapping
  - Volume-conserving correction (Jennings 2013)
  - Physical meaning of thresholds (δv=-2.71, δNL=-0.80, δc=1.686)
  - Power spectrum computation via colossus
- 2.4 Void Identification Methods
- 2.5 Impact of Tracer Bias and Sparsity
- 2.6 Cleaning and Mapping to Theoretical Predictions
- 2.7 Observational Measurements: The BOSS Survey
- 2.8 **NEW**: Bayesian Parameter Estimation for Void Cosmology
  - Affine-invariant ensemble MCMC (emcee)
  - Gaussian likelihood with Poisson covariance
  - The calibration requirement
- 2.9 Toward Dynamical Void Definitions (BitVF, Verza et al. 2024)
- 2.10 Summary and Motivation

### Chapter 3: Methodology and Implementation (~12 pages)
- 3.1 Overview of Analysis Pipeline
- 3.2 Simulation Data: The Aletheia Suite (replace generic → specific: box=250 Mpc/h, Ωm=0.315, σ8=0.811, z=0, DM-only)
- 3.3 Tracer Subsamples and Reconstruction Fields (BIT, NREC50, REVOLVER)
- 3.4 Void Identification: Voronoi Tessellation and Watershed Transform
- 3.5 Void Cleaning Pipeline (TH→OV00→border)
- 3.6 Validation of Void Definition (density profiles)
- 3.7 VSF Computation
- 3.8 **NEW**: Cleaning Validation Framework (ForArun cross-check, jackknife, octant variance, ensemble hypothesis)
- 3.9 **NEW**: MCMC Parameter Estimation Pipeline
  - Vdn forward model (reference vsf_model.py equations)
  - emcee configuration: 32 walkers, priors table
  - Three progressive configurations (2D all-bins, 2D R>6, 3D δv-free)
  - Cluster production: 8 runs (4 subsamples × 2 fields)
  - Pre-MCMC validation step
- 3.10 Sensitivity Analysis Framework
- 3.11 Summary

### Chapter 4: Results (~20 pages, largest chapter)
- 4.1 Impact of Sparsity on the VSF (existing, with figures)
- 4.2 Comparison of Void Finding Methods (existing, REVOLVER vs BitVF)
- 4.3 Universality of the VSF (existing)
- 4.4 **NEW**: Pipeline Validation Results
  - Fig: NREC50_validation_4panel.png — cleaned VSF vs ForArun for all 4 subsamples
  - Fig: revolver_crosscheck.png — BitVF vs REVOLVER independent verification
  - Fig: peak_shift_diagnosis.png — raw catalog amplitude diagnosis
  - Fig: ensemble_hypothesis.png — single-realization vs ensemble explanation
  - Fig: jackknife_vs_poisson.png — error estimation comparison
  - Fig: variance_octants.png — cosmic variance across octants
- 4.5 **NEW**: Three-Panel VSF Comparison (BIT / NREC50 / REVOLVER)
  - Fig: vsf_comparison_v4.png — the definitive comparison figure
- 4.6 **NEW**: Pre-MCMC Model Validation (CRITICAL SECTION)
  - Fig: vsf_theory_vs_data.png — Vdn at true cosmology vs data
  - Fig: vsf_residuals.png — systematic residual analysis
  - Fig: vsf_delta_nl_scan.png — δNL parameter scan
  - Quantify disagreement: residuals table by radius range
  - Honest statement: VdN does NOT match data at known cosmology
- 4.7 **NEW**: MCMC Parameter Estimation Results
  - 4.7.1 Progressive MCMC Configurations
    - Run 1 (2D all bins): corner_BIT_20p_2D.png, bestfit_BIT_20p_2D.png → failure
    - Run 2 (2D R>6): corner_BIT_20p_2D_rmin6.png, bestfit_rmin6.png → biased
    - Run 3 (3D δv free): corner_BIT_20p_3D.png → promising
  - 4.7.2 Production MCMC Results (8 cluster runs)
    - Summary table of all 8 runs with Ωm, σ8, δv estimates
    - Fig: corner_NREC50_20p_3D.png — representative NREC50 posterior
    - Fig: corner_NREC50_2_5p_3D.png — best σ8 recovery
    - Fig: bestfit_NREC50_20p_3D.png — best-fit VSF overlay
    - Fig: bestfit_NREC50_2_5p_3D.png
    - Fig: traces_NREC50_2_5p_3D.png — convergence evidence
    - Key findings: Ωm→prior floor, σ8 recovered, δv validates theory
- 4.8 **NEW**: Diagnostic Analysis
  - Fig: landscape_NREC50.png — χ² landscape
  - Fig: bestfit_compare.png — local vs reference data comparison
  - Root cause: VdN small-R overprediction drives Ωm bias
- 4.9 Summary of Results (expanded)

### Chapter 5: Discussion (~10 pages)
- 5.1 Interpretation of Results (rewrite to cover both phases)
- 5.2 **NEW**: The Calibration Problem (MOST IMPORTANT NEW SECTION)
  - What calibration means: ensuring model matches data at known cosmology before blinding
  - Two complementary calibration strategies that should have been applied:
    (a) Mock-based correction factor for the VdN model (Contarini et al. 2023 approach)
    (b) Cross-checking against alternative theoretical models — moving-barrier (Verza et al. 2024), unified multiplicity (Paranjape et al. 2024), Vdn variants — before committing to a single forward model for MCMC
  - Why neither was done: the basic VdN model was used as-is without comparison to alternatives or mock calibration
  - Root causes of VdN disagreement: non-sphericity, resolution effects, spatial exclusion
  - Reference Contarini et al. (2023) calibration approach
  - The moving-barrier model (Verza et al. 2024) as the natural cross-check we should have done
  - Lesson learned: pipeline + (calibration AND model cross-validation), not pipeline alone
- 5.3 Limitations of Geometric Void Finders
- 5.4 Limitations of the Present Analysis (single box, Poisson covariance, no RSD, no survey mask)
- 5.5 Implications for Future Void Cosmology
- 5.6 Summary

### Chapter 6: Conclusion (~2 pages)
- Expanded to reflect both phases
- Clear statement: Phase 1 successful, Phase 2 reveals model limitation requiring calibration

### Chapter 7: Future Work (~2 pages)
- Calibration strategy (mock-based forward modeling)
- Moving-barrier model (Verza et al. 2024)
- Full covariance matrix (jackknife or mock-based)
- Application to BOSS DR12
- DESI/Euclid void prospects

### Appendices
- A: Dynamical Bias Correction (keep existing)
- B: **NEW**: VdN Model Implementation Details (equations + code structure)
- C: Pipeline Implementation Details (existing Appendix B)
- D: **NEW**: MCMC Run Summary Table (full 8-run results)

---

## New Bibliography Entries (~15 additions)

1. Foreman-Mackey et al. 2013 (emcee)
2. Goodman & Weare 2010 (affine-invariant ensemble samplers)
3. Verza et al. 2024 (moving-barrier model)
4. Diemer 2018 (COLOSSUS toolkit)
5. Hamaus et al. 2020 (precision cosmology with voids in BOSS)
6. Massara et al. 2015 (voids + massive neutrinos)
7. Pisani et al. 2015 (counting voids to probe dark energy, PhysRevD)
8. Press & Schechter 1974 (PS formalism)
9. DESI Collaboration 2024 (first-year BAO)
10. Planck Collaboration 2020 (Planck 2018 cosmological parameters)
11. Springel 2005 (GADGET/N-body simulations)
12. Hahn et al. 2015 or similar (reconstruction techniques)
13. Paz et al. 2013 (void density profiles)
14. Pollina et al. 2017/2019 (void cosmology)
15. Ronconi et al. 2019 (VSF methodology update)

---

## Implementation Order

1. **Copy figures** into `figures/` directory (23 new figures)
2. **Update references.bib** with ~15 new entries
3. **Rewrite main.tex** as a complete document:
   - Front matter (title, abstract, acknowledgments)
   - Chapters 1-7 with all new sections
   - Appendices A-D
   - All figure references with detailed captions
4. **Verification**: compile with pdflatex to check for errors

---

## Verification Plan

1. Check all `\ref{}` and `\cite{}` commands resolve
2. Verify all figure files exist in `figures/`
3. Compile with `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
4. Check figure placement and caption quality
5. Verify page count is appropriate (~60-70 pages)