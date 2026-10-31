# Thesis Chapter Structure & Figure Mapping Reference

This document maps chapters, sections, and authoritative figures in `main_thesis_docs/main.tex`.

---

## 1. Thesis Chapter Structure

| Chapter | Title / Topic | Primary Key Scripts |
| :--- | :--- | :--- |
| **1** | Introduction: Large Scale Structure & Voids | Theoretical review |
| **2** | Excursion Set Theory & Jennings et al. Vdn | `scripts/vsf_model.py` |
| **3** | Void Finders: Dynamical BitVF vs Watershed REVOLVER | `scripts/run_analysis.sh` |
| **4** | Parameter Inference in Dark Matter Simulations | `scripts/mcmc_diagnostic_plots.py` |
| **5** | Halo Void Statistics, Barrier Universality & Model Compatibility | `scripts/plot_calibration_vsf_residuals_grid.py`, `scripts/compare_barrier_universality.py` |
| **6** | Conclusions & Prospects for Euclid / DESI | Summary synthesis |

---

## 2. Key Thesis Figures & Source Scripts

| Thesis Figure Label | Source Script in `scripts/` | Target in `main_thesis_docs/figures/` |
| :--- | :--- | :--- |
| `fig:calibration_grid` | `plot_calibration_vsf_residuals_grid.py` | `calibration_vsf_residuals_grid.png` |
| `fig:robustness_audit` | `audit_bitvf_model_robustness.py` | `audit_bitvf_model_robustness.png` |
| `fig:barrier_universality` | `compare_barrier_universality.py` | `compare_barrier_universality.png` |
| `fig:mcmc_comparison` | `plot_mcmc_finder_comparison.py` | `plot_mcmc_finder_comparison.png` |

---

## 3. Post-Compilation Checks

After compiling `main.tex`:
1. Check that total page count and table of contents are up to date.
2. Confirm there are no dangling citations:
   ```bash
   grep -i "citation.*undefined" main_thesis_docs/main.log
   ```
3. Confirm there are no broken references:
   ```bash
   grep -i "reference.*undefined" main_thesis_docs/main.log
   ```
