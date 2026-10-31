# `scripts/` — Production pipeline + diagnostics

After Phase 1 cleanup (May 2026). All earlier prototypes, "copy" duplicates,
and superseded plotting scripts are in `_archive/`.

## Production pipeline (run in order)

| Script | Purpose |
|--------|---------|
| `clean_voids.py` | Cleaning pipeline: TH (`delta_avg ≤ −0.80`) → strict OV00 overlap on `r_ext` → 20 h⁻¹Mpc center border. Reads `data/raw/`, writes `data/cleaned/cleaned_recon_cat_*.csv` and `cleaned_tdf_cat_*.csv`. |
| `compute_vsf.py` | Computes the 3-panel thesis-style VSF figure (TDF / OT / REVOLVER). Uses ForArun's exact 20-bin scheme and `V_eff = 250³`. Writes `plots/vsf_comparison.png`. |
| `vsf_model.py` | Vdn theoretical VSF model — used as the likelihood forward model for MCMC. Imported by `run_mcmc_vsf.py` and `validate_vsf_model.py`. |
| `validate_vsf_model.py` | Sanity check that the Vdn model produces sensible VSFs at fiducial cosmology. |
| `run_mcmc_vsf.py` | emcee sampler over (Ω_m, σ_8, δ_v). Reads ForArun `.dat` files as observations; writes chains to `data/mcmc_chains/`. |
| `submit_mcmc.sh`, `submitMCMC.sh` | SLURM submission scripts for cluster MCMC runs. |

## Phase 1 diagnostics (May 2026)

| Script | What it produces |
|--------|------------------|
| `diagnose_cleaning_v2.py` | Ablation grid over (CS × TH × sort × overlap × border × V_eff). Generates SSR rankings and 4-panel NREC50 + 2-panel BIT validation plots. |
| `diagnose_peak_shift.py` | (raw / +TH / cleaned / ForArun) overlay per subsample. Shows that even pre-cleaning VSFs lie below ForArun at sparse densities. |
| `sweep_veff_per_subsample.py` | Per-subsample V_eff optimization. Confirms L = 250 within 8% of optimal for all subsamples. |
| `variance_octants.py` | Empirical intra-realization variance from 8-octant subdivision. Lower bound on inter-realization scatter. |
| `revolver_crosscheck.py` | BitVF (this work) vs REVOLVER (independent finder) vs ForArun. Validates BitVF qualitative behaviour. |
| `thesis_figure_ensemble.py` | **Headline figure** for the ensemble hypothesis: 2 rows (20p / 2_5p) × 3 cols (peak-shift / variance / REVOLVER). |

## Plotting helpers (kept, may be reused)

| Script | Purpose |
|--------|---------|
| `plot_thesis.py` | Thesis-style VSF plot helper (currently unused by main pipeline; retained for one-off thesis figure tuning). |
| `plot_void_cross_sections.py` | 2D slice visualization of voids in a chosen plane. |

## Inputs / outputs at a glance

```
data/raw/                       <- BitVF C++ outputs (one realization per fraction)
data/subsamples/                <- DM particle subsamples (input to BitVF)
data/ForArun/                   <- Simone data ensemble VSFs (.dat)
data/cleaned/                   <- production cleaning outputs (this work)
  ├─ cleaned_recon_cat_*.csv    (NREC50, all 4 subsamples)
  ├─ cleaned_tdf_cat_*.csv      (BIT, 2.5p / 5p only)
  ├─ cleaned_revolver_cat_*.dat (REVOLVER reference catalogs)
  └─ _archive/                  (old iterations)
data/mcmc_chains/               <- MCMC outputs from cluster
plots/                          <- all generated figures
plots/cleaning_validation/      <- Phase 1 diagnostic plots
```

## Documentation

Project narrative and results write-ups live in `../LLM_suggestions/`:

- `walkthrough.md` — comprehensive defense reference (Appendix A summarizes Phase 1)
- `raw_catalog_investigation.md` — full Phase 1 closing document
- `mcmc_cluster_analysis.md` — Phase 2 results
- `task.md` — task tracker (Phase 1 closed)
