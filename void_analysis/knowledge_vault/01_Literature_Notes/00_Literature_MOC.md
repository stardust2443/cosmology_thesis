# 📚 Literature Map of Content (MOC)

Literature notes synthesized using the **Karpathy Method**: Extract core physics, exact equations, stated vs. unstated assumptions, and direct points of contact with our thesis data.

---

## 📑 Core Papers in This Vault

| Paper | Key Contribution | Primary Point of Contact with Thesis |
| :--- | :--- | :--- |
| [[01_Literature_Notes/Sartori2025_BitVF\|Sartori et al. (2025)]] | Back-in-time void finder (BitVF) using displacement field divergence $\theta(\mathbf{x}) = \nabla \cdot \mathbf{\Psi}$ | Phase 1 validation (matching ForArun within $1\%\text{--}7\%$) and BitVF halo catalog generation. |
| [[01_Literature_Notes/Jennings2013_Vdn\|Jennings et al. (2013)]] | Excursion-set Void Size Function (Vdn) with spherical volume conservation | Theoretical VSF model used in CBL; requires `TOUCH` non-overlapping spheres. |
| [[01_Literature_Notes/Nadathur2019_REVOLVER\|Nadathur et al. (2019)]] | REVOLVER void finder using ZOBOV Voronoi watershed basins | Phase 3 benchmark finder; exhibits intrinsic structural mismatch with Vdn ($\chi^2/\nu \approx 3.1\text{--}8.1$). |
| [[01_Literature_Notes/Contarini2021_VoidCosmology\|Contarini et al. (2021, 2023)]] | Cosmological parameter constraints with cosmic voids in CosmoBolognaLib | Baseline for linear void bias modeling and MCMC likelihood formulations. |
| [[01_Literature_Notes/Marulli2016_CBL\|Marulli et al. (2016)]] | CosmoBolognaLib: C++ library for cosmological statistics | Software engine used for all 2-point statistics, VSFs, and covariance calculations. |

---

## 🔗 Zotero & NotebookLM Integration
- All papers in this directory are indexed in the master BibTeX file: [`main_thesis_docs/references.bib`](file:///Users/arun/projects/my_cosmo_analysis/void_analysis/main_thesis_docs/references.bib).
- Corresponding full-text PDFs can be synced to local storage or uploaded to your Google NotebookLM workspace.
