---
name: presentation-mastery
description: >-
  Use this skill when preparing, designing, or rehearsing scientific slide decks and defense talks,
  especially for meetings or defenses with Dr. Simone Sartori, Dr. Sofia Contarini, Prof. Federico Marulli,
  and Prof. Lauro Moscardini.
---

# Scientific Presentation & Committee Mastery Runbook

This skill governs the slide architecture, oral narrative, and defensive rhetoric for presenting cosmological void statistics to senior committee members and supervisors.

---

## 1. Committee Stakeholder Heuristics

When structuring any slide deck or defense script, calibrate directly to the four committee members:

| Committee Member | Primary Value | Triggers / Pet Peeves | What Wins Their Approval |
| :--- | :--- | :--- | :--- |
| **Prof. Federico Marulli** *(Supervisor)* | Methodological discipline, clean residuals, defensible statistical bounds. | Connecting discrete data points with lines; unsubstantiated cosmological claims; hiding bad $\chi^2$. | Showing standardized pull panels $(d-m)/\sigma_{\mathrm{JK}}$ with $\pm 1\sigma$ bands; conservative error bounds; addressing Sample 50P Poisson noise upfront. |
| **Dr. Simone Sartori** *(BitVF Creator)* | BitVF algorithmic elegance, displacement divergence physics. | Dismissing BitVF without fair benchmark; confusing Lagrangian divergence with Eulerian density cuts. | Showing that BitVF matches Vdn physics ($\chi^2/\nu \approx 0.9\text{--}1.7$) while watershed finders fail; proving BitVF's barrier is scale-invariant to $< 1\%$. |
| **Dr. Sofia Contarini** *(Void Cosmology Expert)* | Theoretical rigor in excursion set models, void bias, tracer dependence. | Treating void bias as a black-box parameter; ignoring parameter degeneracies. | Presenting the linear barrier universality ($B_{\mathrm{slope}} \approx 0.91$) with $0.07\sigma$ precision; demonstrating joint clustering constraints. |
| **Prof. Lauro Moscardini** *(Department Head / Euclid Lead)* | Survey feasibility, cosmological return, high-level impact. | Over-focusing on niche coding bugs without linking to Euclid / DESI; lack of big-picture narrative. | Clear transition from N-body simulations to Euclid flagship mocks; computational scaling; dark energy & neutrino mass sensitivity. |

---

## 2. Standard 5-Act Slide Narrative Arc

Every defense or group presentation must follow this 5-act structure:

1. **Act 1: The Epistemological Problem**
   - Voids are defined algorithmically. Excursion-set models (Jennings et al. Vdn) assume isolated, volume-conserving spherical underdensities.
   - Core Question: *Which void finders define objects that are physically compatible with theoretical excursion-set models?*
2. **Act 2: Phase 1 — Validation & Catalog Construction**
   - Matching Simone Sartori's ForArun reference to $1\%\text{--}7\%$ precision across dark matter downsamplings.
   - Proof that `TOUCH` non-overlapping spheres preserve volume conservation, while `CENTER` double-counts volume.
3. **Act 3: Phase 2 — Dark Matter Parameter Recovery**
   - Unbiased recovery of Aletheia truth ($\Omega_m = 0.315, \sigma_8 = 0.811$) with informative priors.
4. **Act 4: Phase 3 — The Halo Cross-Finder Benchmark (The Climax)**
   - BitVF vs. REVOLVER across 4 halo catalogs (100P, 75P, 50P, 30P).
   - Goodness-of-fit showdown: BitVF $\chi^2/\nu = 0.90\text{--}1.67$ vs. REVOLVER $\chi^2/\nu = 3.14\text{--}8.12$.
   - The S-curve tilt mechanism explaining REVOLVER's apparent MCMC bias ($\Omega_m \sim 0.44$).
   - Universal linear barrier: $B_{\mathrm{slope}} = 0.914 \pm 0.043$ vs $0.910 \pm 0.046$ ($\Delta = 0.07\sigma$).
   - 5-part robustness audit: Scale invariance ($< 1\%$ drift), tail truncation, and joint multi-tracer barrier fit.
5. **Act 5: Euclid Outlook & Conclusion**
   - Translating simulation calibrations to Euclid and DESI galaxy mocks.

---

## 3. Visual & Slide Standards

- **Plots**: Every VSF slide must use figures generated via the `advisor-scientific-plotting` skill:
  - Discrete data markers with error bars (no connecting lines).
  - Continuous theory curves.
  - Lower residual pull subpanel with shaded $\pm 1\sigma$ and $\pm 2\sigma$ reference bands.
  - Log axis limits set cleanly (e.g. `ylim(2e-9, 2e-5)`).
- **Typography**: 1 idea per slide, bold key numbers (e.g. **$\mathbf{0.07\sigma}$**, **$\mathbf{\chi^2/\nu = 0.90}$**).
- **Backup Slides**: Always maintain dedicated backup slides with the 5 robustness audits and the per-bin $\chi^2$ gradients.
