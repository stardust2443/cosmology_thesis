---
tags: [concept, mcmc, convergence, prior-wall, diagnostics, emcee]
aliases: [MCMC Prior-Wall Diagnostics, Prior Wall Pinning]
---

# 🧠 First-Principles Concept: MCMC Prior-Wall Diagnostics

> **First Principle**: When an MCMC chain hugs the prior boundary (e.g. $\Omega_m = 0.50$ or $\sigma_8 = 1.20$), the novice response is to widen the prior. The Karpathy first-principles response is to calculate the **per-bin $\chi^2$ gradients** and check if the minimum of the likelihood function lies in unphysical parameter space due to **theoretical model mismatch**.

---

## 1. The Mathematical Mechanism of Prior-Wall Pinning

Given data $d$ and model $m(\theta)$:
$$\chi^2(\theta) = (d - m(\theta))^T C^{-1} (d - m(\theta))$$

If the theoretical model $m(\theta)$ has a structural shape that **cannot** match the data points $d$ regardless of parameter choice:
$$\nabla_\theta \chi^2(\theta) \neq 0 \quad \forall \, \theta \in \mathcal{P}_{\mathrm{physical}}$$
The optimizer will push parameters along the steepest descent direction until it encounters the hard constraint boundary defined by the prior $\mathcal{P}(\theta)$:
$$\ln \mathcal{L}(\theta) + \ln \mathcal{P}(\theta)$$
At the boundary, the probability is clipped to zero ($-\infty$ in log space), pinning the walkers against the wall.

---

## 2. REVOLVER vs. BitVF Case Study

In our Phase 2 & 3 MCMC runs:

| Finder | Data Vector | Prior Range on $\Omega_m$ | Posterior Median | Chain Status |
| :--- | :--- | :---: | :---: | :--- |
| **BitVF** | TOUCH/OV00 | $[0.10, 0.50]$ | $\mathbf{0.318 \pm 0.012}$ | Converged in interior (Recovers truth $0.315$) |
| **REVOLVER** | Unmerged Basins | $[0.10, 0.50]$ | $\mathbf{0.442 \pm 0.018}$ | Pushed to upper boundary |

### Diagnosing the Cause:
1. We evaluated the per-bin residual pull $\chi_i^2 = (d_i - m_i)^2 / \sigma_i^2$ at the truth, at the local posterior mode, and across parameter sweeps (see `scripts/mcmc_diagnostic_plots.py`).
2. Result: REVOLVER's data vector exhibits an intrinsic S-curve tilt (large voids underdense, intermediate voids overdense relative to Vdn).
3. Increasing $\Omega_m$ lowers the high-radius tail of the Vdn prediction, attempting to compensate for the S-curve.
4. Widening the prior to $\Omega_m \in [0.1, 0.8]$ only causes REVOLVER to drift further to $\Omega_m \sim 0.60$, proving that the problem is not prior truncation, but **model-finder incompatibility**.

---

## 3. Defense Rule for Federico Marulli & Simone Sartori
- When presenting posterior contours:
  *"BitVF recovers the simulation truth $\Omega_m = 0.315, \sigma_8 = 0.811$ within $1\sigma$. The REVOLVER bias is mathematically traceable to the S-curve shape mismatch between watershed basins and excursion-set theory, driving the MCMC sampler toward the upper prior boundary."*

---

## 🔗 Related Notes
- [[02_Permanent_Concepts/Model_Finder_Compatibility|Model-Finder Compatibility]]
- [[.agents/skills/cosmo-mcmc-inference/SKILL.md|cosmo-mcmc-inference Skill]]
