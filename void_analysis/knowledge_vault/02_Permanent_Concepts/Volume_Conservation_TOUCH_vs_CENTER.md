---
tags: [concept, void-cleaning, volume-conservation, statistics, vdn]
aliases: [Volume Conservation, TOUCH vs CENTER]
---

# 🧠 First-Principles Concept: Volume Conservation (TOUCH vs. CENTER)

> **First Principle**: Excursion set theory assumes space is partitioned into non-overlapping regions associated with isolated peaks or underdense troughs. Double-counting volume invalidates the normalization of the probability density function $f(\ln\sigma)$.

---

## 1. The Two Cleaning Algorithms

When BitVF identifies proto-void candidates, they frequently overlap in Eulerian space. Two distinct filtering philosophies exist:

### A. `TOUCH` (OV00 — Non-Overlapping Spheres)
- Spheres are sorted by radius (or central depth).
- Larger voids are retained; any subsequent void whose boundary intersects with an existing void is discarded.
- **Overlap allowed**: $0\%$.
- **Physics**: Strictly preserves volume conservation. Each point in space belongs to at most one void.

### B. `CENTER` (OV05 — Concentric / Partial Overlap)
- Spheres are allowed to overlap as long as the center of one void does not lie within the radius of another void.
- **Overlap allowed**: Up to $50\%$ volumetric intersection.
- **Physics**: Severely double-counts space. A single underdensity contributes to multiple volume integrals.

---

## 2. Experimental Proof: Audit 4 Results

From our 5-part BitVF model robustness audit across the Aletheia halo catalogs:

| Catalog | `TOUCH` $\chi^2/\nu$ | `CENTER` $\chi^2/\nu$ | Ratio ($\chi^2_{\mathrm{CENTER}} / \chi^2_{\mathrm{TOUCH}}$) |
| :--- | :---: | :---: | :---: |
| **100P** | **0.90** | **23.14** | $\approx 25\times$ worse |
| **75P**  | **1.36** | **28.62** | $\approx 21\times$ worse |
| **50P**  | **1.83** (clean) | **18.45** | $\approx 10\times$ worse |
| **30P**  | **1.67** | **19.80** | $\approx 12\times$ worse |

### Why `CENTER` Blows Up $\chi^2/\nu$:
1. Double-counting dramatically inflates the measured number density $\mathrm{d}n/\mathrm{d}\ln R$ at intermediate scales ($R \sim 15\text{--}25\,h^{-1}\mathrm{Mpc}$).
2. The theoretical Vdn model curve cannot bend upward at intermediate scales without violating mass conservation.
3. The fit breaks down completely ($\chi^2/\nu \approx 18\text{--}29$).

---

## 3. Defense Rule for Sartori & Marulli
If asked: *"Why did you not use the CENTER cleaning algorithm that Simone previously explored?"*
- **Answer**: *"Jennings et al. Vdn is derived strictly under the volume conservation condition. When we allow overlapping spheres under `CENTER`, we double-count volume, which blows up $\chi^2/\nu$ from $0.90$ to $23.14$. We proved in Audit 4 that `TOUCH` non-overlapping spheres are a mathematical requirement for excursion-set modeling."*

---

## 🔗 Related Notes
- [[02_Permanent_Concepts/Excursion_Set_Void_Theory|Excursion Set Void Theory]]
- [[02_Permanent_Concepts/Model_Finder_Compatibility|Model-Finder Compatibility]]
- [[01_Literature_Notes/Jennings2013_Vdn|Jennings et al. (2013)]]
