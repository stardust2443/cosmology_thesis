---
title: "The abundance of voids and the excursion set formalisms"
authors: [Elise Jennings, Yan-Chuan Cai, David Weinberg]
year: 2013
journal: "MNRAS 432, 4"
tags: [paper, vdn, excursion-set, theory, vsf]
---

# 📖 Literature Note: Jennings et al. (2013) — Vdn Model

## 1. Executive Summary
Jennings et al. derive an excursion-set formulation for the Void Size Function (VSF) that accounts for the expansion of underdense regions and explicitly enforces **volume conservation**, distinguishing it from the earlier Sheth & van de Weygaert (2004, SvdW) model.

## 2. Mathematical Foundation
1. **The Volume Conservation Condition**:
   $$\sum_i V(R_i) \, n(R_i) = 1 - F_{\mathrm{collapse}}$$
   Space is completely partitioned into non-overlapping expanding voids and collapsing halos.
2. **The Differential Abundance**:
   $$\frac{\mathrm{d}n}{\mathrm{d}\ln R} = \frac{f(\ln \sigma)}{V(R)} \left| \frac{\mathrm{d}\ln \sigma}{\mathrm{d}\ln R} \right|$$
   where $V(R) = \frac{4\pi}{3} R^3$.
3. **First Crossing Distribution $f(\ln\sigma)$**:
   Two-barrier problem with linear underdensity threshold $\delta_v \approx -2.71$ (void expansion limit) and linear collapse threshold $\delta_c = 1.686$ (void-in-cloud prevention).
4. **Volume Mapping**:
   Eulerian radius $R$ is related to Lagrangian radius $r_L$ by:
   $$R = r_L \, (1 + \Delta_v)^{-1/3}$$
   where $\Delta_v \approx -0.80$ is the non-linear underdensity at shell crossing.

## 3. Critical Insight for Our Project
- The model **requires isolated, non-overlapping spherical voids**.
- If applied to overlapping catalogs (`CENTER`) or irregular unmerged watershed basins (`REVOLVER`), the mathematical volume conservation condition is broken, resulting in artificially elevated $\chi^2/\nu$.

## 🔗 Related Notes
- [[02_Permanent_Concepts/Volume_Conservation_TOUCH_vs_CENTER|Volume Conservation (TOUCH vs CENTER)]]
- [[02_Permanent_Concepts/Excursion_Set_Void_Theory|Excursion Set Void Theory]]
- [[02_Permanent_Concepts/Model_Finder_Compatibility|Model-Finder Compatibility]]
