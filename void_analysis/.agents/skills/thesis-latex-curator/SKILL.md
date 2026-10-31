---
name: thesis-latex-curator
description: >-
  Use this skill when editing, compiling, or auditing the PhD thesis LaTeX source
  (main_thesis_docs/main.tex), managing bibliography citations (references.bib), synchronizing figures
  from plots/ to main_thesis_docs/figures/, resolving LaTeX compilation errors, or enforcing notation consistency.
---

# PhD Thesis LaTeX Curation & Build Runbook

This skill governs the compilation, figure synchronization, and academic notation consistency for Arun Mitra's PhD thesis (`main_thesis_docs/main.tex`).

---

## 1. Directory Layout & Primary Files

All thesis files live inside `main_thesis_docs/`:
- **Master LaTeX Source**: `main_thesis_docs/main.tex`
- **Bibliography**: `main_thesis_docs/references.bib`
- **Embedded Figures**: `main_thesis_docs/figures/`
- **Output PDF**: `main_thesis_docs/main.pdf`

---

## 2. Compilation Protocol

When compiling the thesis document, always run inside the `main_thesis_docs/` directory:

```bash
cd main_thesis_docs

# Standard 3-pass build with BibTeX
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Or using `latexmk` if available:
```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

### Build Error Triage
1. **Undefined References / Question Marks (`??`)**:
   - Check `main.log` for `LaTeX Warning: Reference '...' undefined`.
   - Ensure `\label{sec:...}` or `\label{fig:...}` matches the `\ref{...}` tag exactly.
2. **Missing Citations**:
   - Check `main.blg` for `Warning--I didn't find a database entry for "..."`.
   - Add missing BibTeX entries to `main_thesis_docs/references.bib`.
3. **Float / Figure Overflow**:
   - Use `[tbp]` placement specifiers rather than hard `[h!]` for large 2-column or grid figures.

---

## 3. Figure Synchronization Workflow

When a script updates or creates a figure in `plots/`:
1. **Verification**: Confirm the figure follows the rules in `advisor-scientific-plotting` (pull panel, discrete points without lines, log limits).
2. **Synchronization**: Copy the updated figure to `main_thesis_docs/figures/`:
   ```bash
   cp plots/halos/<figure_name>.png main_thesis_docs/figures/
   ```
3. **Document Reference**: Check `main.tex` to ensure the caption, label, and text discussion accurately reflect the latest numbers and findings.

---

## 4. Standardized Notation & Typography Rules

Always maintain strict LaTeX notation consistency:

| Concept | Correct LaTeX | Incorrect / Prohibited |
| :--- | :--- | :--- |
| Comoving distance | `$h^{-1}\mathrm{Mpc}$` | `$h^{-1} Mpc$`, `$Mpc/h$` |
| Number density | `$h^3\,\mathrm{Mpc}^{-3}$` | `$h^3/Mpc^3$`, `$Mpc^{-3}$` |
| Matter density | `$\Omega_m$` | `$\Omega_M$`, `$Om$` |
| Amplitude of fluctuations | `$\sigma_8$` | `$\sigma_8$`, `$s_8$` |
| Reduced chi-square | `$\chi^2/\nu$` | `$\chi^2/\mathrm{dof}$`, `$\chi^2_{red}$` |
| Void bias | `$b_{\mathrm{void}}$` | `$b_{void}$` (un-italicized subscript) |
| Effective tracer bias | `$b_{\mathrm{eff}}$` | `$b_{eff}$` |
| Barrier regression | `$b_{\mathrm{void}} = B_{\mathrm{slope}} b_{\mathrm{eff}} + B_{\mathrm{offset}}$` | arbitrary symbols |

See [thesis_workflow.md](./references/thesis_workflow.md) for chapter structure and figure mapping.
