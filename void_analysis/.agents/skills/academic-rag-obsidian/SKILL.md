---
name: academic-rag-obsidian
description: >-
  Use this skill when capturing literature from Consensus, syncing papers with Zotero/BibTeX,
  generating atomic first-principles notes in Obsidian using the Karpathy Method, or preparing
  curated source files for Google NotebookLM.
---

# Academic RAG & Obsidian Synthesis Runbook (Karpathy Method)

This skill connects **Consensus.app**, **Zotero / BibTeX**, **Google NotebookLM**, and the local **Obsidian Knowledge Vault** (`knowledge_vault/`), using Andrej Karpathy's first-principles method to eliminate black boxes and maintain an active knowledge graph.

---

## 1. The Karpathy First-Principles Note Standard

Every concept or paper note generated must adhere to these four standards:

1. **No Black-Box Explanations**:
   - Never say "the library calculates the likelihood". Detail the exact mathematical term:
     $$\ln\mathcal{L} = -\frac{1}{2} (d - m)^T C^{-1} (d - m) - \frac{1}{2} \ln|C|$$
2. **Explicit Stated vs. Unstated Assumptions**:
   - Highlight the hidden assumptions in every model (e.g., Jennings Vdn assumes strictly non-overlapping spherical voids; REVOLVER assumes Voronoi polygon volumes can be mapped to effective spheres).
3. **Bi-Directional Knowledge Linking**:
   - Every note must contain links to parent concepts `[[00_Master_MOC]]`, related literature `[[01_Literature_Notes/...]]`, and empirical results `[[02_Permanent_Concepts/...]]`.
4. **Toy Model to Full Scale**:
   - Summarize the synthetic intuition with a toy problem or scaling relation before detailing the full $N$-body simulation.

---

## 2. Note Template Specifications

### A. Literature Note Template (`knowledge_vault/01_Literature_Notes/<AuthorYear>.md`)
```markdown
---
title: "Paper Title"
authors: [Author 1, Author 2]
year: 202X
journal: "MNRAS / ApJ / PRD"
arxiv_id: "XXXX.XXXXX"
bibtex_key: "AuthorYear"
tags: [paper, void-cosmology, ...]
---

# 📖 Literature Note: Author et al. (Year)

## 1. Executive Summary & Core Contribution
...

## 2. Mathematical Formulation & Key Equations
...

## 3. Stated vs. Unstated Assumptions
- **Stated**: ...
- **Unstated / Critical Weakness**: ...

## 4. Direct Connection to Our Void Analysis & Thesis
- How this paper's findings compare with our Aletheia BitVF / REVOLVER results.

## 🔗 Related Notes
- [[02_Permanent_Concepts/...]]
```

### B. Atomic Permanent Concept Template (`knowledge_vault/02_Permanent_Concepts/<Concept>.md`)
```markdown
---
tags: [concept, ...]
aliases: [...]
---

# 🧠 First-Principles Concept: [Concept Name]

> **Core Thesis**: One-sentence fundamental truth.

## 1. First-Principles Derivation / Theory
...

## 2. Algorithmic / Experimental Manifestation
...

## 3. Common Pitfalls & Defense Counter-Arguments
...

## 🔗 Related Notes
...
```

---

## 3. Tool Integrations: Zotero, Consensus, NotebookLM

### Zotero & BibTeX Sync:
- Maintain source-of-truth citations in `main_thesis_docs/references.bib`.
- Ensure `bibtex_key` matches the citation tag in `main.tex` (`\cite{...}`).

### Sourcing from Consensus.app:
- Query Consensus for literature debates (e.g. *"linear void barrier slope universality"*).
- When ingesting Consensus findings, extract:
  - Consensus percentage (e.g. "90% agree that void size function constrains dark energy").
  - Outliers / contradictory papers (add to "Stated vs. Unstated Assumptions").

### Preparing Sources for Google NotebookLM:
- Synthesize comprehensive markdown documents in `knowledge_vault/` and compile them into clean PDFs or text files to upload directly into NotebookLM notebooks.
