---
name: phd-outreach-engine
description: >-
  Use this skill when researching potential PhD advisors, harvesting their recent 2-3 arXiv preprints,
  saving PDFs locally in knowledge_vault/05_Advisor_Preprints/, creating Obsidian advisor dossiers,
  linking citations to Zotero, and drafting European-style PhD application pitches.
---

# European PhD Intelligence & Advisor Outreach Engine

This skill automates the identification, preprint harvesting, and tailored outreach to European and UK Principal Investigators (PIs) in observational cosmology, large-scale structure, cosmic voids, and galaxy surveys.

---

## 1. The European Academic Reality

In European institutions (Germany Max Planck / TV-L E13, Netherlands NWO, UK UKRI/STFC, Italy INAF, Switzerland Swiss NSF, France CNRS):
- PhD positions are funded either through **centralized doctoral programs** (IMPRS, Durham, Leiden) or **PI-controlled research grants** (ERC Starting/Consolidator grants, €1.5M–€2.0M).
- PIs often have open positions months before formal advertisements appear.
- **What works**: A concise (150–250 words), deeply technical email referencing a specific finding in the PI's preprint from the last 6 months, demonstrating that you have immediate code and theory skills to advance their project.
- **What fails**: Generic flattery (*"I have read your esteemed papers with great interest"*), vague research goals, or generic CV attachments.

---

## 2. Automated Preprint Harvesting & Local Archiving

Whenever a candidate advisor is identified from `professors_outreach_europe_50.csv` or `phd_tracker/`:

1. Run the local harvester:
   ```bash
   /opt/homebrew/bin/python3.12 scripts/harvest_advisor_preprints.py --author "Author Name" --max-papers 3
   ```
2. **What this does**:
   - Queries the official arXiv API (`astro-ph.CO`, `astro-ph.GA`).
   - Downloads the 2–3 most recent PDFs into:
     `knowledge_vault/05_Advisor_Preprints/<Author_LastName>/`
   - Generates an Obsidian literature note in `knowledge_vault/04_PhD_Outreach_Dossiers/` linking the paper's methods to your thesis achievements.
   - Appends the BibTeX record to `main_thesis_docs/references.bib` or a local `.bib` file for instant Zotero import.
   - Makes the text queryable by NotebookLM and Consensus.

---

## 3. The 4-Paragraph High-Conversion Outreach Structure

Draft cold emails using this proven European template:

```markdown
Subject: Prospective PhD candidate: Cosmic void statistics & large-scale structure (Arun Mitra)

Dear Prof. [Last Name],

I recently read your preprint, "[Exact Title]" ([arXiv:XXXX.XXXXX]), and was particularly interested in your approach to [specific methodology, e.g., void clustering in DESI / cross-correlating voids with CMB lensing].

I am completing my Master's thesis at the University of Bologna under Prof. Federico Marulli, focusing on cosmic void statistics and cosmological parameter inference using the Aletheia simulation suite and CosmoBolognaLib. In our work, we benchmarked dynamical displacement reconstruction (BitVF) against Voronoi watershed algorithms (REVOLVER), proving that linear void barrier scaling ($B_{\mathrm{slope}} = 0.914 \pm 0.043$) is universal to $0.07\sigma$ and identifying how watershed shape mismatch drives apparent MCMC biases in $\Omega_m$.

Given your group's leadership in [PI's specific survey or project, e.g. Euclid LSS analysis / DESI void science], I am eager to explore PhD opportunities in your team starting in [Target Month/Year, e.g. Autumn 2027], funded either through your active grants or institutional programs.

I have attached my CV and a brief thesis summary. If you have 15 minutes in the coming weeks, I would welcome the opportunity to discuss your group's upcoming directions.

Sincerely,

Arun Mitra
Master's Candidate in Astrophysics & Cosmology, University of Bologna
[Link to GitHub / Thesis Summary]
```

---

## 4. Tracking and CRM Protocol
- Every sent pitch, meeting log, and status update must be logged in `phd_tracker/` and `professors_outreach_europe_50.csv`.
- Maintain a 10-day follow-up reminder rule.
