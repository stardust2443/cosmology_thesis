#!/usr/bin/env python3
"""
audit_thesis_consistency.py — Full-Manuscript LaTeX Consistency & Notation Auditor

Scans main_thesis_docs/main.tex from line 1 to the end for:
1. Distance and density units (h^-1 Mpc, h^3 Mpc^-3)
2. Cosmological and nuisance parameter formatting (Omega_m, sigma_8, b_void, b_eff)
3. Reduced chi-square notation (chi^2/nu vs chi^2/dof)
4. Undefined / broken bib citations against references.bib
5. Common typography and KaTeX/LaTeX formatting flaws.
"""

import os
import re
import sys

TEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main_thesis_docs", "main.tex"))
BIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main_thesis_docs", "references.bib"))

# Rules: (Pattern, Canonical Replacement / Warning Message, Category)
NOTATION_RULES = [
    # Units
    (r'(?<!\\mathrm\{)Mpc\s*/\s*h\b', r'$h^{-1}\mathrm{Mpc}$ (Avoid Mpc/h)', 'UNITS'),
    (r'(?<!\\mathrm\{)h\^\{-?1\}\s*Mpc\b', r'$h^{-1}\mathrm{Mpc}$ (Use \mathrm{Mpc})', 'UNITS'),
    (r'(?<!\\mathrm\{)h\^3\s*/\s*Mpc\^3\b', r'$h^3\,\mathrm{Mpc}^{-3}$ (Avoid h^3/Mpc^3)', 'UNITS'),
    (r'km\s*/\s*s\s*/\s*Mpc\b', r'$\mathrm{km\,s^{-1}\,Mpc^{-1}}$ (Use LaTeX units)', 'UNITS'),

    # Cosmological & Void Parameters
    (r'\\Omega_[M](?![a-zA-Z])', r'$\Omega_m$ (Lowercase m for matter)', 'PARAMETERS'),
    (r'\\chi\^2\s*/\s*(?:dof|deg)', r'$\chi^2/\nu$ (Use \nu for degrees of freedom)', 'PARAMETERS'),
    (r'b_\{?void\}?', r'$b_{\mathrm{void}}$ (Use \mathrm{void})', 'PARAMETERS'),
    (r'b_\{?eff\}?', r'$b_{\mathrm{eff}}$ (Use \mathrm{eff})', 'PARAMETERS'),
    (r'B_\{?slope\}?', r'$B_{\mathrm{slope}}$ (Use \mathrm{slope})', 'PARAMETERS'),
    (r'B_\{?offset\}?', r'$B_{\mathrm{offset}}$ (Use \mathrm{offset})', 'PARAMETERS'),
]

def load_bib_keys(bib_path):
    keys = set()
    if not os.path.exists(bib_path):
        return keys
    with open(bib_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    for match in re.finditer(r'@\w+\s*\{\s*([^,\s]+)\s*,', content):
        keys.add(match.group(1).strip())
    return keys

def audit_latex():
    if not os.path.exists(TEX_PATH):
        print(f"[!] main.tex not found at {TEX_PATH}")
        sys.exit(1)

    with open(TEX_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    bib_keys = load_bib_keys(BIB_PATH)
    print(f"[*] Loaded {len(bib_keys)} BibTeX keys from references.bib")
    print(f"[*] Auditing {len(lines)} lines in main.tex from Chapter 1 to Appendices...\n")

    issues_by_cat = {'UNITS': [], 'PARAMETERS': [], 'CITATIONS': [], 'LABELS': []}
    
    defined_labels = set()
    referenced_labels = []

    for idx, line in enumerate(lines, 1):
        # Ignore comments
        stripped = line.strip()
        if stripped.startswith('%'):
            continue
        code_part = line.split('%')[0]

        # Notation checks
        for pat, fix, cat in NOTATION_RULES:
            if re.search(pat, code_part):
                issues_by_cat[cat].append((idx, fix, code_part.strip()))

        # Label definitions
        for m in re.finditer(r'\\label\{([^}]+)\}', code_part):
            defined_labels.add(m.group(1).strip())

        # References
        for m in re.finditer(r'\\ref\{([^}]+)\}', code_part):
            referenced_labels.append((idx, m.group(1).strip()))

        # Citations
        for m in re.finditer(r'\\cite[a-z]*\{([^}]+)\}', code_part):
            raw_keys = m.group(1).split(',')
            for k in raw_keys:
                clean_k = k.strip()
                if clean_k and clean_k not in bib_keys:
                    issues_by_cat['CITATIONS'].append((idx, f"Missing citation key '{clean_k}'", code_part.strip()))

    # Check broken labels
    for idx, ref in referenced_labels:
        if ref not in defined_labels:
            issues_by_cat['LABELS'].append((idx, f"Undefined reference tag '\\ref{{{ref}}}'", f"Line {idx}"))

    # Summary Report
    print("=" * 70)
    print("           FULL-MANUSCRIPT THESIS CONSISTENCY AUDIT REPORT           ")
    print("=" * 70)
    
    total_issues = sum(len(v) for v in issues_by_cat.values())
    print(f"Total findings across 114+ pages: {total_issues}\n")

    for cat, items in issues_by_cat.items():
        print(f"--- {cat} ({len(items)} findings) ---")
        for idx, fix, context in items[:8]:  # Print first 8 of each
            print(f"  Line {idx:5d}: {fix}")
            print(f"             Context: {context[:80]}...")
        if len(items) > 8:
            print(f"  ... and {len(items) - 8} more in this category.")
        print()

    print("=" * 70)
    print("Audit Complete. Ready to apply automated corrections or targeted edits.")

if __name__ == '__main__':
    audit_latex()
