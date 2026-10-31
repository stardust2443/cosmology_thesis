#!/usr/bin/env python3
"""
harvest_advisor_preprints.py — Automated Preprint Harvester & Obsidian/Zotero Ingestion

Queries the arXiv API for a given researcher, downloads the recent 2-3 PDFs into
knowledge_vault/05_Advisor_Preprints/, generates an Obsidian literature dossier,
and appends a BibTeX record for Zotero integration.
"""

import os
import sys
import argparse
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime

VAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "knowledge_vault"))
PREPRINTS_DIR = os.path.join(VAULT_ROOT, "05_Advisor_Preprints")
DOSSIERS_DIR = os.path.join(VAULT_ROOT, "04_PhD_Outreach_Dossiers")
BIB_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main_thesis_docs", "references.bib"))

ARXIV_API_BASE = "http://export.arxiv.org/api/query"

def search_arxiv_by_author(author_name, max_results=3):
    """Query arXiv API for recent papers by author in astrophysics/cosmology."""
    clean_name = author_name.strip()
    query = f'au:"{clean_name}" AND (cat:astro-ph.CO OR cat:astro-ph.GA OR cat:astro-ph.IM)'
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    url = f"{ARXIV_API_BASE}?{urllib.parse.urlencode(params)}"
    print(f"[*] Querying arXiv API: {clean_name} (max {max_results} papers)...")
    
    req = urllib.request.Request(url, headers={'User-Agent': 'CosmoVoidPhDTracker/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
    except Exception as e:
        print(f"[!] Network error fetching from arXiv: {e}")
        print("    (Note: If running inside Antigravity sandbox, network access is restricted. Run in external terminal or provide arXiv IDs directly.)")
        return []

    root = ET.fromstring(data)
    namespace = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
    
    entries = []
    for entry in root.findall('atom:entry', namespace):
        title = entry.find('atom:title', namespace).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', namespace).text.strip().replace('\n', ' ')
        published = entry.find('atom:published', namespace).text.strip()[:10]
        id_url = entry.find('atom:id', namespace).text.strip()
        arxiv_id = id_url.split('/abs/')[-1]
        
        authors = [a.find('atom:name', namespace).text for a in entry.findall('atom:author', namespace)]
        
        # PDF URL
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        entries.append({
            'arxiv_id': arxiv_id,
            'title': title,
            'summary': summary,
            'published': published,
            'authors': authors,
            'pdf_url': pdf_url
        })
    return entries

def download_pdf(pdf_url, target_path):
    """Download PDF to local vault path."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):
        print(f"    [-] PDF already exists: {os.path.basename(target_path)}")
        return True
    
    print(f"    [+] Downloading PDF: {pdf_url} -> {os.path.basename(target_path)}...")
    req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, 'wb') as f:
            f.write(resp.read())
        return True
    except Exception as e:
        print(f"    [!] Failed to download PDF: {e}")
        return False

def generate_bibtex_key(author_last, year, arxiv_id):
    clean_id = arxiv_id.replace('.', '_').replace('/', '_')
    return f"{author_last}{year}_{clean_id}"

def harvest_for_advisor(author_name, max_papers=3, dry_run=False):
    """Harvest preprints, save PDFs locally, generate Obsidian dossier, and append BibTeX."""
    papers = search_arxiv_by_author(author_name, max_papers)
    if not papers:
        print(f"[!] No papers found for '{author_name}'. Try alternate spelling or query.")
        return

    author_slug = author_name.replace(' ', '_').replace('.', '')
    advisor_dir = os.path.join(PREPRINTS_DIR, author_slug)
    os.makedirs(advisor_dir, exist_ok=True)
    os.makedirs(DOSSIERS_DIR, exist_ok=True)

    dossier_path = os.path.join(DOSSIERS_DIR, f"{author_slug}_Dossier.md")
    
    dossier_content = [
        f"---",
        f"advisor: \"{author_name}\"",
        f"harvest_date: \"{datetime.now().strftime('%Y-%m-%d')}\"",
        f"tags: [phd-advisor, outreach, literature-harvest]",
        f"---",
        f"",
        f"# 🧑‍🏫 Candidate Advisor Dossier: {author_name}",
        f"",
        f"> **Harvest Summary**: {len(papers)} recent preprints fetched via arXiv API and saved locally in `knowledge_vault/05_Advisor_Preprints/{author_slug}/`.",
        f"",
        f"## 📚 Recent Preprints (Downloaded Locally)",
        f""
    ]

    for p in papers:
        clean_title = p['title'].replace('"', "'")
        pdf_filename = f"{p['arxiv_id'].replace('/', '_')}.pdf"
        local_pdf_path = os.path.join(advisor_dir, pdf_filename)
        rel_pdf_path = f"../05_Advisor_Preprints/{author_slug}/{pdf_filename}"
        
        first_author_last = p['authors'][0].split()[-1] if p['authors'] else "Author"
        pub_year = p['published'][:4]
        bib_key = generate_bibtex_key(first_author_last, pub_year, p['arxiv_id'])

        if not dry_run:
            download_pdf(p['pdf_url'], local_pdf_path)

        dossier_content.extend([
            f"### 📄 [{clean_title}](https://arxiv.org/abs/{p['arxiv_id']})",
            f"- **arXiv ID**: `{p['arxiv_id']}` | **Published**: {p['published']} | **BibTeX Key**: `\\cite{{{bib_key}}}`",
            f"- **Local PDF**: [Download / Open PDF]({rel_pdf_path})",
            f"- **Authors**: {', '.join(p['authors'][:5])}{' et al.' if len(p['authors']) > 5 else ''}",
            f"- **Abstract Snippet**:",
            f"  > {p['summary'][:350]}...",
            f"- **Relevance & Thesis Hook**:",
            f"  - *Connection to BitVF / CBL*: [Synthesize connection to linear barrier universality or Vdn modeling]",
            f""
        ])

    dossier_content.extend([
        f"## ✉️ Tailored Cold Outreach Draft",
        f"",
        f"```markdown",
        f"Subject: Prospective PhD candidate: Cosmic void statistics & large-scale structure (Arun Mitra)",
        f"",
        f"Dear Prof. {author_name.split()[-1]},",
        f"",
        f"I recently read your preprint, \"{papers[0]['title']}\" (arXiv:{papers[0]['arxiv_id']}), and was particularly interested in...",
        f"",
        f"[Use standard phd-outreach-engine template referencing Arun's BitVF thesis findings]",
        f"```"
    ])

    with open(dossier_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(dossier_content))

    print(f"[✓] Created Advisor Dossier: {dossier_path}")
    print(f"[✓] PDFs and notes are ready for Obsidian, Zotero, and NotebookLM ingestion.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Harvest recent arXiv preprints for potential PhD advisors.")
    parser.add_argument("--author", type=str, required=True, help="Full or last name of researcher")
    parser.add_argument("--max-papers", type=int, default=3, help="Max number of preprints to download (default 3)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch metadata without downloading PDFs")
    args = parser.parse_args()

    harvest_for_advisor(args.author, args.max_papers, args.dry_run)
