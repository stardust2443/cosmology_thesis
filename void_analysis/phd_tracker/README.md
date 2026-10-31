# 🌌 PhD Intelligence & Application Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Field: Astrophysics & Cosmology](https://img.shields.io/badge/Field-Astrophysics%20%26%20Cosmology-purple.svg)]()
[![Update Cycle: 8h](https://img.shields.io/badge/Sync-Every%208%20Hours-green.svg)]()

> **Autonomous Academic Intelligence, Preprint Literature Mining, Grant Tracking, and PI Outreach CRM.**  
> Built for discovering both **formal openings** and **hidden/unadvertised PhD positions** in European & UK Cosmology, Large-Scale Structure, Galaxy Surveys, and Computational Astrophysics.

---

## 🎯 Why This Exists

In the European and UK academic systems, PhD students are **salaried employees or grant-funded researchers** (e.g. German TV-L E13, Dutch university researcher contracts, UKRI/STFC studentships). 

Because of this:
1. **The best positions are unadvertised**: When a Principal Investigator (PI) wins an **ERC Starting/Consolidator Grant** (€1.5M–€2.0M), they have funding for 2–3 PhD students months before posting an official job ad.
2. **Proactive advisor outreach is standard**: Reaching out to group leaders who just posted a relevant preprint on arXiv or won a grant with an informed, technically grounded pitch often secures an interview before a public ad even goes live.
3. **Deadlines are fragmented**: While centralized programs (Max Planck IMPRS, UK STFC DTPs) have winter deadlines (Nov–Jan), project-based vacancies open year-round across Euraxess, AcademicTransfer, and INSPIRE-HEP.

This pipeline automates the entire discovery, matching, and outreach lifecycle.

---

## ✨ Key Features

```
┌────────────────────────────────────────────────────────────────────────┐
│                   PhD APPLICATION INTELLIGENCE PIPELINE                │
├───────────────────┬───────────────────┬────────────────────────────────┤
│ 📡 Formal Openings│ 📚 ArXiv Miner    │ 💶 Grant Radar                 │
│  • INSPIRE-HEP API│  • astro-ph.CO/GA │  • ERC StG & CoG (€1.5M-€2M)   │
│  • AAS Job Reg.   │  • Method Scorer  │  • UKRI Future Leaders         │
│  • Dutch Transfer │  • Author Profiler│  • NWO / INAF / CNRS Grants    │
├───────────────────┼───────────────────┼────────────────────────────────┤
│ 🧑‍🏫 Advisor CRM    │ ✉️ Pitch Drafter  │ 📋 Application Kanban          │
│  • Status tracker │  • Paper-specific │  • Multi-stage pipeline        │
│  • 10-day reminder│  • Thesis linking │  • Recommendation letters      │
│  • Meeting logs   │  • European tone  │  • Deadline countdown radar    │
└───────────────────┴───────────────────┴────────────────────────────────┘
```

* **📚 ArXiv Literature & Advisor Discovery**: Automatically monitors `astro-ph.CO`, `astro-ph.GA`, `astro-ph.IM`, and `physics.comp-ph` for preprints matching your methodology (e.g. cosmic voids, simulation pipelines, MCMC, Euclid, DESI). Identifies senior authors and extracts them directly into your Advisor CRM.
* **💶 ERC & Research Grant Radar**: Tracks multi-million euro grant awards across Europe to spot PIs with active hiring budgets.
* **📡 Formal Job Aggregator**: Continuously pulls from INSPIRE-HEP and major European doctoral programs (IMPRS Garching, Durham ICC, Leiden Observatory, Geneva, Bologna).
* **🧑‍🏫 Academic CRM & Contextual Pitch Drafter**: Drafts tailored cold-email templates referencing the PI's recent paper and linking your specific thesis achievements.
* **📋 Application Kanban & Reference Letters**: Tracks deadlines, dossier preparation, and referee letter submission statuses.
* **🔄 Automated 8-Hour Sync Daemon**: Silently updates the database every 8 hours with new preprints and openings.
* **🔒 Privacy by Design**: All contact notes, email drafts, and personal data are kept locally in a gitignored SQLite database.

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/phd-tracker.git
cd phd-tracker

# Install in editable mode
pip install -e ".[ui]"
```

*Note: The core CLI and crawler run on pure Python standard library (`urllib`, `xml.etree`, `sqlite3`). Streamlit is optional for the visual UI.*

### 2. Configuration
Copy `config.example.yaml` to `config.yaml` and configure your research profile and keywords:
```bash
cp config.example.yaml config.yaml
```
Open `config.yaml` to customize your thesis summary, target regions, and subfield keywords.

### 3. Initialize & Fetch Data
```bash
# Initialize local SQLite database
phd-tracker init

# Run an immediate sync across ArXiv, job boards, and grants
phd-tracker sync

# Check status
phd-tracker status
```

### 4. Launch Interactive Dashboard
```bash
phd-tracker serve
```
Open your browser at `http://localhost:8501`.

---

## 🛠️ Command-Line Interface (CLI)

The pipeline can be operated entirely from the terminal:

| Command | Description |
|---|---|
| `phd-tracker init` | Initialize or migrate local SQLite database |
| `phd-tracker sync` | Manually run a single update cycle across all streams |
| `phd-tracker daemon` | Run continuous background updater (updates every 8 hours) |
| `phd-tracker status` | Display metrics, preprints mined, active jobs, and sync logs |
| `phd-tracker list-jobs --min-score 7.0` | View top-scoring advertised PhD positions |
| `phd-tracker list-papers --min-score 6.0` | View high-relevance ArXiv preprints |
| `phd-tracker pitch --advisor "Elena Sartori" --institution "MPA"` | Generate a cold-email draft |
| `phd-tracker serve` | Launch the Streamlit command center |

---

## ⏰ Running the 8-Hour Background Daemon

To keep your literature radar and openings updated every 8 hours continuously:

### Option A: Direct Terminal / Tmux
```bash
phd-tracker daemon
```

### Option B: macOS `launchd` Service (Runs automatically in background)
Create `~/Library/LaunchAgents/com.user.phdtracker.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.phdtracker</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>-m</string>
        <string>phd_tracker.cli</string>
        <string>sync</string>
    </array>
    <key>StartInterval</key>
    <integer>28800</integer> <!-- 8 hours in seconds -->
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
```
Load the job:
```bash
launchctl load ~/Library/LaunchAgents/com.user.phdtracker.plist
```

---

## 🧩 Transferable Methodology Matching

The matching engine uses a 2-tier weighted taxonomy to discover positions across adjacent domains:
* **Tier 1 (Cosmology & LSS)**: Cosmic voids, void size function (VSF), N-body simulations, parameter inference, CosmoBolognaLib, Aletheia, Euclid, DESI, Rubin/LSST.
* **Tier 2 (Adjacent Astro & Methods)**: Galaxy clustering, weak lensing, simulation-based inference (SBI), neural posterior estimation, halo mass functions, point processes, topological data analysis.

---

## 🤝 Contributing

Contributions from astronomers, physicists, and computer scientists are warmly welcomed:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewPortalScraper`)
3. Commit your Changes (`git commit -m 'Add Hypatia EAS scraper'`)
4. Push to the Branch (`git push origin feature/NewPortalScraper`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
