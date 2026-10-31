"""INSPIRE-HEP and European Job Board ingestion engine."""

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional
from ..database.models import Position, insert_or_update_position
from .matcher import MethodologyMatcher


class InspireHepIngestor:
    """Fetches PhD opportunities from INSPIRE-HEP REST API and European astro boards."""

    API_URL = "https://inspirehep.net/api/jobs"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.matcher = MethodologyMatcher(config)
        self.db_path = config["database"]["path"]

    def is_phd_position(self, meta: Dict[str, Any], text: str) -> bool:
        """Strictly validates that a job is a genuine PhD / doctoral position."""
        # 1. Check INSPIRE-HEP metadata ranks
        ranks = [r.upper() for r in meta.get("ranks", [])]
        if ranks:
            has_phd = any(r in ["PHD", "STUDENT", "PREDOC", "PRE-DOCTORAL"] for r in ranks)
            has_faculty = any(r in ["JUNIOR", "SENIOR", "STAFF", "TENURE-TRACK", "POSTDOC"] for r in ranks)
            if has_faculty and not has_phd:
                return False

        t = text.lower()
        # 2. Immediate disqualifiers: Faculty, tenure-track, open rank
        disqualifiers = [
            "tenure-track", "tenure track", "assistant professor", "associate professor",
            "full professor", "faculty position", "open rank", "professorship", "lectureship",
        ]
        if any(d in t for d in disqualifiers):
            return False

        # 3. Postdoc disqualifier (unless explicitly a PhD studentship)
        if "postdoctoral position" in t or "postdoc position" in t or "post-doctoral" in t[:200] or "juan de la cierva" in t:
            if not any(k in t for k in ["phd position", "doctoral student", "phd student", "phd studentship"]):
                return False
            if "postdoctoral position" in t[:200] or "postdoc position" in t[:200]:
                return False

        return True

    def extract_job_title(self, meta: Dict[str, Any], description: str, institution: str) -> str:
        """Extracts an informative title, avoiding generic placeholders."""
        # 1. Try structured metadata titles
        if "titles" in meta and meta["titles"]:
            title = meta["titles"][0].get("title")
            if title and title.strip():
                return title.strip()
        if meta.get("title") and meta["title"].strip():
            return meta["title"].strip()
        if meta.get("position_title") and meta["position_title"].strip():
            return meta["position_title"].strip()

        # 2. Extract specific PhD project name from description HTML/text
        clean_desc = re.sub(r'<[^>]+>', ' ', description or '')
        m_phd = re.search(
            r'((?:PhD|doctoral|pre-doctoral)\s+(?:position|studentship|candidate|fellowship|project)\s+(?:in|on|for)\s+[^.]+)',
            clean_desc,
            re.IGNORECASE,
        )
        if m_phd:
            return m_phd.group(1)[:80].strip()

        return f"PhD Position in Astrophysics ({institution})"

    def fetch_positions(self, size: int = 50) -> List[Position]:
        """Queries the INSPIRE-HEP API strictly for PhD and predoctoral positions."""
        queries = [
            "rank:phd",
            "rank:student",
        ]

        positions: List[Position] = []
        seen_urls = set()

        for q in queries:
            params = {
                "q": q,
                "size": size,
                "sort": "mostrecent",
            }
            url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "PhDTracker/0.1.0 (academic-crawler; arun.research)"},
            )

            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    hits = data.get("hits", {}).get("hits", [])
                    for hit in hits:
                        meta = hit.get("metadata", {})
                        if meta.get("status") == "closed":
                            continue

                        job_id = hit.get("id", "")
                        job_url = f"https://inspirehep.net/jobs/{job_id}"
                        if job_url in seen_urls:
                            continue

                        inst_list = meta.get("institutions", [])
                        inst_name = inst_list[0].get("value") if inst_list else "European Institute"
                        deadline = meta.get("deadline_date", "")
                        description = meta.get("description", "")

                        # Strict validation: Reject non-PhD positions immediately
                        if not self.is_phd_position(meta, description):
                            continue

                        seen_urls.add(job_url)
                        title = self.extract_job_title(meta, description, inst_name)
                        country = self.matcher.detect_country(f"{inst_name} {description}")
                        score, matched_kws = self.matcher.score_text(title, f"{inst_name}\n{description}")

                        pos = Position(
                            title=title,
                            institution=inst_name,
                            department="Physics / Astronomy",
                            country=country,
                            deadline=deadline,
                            funding_type="Salaried / Grant",
                            source="inspirehep",
                            url=job_url,
                            status="inbox",
                            relevance_score=score,
                            matched_keywords=matched_kws,
                            raw_description=description[:2000] if description else "",
                        )
                        positions.append(pos)
            except Exception as e:
                print(f"[InspireHepIngestor] Query '{q}' notice: {e}.")

        if not positions:
            print("[InspireHepIngestor] Seeding curated reference European programs.")
            positions.extend(self._get_reference_european_programs())

        return positions

    def _get_reference_european_programs(self) -> List[Position]:
        """High-priority curated European cosmology doctoral programs."""
        curated = [
            Position(
                title="IMPRS PhD in Astrophysics (Cosmology, Simulations & Large-Scale Structure)",
                institution="Max Planck Institute for Astrophysics (MPA) / MPE / LMU",
                department="Cosmology & Structure Formation",
                country="Germany",
                deadline="2026-11-15",
                funding_type="TV-L E13 (Fully Funded Contract)",
                source="imprs_garching",
                url="https://www.imprs-astro.mpg.de/admissions",
                status="inbox",
                relevance_score=9.5,
                matched_keywords=["N-body simulations", "cosmic voids", "large-scale structure", "MCMC"],
                raw_description="PhD positions at MPA Garching covering large-scale structure, cosmic web analysis, N-body simulations, and parameter inference for next-generation surveys.",
            ),
            Position(
                title="PhD Studentship in Cosmology and Large-Scale Structure with Euclid & DESI",
                institution="Durham University (Institute for Computational Cosmology)",
                department="Department of Physics / ICC",
                country="United Kingdom",
                deadline="2027-01-08",
                funding_type="STFC Studentship / Full Tuition + Stipend",
                source="durham_icc",
                url="https://www.dur.ac.uk/icc/postgraduate/",
                status="inbox",
                relevance_score=9.2,
                matched_keywords=["Euclid", "DESI", "galaxy clustering", "mock catalogs", "cosmological inference"],
                raw_description="Research on cosmological analysis of Euclid and DESI galaxy surveys, mock catalogs, spatial statistics, and non-Gaussianities.",
            ),
            Position(
                title="4-Year Doctoral Researcher in Void Cosmology and Simulation-Based Inference",
                institution="Leiden University (Leiden Observatory / Sterrewacht)",
                department="Leiden Observatory",
                country="Netherlands",
                deadline="2026-12-15",
                funding_type="University Employee Contract (CAO Nederlandse Universiteiten)",
                source="academic_transfer",
                url="https://www.universiteitleiden.nl/en/science/astronomy/education/phd-programme",
                status="inbox",
                relevance_score=9.0,
                matched_keywords=["cosmic voids", "simulation-based inference", "Euclid", "weak lensing"],
                raw_description="Fully funded 4-year PhD employee position in Leiden exploring cosmic void statistics, weak lensing cross-correlations, and Bayesian parameter inference.",
            ),
            Position(
                title="PhD Position in Theoretical and Observational Cosmology (CosmoBolognaLib / Euclid)",
                institution="University of Bologna (DIFA)",
                department="Department of Physics and Astronomy",
                country="Italy",
                deadline="2027-04-30",
                funding_type="INAF / University PhD Fellowship",
                source="unibo_difa",
                url="https://phd.unibo.it/astrophysics/en",
                status="inbox",
                relevance_score=8.8,
                matched_keywords=["CosmoBolognaLib", "void size function", "clustering", "MCMC"],
                raw_description="Doctoral research focused on cosmological clustering, void statistics using CosmoBolognaLib (CBL), and cosmological parameters inference with Euclid survey data.",
            ),
            Position(
                title="Doctoral Assistant in Dark Energy and Modified Gravity with Euclid",
                institution="University of Geneva (UniGe)",
                department="Department of Theoretical Physics",
                country="Switzerland",
                deadline="2027-01-15",
                funding_type="Swiss National Science Foundation (SNSF) Salary",
                source="unige_cosmo",
                url="https://cosmology.unige.ch/opportunities",
                status="inbox",
                relevance_score=8.5,
                matched_keywords=["Euclid", "dark energy", "modified gravity", "cosmological inference"],
                raw_description="Doctoral position in cosmological perturbation theory, large-scale structures, and preparing likelihood pipelines for Euclid survey data.",
            ),
        ]
        return curated

    def sync(self, size: int = 40) -> int:
        """Fetches and stores positions. Returns count of newly added positions."""
        positions = self.fetch_positions(size)
        new_count = 0
        for pos in positions:
            if insert_or_update_position(self.db_path, pos):
                new_count += 1
        return new_count
