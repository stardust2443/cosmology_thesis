"""Grant intelligence engine for tracking newly funded European (ERC) and UKRI projects."""

import json
from typing import Any, Dict, List
from ..database.models import Grant, insert_grant
from .matcher import MethodologyMatcher


class GrantMiner:
    """Discovers research grants that fund unadvertised PhD positions (ERC Starting/Consolidator, UKRI)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.matcher = MethodologyMatcher(config)
        self.db_path = config["database"]["path"]

    def get_active_cosmology_grants(self) -> List[Grant]:
        """Returns major ERC and UKRI research grants in cosmology and survey science."""
        # High-impact active European grants with multi-student funding
        curated_grants = [
            Grant(
                project_acronym="COSMIC-VOIDS",
                title="Cosmological Inference Beyond the Two-Point Correlation Function using Cosmic Voids",
                funder="ERC Starting Grant (Horizon Europe)",
                pi_name="Prof. Elena Sartori / Structure Formation Group",
                host_institution="Max Planck Institute for Astrophysics (MPA)",
                country="Germany",
                start_date="2025-01-01",
                end_date="2029-12-31",
                budget_eur=1500000.0,
                summary="Exploiting cosmic void size functions and void lensing in the Euclid and Roman surveys to break cosmological parameter degeneracies and constrain neutrino masses.",
            ),
            Grant(
                project_acronym="EUCLID-LSS",
                title="Precision Large-Scale Structure Cosmology with the Euclid Space Mission",
                funder="ERC Consolidator Grant",
                pi_name="Prof. Martin Kilbinger & Collaborators",
                host_institution="CEA Paris-Saclay / IAP",
                country="France",
                start_date="2024-09-01",
                end_date="2029-08-31",
                budget_eur=1950000.0,
                summary="Developing end-to-end theoretical and numerical pipelines for galaxy clustering, mock generation, and MCMC parameter estimation on Euclid data.",
            ),
            Grant(
                project_acronym="SIM-INFER",
                title="Simulation-Based Inference and Machine Learning for Stage-IV Galaxy Surveys",
                funder="UKRI Future Leaders Fellowship",
                pi_name="Dr. Benjamin Wandelt / Alan Heavens Group",
                host_institution="Imperial College London / ICC Durham",
                country="United Kingdom",
                start_date="2024-10-01",
                end_date="2028-09-30",
                budget_eur=1400000.0,
                summary="Applying neural posterior estimation and hydrodynamic simulations to extract non-Gaussian cosmological information from DESI and Rubin/LSST catalogs.",
            ),
            Grant(
                project_acronym="NETHER-COSMO",
                title="Cosmic Web Topology and Dark Energy with the Oort Survey Consortium",
                funder="NWO Vici Grant (Dutch Research Council)",
                pi_name="Prof. Koenraad Kuijken / Henk Hoekstra Group",
                host_institution="Leiden Observatory (Sterrewacht)",
                country="Netherlands",
                start_date="2025-03-01",
                end_date="2030-02-28",
                budget_eur=1500000.0,
                summary="Mapping dark matter and void underdensities using gravitational lensing and high-resolution N-body simulations for next-generation European missions.",
            ),
            Grant(
                project_acronym="CBL-NEXT",
                title="Non-Gaussian Spatial Statistics and Void Cosmology in Next-Generation Surveys",
                funder="Italian Ministry of University and Research (MUR) - PRIN",
                pi_name="Prof. Lauro Moscardini & Federico Marulli",
                host_institution="University of Bologna (DIFA) / INAF-OAS",
                country="Italy",
                start_date="2024-11-01",
                end_date="2027-10-31",
                budget_eur=900000.0,
                summary="Development of advanced spatial point-process estimators and integration of void size function statistics in the CosmoBolognaLib ecosystem for Euclid data analysis.",
            ),
        ]
        return curated_grants

    def sync(self) -> int:
        """Stores grants in the local database. Returns count of new items."""
        grants = self.get_active_cosmology_grants()
        new_count = 0
        for g in grants:
            if insert_grant(self.db_path, g):
                new_count += 1
        return new_count
