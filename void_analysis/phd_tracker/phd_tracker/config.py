"""Configuration loader for the PhD Application & Academic Intelligence Pipeline."""

import os
from pathlib import Path
from typing import Any, Dict, List

# Default search directories for config.yaml
DEFAULT_CONFIG_LOCATIONS = [
    Path("config.yaml"),
    Path(__file__).parent.parent / "config.yaml",
    Path.home() / ".config" / "phd_tracker" / "config.yaml",
]

DEFAULT_CONFIG: Dict[str, Any] = {
    "sync": {
        "interval_hours": 8,
        "auto_sync": True,
        "notify_on_new": True,
    },
    "user_profile": {
        "name": "Arun",
        "primary_focus": "Cosmology & Large-Scale Structure",
        "target_degree": "PhD",
        "target_start": "2025/2026",
        "preferred_regions": ["Germany", "UK", "Netherlands", "Switzerland", "France", "Italy"],
        "thesis_summary": "Cosmic voids, VSF, N-body simulations (Aletheia), and MCMC cosmological inference.",
        "key_skills": ["Cosmic voids", "N-body simulations", "MCMC", "Survey science"],
    },
    "keywords": {
        "tier_1": [
            "cosmic voids", "void size function", "large-scale structure",
            "cosmological inference", "N-body simulations", "Euclid", "DESI",
            "Rubin", "LSST", "MCMC", "CosmoBolognaLib", "Aletheia", "BitVF"
        ],
        "tier_2": [
            "galaxy clustering", "weak lensing", "halo mass function",
            "simulation-based inference", "neural posterior estimation",
            "cosmic web", "dark energy", "modified gravity", "mock catalogs"
        ],
        "exclude": ["stellar evolution", "exoplanet atmosphere", "heliophysics"],
    },
    "arxiv": {
        "categories": ["astro-ph.CO", "astro-ph.GA", "astro-ph.IM", "physics.comp-ph", "stat.ML"],
        "max_results_per_query": 40,
        "days_back": 180,
    },
    "sources": {
        "inspire_hep": True,
        "aas_jobs": True,
        "academic_transfer": True,
        "euraxess": True,
        "erc_grants": True,
    },
    "database": {
        "path": "data/tracker.db",
    },
}


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Loads configuration from YAML file, merging with default fallback."""
    cfg = DEFAULT_CONFIG.copy()

    target_path = None
    if config_path and Path(config_path).is_file():
        target_path = Path(config_path)
    else:
        for loc in DEFAULT_CONFIG_LOCATIONS:
            if loc.is_file():
                target_path = loc
                break

    if target_path:
        try:
            import yaml
            with open(target_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded and isinstance(loaded, dict):
                    # Recursive merge
                    for k, v in loaded.items():
                        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                            cfg[k].update(v)
                        else:
                            cfg[k] = v
        except ImportError:
            # Simple fallback if pyyaml is missing
            pass
        except Exception as e:
            print(f"[Warning] Failed to parse {target_path}: {e}. Using defaults.")

    # Ensure data directory exists
    db_path = Path(cfg["database"]["path"])
    if not db_path.is_absolute():
        # resolve relative to repo root
        root_dir = Path(__file__).parent.parent
        db_path = root_dir / db_path
        cfg["database"]["path"] = str(db_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    return cfg
