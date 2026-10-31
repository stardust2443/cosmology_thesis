"""Relevance scoring and transferable methodology matching engine."""

import re
from typing import Dict, List, Set, Tuple


class MethodologyMatcher:
    """Evaluates text against multi-tiered keywords, survey acronyms, and methods."""

    def __init__(self, config: Dict):
        self.config = config
        kw = config.get("keywords", {})
        self.tier_1 = [k.lower() for k in kw.get("tier_1", [])]
        self.tier_2 = [k.lower() for k in kw.get("tier_2", [])]
        self.exclude = [k.lower() for k in kw.get("exclude", [])]
        self.preferred_regions = [r.lower() for r in config.get("user_profile", {}).get("preferred_regions", [])]

    def score_text(self, title: str, text: str) -> Tuple[float, List[str]]:
        """
        Computes a relevance score (0.0 - 10.0) and returns matched keyword list.
        Title matches receive double weight.
        """
        full_text = f"{title}\n{text}".lower()
        title_lower = title.lower()

        # Check negative exclusion keywords first
        for ex in self.exclude:
            # Word boundary match for exclusions
            if re.search(r"\b" + re.escape(ex) + r"\b", full_text):
                return 0.0, [f"Excluded: {ex}"]

        matched_keywords: Set[str] = set()
        raw_score = 0.0

        # Tier 1 matches (Methodology & Core Domain)
        for term in self.tier_1:
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, title_lower):
                raw_score += 4.0
                matched_keywords.add(term)
            elif re.search(pattern, full_text):
                raw_score += 2.5
                matched_keywords.add(term)

        # Tier 2 matches (Adjacent Topics)
        for term in self.tier_2:
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, title_lower):
                raw_score += 2.0
                matched_keywords.add(term)
            elif re.search(pattern, full_text):
                raw_score += 1.2
                matched_keywords.add(term)

        # Bonus for survey acronyms in title/text
        surveys = ["euclid", "desi", "rubin", "lsst", "aletheia", "cosmobolognalib", "cbl", "bitvf"]
        for s in surveys:
            if re.search(r"\b" + re.escape(s) + r"\b", full_text):
                raw_score += 1.0
                matched_keywords.add(s.upper())

        # Normalize score to 0.0 - 10.0
        final_score = min(round(raw_score, 1), 10.0)
        return final_score, sorted(list(matched_keywords))

    def detect_country(self, text: str) -> str:
        """Heuristic detector for European / UK institution locations."""
        text_lower = text.lower()
        country_patterns = {
            "Germany": ["germany", "deutschland", "max planck", "garching", "heidelberg", "munich", "bonn", "potsdam", "mpa", "mpe", "mpia"],
            "United Kingdom": ["united kingdom", "uk", "durham", "edinburgh", "cambridge", "oxford", "ucl", "portsmouth", "sussex", "manchester", "stfc"],
            "Netherlands": ["netherlands", "leiden", "groningen", "amsterdam", "utrecht", "radboud", "nwo"],
            "Switzerland": ["switzerland", "epfl", "eth zurich", "geneva", "unige", "lausanne", "zurich"],
            "France": ["france", "iap", "paris", "saclay", "marseille", "cnrs", "cea"],
            "Italy": ["italy", "italia", "sissa", "bologna", "trieste", "inaf", "padova", "roma"],
            "Spain": ["spain", "españa", "iac", "ift", "barcelona", "madrid", "canarias"],
        }
        for country, keywords in country_patterns.items():
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                    return country
        return "International / Other"
