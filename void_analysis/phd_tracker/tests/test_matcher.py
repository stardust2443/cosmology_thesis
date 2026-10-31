"""Matcher unit tests."""

from phd_tracker.config import DEFAULT_CONFIG
from phd_tracker.engines.matcher import MethodologyMatcher


def test_matcher_scoring():
    matcher = MethodologyMatcher(DEFAULT_CONFIG)

    # High match test
    title = "Constraining Cosmological Parameters with Cosmic Voids and Euclid Mock Catalogs"
    text = "We use N-body simulations and MCMC to infer Omega_m and sigma_8 via the void size function."
    score, kws = matcher.score_text(title, text)
    assert score >= 8.0
    assert "cosmic voids" in kws
    assert "EUCLID" in kws
    assert "n-body simulations" in [k.lower() for k in kws]

    # Exclusion test
    ex_title = "Observations of Exoplanet Atmosphere and Solar Flare Dynamics"
    ex_text = "Heliophysics and stellar evolution analysis."
    ex_score, ex_kws = matcher.score_text(ex_title, ex_text)
    assert ex_score == 0.0
    assert any("Excluded" in k for k in ex_kws)

    # Country detection
    country = matcher.detect_country("Leiden Observatory, Sterrewacht, Netherlands")
    assert country == "Netherlands"

    country_de = matcher.detect_country("Max Planck Institute for Astrophysics, Garching")
    assert country_de == "Germany"

    country_uk = matcher.detect_country("Institute for Computational Cosmology, Durham University, UK")
    assert country_uk == "United Kingdom"

    print("✓ All matcher unit tests passed!")


if __name__ == "__main__":
    test_matcher_scoring()
