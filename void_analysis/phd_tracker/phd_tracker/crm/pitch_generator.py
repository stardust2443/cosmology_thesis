"""Contextual pitch generator for academic cold-email outreach."""

from typing import Any, Dict, Optional


class AcademicPitchGenerator:
    """Generates personalized, high-yield academic outreach emails to prospective PhD advisors."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.profile = config.get("user_profile", {})
        self.user_name = self.profile.get("name", "Arun")
        self.thesis_summary = self.profile.get(
            "thesis_summary",
            "validation of BitVF back-in-time void finder on Aletheia N-body simulations and cosmological parameter inference via CosmoBolognaLib (CBL) MCMC."
        )

    def generate_pitch(
        self,
        advisor_name: str,
        institution: str,
        paper_title: Optional[str] = None,
        paper_arxiv_id: Optional[str] = None,
        grant_name: Optional[str] = None,
        specific_angle: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generates a tailored academic email draft based on European research norms:
        concise, technically grounded, and directly connecting previous research to their project.
        """
        # Determine hook
        if paper_title:
            subject = f"Prospective PhD Inquiry: Cosmic Voids & LSS Inference (re: {paper_title[:45]}...)"
            hook = (
                f"I recently read your preprint on arXiv ({f'arXiv:{paper_arxiv_id} - ' if paper_arxiv_id else ''}\"{paper_title}\") "
                f"with great enthusiasm. Your work on this problem strongly aligns with my own research interests in large-scale structure and simulation analysis."
            )
        elif grant_name:
            subject = f"Prospective PhD Inquiry: {grant_name} Project at {institution}"
            hook = (
                f"I have been following your research at {institution}, and was particularly excited to learn about your {grant_name} project. "
                f"Your group's focus on structure formation and precision cosmological analysis is a natural next step for my academic path."
            )
        else:
            subject = f"Prospective PhD Inquiry: Cosmology & Structure Formation Group at {institution}"
            hook = (
                f"I am writing to express my strong interest in pursuing a PhD under your supervision in the {institution} cosmology group, "
                f"starting in 2025/2026."
            )

        # Technical alignment paragraph
        method_alignment = (
            f"In my recent thesis research, I focused on {self.thesis_summary}. Specifically, I validated the void finder "
            f"against simulated N-body density fields (Aletheia box), constructed cleaned void size function (VSF) catalogs, "
            f"and executed joint MCMC parameter inference using the CBL native sampler to constrain (Omega_m, sigma_8) and break parameter degeneracies."
        )

        # Contribution pitch
        if specific_angle:
            contribution = (
                f"I would be thrilled to bring this background in void statistics, high-performance C++/Python simulation pipelines, "
                f"and MCMC inference to your group, particularly towards {specific_angle}."
            )
        else:
            contribution = (
                f"With my experience in mock catalogs, spatial clustering statistics, and Bayesian parameter inference, "
                f"I am eager to contribute immediately to your group's upcoming work on survey data (such as Euclid, DESI, or simulation mock suites)."
            )

        body = f"""Dear Professor {advisor_name.split()[-1]},

{hook}

{method_alignment}

{contribution}

Are you currently considering taking on new doctoral students for the upcoming cycle? I would welcome the opportunity to discuss any potential openings or project alignment for 10–15 minutes over Zoom at your convenience.

I have attached my academic CV and summary of research accomplishments. Thank you very much for your time and consideration.

Sincerely,

{self.user_name}
"""
        return {
            "subject": subject,
            "body": body,
        }
