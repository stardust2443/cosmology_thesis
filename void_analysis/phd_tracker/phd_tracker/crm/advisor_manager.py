"""Advisor pipeline and outreach management."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from ..database.models import (
    Advisor,
    Application,
    get_advisors,
    save_advisor,
    save_application,
)


class AdvisorManager:
    """Manages advisor pipeline, relationship statuses, and follow-up reminders."""

    STATUS_STAGES = [
        "Uncontacted",
        "Drafting",
        "Email Sent",
        "Follow-up Needed",
        "Meeting Scheduled",
        "Funding Confirmed",
        "No Vacancy",
    ]

    def __init__(self, db_path: str):
        self.db_path = db_path

    def add_advisor_from_author(
        self,
        name: str,
        institution: str,
        research_areas: str = "",
        key_surveys_or_codes: str = "",
        country: str = "Europe",
        email: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Adds a new prospective advisor to the CRM."""
        advisor = Advisor(
            name=name,
            institution=institution,
            email=email,
            country=country,
            research_areas=research_areas,
            key_surveys_or_codes=key_surveys_or_codes,
            outreach_status="Uncontacted",
            notes=notes,
        )
        return save_advisor(self.db_path, advisor)

    def mark_email_sent(self, advisor_id: int, follow_up_days: int = 10) -> None:
        """Updates status to 'Email Sent' and automatically sets follow-up date."""
        today = datetime.now().strftime("%Y-%m-%d")
        follow_up = (datetime.now() + timedelta(days=follow_up_days)).strftime("%Y-%m-%d")

        advisors = get_advisors(self.db_path)
        target = next((a for a in advisors if a["id"] == advisor_id), None)
        if target:
            adv = Advisor(
                id=target["id"],
                name=target["name"],
                email=target["email"],
                institution=target["institution"],
                department=target["department"],
                country=target["country"],
                research_areas=target["research_areas"],
                key_surveys_or_codes=target["key_surveys_or_codes"],
                outreach_status="Email Sent",
                last_contact_date=today,
                follow_up_date=follow_up,
                notes=target["notes"],
            )
            save_advisor(self.db_path, adv)

    def promote_to_application(self, advisor_id: int, title: Optional[str] = None) -> int:
        """Promotes an advisor contact directly into an active application on the Kanban board."""
        advisors = get_advisors(self.db_path)
        target = next((a for a in advisors if a["id"] == advisor_id), None)
        if not target:
            raise ValueError(f"Advisor with ID {advisor_id} not found.")

        app_title = title or f"PhD Position with {target['name']}"
        app = Application(
            advisor_id=target["id"],
            title=app_title,
            institution=target["institution"],
            stage="Contacted PI",
            decision="Pending",
            notes=f"Created from Advisor CRM: {target['name']}. Prior notes: {target.get('notes', '')}",
        )
        return save_application(self.db_path, app)
