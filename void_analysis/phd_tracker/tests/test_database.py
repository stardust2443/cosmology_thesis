"""Database unit tests for PhD Tracker."""

import os
import tempfile
from phd_tracker.database import (
    Advisor,
    Application,
    ArxivPaper,
    Grant,
    Position,
    get_advisors,
    get_all_letters,
    get_applications,
    get_arxiv_papers,
    get_grants,
    get_positions,
    get_sync_status,
    init_db,
    insert_grant,
    insert_or_update_arxiv_paper,
    insert_or_update_position,
    log_sync_event,
    save_advisor,
    save_application,
    save_letter_request,
    toggle_star_arxiv_paper,
    update_position_status,
)


def test_database_flow():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name

    try:
        init_db(temp_db)

        # 1. Test ArXiv Paper insertion
        paper = ArxivPaper(
            arxiv_id="2410.99999",
            title="Void Size Functions in Aletheia Simulations",
            abstract="We present cosmological inference using cosmic voids and CBL native sampler.",
            authors=["Arun", "Elena Sartori"],
            categories=["astro-ph.CO"],
            published_date="2026-10-20",
            matched_keywords=["cosmic voids", "N-body simulations", "MCMC"],
            relevance_score=9.5,
        )
        is_new = insert_or_update_arxiv_paper(temp_db, paper)
        assert is_new is True

        papers = get_arxiv_papers(temp_db, min_score=8.0)
        assert len(papers) == 1
        assert papers[0]["arxiv_id"] == "2410.99999"

        # Star toggle
        is_starred = toggle_star_arxiv_paper(temp_db, papers[0]["id"])
        assert is_starred is True

        # 2. Test Position insertion & status
        pos = Position(
            title="PhD in Void Cosmology",
            institution="Leiden University",
            url="https://leiden.edu/phd/1",
            source="test",
            relevance_score=8.5,
            country="Netherlands",
            deadline="2026-12-01",
        )
        assert insert_or_update_position(temp_db, pos) is True
        positions = get_positions(temp_db, min_score=8.0)
        assert len(positions) == 1

        update_position_status(temp_db, positions[0]["id"], "shortlisted")
        shortlisted = get_positions(temp_db, status="shortlisted")
        assert len(shortlisted) == 1

        # 3. Test Advisor CRM
        adv_id = save_advisor(
            temp_db,
            Advisor(name="Prof. Elena Sartori", institution="MPA Garching", country="Germany"),
        )
        advisors = get_advisors(temp_db)
        assert len(advisors) == 1
        assert advisors[0]["id"] == adv_id

        # 4. Test Application & Letters
        app_id = save_application(
            temp_db,
            Application(
                position_id=positions[0]["id"],
                advisor_id=adv_id,
                title="PhD in Void Cosmology",
                institution="Leiden University",
                stage="Identified",
            ),
        )
        apps = get_applications(temp_db)
        assert len(apps) == 1

        save_letter_request(temp_db, app_id=app_id, referee_name="Prof. Advisor", status="Requested")
        letters = get_all_letters(temp_db)
        assert len(letters) == 1
        assert letters[0]["referee_name"] == "Prof. Advisor"

        # 5. Test Sync Log
        log_sync_event(
            temp_db,
            source="test_runner",
            records_fetched=10,
            status="success",
            message="OK",
            started_at="2026-10-22T10:00:00",
            finished_at="2026-10-22T10:00:02",
            duration_sec=2.0,
        )
        logs = get_sync_status(temp_db)
        assert len(logs) == 1
        assert logs[0]["records_fetched"] == 10

        print("✓ All database unit tests passed!")
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)


if __name__ == "__main__":
    test_database_flow()
