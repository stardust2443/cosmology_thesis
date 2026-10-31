"""Data models and CRUD operations for the PhD tracker database."""

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from .schema import get_connection


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    published_date: str
    matched_keywords: List[str]
    relevance_score: float
    id: Optional[int] = None
    primary_advisor_id: Optional[int] = None
    is_starred: bool = False


@dataclass
class Position:
    title: str
    institution: str
    url: str
    source: str
    department: Optional[str] = None
    country: Optional[str] = None
    deadline: Optional[str] = None
    funding_type: Optional[str] = None
    status: str = "inbox"
    relevance_score: float = 0.0
    matched_keywords: List[str] = None
    raw_description: Optional[str] = None
    id: Optional[int] = None


@dataclass
class Advisor:
    name: str
    institution: str
    id: Optional[int] = None
    email: Optional[str] = None
    department: Optional[str] = None
    country: Optional[str] = None
    research_areas: Optional[str] = None
    key_surveys_or_codes: Optional[str] = None
    outreach_status: str = "Uncontacted"
    last_contact_date: Optional[str] = None
    follow_up_date: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Grant:
    title: str
    funder: str
    id: Optional[int] = None
    project_acronym: Optional[str] = None
    pi_name: Optional[str] = None
    host_institution: Optional[str] = None
    country: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    budget_eur: Optional[float] = None
    summary: Optional[str] = None
    advisor_id: Optional[int] = None


@dataclass
class Application:
    title: str
    institution: str
    id: Optional[int] = None
    position_id: Optional[int] = None
    advisor_id: Optional[int] = None
    stage: str = "Identified"
    decision: str = "Pending"
    portal_url: Optional[str] = None
    submission_deadline: Optional[str] = None
    submitted_date: Optional[str] = None
    interview_date: Optional[str] = None
    notes: Optional[str] = None


def insert_or_update_arxiv_paper(db_path: str, paper: ArxivPaper) -> bool:
    """Inserts a paper or updates score/keywords if already exists. Returns True if new."""
    authors_str = json.dumps(paper.authors)
    categories_str = ",".join(paper.categories)
    keywords_str = ",".join(paper.matched_keywords or [])

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM arxiv_papers WHERE arxiv_id = ?", (paper.arxiv_id,))
        row = cursor.fetchone()

        if row:
            cursor.execute(
                """
                UPDATE arxiv_papers
                SET relevance_score = MAX(relevance_score, ?),
                    matched_keywords = ?
                WHERE id = ?
                """,
                (paper.relevance_score, keywords_str, row["id"]),
            )
            conn.commit()
            return False
        else:
            cursor.execute(
                """
                INSERT INTO arxiv_papers (
                    arxiv_id, title, abstract, authors_json, categories,
                    published_date, matched_keywords, relevance_score, is_starred
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper.arxiv_id, paper.title, paper.abstract, authors_str,
                    categories_str, paper.published_date, keywords_str,
                    paper.relevance_score, 1 if paper.is_starred else 0
                ),
            )
            conn.commit()
            return True


def get_arxiv_papers(
    db_path: str,
    min_score: float = 0.0,
    limit: int = 50,
    starred_only: bool = False,
    search_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves papers sorted by relevance score and recency."""
    query = "SELECT * FROM arxiv_papers WHERE relevance_score >= ?"
    params: List[Any] = [min_score]

    if starred_only:
        query += " AND is_starred = 1"

    if search_query:
        query += " AND (title LIKE ? OR abstract LIKE ? OR authors_json LIKE ?)"
        wildcard = f"%{search_query}%"
        params.extend([wildcard, wildcard, wildcard])

    query += " ORDER BY is_starred DESC, relevance_score DESC, published_date DESC LIMIT ?"
    params.append(limit)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["authors"] = json.loads(d["authors_json"])
            except Exception:
                d["authors"] = []
            results.append(d)
        return results


def toggle_star_arxiv_paper(db_path: str, paper_id: int) -> bool:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE arxiv_papers SET is_starred = 1 - is_starred WHERE id = ?", (paper_id,))
        conn.commit()
        cursor.execute("SELECT is_starred FROM arxiv_papers WHERE id = ?", (paper_id,))
        return bool(cursor.fetchone()["is_starred"])


def insert_or_update_position(db_path: str, pos: Position) -> bool:
    keywords_str = ",".join(pos.matched_keywords or [])
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM positions WHERE url = ?", (pos.url,))
        row = cursor.fetchone()
        if row:
            cursor.execute(
                """
                UPDATE positions
                SET relevance_score = MAX(relevance_score, ?),
                    matched_keywords = ?
                WHERE id = ?
                """,
                (pos.relevance_score, keywords_str, row["id"]),
            )
            conn.commit()
            return False
        else:
            cursor.execute(
                """
                INSERT INTO positions (
                    title, institution, department, country, deadline,
                    funding_type, source, url, status, relevance_score,
                    matched_keywords, raw_description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pos.title, pos.institution, pos.department, pos.country,
                    pos.deadline, pos.funding_type, pos.source, pos.url,
                    pos.status, pos.relevance_score, keywords_str, pos.raw_description
                ),
            )
            conn.commit()
            return True


def get_positions(
    db_path: str,
    status: Optional[str] = None,
    min_score: float = 0.0,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    query = "SELECT * FROM positions WHERE relevance_score >= ?"
    params: List[Any] = [min_score]
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY relevance_score DESC, deadline ASC LIMIT ?"
    params.append(limit)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(r) for r in cursor.fetchall()]


def update_position_status(db_path: str, position_id: int, new_status: str) -> None:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE positions SET status = ? WHERE id = ?", (new_status, position_id))
        conn.commit()


def save_advisor(db_path: str, advisor: Advisor) -> int:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if advisor.id:
            cursor.execute(
                """
                UPDATE advisors
                SET name = ?, email = ?, institution = ?, department = ?, country = ?,
                    research_areas = ?, key_surveys_or_codes = ?, outreach_status = ?,
                    last_contact_date = ?, follow_up_date = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    advisor.name, advisor.email, advisor.institution, advisor.department,
                    advisor.country, advisor.research_areas, advisor.key_surveys_or_codes,
                    advisor.outreach_status, advisor.last_contact_date, advisor.follow_up_date,
                    advisor.notes, advisor.id
                ),
            )
            conn.commit()
            return advisor.id
        else:
            cursor.execute(
                """
                INSERT INTO advisors (
                    name, email, institution, department, country, research_areas,
                    key_surveys_or_codes, outreach_status, last_contact_date,
                    follow_up_date, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    advisor.name, advisor.email, advisor.institution, advisor.department,
                    advisor.country, advisor.research_areas, advisor.key_surveys_or_codes,
                    advisor.outreach_status, advisor.last_contact_date, advisor.follow_up_date,
                    advisor.notes
                ),
            )
            conn.commit()
            return cursor.lastrowid


def get_advisors(db_path: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
    query = "SELECT * FROM advisors"
    params: List[Any] = []
    if status:
        query += " WHERE outreach_status = ?"
        params.append(status)
    query += " ORDER BY updated_at DESC"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(r) for r in cursor.fetchall()]


def insert_grant(db_path: str, grant: Grant) -> bool:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM grants WHERE title = ? AND host_institution = ?",
            (grant.title, grant.host_institution)
        )
        if cursor.fetchone():
            return False
        cursor.execute(
            """
            INSERT INTO grants (
                project_acronym, title, funder, pi_name, host_institution,
                country, start_date, end_date, budget_eur, summary, advisor_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                grant.project_acronym, grant.title, grant.funder, grant.pi_name,
                grant.host_institution, grant.country, grant.start_date, grant.end_date,
                grant.budget_eur, grant.summary, grant.advisor_id
            )
        )
        conn.commit()
        return True


def get_grants(db_path: str, limit: int = 50) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM grants ORDER BY start_date DESC LIMIT ?", (limit,))
        return [dict(r) for r in cursor.fetchall()]


def save_application(db_path: str, app: Application) -> int:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if app.id:
            cursor.execute(
                """
                UPDATE applications
                SET title = ?, institution = ?, stage = ?, decision = ?, portal_url = ?,
                    submission_deadline = ?, submitted_date = ?, interview_date = ?,
                    notes = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    app.title, app.institution, app.stage, app.decision, app.portal_url,
                    app.submission_deadline, app.submitted_date, app.interview_date,
                    app.notes, app.id
                )
            )
            conn.commit()
            return app.id
        else:
            cursor.execute(
                """
                INSERT INTO applications (
                    position_id, advisor_id, title, institution, stage, decision,
                    portal_url, submission_deadline, submitted_date, interview_date, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    app.position_id, app.advisor_id, app.title, app.institution, app.stage,
                    app.decision, app.portal_url, app.submission_deadline, app.submitted_date,
                    app.interview_date, app.notes
                )
            )
            conn.commit()
            return cursor.lastrowid


def get_applications(db_path: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
    query = "SELECT * FROM applications"
    params: List[Any] = []
    if stage:
        query += " WHERE stage = ?"
        params.append(stage)
    query += " ORDER BY submission_deadline ASC, updated_at DESC"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(r) for r in cursor.fetchall()]


def log_sync_event(
    db_path: str,
    source: str,
    records_fetched: int,
    status: str,
    message: str,
    started_at: str,
    finished_at: str,
    duration_sec: float,
) -> None:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO sync_logs (
                source, records_fetched, status, message, started_at, finished_at, duration_sec
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (source, records_fetched, status, message, started_at, finished_at, duration_sec)
        )
        conn.commit()


def get_sync_status(db_path: str) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sync_logs ORDER BY finished_at DESC LIMIT 10")
        return [dict(r) for r in cursor.fetchall()]


def save_letter_request(
    db_path: str,
    app_id: int,
    referee_name: str,
    referee_email: Optional[str] = None,
    status: str = "To Request",
    requested_date: Optional[str] = None,
    submitted_date: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO recommendation_letters (
                application_id, referee_name, referee_email, status, requested_date, submitted_date, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (app_id, referee_name, referee_email, status, requested_date, submitted_date, notes)
        )
        conn.commit()
        return cursor.lastrowid


def get_letters_for_application(db_path: str, app_id: int) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM recommendation_letters WHERE application_id = ?", (app_id,))
        return [dict(r) for r in cursor.fetchall()]


def get_all_letters(db_path: str) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT l.*, a.title AS app_title, a.institution AS app_institution, a.submission_deadline
            FROM recommendation_letters l
            JOIN applications a ON l.application_id = a.id
            ORDER BY a.submission_deadline ASC
        """)
        return [dict(r) for r in cursor.fetchall()]
