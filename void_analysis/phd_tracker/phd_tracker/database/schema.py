"""Database schema initialization and migration for PhD Application Tracker."""

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS advisors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT,
    institution TEXT,
    department TEXT,
    country TEXT,
    research_areas TEXT,
    key_surveys_or_codes TEXT,
    outreach_status TEXT DEFAULT 'Uncontacted',
    last_contact_date TEXT,
    follow_up_date TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS arxiv_papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    authors_json TEXT,
    categories TEXT,
    published_date TEXT,
    matched_keywords TEXT,
    relevance_score REAL DEFAULT 0.0,
    primary_advisor_id INTEGER,
    is_starred INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(primary_advisor_id) REFERENCES advisors(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    institution TEXT NOT NULL,
    department TEXT,
    country TEXT,
    deadline TEXT,
    funding_type TEXT,
    source TEXT,
    url TEXT UNIQUE,
    status TEXT DEFAULT 'inbox',
    relevance_score REAL DEFAULT 0.0,
    matched_keywords TEXT,
    raw_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS grants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_acronym TEXT,
    title TEXT NOT NULL,
    funder TEXT,
    pi_name TEXT,
    host_institution TEXT,
    country TEXT,
    start_date TEXT,
    end_date TEXT,
    budget_eur REAL,
    summary TEXT,
    advisor_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(advisor_id) REFERENCES advisors(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER,
    advisor_id INTEGER,
    title TEXT NOT NULL,
    institution TEXT NOT NULL,
    stage TEXT DEFAULT 'Identified',
    decision TEXT DEFAULT 'Pending',
    portal_url TEXT,
    submission_deadline TEXT,
    submitted_date TEXT,
    interview_date TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(position_id) REFERENCES positions(id) ON DELETE SET NULL,
    FOREIGN KEY(advisor_id) REFERENCES advisors(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS recommendation_letters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER NOT NULL,
    referee_name TEXT NOT NULL,
    referee_email TEXT,
    status TEXT DEFAULT 'To Request',
    requested_date TEXT,
    submitted_date TEXT,
    notes TEXT,
    FOREIGN KEY(application_id) REFERENCES applications(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sync_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    records_fetched INTEGER DEFAULT 0,
    status TEXT NOT NULL,
    message TEXT,
    started_at TEXT,
    finished_at TEXT,
    duration_sec REAL
);

-- Indexes for lightning-fast queries
CREATE INDEX IF NOT EXISTS idx_arxiv_date ON arxiv_papers(published_date);
CREATE INDEX IF NOT EXISTS idx_arxiv_score ON arxiv_papers(relevance_score);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_deadline ON positions(deadline);
CREATE INDEX IF NOT EXISTS idx_advisors_status ON advisors(outreach_status);
CREATE INDEX IF NOT EXISTS idx_applications_stage ON applications(stage);
"""


def init_db(db_path: str) -> None:
    """Initializes the SQLite database with full schema and indexes."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def get_connection(db_path: str) -> sqlite3.Connection:
    """Returns a connection with Row factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn
