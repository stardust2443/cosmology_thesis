"""Command-line interface for the PhD Application & Academic Intelligence Pipeline."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
# Support both package execution and direct script execution
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from phd_tracker.config import load_config
    from phd_tracker.crm.advisor_manager import AdvisorManager
    from phd_tracker.crm.pitch_generator import AcademicPitchGenerator
    from phd_tracker.database import (
        get_advisors,
        get_applications,
        get_arxiv_papers,
        get_grants,
        get_positions,
        get_sync_status,
        init_db,
    )
    from phd_tracker.scheduler.sync_daemon import SyncDaemon
else:
    from .config import load_config
    from .crm.advisor_manager import AdvisorManager
    from .crm.pitch_generator import AcademicPitchGenerator
    from .database import (
        get_advisors,
        get_applications,
        get_arxiv_papers,
        get_grants,
        get_positions,
        get_sync_status,
        init_db,
    )
    from .scheduler.sync_daemon import SyncDaemon


def cmd_init(args, cfg):
    db_path = cfg["database"]["path"]
    init_db(db_path)
    print(f"✓ Initialized database at: {db_path}")


def cmd_sync(args, cfg):
    db_path = cfg["database"]["path"]
    init_db(db_path)
    daemon = SyncDaemon(cfg)
    print("Fetching newest ArXiv preprints, job openings, and research grants...")
    results = daemon.run_sync_cycle()
    print("✓ Sync finished!")
    print(f"  - ArXiv papers added/updated: {results['arxiv']}")
    print(f"  - Job positions added/updated: {results['positions']}")
    print(f"  - Grants added/updated: {results['grants']}")


def cmd_daemon(args, cfg):
    db_path = cfg["database"]["path"]
    init_db(db_path)
    daemon = SyncDaemon(cfg)
    hours = cfg.get("sync", {}).get("interval_hours", 8)
    print(f"Starting continuous daemon (syncing every {hours} hours)... Press Ctrl+C to stop.")
    daemon.start_background()
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        daemon.stop()
        print("Daemon stopped.")


def cmd_status(args, cfg):
    db_path = cfg["database"]["path"]
    init_db(db_path)
    papers = get_arxiv_papers(db_path, limit=1000)
    positions = get_positions(db_path, limit=1000)
    advisors = get_advisors(db_path)
    applications = get_applications(db_path)
    sync_logs = get_sync_status(db_path)

    print("\n=== PhD Tracker & Intelligence Status ===")
    print(f"Database: {db_path}")
    print(f"Preprints tracked:    {len(papers)}")
    print(f"Job positions:        {len(positions)}")
    print(f"Advisors in CRM:      {len(advisors)}")
    print(f"Active applications:  {len(applications)}")
    print("------------------------------------------")
    if sync_logs:
        last = sync_logs[0]
        print(f"Last sync: {last['finished_at']} ({last['source']} -> {last['status']})")
    else:
        print("No sync history recorded yet. Run `phd-tracker sync`.")
    print("==========================================\n")


def cmd_list_jobs(args, cfg):
    db_path = cfg["database"]["path"]
    init_db(db_path)
    positions = get_positions(db_path, min_score=args.min_score, limit=args.limit)
    print(f"\nFound {len(positions)} job openings (min score: {args.min_score}):\n")
    for p in positions:
        print(f"[{p['relevance_score']:.1f}/10] {p['title']}")
        print(f"       Inst: {p['institution']} ({p['country']})")
        print(f"       Deadline: {p['deadline'] or 'Rolling'} | Funding: {p['funding_type']}")
        print(f"       URL: {p['url']}\n")


def cmd_list_papers(args, cfg):
    db_path = cfg["database"]["path"]
    init_db(db_path)
    papers = get_arxiv_papers(db_path, min_score=args.min_score, limit=args.limit)
    print(f"\nFound {len(papers)} preprints (min score: {args.min_score}):\n")
    for p in papers:
        star = "★ " if p["is_starred"] else ""
        print(f"{star}[{p['relevance_score']:.1f}/10] {p['title']}")
        print(f"       Authors: {', '.join(p['authors'][:3])}{'...' if len(p['authors']) > 3 else ''}")
        print(f"       ArXiv: arXiv:{p['arxiv_id']} | Date: {p['published_date']}")
        print(f"       Keywords: {p['matched_keywords']}\n")


def cmd_list_advisors(args, cfg):
    db_path = cfg["database"]["path"]
    init_db(db_path)
    advisors = get_advisors(db_path, status=args.status)
    print(f"\nTarget Advisors in CRM ({len(advisors)} found):\n")
    for a in advisors:
        print(f"• {a['name']} — {a['institution']} ({a.get('country') or 'Europe'})")
        print(f"  Status: {a['outreach_status']} | Follow-up: {a.get('follow_up_date') or 'None'}")
        if a.get("research_areas"):
            print(f"  Research: {a['research_areas']}")
        if a.get("notes"):
            print(f"  Notes: {a['notes']}")
        print()


def cmd_list_grants(args, cfg):
    db_path = cfg["database"]["path"]
    init_db(db_path)
    grants = get_grants(db_path, limit=args.limit)
    print(f"\nResearch Grants Radar ({len(grants)} found):\n")
    for g in grants:
        print(f"[{g.get('project_acronym') or 'GRANT'}] {g['title']}")
        print(f"  PI: {g.get('pi_name')} | Inst: {g.get('host_institution')} ({g.get('country')})")
        print(f"  Funder: {g.get('funder')} | Budget: €{g.get('budget_eur', 0):,.0f}")
        print()


def cmd_pitch(args, cfg):
    gen = AcademicPitchGenerator(cfg)
    draft = gen.generate_pitch(
        advisor_name=args.advisor,
        institution=args.institution,
        paper_title=args.paper,
        specific_angle=args.angle,
    )
    print("\n" + "=" * 50)
    print(f"SUBJECT: {draft['subject']}")
    print("=" * 50)
    print(draft["body"])
    print("=" * 50 + "\n")


def cmd_serve(args, cfg):
    try:
        import streamlit
    except ImportError:
        print("\n" + "!" * 60)
        print("Streamlit is required for the visual UI, but is not yet installed.")
        print(f"Install it with:  {sys.executable} -m pip install streamlit pandas")
        print("or run:          pip3 install streamlit pandas")
        print("!" * 60 + "\n")
        return

    app_path = Path(__file__).parent / "ui" / "app.py"
    print(f"Launching Streamlit dashboard from: {app_path}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    if args.port:
        cmd.extend(["--server.port", str(args.port)])
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        prog="phd-tracker",
        description="Autonomous Academic Intelligence & PhD Application Pipeline",
    )
    parser.add_argument("--config", "-c", help="Path to config.yaml", default=None)
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # init
    subparsers.add_parser("init", help="Initialize local SQLite database")

    # sync
    subparsers.add_parser("sync", help="Run a manual sync cycle across all sources")

    # daemon
    subparsers.add_parser("daemon", help="Run background 8-hour update daemon")

    # status
    subparsers.add_parser("status", help="Show system metrics and sync history")

    # list-jobs
    p_jobs = subparsers.add_parser("list-jobs", help="List matched job postings")
    p_jobs.add_argument("--min-score", type=float, default=0.0, help="Minimum relevance score")
    p_jobs.add_argument("--limit", type=int, default=20, help="Number of records to show")

    # list-papers
    p_papers = subparsers.add_parser("list-papers", help="List mined ArXiv preprints")
    p_papers.add_argument("--min-score", type=float, default=0.0, help="Minimum relevance score")
    p_papers.add_argument("--limit", type=int, default=20, help="Number of records to show")

    # list-advisors
    p_adv = subparsers.add_parser("list-advisors", help="List target advisors in CRM")
    p_adv.add_argument("--status", help="Filter by outreach status (e.g. Uncontacted, Email Sent)")

    # list-grants
    p_grants = subparsers.add_parser("list-grants", help="List funded ERC/UKRI research grants")
    p_grants.add_argument("--limit", type=int, default=20, help="Number of records to show")

    # pitch
    p_pitch = subparsers.add_parser("pitch", help="Draft a cold-email pitch to an advisor")
    p_pitch.add_argument("--advisor", required=True, help="Advisor full name")
    p_pitch.add_argument("--institution", required=True, help="Advisor institution")
    p_pitch.add_argument("--paper", help="Paper title to reference")
    p_pitch.add_argument("--angle", help="Specific scientific angle")

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch interactive Streamlit UI")
    p_serve.add_argument("--port", type=int, default=8501, help="Port to serve on")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "init":
        cmd_init(args, cfg)
    elif args.command == "sync":
        cmd_sync(args, cfg)
    elif args.command == "daemon":
        cmd_daemon(args, cfg)
    elif args.command == "status":
        cmd_status(args, cfg)
    elif args.command == "list-jobs":
        cmd_list_jobs(args, cfg)
    elif args.command == "list-papers":
        cmd_list_papers(args, cfg)
    elif args.command == "list-advisors":
        cmd_list_advisors(args, cfg)
    elif args.command == "list-grants":
        cmd_list_grants(args, cfg)
    elif args.command == "pitch":
        cmd_pitch(args, cfg)
    elif args.command == "serve":
        cmd_serve(args, cfg)
    else:
        cmd_status(args, cfg)


if __name__ == "__main__":
    main()
