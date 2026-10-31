"""Streamlit Command Center for the PhD Application & Academic Intelligence Pipeline."""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for clean imports
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

from phd_tracker.config import load_config
from phd_tracker.crm.advisor_manager import AdvisorManager
from phd_tracker.crm.pitch_generator import AcademicPitchGenerator
from phd_tracker.database import (
    Advisor,
    Application,
    get_advisors,
    get_all_letters,
    get_applications,
    get_arxiv_papers,
    get_grants,
    get_positions,
    get_sync_status,
    init_db,
    save_advisor,
    save_application,
    save_letter_request,
    toggle_star_arxiv_paper,
    update_position_status,
)
from phd_tracker.scheduler.sync_daemon import SyncDaemon

# Page Configuration
st.set_page_config(
    page_title="PhD Intelligence & Application Pipeline",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished academic UI
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-eu { background-color: #e3f2fd; color: #0d47a1; }
    .badge-uk { background-color: #ede7f6; color: #4a148c; }
    .badge-de { background-color: #fff3e0; color: #e65100; }
    .badge-score { background-color: #e8f5e9; color: #1b5e20; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_app_state():
    cfg = load_config()
    db_path = cfg["database"]["path"]
    init_db(db_path)
    return cfg, db_path


cfg, db_path = load_app_state()
advisor_mgr = AdvisorManager(db_path)
pitch_gen = AcademicPitchGenerator(cfg)
daemon = SyncDaemon(cfg)

# Sidebar
st.sidebar.title("🌌 PhD Intel & CRM")
st.sidebar.caption(f"Profile: **{cfg.get('user_profile', {}).get('name', 'Arun')}**")
st.sidebar.caption(f"Target: **Cosmology & LSS / Europe & UK**")
st.sidebar.markdown("---")

sync_logs = get_sync_status(db_path)
last_sync = sync_logs[0]["finished_at"][:19].replace("T", " ") if sync_logs else "Never"
interval_hrs = cfg.get("sync", {}).get("interval_hours", 8)

st.sidebar.markdown(f"**Update Cycle:** Every {interval_hrs} hours")
st.sidebar.markdown(f"**Last Sync:** `{last_sync}`")

if st.sidebar.button("🔄 Trigger Sync Now", use_container_width=True):
    with st.spinner("Executing 8-Hour Sync Cycle across ArXiv, INSPIRE-HEP, and Grants..."):
        res = daemon.run_sync_cycle()
        st.sidebar.success(f"Synced! (+{res['arxiv']} papers, +{res['positions']} jobs, +{res['grants']} grants)")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
selected_tab = st.sidebar.radio(
    "Go to",
    [
        "📡 Formal Openings Radar",
        "📚 ArXiv Literature & Advisor Discovery",
        "💶 ERC / Grant Radar",
        "🧑‍🏫 Advisor CRM & Outreach",
        "✉️ Contextual Pitch Drafter",
        "📋 Application Kanban & Letters",
        "⚙️ Sync & System Settings",
    ],
)

# Header Metrics
papers = get_arxiv_papers(db_path, limit=1000)
positions = get_positions(db_path, limit=1000)
advisors = get_advisors(db_path)
applications = get_applications(db_path)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Literature Preprints", len(papers), help="ArXiv preprints mined in your subfields")
c2.metric("Open Job Positions", len(positions), help="Advertised PhD openings in Europe & UK")
c3.metric("Target Advisors in CRM", len(advisors), help="Prospective supervisors to contact")
c4.metric("Active Applications", len(applications), help="Applications currently in pipeline")

st.markdown("---")


# ==========================================
# 1. FORMAL OPENINGS RADAR
# ==========================================
if selected_tab == "📡 Formal Openings Radar":
    st.subheader("📡 Formal European & UK PhD Openings")
    st.caption("Aggregated from INSPIRE-HEP, AAS Job Register, and European university boards.")

    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
    min_score = f_col1.slider("Minimum Relevance Score", 0.0, 10.0, 5.0, 0.5)
    country_filter = f_col2.selectbox(
        "Country Filter",
        ["All", "Germany", "United Kingdom", "Netherlands", "Switzerland", "France", "Italy", "Spain"],
    )
    status_filter = f_col3.selectbox("Status", ["All", "inbox", "shortlisted", "archived"])

    filtered_pos = get_positions(db_path, min_score=min_score, limit=100)
    if country_filter != "All":
        filtered_pos = [p for p in filtered_pos if p.get("country") == country_filter]
    if status_filter != "All":
        filtered_pos = [p for p in filtered_pos if p.get("status") == status_filter]

    st.write(f"Showing **{len(filtered_pos)}** matched positions")

    for pos in filtered_pos:
        with st.container():
            score_color = "#2e7d32" if pos["relevance_score"] >= 8.0 else "#f57c00"
            st.markdown(
                f"""
                <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin-bottom: 12px; background: white;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #1a237e;">{pos['title']}</h4>
                        <span style="background: {score_color}; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold;">
                            ★ {pos['relevance_score']:.1f}/10
                        </span>
                    </div>
                    <p style="margin: 5px 0; color: #424242;">
                        🏛️ <b>{pos['institution']}</b> ({pos.get('country', 'Europe')}) &nbsp;|&nbsp; 
                        📅 Deadline: <b>{pos.get('deadline') or 'Rolling / Open'}</b> &nbsp;|&nbsp; 
                        💶 Funding: <b>{pos.get('funding_type') or 'Funded'}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            kws = (pos.get("matched_keywords") or "").split(",")
            kw_html = " ".join([f"<span class='badge' style='background:#e0f2f1; color:#004d40;'>{k}</span>" for k in kws if k])
            if kw_html:
                st.markdown(kw_html, unsafe_allow_html=True)

            b1, b2, b3 = st.columns([1, 1.5, 4])
            if pos["status"] != "shortlisted" and b1.button("⭐ Shortlist", key=f"short_{pos['id']}"):
                update_position_status(db_path, pos["id"], "shortlisted")
                st.rerun()
            if b2.button("📋 Add to Kanban Board", key=f"kanban_{pos['id']}"):
                save_application(
                    db_path,
                    Application(
                        position_id=pos["id"],
                        title=pos["title"],
                        institution=pos["institution"],
                        stage="Identified",
                        submission_deadline=pos.get("deadline"),
                        portal_url=pos.get("url"),
                    ),
                )
                st.success("Added to Application Kanban!")
                st.rerun()
            if pos.get("url"):
                b3.markdown(f"[🔗 View Official Posting]({pos['url']})")

            with st.expander("Show Description Details"):
                desc = pos.get("raw_description") or "No detailed description provided."
                # Strip raw HTML tags for clean readability
                clean_desc = re.sub(r'</?(?:div|strong|b|p|br|span|h[1-6])[^>]*>', ' ', desc)
                clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
                st.write(clean_desc)
            st.markdown("---")


# ==========================================
# 2. ARXIV LITERATURE & ADVISOR DISCOVERY
# ==========================================
elif selected_tab == "📚 ArXiv Literature & Advisor Discovery":
    st.subheader("📚 ArXiv Preprint Radar & Unadvertised Advisor Discovery")
    st.caption("Mines recent preprints in `astro-ph.CO`, `astro-ph.GA`, `astro-ph.IM`, and computational methods.")

    a_col1, a_col2 = st.columns([3, 1])
    search_query = a_col1.text_input("🔍 Filter by topic, method, or author name", placeholder="e.g. cosmic voids, Euclid, Sartori, CBL")
    min_paper_score = a_col2.slider("Min Match Score", 0.0, 10.0, 4.0, 0.5)

    paper_list = get_arxiv_papers(db_path, min_score=min_paper_score, search_query=search_query or None)

    st.write(f"Displaying **{len(paper_list)}** matching preprints")

    for p in paper_list:
        with st.container():
            star_label = "★" if p.get("is_starred") else "☆"
            st.markdown(
                f"""
                <div style="border-left: 4px solid #3f51b5; padding-left: 10px; margin-bottom: 8px;">
                    <h4 style="margin: 0; color: #1a237e;">{p['title']}</h4>
                    <p style="margin: 4px 0; color: #555;">
                        <b>Authors:</b> {', '.join(p['authors'][:4])}{'...' if len(p['authors']) > 4 else ''} | 
                        <b>Published:</b> {p['published_date']} | 
                        <a href="https://arxiv.org/abs/{p['arxiv_id']}" target="_blank">arXiv:{p['arxiv_id']}</a>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            kws = (p.get("matched_keywords") or "").split(",")
            kw_html = " ".join([f"<span class='badge' style='background:#ede7f6; color:#311b92;'>{k}</span>" for k in kws if k])
            if kw_html:
                st.markdown(kw_html, unsafe_allow_html=True)

            with st.expander("📖 Abstract"):
                st.write(p["abstract"])

            col_btn1, col_btn2, col_btn3 = st.columns([1.5, 2, 3])
            if col_btn1.button(f"{star_label} Star Paper", key=f"star_{p['id']}"):
                toggle_star_arxiv_paper(db_path, p["id"])
                st.rerun()

            # 1-Click Track Author
            senior_author = p["authors"][-1] if p["authors"] else "Group Leader"
            if col_btn2.button(f"🧑‍🏫 Track Author: {senior_author[:18]}...", key=f"track_{p['id']}"):
                adv_id = advisor_mgr.add_advisor_from_author(
                    name=senior_author,
                    institution="Discovered via ArXiv: " + p["arxiv_id"],
                    research_areas=", ".join(kws[:3]),
                    notes=f"Discovered from preprint arXiv:{p['arxiv_id']} - {p['title'][:60]}",
                )
                st.success(f"Added {senior_author} to Advisor CRM!")

            if col_btn3.button("✉️ Draft Cold Email for this Paper", key=f"pitch_{p['id']}"):
                st.session_state["draft_target_paper"] = p["title"]
                st.session_state["draft_target_arxiv"] = p["arxiv_id"]
                st.session_state["draft_target_advisor"] = senior_author
                st.info("Draft loaded! Switch to '✉️ Contextual Pitch Drafter' tab to view.")

            st.markdown("---")


# ==========================================
# 3. ERC / GRANT RADAR
# ==========================================
elif selected_tab == "💶 ERC / Grant Radar":
    st.subheader("💶 Research Grant Radar (ERC Starting/Consolidator & UKRI)")
    st.caption("Newly funded grants (€1.5M - €2M) indicate PIs who have active budget to hire PhD students right now.")

    grants = get_grants(db_path)
    for g in grants:
        with st.container():
            st.markdown(
                f"""
                <div style="border: 1px solid #c8e6c9; border-radius: 8px; padding: 15px; margin-bottom: 12px; background: #fafafa;">
                    <div style="display: flex; justify-content: space-between;">
                        <h4 style="margin: 0; color: #2e7d32;">[{g.get('project_acronym', 'GRANT')}] {g['title']}</h4>
                        <span style="font-weight: bold; color: #1b5e20;">€{g.get('budget_eur', 0):,.0f}</span>
                    </div>
                    <p style="margin: 5px 0;">
                        👤 <b>PI: {g.get('pi_name')}</b> &nbsp;|&nbsp; 
                        🏛️ <b>{g.get('host_institution')}</b> ({g.get('country')}) &nbsp;|&nbsp; 
                        🎯 Funder: <b>{g.get('funder')}</b>
                    </p>
                    <p style="color: #424242; font-size: 0.95rem;">{g.get('summary')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(f"🧑‍🏫 Add PI ({g.get('pi_name')}) to Advisor CRM", key=f"grant_adv_{g['id']}"):
                adv_id = advisor_mgr.add_advisor_from_author(
                    name=g.get("pi_name") or "PI",
                    institution=g.get("host_institution") or "European Institute",
                    country=g.get("country") or "Europe",
                    notes=f"PI on {g.get('funder')}: {g.get('title')}",
                )
                st.success(f"Added {g.get('pi_name')} to Advisor CRM!")
            st.markdown("---")


# ==========================================
# 4. ADVISOR CRM & OUTREACH
# ==========================================
elif selected_tab == "🧑‍🏫 Advisor CRM & Outreach":
    st.subheader("🧑‍🏫 Advisor CRM & Outreach Tracker")
    st.caption("Manage informal communications, track response sentiments, and schedule follow-up emails.")

    with st.expander("➕ Add New Prospective Advisor Manually"):
        with st.form("add_adv_form"):
            c_name, c_inst, c_country = st.columns(3)
            name = c_name.text_input("Advisor Name", placeholder="e.g. Prof. Elena Sartori")
            inst = c_inst.text_input("Institution", placeholder="e.g. MPA Garching")
            country = c_country.text_input("Country", placeholder="e.g. Germany")
            c_email, c_topic = st.columns(2)
            email = c_email.text_input("Email", placeholder="advisor@institute.org")
            topic = c_topic.text_input("Research Synergy", placeholder="Cosmic voids, N-body, Euclid")
            notes = st.text_area("Initial Notes / Relevant Papers")
            if st.form_submit_button("Save Advisor to CRM"):
                if name and inst:
                    advisor_mgr.add_advisor_from_author(
                        name=name, institution=inst, country=country, email=email, research_areas=topic, notes=notes
                    )
                    st.success(f"Saved {name} to CRM!")
                    st.rerun()

    advisors_list = get_advisors(db_path)
    if not advisors_list:
        st.info("No advisors in CRM yet. Use the ArXiv Explorer or Grant Radar to add target PIs!")
    else:
        for adv in advisors_list:
            with st.container():
                st.markdown(
                    f"""
                    <div style="border: 1px solid #cfd8dc; border-radius: 8px; padding: 12px; margin-bottom: 10px; background: white;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0; color: #263238;">👤 {adv['name']}</h4>
                            <span class="badge" style="background: #e1f5fe; color: #0277bd; font-size: 0.9rem;">
                                {adv['outreach_status']}
                            </span>
                        </div>
                        <p style="margin: 4px 0; color: #546e7a;">
                            🏛️ <b>{adv['institution']}</b> ({adv.get('country') or 'Europe'}) &nbsp;|&nbsp; 
                            ✉️ {adv.get('email') or 'Email not recorded yet'}
                        </p>
                        <p style="margin: 4px 0; color: #37474f;"><b>Research:</b> {adv.get('research_areas') or 'General Cosmology'}</p>
                        {f"<p style='color: #c62828; margin: 4px 0;'>⏰ <b>Follow-up Due:</b> {adv['follow_up_date']}</p>" if adv.get('follow_up_date') else ""}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_a1, col_a2, col_a3, col_a4 = st.columns([1.5, 1.5, 1.5, 2])
                new_status = col_a1.selectbox(
                    "Update Status",
                    AdvisorManager.STATUS_STAGES,
                    index=AdvisorManager.STATUS_STAGES.index(adv["outreach_status"]) if adv["outreach_status"] in AdvisorManager.STATUS_STAGES else 0,
                    key=f"status_sel_{adv['id']}",
                )
                if new_status != adv["outreach_status"]:
                    adv_obj = Advisor(
                        id=adv["id"],
                        name=adv["name"],
                        email=adv["email"],
                        institution=adv["institution"],
                        country=adv["country"],
                        research_areas=adv["research_areas"],
                        outreach_status=new_status,
                        notes=adv["notes"],
                    )
                    save_advisor(db_path, adv_obj)
                    st.rerun()

                if col_a2.button("📬 Mark Sent (+10d Reminder)", key=f"sent_{adv['id']}"):
                    advisor_mgr.mark_email_sent(adv["id"], follow_up_days=10)
                    st.success("Updated to 'Email Sent' with a 10-day follow-up reminder!")
                    st.rerun()

                if col_a3.button("📋 Promote to Application", key=f"prom_{adv['id']}"):
                    advisor_mgr.promote_to_application(adv["id"])
                    st.success("Promoted to Kanban board!")
                    st.rerun()

                if col_a4.button("✉️ Load in Pitch Drafter", key=f"pitch_load_{adv['id']}"):
                    st.session_state["draft_target_advisor"] = adv["name"]
                    st.session_state["draft_target_inst"] = adv["institution"]
                    st.info("Loaded into Pitch Drafter tab!")

            st.markdown("---")


# ==========================================
# 5. CONTEXTUAL PITCH DRAFTER
# ==========================================
elif selected_tab == "✉️ Contextual Pitch Drafter":
    st.subheader("✉️ Contextual Academic Cold-Email Drafter")
    st.caption("Generates tailored European academic inquiry emails linking your thesis work to their recent paper or grant.")

    adv_name = st.text_input(
        "Target Professor / PI Name",
        value=st.session_state.get("draft_target_advisor", "Prof. Elena Sartori"),
    )
    inst_name = st.text_input(
        "Institution",
        value=st.session_state.get("draft_target_inst", "Max Planck Institute for Astrophysics (MPA)"),
    )
    paper_title = st.text_input(
        "Recent ArXiv Paper Title (Optional)",
        value=st.session_state.get("draft_target_paper", "Cosmic Voids in the Aletheia Simulation Suite"),
    )
    arxiv_id = st.text_input(
        "ArXiv ID (Optional)",
        value=st.session_state.get("draft_target_arxiv", "2410.12345"),
    )
    specific_angle = st.text_input(
        "Specific Angle / Contribution",
        placeholder="e.g. extending BitVF void catalogs to Euclid mock surveys or SBI parameter inference",
    )

    if st.button("🚀 Generate Tailored Pitch Draft", use_container_width=True):
        draft = pitch_gen.generate_pitch(
            advisor_name=adv_name,
            institution=inst_name,
            paper_title=paper_title or None,
            paper_arxiv_id=arxiv_id or None,
            specific_angle=specific_angle or None,
        )

        st.markdown("#### Subject Line:")
        st.code(draft["subject"], language="text")

        st.markdown("#### Email Body:")
        st.text_area("Ready to customize and send:", value=draft["body"], height=320)
        st.info("Tip: Always attach your Academic CV (with BitVF and CBL MCMC highlights) and unofficial transcripts as a clean PDF.")


# ==========================================
# 6. APPLICATION KANBAN & LETTERS
# ==========================================
elif selected_tab == "📋 Application Kanban & Letters":
    st.subheader("📋 Application Pipeline & Referee Letter Tracker")

    stages = ["Identified", "Contacted PI", "Preparing Dossier", "Letters Requested", "Submitted", "Interview", "Decision"]
    apps = get_applications(db_path)

    st.markdown("### Kanban Pipeline")
    cols = st.columns(len(stages))
    for idx, stage in enumerate(stages):
        with cols[idx]:
            stage_apps = [a for a in apps if a["stage"] == stage]
            st.markdown(f"**{stage}** ({len(stage_apps)})")
            for sa in stage_apps:
                st.markdown(
                    f"""
                    <div style="background: #eef2f6; border-radius: 6px; padding: 8px; margin-bottom: 8px; font-size: 0.85rem;">
                        <b>{sa['title'][:25]}...</b><br>
                        🏛️ {sa['institution'][:20]}<br>
                        📅 {sa.get('submission_deadline') or 'No deadline set'}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                next_stage = st.selectbox(
                    "Move",
                    stages,
                    index=stages.index(sa["stage"]),
                    key=f"move_{sa['id']}",
                    label_visibility="collapsed",
                )
                if next_stage != sa["stage"]:
                    sa_obj = Application(
                        id=sa["id"],
                        title=sa["title"],
                        institution=sa["institution"],
                        stage=next_stage,
                        submission_deadline=sa["submission_deadline"],
                    )
                    save_application(db_path, sa_obj)
                    st.rerun()

    st.markdown("---")
    st.subheader("📬 Recommendation Letter Tracker")

    with st.expander("➕ Log New Recommendation Letter Request"):
        with st.form("ref_letter_form"):
            app_choices = {f"{a['title']} ({a['institution']})": a["id"] for a in apps}
            if not app_choices:
                st.warning("Create at least one application before logging letters.")
            else:
                sel_app = st.selectbox("Select Application", list(app_choices.keys()))
                ref_name = st.text_input("Referee Name", placeholder="e.g. Master's Thesis Advisor")
                ref_email = st.text_input("Referee Email")
                ref_status = st.selectbox("Status", ["To Request", "Requested", "Confirmed Submitted"])
                if st.form_submit_button("Save Letter Tracker"):
                    save_letter_request(
                        db_path,
                        app_id=app_choices[sel_app],
                        referee_name=ref_name,
                        referee_email=ref_email,
                        status=ref_status,
                    )
                    st.success("Letter tracker saved!")
                    st.rerun()

    letters = get_all_letters(db_path)
    if letters:
        st.table([
            {
                "Referee": l["referee_name"],
                "Application": l["app_title"],
                "Institution": l["app_institution"],
                "Status": l["status"],
                "Deadline": l["submission_deadline"],
            }
            for l in letters
        ])


# ==========================================
# 7. SYNC & SYSTEM SETTINGS
# ==========================================
elif selected_tab == "⚙️ Sync & System Settings":
    st.subheader("⚙️ 8-Hour Sync Engine & System Logs")

    st.markdown(
        f"""
        ### Automated 8-Hour Sync Daemon
        This pipeline is built to automatically crawl and update every **{interval_hrs} hours**.
        
        To run the continuous sync daemon in your terminal or background:
        ```bash
        phd-tracker daemon --interval {interval_hrs}
        ```
        """
    )

    st.markdown("### Execution History")
    logs = get_sync_status(db_path)
    if logs:
        st.table([
            {
                "Source": lg["source"],
                "Records Fetched": lg["records_fetched"],
                "Status": lg["status"],
                "Finished At": lg["finished_at"],
                "Duration (s)": lg["duration_sec"],
                "Message": lg["message"],
            }
            for lg in logs
        ])
    else:
        st.info("No sync history recorded yet.")
