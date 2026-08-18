import os
import sys

try:
    import moviepy
except ImportError:
    for path in [
        os.path.join(os.getcwd(), "mock_env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "mock_env"),
        os.path.join(os.path.dirname(__file__), "mock_env")
    ]:
        if os.path.exists(path):
            if path not in sys.path:
                sys.path.insert(0, path)
            import sitecustomize
            break

import streamlit as st
import tempfile
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from pipeline import run_pipeline
import db_storage

st.set_page_config(page_title="Meeting Intelligence", page_icon="🎙️", layout="wide")

st.title("🎙️ Meeting / Lecture Intelligence")
st.caption("Upload a video or audio file to get a transcript, summary, speakers, sentiment, and engagement score.")

with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox("Whisper model size", ["tiny", "base", "small", "medium", "large"], index=1,
                               help="Bigger = more accurate but slower")
    do_video = st.checkbox("Analyze video engagement", value=True)
    do_diarize = st.checkbox("Detect speakers (diarization)", value=True,
                              help="Requires HF_TOKEN environment variable - see README")
    
    st.header("History")
    db_type = "MySQL (Remote)" if os.environ.get("MYSQL_HOST") else "SQLite (Local)"
    st.caption(f"Active database: **{db_type}**")
    past_reports = db_storage.get_all_reports()
    if past_reports:
        selected_report_meta = st.selectbox(
            "Load past report",
            past_reports,
            format_func=lambda x: f"{x['timestamp']} - {os.path.basename(x['source_file'])}"
        )
        if st.button("Load Selected Report"):
            st.session_state.loaded_report = db_storage.get_report_by_id(selected_report_meta["id"])
        if st.button("Delete Selected Report"):
            db_storage.delete_report(selected_report_meta["id"])
            st.session_state.loaded_report = None
            st.rerun()
    else:
        st.write("No reports in history yet.")

def display_report(report):
    st.subheader("📝 Summary")
    st.write(report["summary"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔑 Keywords")
        st.write(", ".join(report["keywords"]) if report["keywords"] else "None found")

        st.subheader("✅ Action Items")
        if report["action_items"]:
            for item in report["action_items"]:
                st.write(f"- ({item['start']}s) {item['text']}")
        else:
            st.write("None detected")

    with col2:
        if report.get("video_engagement"):
            st.subheader("📹 Engagement")
            st.metric("Engagement score", f"{report['video_engagement']['engagement_score']}/100")
            st.write(f"Avg. faces visible: {report['video_engagement']['average_faces_visible']}")

    st.subheader("💬 Transcript with Speakers & Sentiment")
    if report.get("diarized_segments"):
        for seg in report["diarized_segments"]:
            st.write(f"**[{seg['speaker']}]** ({seg['start']}s–{seg['end']}s): {seg['text']}")
    else:
        st.write(report["full_transcript"])

    st.download_button(
        "Download full report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="meeting_report.json",
        mime="application/json",
    )

if "loaded_report" not in st.session_state:
    st.session_state.loaded_report = None

uploaded_file = st.file_uploader(
    "Drop a video or audio file here",
    type=["mp4", "mov", "mkv", "avi", "webm", "mp3", "wav", "m4a", "aac", "flac"],
)

if uploaded_file is not None:
    if st.button("Run Analysis", type="primary"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.status("Running pipeline...", expanded=True) as status:
                st.write("Extracting audio, transcribing, analyzing...")
                report = run_pipeline(
                    input_path=input_path,
                    output_dir=tmp_dir,
                    model_size=model_size,
                    do_video=do_video,
                    do_diarize=do_diarize,
                )
                status.update(label="Done!", state="complete")
            st.session_state.loaded_report = report
            st.rerun()

if st.session_state.loaded_report is not None:
    display_report(st.session_state.loaded_report)
else:
    st.info("Upload a file above or select a past report from history to get started.")
