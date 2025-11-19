"""Streamlit interface for medical transcription and note generation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

import app
st.set_page_config(page_title="AI Scribe", layout="wide")


def inject_styles() -> None:
    """Inject custom CSS for a more polished layout."""

    st.markdown(
        """
        <style>
        .ai-scribe-hero {
            background: linear-gradient(135deg, #0f172a, #1e3a8a);
            padding: 30px;
            border-radius: 18px;
            color: #f8fafc;
            margin-bottom: 24px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.25);
        }
        .ai-scribe-hero h1 {
            font-size: 2.2rem;
            margin-bottom: 6px;
        }
        .ai-scribe-hero p {
            font-size: 1.05rem;
            opacity: 0.85;
        }
        .stButton > button {
            border-radius: 999px !important;
            padding: 0.6rem 1.6rem !important;
            font-weight: 600 !important;
        }
        .upload-card {
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 16px;
            padding: 24px;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        }
        .metrics-card {
            border-radius: 16px;
            background: #0f172a;
            color: #e0f2fe;
            padding: 24px;
            height: 100%;
            box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.25);
        }
        .metrics-card h3 {
            margin-top: 0;
            margin-bottom: 0.5rem;
        }
        .metrics-card p {
            margin-bottom: 0.75rem;
            color: rgba(226, 232, 240, 0.75);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 10px 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_api_keys() -> None:
    """Validate that required API keys are available."""

    missing = []
    for env_key in ("ASSEMBLYAI_API_KEY", "GEMINI_API_KEY"):
        try:
            app.get_api_key(env_key)
        except RuntimeError:
            missing.append(env_key)

    if missing:
        st.error(
            "Missing API keys: " + ", ".join(missing) +
            ". Add them to your environment or `.env` file and restart the app."
        )
        st.stop()


def save_uploaded_file(uploaded_file) -> Path:
    """Persist uploaded file to a temporary location."""

    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        return Path(tmp.name)


def format_timestamp(milliseconds: Optional[int]) -> str:
    if milliseconds is None:
        return "-"
    total_seconds = max(int(milliseconds // 1000), 0)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def main() -> None:
    inject_styles()

    st.markdown(
        """
        <div class="ai-scribe-hero">
            <h1>🩺 AI Scribe</h1>
            <p>Upload a clinical encounter, let AssemblyAI create a transcript, and have Gemini craft structured SOAP or H&P documentation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ensure_api_keys()

    sidebar = st.sidebar.empty()
    with sidebar.container():
        st.header("Workflow Settings")
        note_format = st.selectbox(
            "Medical note format",
            options=["None", "SOAP", "H&P"],
            index=0,
            help="Choose the structure Gemini should use when summarizing the transcript.",
        )
        poll_interval = st.slider(
            "Polling interval (seconds)",
            min_value=1,
            max_value=10,
            value=3,
            help="Lower values return results faster but increase API polling.",
        )
        st.markdown(
            """
            ---
            **Tips**
            - Supports MP3, WAV, M4A, AAC, FLAC, and OGG.
            - For best accuracy, use high-quality audio with a single primary speaker.
            """
        )

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Upload encounter audio")
        st.caption("Securely process audio using AssemblyAI. Files are streamed and removed after transcription.")
        upload_container = st.container()
        with upload_container:
            st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Drop a file or browse", type=["mp3", "wav", "m4a", "aac", "flac", "ogg"],
                accept_multiple_files=False,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            action_disabled = uploaded is None
            transcribe_btn = st.button("Generate Documentation", type="primary", disabled=action_disabled)

    with right:
        st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
        st.markdown("### Session Snapshot")
        st.markdown(f"- **Gemini model**: `{app.get_gemini_model()}`")
        st.markdown(f"- **Note format**: `{note_format}`")
        st.markdown(f"- **Polling interval**: `{poll_interval}s`")
        st.markdown(
            "If you encounter model errors, confirm that your API key has access to the selected Gemini version."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is None:
        st.info("Upload an audio file to enable transcription.")
        return

    if transcribe_btn:
        with st.spinner("Processing audio…"):
            temp_path = save_uploaded_file(uploaded)
            try:
                assembly_key = app.get_api_key("ASSEMBLYAI_API_KEY")
                headers = {"authorization": assembly_key}

                audio_url = app.upload_audio(str(temp_path), headers)
                transcript_id = app.request_transcription(audio_url, headers)
                result = app.poll_transcription(transcript_id, headers, poll_interval=poll_interval)
                diarized_segments = app.get_diarized_segments(result)
                transcript_text = result.get("text", "").strip()

                if not transcript_text:
                    st.error("Transcription completed but returned empty text.")
                    return

                st.success("Transcription ready! Review the output below.")
                tab_titles = ["Transcript"]
                note_tab_index: Optional[int] = None
                summary_tab_index: Optional[int] = None
                diarization_tab_index: Optional[int] = None

                if note_format != "None":
                    note_tab_index = len(tab_titles)
                    tab_titles.append("Medical Note")
                else:
                    summary_tab_index = len(tab_titles)
                    tab_titles.append("Summary")

                if diarized_segments:
                    diarization_tab_index = len(tab_titles)
                    tab_titles.append("Speaker Timeline")

                tabs = st.tabs(tab_titles)

                with tabs[0]:
                    st.text_area("Transcribed Text", transcript_text, height=260, key="transcript_text")
                    st.download_button(
                        "Download transcript",
                        data=transcript_text,
                        file_name="transcript.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

                if note_format != "None":
                    assert note_tab_index is not None
                    with tabs[note_tab_index]:
                        target_format = "soap" if note_format == "SOAP" else "hp"
                        try:
                            gemini_key = app.get_api_key("GEMINI_API_KEY")
                            note = app.generate_clinical_note(transcript_text, target_format, gemini_key)
                            header = "SOAP Note" if target_format == "soap" else "H&P Note"
                            st.text_area(header, note, height=400, key="clinical_note")
                            st.download_button(
                                f"Download {header}",
                                data=note,
                                file_name=f"{header.lower().replace(' ', '_')}.txt",
                                mime="text/plain",
                                use_container_width=True,
                            )
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Failed to generate {note_format} note: {exc}")
                else:
                    assert summary_tab_index is not None
                    with tabs[summary_tab_index]:
                        st.write("Note generation skipped. Select a format in the sidebar to enable Gemini summaries.")

                if diarized_segments:
                    assert diarization_tab_index is not None
                    with tabs[diarization_tab_index]:
                        st.markdown("#### Speaker Timeline")
                        for seg in diarized_segments:
                            start = format_timestamp(seg.get("start"))
                            end = format_timestamp(seg.get("end"))
                            speaker = seg.get("speaker", "Speaker")
                            text = seg.get("text", "")
                            st.markdown(
                                f"**{speaker}** · {start} – {end}" "\n" f"{text}",
                                help="Speaker diarization provided by AssemblyAI.",
                            )
                        st.download_button(
                            "Download speaker timeline",
                            data="\n".join(
                                f"{seg.get('speaker')}: {seg.get('text')}" for seg in diarized_segments
                            ),
                            file_name="speaker_timeline.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Transcription failed: {exc}")
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
