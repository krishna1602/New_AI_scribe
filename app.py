"""Streamlit AI scribe application."""
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Dict

import assemblyai as aai
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

# --- Environment setup -----------------------------------------------------
load_dotenv()
ASSEMBLY_KEY = os.getenv("ASSEMBLE_API_KEY") or os.getenv("ASSEMBLY_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if ASSEMBLY_KEY:
    aai.settings.http_timeout = 180  # allow large uploads without timing out

# --- Templates --------------------------------------------------------------
DEFAULT_PROMPTS: Dict[str, str] = {
    "SOAP": """
You are a medical scribe. Build a SOAP note in Markdown with these sections:
1. Subjective
   - Chief complaint
   - History of present illness (chronological, include context)
   - Pertinent ROS and social history
2. Objective
   - Vitals (show "Not discussed" if missing)
   - Focused physical exam grouped by system
   - Pertinent labs/imaging mentioned in the transcript
3. Assessment
   - Number each active problem with brief reasoning
   - Mention disease severity/stage or differential when unclear
4. Plan
   - For each problem list diagnostics, therapeutics, lifestyle, patient education
   - Include follow-up interval, disposition, and counseling highlights
5. Key takeaways
   - 3 bullets people/action oriented
Use professional tone, avoid hallucination, quote "Patient denies" style statements verbatim when possible.
""".strip(),
    "H&P": """
You are a medical scribe. Draft a History & Physical (H&P) in Markdown:
- Identifying information (age, sex, relevant PMH)
- Chief complaint
- HPI with OLDCARTS style structure
- Past medical, surgical, medication, allergy, family, and social history
- Review of systems grouping positives then pertinent negatives
- Physical Exam (organized head-to-toe)
- Diagnostics (labs/imaging mentioned)
- Assessment & Plan summarizing top problems, reasoning, and next steps
End with Disposition + Follow-up Instructions.
Only rely on transcript content; if data missing write "Not discussed".
""".strip(),
}

INSTRUCTION_FILE = Path(".txt")


def load_reference_text() -> str:
    """Return user-provided template guidance if available."""
    if INSTRUCTION_FILE.exists():
        try:
            return INSTRUCTION_FILE.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            return ""
    return ""


def init_state() -> None:
    """Ensure session_state defaults exist."""
    st.session_state.setdefault("transcript_text", "")
    st.session_state.setdefault("ai_note", "")
    st.session_state.setdefault("note_type", "SOAP")
    st.session_state.setdefault("templates", DEFAULT_PROMPTS.copy())


@st.cache_resource(show_spinner=False)
def get_gemini_model() -> genai.GenerativeModel | None:
    if not GEMINI_KEY:
        return None
    genai.configure(api_key=GEMINI_KEY)
    return genai.GenerativeModel("gemini-2.5-flash")


def transcribe_audio_file(upload, api_key: str) -> str:
    """Send audio to AssemblyAI and return transcript text."""
    aai.settings.api_key = api_key
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix) as tmp:
        tmp.write(upload.getbuffer())
        tmp_path = tmp.name
    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(transcript.error)
    return transcript.text or ""


def generate_note(transcript_text: str, prompt: str, note_type: str) -> str:
    model = get_gemini_model()
    if not model:
        raise RuntimeError("Gemini API key missing.")
    full_prompt = textwrap.dedent(
        f"""
        You are assisting a clinician. Using the transcript below, create a {note_type} note.
        Transcript:
        \"\"\"{transcript_text}\"\"\"
        Guidance:
        {prompt}
        """
    ).strip()
    response = model.generate_content(full_prompt)
    if not hasattr(response, "text") or not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    return response.text.strip()


# --- UI --------------------------------------------------------------------
st.set_page_config(page_title="AI Clinical Scribe", page_icon="🩺", layout="wide")
st.markdown(
    """
    <style>
    .status-badge {padding:0.35rem 0.7rem;border-radius:999px;font-size:0.8rem;display:inline-block;}
    .status-ok {background:#E3FCEF;color:#067647;}
    .status-missing {background:#FDEBEC;color:#B42318;}
    .stTextArea textarea {font-family:"SFMono-Regular",Consolas,monospace;font-size:0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

init_state()
reference_text = load_reference_text()

st.title("AI Clinical Scribe")
st.caption("Upload a conversation, generate SOAP or H&P documentation, and export instantly.")

status_col1, status_col2 = st.columns(2)
status_col1.markdown(
    f'<span class="status-badge {"status-ok" if ASSEMBLY_KEY else "status-missing"}">'
    f'AssemblyAI: {"Ready" if ASSEMBLY_KEY else "Missing key"}</span>',
    unsafe_allow_html=True,
)
status_col2.markdown(
    f'<span class="status-badge {"status-ok" if GEMINI_KEY else "status-missing"}">'
    f'Gemini: {"Ready" if GEMINI_KEY else "Missing key"}</span>',
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("1 · Intake")
    audio_file = st.file_uploader("Upload encounter audio", type=["mp3", "m4a", "wav"], help="AssemblyAI key required")
    if st.button("Transcribe audio", disabled=audio_file is None or not ASSEMBLY_KEY, use_container_width=True):
        if not ASSEMBLY_KEY:
            st.error("Add your AssemblyAI key in .env to enable transcription.")
        elif audio_file is None:
            st.warning("Upload an audio file first.")
        else:
            with st.spinner("Transcribing with AssemblyAI…"):
                try:
                    transcript = transcribe_audio_file(audio_file, ASSEMBLY_KEY)
                    st.session_state.transcript_text = transcript
                    st.success("Transcription complete.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Transcription failed: {exc}")

    transcript_input = st.text_area(
        "Transcript (auto-filled from AssemblyAI or paste manually)",
        value=st.session_state.transcript_text,
        height=320,
    )
    if transcript_input != st.session_state.transcript_text:
        st.session_state.transcript_text = transcript_input

    if reference_text:
        with st.expander("Reference template from .txt"):
            st.code(reference_text)

with right_col:
    st.subheader("2 · Structured Note Builder")
    note_type = st.radio("Select format", options=list(DEFAULT_PROMPTS.keys()), horizontal=True)
    st.session_state.note_type = note_type

    editor_key = f"prompt_editor_{note_type}"
    if editor_key not in st.session_state:
        st.session_state[editor_key] = st.session_state.templates[note_type]

    with st.expander("Customize instructions", expanded=False):
        prompt_edit = st.text_area(
            label=f"{note_type} guidance",
            key=editor_key,
            height=220,
        )
        st.session_state.templates[note_type] = prompt_edit or DEFAULT_PROMPTS[note_type]

    if st.button(
        "Generate clinical note",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.transcript_text or not GEMINI_KEY,
    ):
        if not st.session_state.transcript_text:
            st.warning("Paste or transcribe a conversation first.")
        elif not GEMINI_KEY:
            st.error("Add your Gemini key in .env to enable drafting.")
        else:
            with st.spinner("Writing note with Gemini…"):
                try:
                    note = generate_note(
                        st.session_state.transcript_text,
                        st.session_state.templates[note_type],
                        note_type,
                    )
                    st.session_state.ai_note = note
                    st.success("Clinical note ready.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Gemini error: {exc}")

    tabs = st.tabs(["Clinical Note", "Transcript", "Download"])
    with tabs[0]:
        if st.session_state.ai_note:
            st.markdown(st.session_state.ai_note)
        else:
            st.info("Generate a note to see it here.")

    with tabs[1]:
        if st.session_state.transcript_text:
            st.markdown(st.session_state.transcript_text)
        else:
            st.info("Transcript will appear after transcription or pasting conversation text.")

    with tabs[2]:
        if st.session_state.ai_note:
            st.download_button(
                label="Download Markdown",
                data=st.session_state.ai_note,
                file_name=f"{note_type.lower()}_note.md",
                mime="text/markdown",
            )
        else:
            st.caption("No note available yet.")

st.caption("Need help? Set ASSEMBLE_API_KEY and GEMINI_API_KEY in .env, then run `streamlit run main.py`.")
