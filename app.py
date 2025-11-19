"""Command-line tool to transcribe audio with AssemblyAI and optionally
generate SOAP or H&P clinical notes using Gemini."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Generator, Optional, List

import requests


ASSEMBLY_BASE_URL = "https://api.assemblyai.com"
ASSEMBLY_UPLOAD_ENDPOINT = f"{ASSEMBLY_BASE_URL}/v2/upload"
ASSEMBLY_TRANSCRIPT_ENDPOINT = f"{ASSEMBLY_BASE_URL}/v2/transcript"
GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
FALLBACK_GEMINI_MODELS = (
    "gemini-2.5-flash-latest",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
)


class TranscriptionError(RuntimeError):
    """Raised when the transcription API reports an error."""


def load_env_file(path: str = ".env") -> None:
    """Populate os.environ with values from a simple .env file if present."""

    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                if "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                cleaned = value.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), cleaned)
    except OSError as exc:
        raise RuntimeError(f"Failed to read .env file at {path}: {exc}") from exc


def get_api_key(env_name: str) -> str:
    """Return API key from environment or raise a helpful error."""

    load_env_file()
    key = os.getenv(env_name)
    if not key:
        raise RuntimeError(
            f"Environment variable {env_name} is required."
            " Set it in your shell or the .env file."
        )
    return key


def get_gemini_model() -> str:
    """Return the Gemini model name, allowing override via environment."""

    load_env_file()
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def read_file_chunks(file_path: str, chunk_size: int = 5_242_880) -> Generator[bytes, None, None]:
    """Yield chunks of a file for streaming upload."""

    with open(file_path, "rb") as audio_file:
        while True:
            data = audio_file.read(chunk_size)
            if not data:
                break
            yield data


def upload_audio(file_path: str, headers: Dict[str, str]) -> str:
    """Upload local audio file to AssemblyAI and return the remote URL."""

    response = requests.post(
        ASSEMBLY_UPLOAD_ENDPOINT,
        headers={**headers, "content-type": "application/octet-stream"},
        data=read_file_chunks(file_path),
        timeout=300,
    )
    response.raise_for_status()
    upload_url = response.json().get("upload_url")
    if not upload_url:
        raise RuntimeError("AssemblyAI upload response missing 'upload_url'.")
    return upload_url


def request_transcription(audio_url: str, headers: Dict[str, str]) -> str:
    """Submit transcription request and return the transcript ID."""

    config = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "format_text": True,
        "punctuate": True,
        "speech_model": "universal",
        "language_detection": True,
    }
    response = requests.post(ASSEMBLY_TRANSCRIPT_ENDPOINT, json=config, headers=headers, timeout=60)
    response.raise_for_status()
    transcript_id = response.json().get("id")
    if not transcript_id:
        raise RuntimeError("AssemblyAI transcription response missing 'id'.")
    return transcript_id


def poll_transcription(transcript_id: str, headers: Dict[str, str], poll_interval: int = 3) -> Dict[str, object]:
    """Poll AssemblyAI until transcription completes or fails."""

    polling_url = f"{ASSEMBLY_TRANSCRIPT_ENDPOINT}/{transcript_id}"
    while True:
        response = requests.get(polling_url, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        status = result.get("status")

        if status == "completed":
            return result
        if status == "error":
            raise TranscriptionError(result.get("error", "Unknown transcription error."))

        time.sleep(poll_interval)


def _format_speaker_label(raw_label: Optional[object]) -> str:
    if raw_label is None:
        return "Speaker"
    label = str(raw_label).strip()
    if not label:
        return "Speaker"
    if len(label) == 1 and label.isalpha():
        return f"Speaker {label.upper()}"
    return f"Speaker {label}"


def get_diarized_segments(transcription: Dict[str, object]) -> List[Dict[str, object]]:
    """Return diarized speaker segments from an AssemblyAI transcription result."""

    segments: List[Dict[str, object]] = []

    utterances = transcription.get("utterances")
    if isinstance(utterances, list) and utterances:
        for entry in utterances:
            text = (entry.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                {
                    "speaker": _format_speaker_label(entry.get("speaker")),
                    "text": text,
                    "start": entry.get("start"),
                    "end": entry.get("end"),
                }
            )
        if segments:
            return segments

    words = transcription.get("words")
    if isinstance(words, list) and words:
        current: Optional[Dict[str, object]] = None
        for word in words:
            speaker_raw = word.get("speaker")
            word_text = word.get("text") or ""
            if not word_text.strip():
                continue

            if current and current.get("_speaker_raw") == speaker_raw:
                current["_words"].append(word_text)
                current["end"] = word.get("end", current.get("end"))
            else:
                if current:
                    segments.append(
                        {
                            "speaker": _format_speaker_label(current.get("_speaker_raw")),
                            "text": " ".join(current["_words"]).strip(),
                            "start": current.get("start"),
                            "end": current.get("end"),
                        }
                    )
                current = {
                    "_speaker_raw": speaker_raw,
                    "_words": [word_text],
                    "start": word.get("start"),
                    "end": word.get("end"),
                }

        if current:
            segments.append(
                {
                    "speaker": _format_speaker_label(current.get("_speaker_raw")),
                    "text": " ".join(current["_words"]).strip(),
                    "start": current.get("start"),
                    "end": current.get("end"),
                }
            )

    return segments


def build_note_prompt(transcript: str, note_format: str) -> str:
    """Return a formatted prompt for Gemini based on the requested note format."""

    if note_format == "soap":
        instructions = (
            "You are a medical documentation assistant.\n"
            "Convert the provided clinical transcript into a concise, accurate, and well-structured SOAP note.\n"
            "Only use information explicitly stated in the transcript. Do NOT add, infer, or assume any data not mentioned.\n"
            "If a section has no data in the transcript, write: “Not mentioned.”\n\n"
            "Format exactly as follows:\n\n"
            "SOAP NOTE\n"
            "Patient Name:\n"
            "DOB:\n"
            "Clinician:\n"
            "Date:\n"
            "Setting: (telemedicine / in-person) — based on transcript\n\n"
            "S – Subjective\n"
            "• Chief Complaint:\n"
            "• History of Present Illness:\n"
            "• Review of Systems (only items mentioned):\n"
            "• Past Medical History:\n"
            "• Medications:\n"
            "• Allergies:\n"
            "• Family History:\n"
            "• Social History:\n\n"
            "O – Objective\n"
            "• Exam findings from transcript\n"
            "(If telehealth and no exam provided, write: “No physical exam performed; assessment based on verbal report.”)\n"
            "• Vitals if mentioned\n\n"
            "A – Assessment\n"
            "• List all clinician-stated assessments or inferred concerns explicitly stated in the transcript\n"
            "• Do NOT generate diagnoses that were not discussed\n\n"
            "P – Plan\n"
            "• Diagnostics ordered or recommended\n"
            "• Treatments/medications advised\n"
            "• Work restrictions\n"
            "• Safety netting / follow-up advice\n\n"
            "Transcript:\n"
            f"{transcript}"
        )
    else:  # note_format == "hp"
        instructions = (
            "You are a medical documentation assistant.\n"
            "Convert the provided transcript into a structured History & Physical (H&P) note.\n"
            "Use only information explicitly stated. Do NOT guess or add details.\n"
            "If a section is missing information, mark it as “Not mentioned.”\n\n"
            "Format exactly as follows:\n\n"
            "HISTORY & PHYSICAL (H&P)\n\n"
            "Patient Name:\n"
            "DOB:\n"
            "Clinician:\n"
            "Date:\n"
            "Setting:\n\n"
            "HISTORY\n"
            "Chief Complaint:\n"
            "History of Present Illness:\n"
            "Past Medical History:\n"
            "Past Surgical History:\n"
            "Medications:\n"
            "Allergies:\n"
            "Family History:\n"
            "Social History:\n"
            "Review of Systems:\n"
            "(Only list items explicitly found in the transcript.)\n\n"
            "PHYSICAL EXAM\n"
            "• If no exam data exists, write: “Not performed in transcript.”\n\n"
            "ASSESSMENT\n"
            "• Summarize the clinician’s diagnostic thinking exactly as discussed.\n"
            "• Do NOT generate new differentials unless mentioned.\n\n"
            "PLAN\n"
            "• Document investigations ordered or recommended\n"
            "• Treatment recommendations\n"
            "• Follow-up instructions\n"
            "• Any disposition (e.g., clinic referral, ER recommendation)\n\n"
            "Transcript:\n"
            f"{transcript}"
        )
    return instructions


def _invoke_gemini(payload: Dict[str, object], api_key: str, model_name: str) -> Dict[str, object]:
    """Call Gemini API for the specified model and return JSON response."""

    url = f"{GEMINI_API_ROOT}/{model_name}:generateContent"
    response = requests.post(
        url,
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def generate_clinical_note(transcript: str, note_format: str, api_key: str) -> str:
    """Call Gemini to convert transcript into the requested note format."""

    prompt = build_note_prompt(transcript, note_format)
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }

    configured_model = get_gemini_model()
    models_to_try = []
    seen_models = set()
    for name in (configured_model, *FALLBACK_GEMINI_MODELS):
        if name and name not in seen_models:
            models_to_try.append(name)
            seen_models.add(name)

    last_404_error: Optional[str] = None

    for model_name in models_to_try:
        try:
            data = _invoke_gemini(payload, api_key, model_name)
        except requests.HTTPError as exc:  # pragma: no cover - network errors
            status = exc.response.status_code if exc.response is not None else None
            if status == 404:
                last_404_error = (
                    f"Gemini model '{model_name}' not available (404). "
                    "Update GEMINI_MODEL or ensure the model is enabled for your API key."
                )
                continue
            raise RuntimeError(
                f"Gemini request failed for model '{model_name}': {exc}"
            ) from exc

        candidates = data.get("candidates") or []
        for candidate in candidates:
            if candidate.get("finishReason") == "STOP" and "content" in candidate:
                parts = candidate["content"].get("parts", [])
                text = "".join(part.get("text", "") for part in parts).strip()
                if text:
                    return text

    if last_404_error:
        raise RuntimeError(last_404_error)

    raise RuntimeError("Gemini response did not contain usable text.")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Transcribe an audio file with AssemblyAI and optionally generate a SOAP or"
            " H&P clinical note using Gemini."
        )
    )
    parser.add_argument("audio_path", help="Path to the audio file to transcribe.")
    parser.add_argument(
        "--note-format",
        choices=["soap", "hp"],
        help="Generate the specified clinical note format using Gemini.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=3,
        help="Seconds to wait between transcription status checks (default: 3).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the transcript (and note if generated).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.audio_path):
        print(f"Audio file not found: {args.audio_path}", file=sys.stderr)
        return 1

    try:
        assembly_key = get_api_key("ASSEMBLYAI_API_KEY")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    headers = {"authorization": assembly_key}

    try:
        print("Uploading audio…", flush=True)
        audio_url = upload_audio(args.audio_path, headers)
        print("Audio uploaded. Starting transcription…", flush=True)
        transcript_id = request_transcription(audio_url, headers)
        result = poll_transcription(transcript_id, headers, poll_interval=args.poll_interval)
    except (requests.RequestException, TranscriptionError, RuntimeError) as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        return 1

    transcript_text = result.get("text", "").strip()
    if not transcript_text:
        print("Transcription completed but no text was returned.", file=sys.stderr)
        return 1

    output_lines = ["Transcript:\n", transcript_text, "\n"]

    note_text: Optional[str] = None
    if args.note_format:
        try:
            gemini_key = get_api_key("GEMINI_API_KEY")
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        try:
            print(f"Generating {args.note_format.upper()} note with Gemini…", flush=True)
            note_text = generate_clinical_note(transcript_text, args.note_format, gemini_key)
        except (requests.RequestException, RuntimeError) as exc:
            print(f"Gemini note generation failed: {exc}", file=sys.stderr)
            return 1

        header = "SOAP Note:" if args.note_format == "soap" else "H&P Note:"
        output_lines.extend([header, "\n", note_text, "\n"])

    output = "\n".join(output_lines).strip()
    print(output)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as out_file:
                out_file.write(output)
            print(f"Saved output to {args.output}")
        except OSError as exc:
            print(f"Failed to write output file: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())