from flask import Blueprint, request, jsonify, current_app, abort
from tensorflow.keras.preprocessing.sequence import pad_sequences

notes_bp = Blueprint("notes", __name__)


def get_summarizer():
    """Retrieve the HF summarization pipeline from app config."""
    summarizer = current_app.config.get("SUMMARIZER")
    if not summarizer:
        current_app.logger.error("Summarizer not configured")
        abort(500, "Summarizer unavailable")
    return summarizer


def extract_text():
    """Parse and validate incoming JSON for the `/process` endpoint."""
    data = request.get_json(silent=True) or {}
    text = data.get("text_input", "").strip()
    if not text:
        abort(400, "No input provided")
    return text


def summarize_text(text: str) -> str:
    """Run the HF pipeline and return the generated summary."""
    pipe = get_summarizer()
    try:
        out = pipe(
            text,
            max_length=current_app.config["MAX_LENGTH_TARGET"],
            min_length=int(current_app.config.get("MIN_LENGTH_TARGET", 5)),
            do_sample=False
        )
        return out[0].get("summary_text", "")
    except Exception as e:
        current_app.logger.error(f"Summarization error: {e}")
        abort(500, "Error during summarization")
