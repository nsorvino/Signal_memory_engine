# coherence/commons.py
"""
Common utilities for mapping RAG hits (Document, score) pairs into normalized event-memory structures.
"""
import hashlib
from datetime import datetime
from typing import Any

from langchain.schema import Document


def generate_event_id(content: str, timestamp: str = "") -> str:
    """
    Deterministically generate a unique ID for an event based on its content and optional timestamp.
    """
    base = content + timestamp
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def normalize_timestamp(ts: Any) -> str:
    """
    Normalize a timestamp to ISO 8601 string if possible.
    Accepts datetime or ISO-like string.
    """
    if isinstance(ts, datetime):
        return ts.isoformat()
    try:
        # attempt parse string
        dt = datetime.fromisoformat(str(ts))
        return dt.isoformat()
    except Exception:
        # fallback to raw string
        return str(ts)


def flag_from_score(score: float) -> str:
    """
    Convert a similarity score (0–1) into a signal flag.
    """
    if score > 0.8:
        return "concern"
    elif score > 0.5:
        return "drifting"
    else:
        return "stable"


SUGGESTIONS = {
    "stable": "No action needed.",
    "drifting": "Consider sending a check-in message.",
    "concern": "Recommend escalation or a one-on-one conversation.",
}


def map_events_to_memory(
    docs_and_scores: list[tuple[Document, float]],
    source_agent: str | None = None,
) -> list[dict[str, Any]]:
    """
    Convert a list of (Document, score) into a list of normalized event dictionaries.

    Each event will include:
      - event_id: unique hash
      - content: the document page_content
      - score: the similarity score
      - source_agent: name of the agent that produced this hit (if provided)
      - timestamp: if present in document.metadata (normalized)
      - tags: list of flags (e.g., "stable", "drifting", "concern")
      - suggestion: action suggestion based on the flag
      - metadata: other metadata fields, if any
    """
    events: list[dict[str, Any]] = []
    for doc, score in docs_and_scores:
        meta = doc.metadata or {}
        ts_raw = meta.get("timestamp")
        timestamp = normalize_timestamp(ts_raw) if ts_raw is not None else None

        flag = flag_from_score(score)
        suggestion = SUGGESTIONS.get(flag, "")

        event: dict[str, Any] = {
            "event_id": generate_event_id(doc.page_content, timestamp or ""),
            "content": doc.page_content.strip(),
            "score": score,
            "tags": [flag],
            "suggestion": suggestion,
        }
        if source_agent:
            event["source_agent"] = source_agent
        if timestamp:
            event["timestamp"] = timestamp
        # include any other metadata
        extra_meta = {k: v for k, v in meta.items() if k != "timestamp"}
        if extra_meta:
            event["metadata"] = extra_meta

        events.append(event)
    return events
