# processing/normalizer.py
# This is a work in progress and is meant for when we have data incoming that needs to be cleaned up.

import re
from datetime import datetime
from typing import Any


def normalize_event(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Take a raw event dict (e.g. chat message, biometric sample, log line)
    and return a clean, well‐typed version with:
      - standardized keys (“text”, “timestamp”, “source”, etc.)
      - ISO-8601 timestamp
      - stripped control characters/HTML
    """
    clean: dict[str, Any] = {}

    # 1) Canonicalize timestamp
    ts = raw.get("timestamp") or raw.get("time") or raw.get("ts")
    try:
        # if it's numeric (epoch), convert to ISO
        if isinstance(ts, int | float):
            clean["timestamp"] = datetime.fromtimestamp(ts).isoformat()
        else:
            clean["timestamp"] = datetime.fromisoformat(str(ts)).isoformat()
    except Exception:
        clean["timestamp"] = datetime.utcnow().isoformat()

    # 2) Canonicalize content
    text = raw.get("text") or raw.get("content") or raw.get("message") or ""
    # strip HTML tags, control chars, collapse whitespace
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text).strip()
    clean["text"] = text

    # 3) Source / agent tag
    clean["source"] = raw.get("agent") or raw.get("source") or "unknown"

    # 4) Carry forward any other metadata
    for k, v in raw.items():
        if k not in ("timestamp", "time", "ts", "text", "content", "message", "agent", "source"):
            clean[k] = v

    return clean
