# tests/test_coherence_commons.py

"""
Coherence stage tests (lightweight & offline)

Targets: coherence/commons.py

What we verify:
1) flag_from_score: correct buckets at boundary values.
2) normalize_timestamp: accepts datetime/ISO strings and preserves unknown formats.
3) generate_event_id: deterministic based on (content + timestamp).
4) map_events_to_memory:
   - trims content
   - carries score through
   - emits correct flag + suggestion
   - includes source_agent when provided
   - normalizes/preserves timestamp appropriately
   - preserves extra metadata but not 'timestamp' duplicated in metadata
   - returns 1:1 number of events vs input documents
"""

import re
from datetime import datetime

from langchain.schema import Document

# Import the module under test
from coherence.commons import (
    SUGGESTIONS,
    flag_from_score,
    generate_event_id,
    map_events_to_memory,
    normalize_timestamp,
)


# ---------- flag_from_score ---------------------------------------------------
def test_flag_from_score_boundaries():
    """
    Thresholds in commons.flag_from_score:
        > 0.8  → "concern"
        > 0.5  → "drifting"
        else   → "stable"
    Check boundary behavior and typical values.
    """
    # Exactly 0.5 should be "stable"
    assert flag_from_score(0.5) == "stable"

    # Just above 0.5 should be "drifting"
    assert flag_from_score(0.50001) == "drifting"
    assert flag_from_score(0.7) == "drifting"

    # Exactly 0.8 should be "drifting"
    assert flag_from_score(0.8) == "drifting"

    # Just above 0.8 should be "concern"
    assert flag_from_score(0.80001) == "concern"
    assert flag_from_score(0.95) == "concern"

    # Low value → stable
    assert flag_from_score(0.0) == "stable"


# ---------- normalize_timestamp ----------------------------------------------
def test_normalize_timestamp_variants():
    """
    normalize_timestamp accepts a datetime or an ISO-like string
    and returns an ISO 8601 string; if it can't parse, it returns str(ts).
    """
    # 1) datetime input → isoformat
    dt = datetime(2024, 1, 2, 3, 4, 5)
    assert normalize_timestamp(dt) == dt.isoformat()

    # 2) ISO string input → isoformat-preserving
    iso_str = "2024-05-06T07:08:09"
    assert normalize_timestamp(iso_str) == iso_str

    # 3) Non-ISO string → returns as-is (stringified)
    weird = "2024/05/06 07:08:09"  # with slashes; not ISO
    assert normalize_timestamp(weird) == weird

    # 4) Non-string, non-datetime → stringified
    assert normalize_timestamp(12345) == "12345"


# ---------- generate_event_id -------------------------------------------------
def test_generate_event_id_deterministic_and_sensitive_to_inputs():
    """
    Event ID should be deterministic given the same (content + timestamp)
    and change if either the content or timestamp changes.
    """
    content = "Hello world"
    ts1 = "2024-01-01T00:00:00"
    ts2 = "2024-01-02T00:00:00"

    a = generate_event_id(content, ts1)
    b = generate_event_id(content, ts1)
    c = generate_event_id(content, ts2)
    d = generate_event_id(content + "!", ts1)

    assert a == b, "Same inputs must produce the same ID"
    assert a != c, "Different timestamp should change the ID"
    assert a != d, "Different content should change the ID"

    assert re.fullmatch(r"[a-f0-9]{32}", a)


# ---------- map_events_to_memory ---------------------------------------------
def test_map_events_to_memory_core_fields_and_flags():
    """
    map_events_to_memory returns a normalized list of dicts containing:
    - event_id (hash)
    - content (trimmed)
    - score
    - tags [flag]
    - suggestion (based on flag)
    - timestamp (if provided)
    - source_agent (if provided)
    - metadata (other metadata, excluding 'timestamp')
    """
    docs_and_scores = [
        # Concern (score > 0.8) + timestamp + extra metadata
        (
            Document(
                page_content="  critical signal here  ",
                metadata={"timestamp": "2024-01-01T12:34:56", "foo": "bar"},
            ),
            0.91,
        ),
        # Drifting (0.5 < score <= 0.8) with non-ISO timestamp
        (
            Document(
                page_content=" borderline case ",
                metadata={"timestamp": "2024/01/02 12:00:00", "source": "unit-test"},
            ),
            0.75,
        ),
        # Stable (score <= 0.5) without timestamp
        (
            Document(
                page_content=" low risk ",
                metadata={"note": "no timestamp here"},
            ),
            0.2,
        ),
    ]

    out = map_events_to_memory(docs_and_scores, source_agent="Oria")

    # 1:1 count
    assert len(out) == len(docs_and_scores)

    # --- First event: concern
    e0 = out[0]
    assert e0["content"] == "critical signal here"  # trimmed
    assert e0["score"] == 0.91
    assert e0["tags"] == ["concern"]
    assert e0["suggestion"] == SUGGESTIONS["concern"]
    assert e0["source_agent"] == "Oria"
    # Timestamp should be preserved as ISO
    assert e0["timestamp"] == "2024-01-01T12:34:56"
    # Metadata should include 'foo' but not duplicate 'timestamp'
    assert "metadata" in e0 and e0["metadata"]["foo"] == "bar"
    assert "timestamp" not in e0.get("metadata", {})

    # --- Second event: drifting, non-ISO timestamp is kept as string
    e1 = out[1]
    assert e1["tags"] == ["drifting"]
    assert e1["suggestion"] == SUGGESTIONS["drifting"]
    # Non-ISO timestamp should round trip as-is (commons keeps string)
    assert e1["timestamp"] == "2024/01/02 12:00:00"
    assert "metadata" in e1
    assert "source" in e1["metadata"]
    assert "timestamp" not in e1["metadata"]

    # --- Third event: stable, no timestamp key at all
    e2 = out[2]
    assert e2["tags"] == ["stable"]
    assert e2["suggestion"] == SUGGESTIONS["stable"]
    assert "timestamp" not in e2  # not provided → not present


def test_map_events_to_memory_empty_input():
    """Empty input should return an empty list (no surprises)."""
    assert map_events_to_memory([]) == []
