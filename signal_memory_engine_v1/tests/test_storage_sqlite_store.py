# tests/test_storage_sqlite_store.py

"""
Storage layer tests (sqlite_store)

Covers:
1) init_db creates the schema and is idempotent.
2) insert_event writes rows with correct coercions (timestamp default, escalate_flag int).
3) list_recent returns newest-first, honors limit, and exposes payload as TEXT (JSON string if dict was given).
4) list_by_user filters by user_id and honors limit.
5) Schema sanity: events table contains expected columns.

These tests never touch external services and use a temporary DB file.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from storage.sqlite_store import (
    init_db,
    insert_event,
    list_by_user,
    list_recent,
)

# ---------- helpers -----------------------------------------------------------


def _open_sqlite(db_path: str) -> sqlite3.Connection:
    """Open a sqlite connection to assert on internal schema/rows."""
    return sqlite3.connect(db_path, check_same_thread=False)


# ---------- tests -------------------------------------------------------------


def test_init_db_idempotent_and_schema(tmp_path: Path):
    """
    init_db should:
      - create required tables if missing
      - insert a single row into schema_meta only once
      - be safe to call multiple times
    """
    db_path = str(tmp_path / "signal.db")

    # First init
    init_db(db_path)
    with _open_sqlite(db_path) as cx:
        # schema_meta must exist and contain exactly one version row
        (count,) = cx.execute("SELECT COUNT(*) FROM schema_meta").fetchone()
        assert count == 1

        # events table should exist with expected columns
        cols = [c[1] for c in cx.execute("PRAGMA table_info(events)")]
        expected = {
            "id",
            "timestamp",
            "user_id",
            "agent_id",
            "signal_type",
            "emotional_tone",
            "drift_score",
            "escalate_flag",
            "payload",
            "relationship_context",
            "diagnostic_notes",
        }
        assert set(cols) == expected

    # Second init — idempotent (should not add another schema_meta row)
    init_db(db_path)
    with _open_sqlite(db_path) as cx:
        (count_again,) = cx.execute("SELECT COUNT(*) FROM schema_meta").fetchone()
        assert count_again == 1, "init_db must be idempotent"


def test_insert_and_list_recent_ordering_and_payload_json(tmp_path: Path):
    """
    insert_event should:
      - auto-fill timestamp if not provided
      - coerce escalate_flag to 0/1
      - JSON-encode dict payloads into TEXT

    list_recent should:
      - return newest-first (DESC id)
      - obey the provided limit
      - surface payload as TEXT (caller can json.loads if needed)
    """
    db_path = str(tmp_path / "signal.db")
    init_db(db_path)

    # Insert first event (dict payload → JSON text in DB)
    e1 = {
        "user_id": "u1",
        "agent_id": "Axis",
        "signal_type": "relational",
        "emotional_tone": 0.8,
        "drift_score": 0.4,
        "escalate_flag": False,  # should become 0
        "payload": {"foo": "bar"},
        "relationship_context": "team-1",
        "diagnostic_notes": "note-1",
    }
    id1 = insert_event(db_path, e1)
    assert isinstance(id1, int) and id1 >= 1

    # Insert second event
    e2 = {
        "user_id": "u2",
        "agent_id": "M",
        "signal_type": "compliance",
        "emotional_tone": 0.2,
        "drift_score": 0.9,
        "escalate_flag": True,  # should become 1
        "payload": "raw-payload",
        "relationship_context": "team-2",
        "diagnostic_notes": "note-2",
    }
    id2 = insert_event(db_path, e2)
    assert id2 == id1 + 1, "IDs should be auto-incrementing"

    # Query: newest first, limit 2
    rows = list_recent(db_path, limit=2)
    assert [r["id"] for r in rows] == [id2, id1]

    # Check storage of fields and coercions
    r2, r1 = rows[0], rows[1]

    # Timestamps present and ISO-like (contain 'T')
    assert "timestamp" in r2 and "T" in r2["timestamp"]
    assert "timestamp" in r1 and "T" in r1["timestamp"]

    # Escalate flags coerced to ints
    assert r2["escalate_flag"] == 1
    assert r1["escalate_flag"] == 0

    # - e2 was a string → same string
    assert r2["payload"] == "raw-payload"

    # - e1 was a dict → JSON string we can parse back
    parsed = json.loads(r1["payload"])
    assert parsed == {"foo": "bar"}


def test_list_by_user_filters_and_limit(tmp_path: Path):
    """
    list_by_user should return only the requested user's events,
    newest-first, and honor the limit.
    """
    db_path = str(tmp_path / "signal.db")
    init_db(db_path)

    # Insert 3 events across two users
    ids = []
    ids.append(
        insert_event(
            db_path,
            {
                "user_id": "alice",
                "agent_id": "Oria",
                "signal_type": "biometric",
                "emotional_tone": 0.1,
                "drift_score": 0.2,
                "escalate_flag": False,
                "payload": {"a": 1},
                "relationship_context": "grp-a",
                "diagnostic_notes": "ok",
            },
        )
    )
    ids.append(
        insert_event(
            db_path,
            {
                "user_id": "alice",
                "agent_id": "Oria",
                "signal_type": "relational",
                "emotional_tone": 0.2,
                "drift_score": 0.3,
                "escalate_flag": False,
                "payload": {"b": 2},
                "relationship_context": "grp-a",
                "diagnostic_notes": "ok",
            },
        )
    )
    ids.append(
        insert_event(
            db_path,
            {
                "user_id": "bob",
                "agent_id": "M",
                "signal_type": "compliance",
                "emotional_tone": 0.9,
                "drift_score": 0.7,
                "escalate_flag": True,
                "payload": {"c": 3},
                "relationship_context": "grp-b",
                "diagnostic_notes": "watch",
            },
        )
    )

    # Fetch only Alice's last 1 event
    rows = list_by_user(db_path, user_id="alice", limit=1)
    assert len(rows) == 1
    assert rows[0]["user_id"] == "alice"
    # Should be the most recent Alice event (the second insert for alice)
    assert rows[0]["id"] == ids[1]


def test_insert_event_infers_timestamp_when_missing(tmp_path: Path):
    """
    If the caller does not supply a timestamp, insert_event should fill it
    with an ISO 8601 UTC timestamp string.
    """
    db_path = str(tmp_path / "signal.db")
    init_db(db_path)

    eid = insert_event(
        db_path,
        {
            "user_id": "u3",
            "agent_id": "Axis",
            "signal_type": "relational",
            "emotional_tone": 0.4,
            "drift_score": 0.1,
            "escalate_flag": False,
            "payload": {"msg": "hi"},
        },
    )

    rows = list_recent(db_path, limit=1)
    assert rows and rows[0]["id"] == eid
    ts = rows[0]["timestamp"]
    assert isinstance(ts, str) and "T" in ts
