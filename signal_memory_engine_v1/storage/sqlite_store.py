import json
import os
import sqlite3
from datetime import datetime

SCHEMA_VERSION = 1


def _conn(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path, check_same_thread=False)


def init_db(db_path: str):
    with _conn(db_path) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS schema_meta(version INTEGER NOT NULL)")
        version = cx.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
        cx.execute(
            """CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            user_id TEXT,
            agent_id TEXT,
            signal_type TEXT,
            emotional_tone REAL,
            drift_score REAL,
            escalate_flag INTEGER NOT NULL,
            payload TEXT,
            relationship_context TEXT,
            diagnostic_notes TEXT
        )"""
        )
        if not version:
            cx.execute("INSERT INTO schema_meta(version) VALUES (?)", (SCHEMA_VERSION,))
        cx.commit()


def insert_event(db_path: str, event: dict) -> int:
    with _conn(db_path) as cx:
        cols = (
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
        )
        payload = (
            json.dumps(event.get("payload"))
            if isinstance(event.get("payload"), dict | list)
            else event.get("payload")
        )
        row = (
            event.get("timestamp") or datetime.utcnow().isoformat(),
            event.get("user_id"),
            event.get("agent_id"),
            event.get("signal_type"),
            event.get("emotional_tone"),
            event.get("drift_score"),
            1 if event.get("escalate_flag") else 0,
            payload,
            event.get("relationship_context"),
            event.get("diagnostic_notes"),
        )
        cx.execute(
            f"INSERT INTO events({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", row
        )
        cx.commit()
        return cx.execute("SELECT last_insert_rowid()").fetchone()[0]


def list_recent(db_path: str, limit: int = 10) -> list[dict]:
    with _conn(db_path) as cx:
        rows = cx.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        cols = [c[1] for c in cx.execute("PRAGMA table_info(events)")]
        return [dict(zip(cols, r)) for r in rows]


def list_by_user(db_path: str, user_id: str, limit: int = 10) -> list[dict]:
    with _conn(db_path) as cx:
        rows = cx.execute(
            "SELECT * FROM events WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit)
        ).fetchall()
        cols = [c[1] for c in cx.execute("PRAGMA table_info(events)")]
        return [dict(zip(cols, r)) for r in rows]
