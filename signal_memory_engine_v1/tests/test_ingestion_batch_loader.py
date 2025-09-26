# tests/test_ingestion_batch_loader.py

"""
Covers BatchLoader.load():
- trims content
- skips empty rows
- preserves tags/agent
- ensures an id exists
"""

import json
from pathlib import Path

from ingestion.batch_loader import BatchLoader


def test_batch_loader_normalizes_entries(tmp_path: Path):
    raw = [
        {
            "content": " First, hydrate the vector store.  ",
            "tags": ["setup"],
            "agent": "Axis",
        },
        {"content": "  "},  # skipped (empty after strip)
        {"id": "custom-123", "content": "Keep an eye on drift.", "tags": []},
    ]
    data_file = tmp_path / "seed_memories.json"
    data_file.write_text(json.dumps(raw), encoding="utf-8")

    loader = BatchLoader(str(data_file))
    rows = loader.load()

    assert len(rows) == 2
    assert rows[0]["content"] == "First, hydrate the vector store."
    assert rows[0]["tags"] == ["setup"]
    assert rows[0]["agent"] == "Axis"
    assert rows[1]["id"] == "custom-123"
    assert rows[1]["tags"] == []
    assert all(r["id"] and r["content"] for r in rows)
