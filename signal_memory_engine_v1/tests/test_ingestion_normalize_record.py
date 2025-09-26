# signal_memory_engine_v1/tests/test_ingestion_ingest_memory.py
import importlib
import json
import os
from pathlib import Path

import pytest


def import_mod():
    # Import the module in "offline/test" mode so init side-effects are stubbed
    os.environ["SME_TEST_MODE"] = "1"
    return importlib.import_module("signal_memory_engine_v1.ingestion.ingest_memory")


class FakeEmbedder:
    def embed_query(self, text: str):
        # small deterministic vector for tests
        return [len(text) * 0.1, 0.2, 0.3]


# ----------------------
# normalize_record tests
# ----------------------
def test_normalize_record_variants():
    mod = import_mod()

    # content under "content"
    rid, content, meta = mod.normalize_record({"id": "abc", "content": "Hello", "x": 1}, "seed")
    assert rid == "abc"
    assert content == "Hello"
    assert meta["source"] == "seed" and meta["x"] == 1

    # content under "text" -> synthesize id
    rid2, content2, meta2 = mod.normalize_record({"text": "Hi"}, "file2")
    assert content2 == "Hi" and meta2["source"] == "file2" and rid2

    # content under "excerpt"
    rid3, content3, meta3 = mod.normalize_record({"excerpt": "Piece"}, "file3")
    assert content3 == "Piece" and meta3["source"] == "file3"

    # content under "event_content"
    rid4, content4, meta4 = mod.normalize_record({"event_content": "Log line"}, "file4")
    assert content4 == "Log line" and meta4["source"] == "file4"

    # missing content -> ValueError
    with pytest.raises(ValueError):
        mod.normalize_record({"id": "nope"}, "bad")


# ----------------------
# ingest / upsert tests
# ----------------------
def test_upsert_batch_chunks(monkeypatch):
    mod = import_mod()

    calls = []

    def fake_upsert(*, vectors):
        calls.append(list(vectors))
        return {"upserted": len(vectors)}

    monkeypatch.setattr(mod.index, "upsert", fake_upsert, raising=True)

    batch = [
        ("id1", [0.1, 0.2], {"a": 1}),
        ("id2", [0.1, 0.2], {"a": 2}),
        ("id3", [0.1, 0.2], {"a": 3}),
        ("id4", [0.1, 0.2], {"a": 4}),
        ("id5", [0.1, 0.2], {"a": 5}),
    ]
    mod.upsert_batch(batch, batch_size=2)

    # expect 3 chunks: [2,2,1]
    assert len(calls) == 3
    assert len(calls[0]) == 2 and len(calls[1]) == 2 and len(calls[2]) == 1


def test_ingest_list_happy_and_skip(monkeypatch, capsys):
    mod = import_mod()
    monkeypatch.setattr(mod, "embedder", FakeEmbedder(), raising=False)

    calls = []

    def fake_upsert(*, vectors):
        calls.append(vectors)
        return {"upserted": len(vectors)}

    monkeypatch.setattr(mod.index, "upsert", fake_upsert, raising=True)

    # one good record + one malformed (missing content) -> should skip 1
    recs = [
        {"content": "hello", "x": 1},
        {"id": "no-content"},
    ]
    mod.ingest_list(recs, source="unit")

    out = capsys.readouterr().out
    assert "[unit] upserted 1 records." in out
    assert len(calls) == 1 and len(calls[0]) == 1


def test_ingest_jsonl_json_csv(monkeypatch, tmp_path: Path, capsys):
    mod = import_mod()
    monkeypatch.setattr(mod, "embedder", FakeEmbedder(), raising=False)

    total_upserted = {"n": 0}

    def fake_upsert(*, vectors):
        total_upserted["n"] += len(vectors)
        return {"upserted": len(vectors)}

    monkeypatch.setattr(mod.index, "upsert", fake_upsert, raising=True)

    # --- jsonl ---
    p_jsonl = tmp_path / "seed_memory.jsonl"
    p_jsonl.write_text("\n".join([json.dumps({"content": "A"}), json.dumps({"text": "B"})]))
    mod.ingest_jsonl(p_jsonl, "seed_memory")

    # --- json (list) ---
    p_json = tmp_path / "drift_simulation.json"
    p_json.write_text(json.dumps([{"excerpt": "C"}]))
    mod.ingest_json(p_json, "drift_simulation")

    # --- csv ---
    p_csv = tmp_path / "breakdown_examples.csv"
    p_csv.write_text("content,x\nD,1\nE,2\n")
    mod.ingest_csv(p_csv, "breakdown_examples")

    # 2 (jsonl) + 1 (json) + 2 (csv) = 5
    assert total_upserted["n"] == 5

    out = capsys.readouterr().out
    assert "[seed_memory] ingesting JSONL" in out
    assert "[drift_simulation] ingesting JSON: " in out
    assert "[breakdown_examples] ingesting CSV: " in out


def test_ingest_json_requires_list_raises(monkeypatch, tmp_path: Path):
    mod = import_mod()
    monkeypatch.setattr(mod, "embedder", FakeEmbedder(), raising=False)

    p_bad = tmp_path / "notalist.json"
    p_bad.write_text(json.dumps({"content": "oops"}))
    with pytest.raises(ValueError):
        mod.ingest_json(p_bad, "notalist")


def test_main_mixed_paths(monkeypatch, tmp_path: Path, capsys):
    mod = import_mod()
    monkeypatch.setattr(mod, "embedder", FakeEmbedder(), raising=False)

    # Good files
    p_jsonl = tmp_path / "good.jsonl"
    p_jsonl.write_text(json.dumps({"content": "x"}) + "\n")

    p_json = tmp_path / "good.json"
    p_json.write_text(json.dumps([{"content": "y"}]))

    p_csv = tmp_path / "good.csv"
    p_csv.write_text("content\nz\n")

    # Unsupported extension
    p_txt = tmp_path / "bad.txt"
    p_txt.write_text("whatever")

    # Missing file
    missing = tmp_path / "missing.json"

    # Point DATA_FILES at our temp files (plus a missing one)
    monkeypatch.setattr(
        mod,
        "DATA_FILES",
        [str(p_jsonl), str(p_json), str(p_csv), str(p_txt), str(missing)],
        raising=False,
    )

    calls = []

    def fake_upsert(*, vectors):
        calls.append(vectors)
        return {"upserted": len(vectors)}

    monkeypatch.setattr(mod.index, "upsert", fake_upsert, raising=True)

    mod.main()

    # We ingested 3 valid files with 1 row each
    assert sum(len(v) for v in calls) == 3

    out = capsys.readouterr().out
    assert "File not found, skipping:" in out
    assert "Unsupported extension .txt" in out
    assert "All ingestion tasks complete." in out
