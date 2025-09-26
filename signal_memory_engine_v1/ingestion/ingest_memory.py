#!/usr/bin/env python
# ingestion/ingest_memory.py

import csv
import json
import os
import uuid
from pathlib import Path
from typing import Any

from vector_store.embeddings import get_embedder
from vector_store.pinecone_index import init_pinecone_index

# ── CONFIG ─────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
INDEX_NAME = os.getenv("PINECONE_INDEX", "signal-engine")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# list every data file you want ingested here
DATA_FILES = [
    "data/seed_memory.jsonl",
    "data/drift_simulation.jsonl",
    "data/breakdown_examples.jsonl",
    "data/seed_memories.json",  # JSON array
    # "data/your_other.csv",     # if you have CSV
]

# ── INITIALIZE ──────────────────────────────────────────────────────────────
index = init_pinecone_index(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV,
    index_name=INDEX_NAME,
    dimension=384,
    metric="cosine",
)

embedder = get_embedder(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-ada-002",
)


def normalize_record(rec: dict[str, Any], source: str) -> tuple[str, str, dict[str, Any]]:
    """
    Turn any input record into (id, content, metadata).
    Looks for common content fields, falls back to uuid.
    """
    # possible content keys in your files:
    for content_key in ("content", "text", "excerpt", "event_content"):
        if content_key in rec:
            content = rec.pop(content_key)
            break
    else:
        raise ValueError(f"No content field in record: {rec}")

    # pick an existing id if present, else make one
    record_id = (
        rec.get("id")
        or rec.get("thread_id")
        or rec.get("timestamp")
        or f"{source}-{uuid.uuid4().hex}"
    )

    # metadata: include source and all remaining fields
    meta = {"source": source}
    meta.update(rec)
    return record_id, content, meta


def upsert_batch(batch: list[tuple[str, list[float], dict[str, Any]]], batch_size: int = 100):
    """Upsert vectors to Pinecone in batches."""
    for i in range(0, len(batch), batch_size):
        index.upsert(vectors=batch[i : i + batch_size])


def ingest_list(recs: list[dict[str, Any]], source: str):
    batch = []
    for rec in recs:
        try:
            rid, content, meta = normalize_record(rec, source)
        except ValueError as e:
            print(f"⚠️  Skipping record: {e}")
            continue
        vec = embedder.embed_query(content)
        batch.append((rid, vec, meta))
    upsert_batch(batch)
    print(f"[{source}] upserted {len(batch)} records.")


def ingest_jsonl(path: Path, source: str):
    print(f"[{source}] ingesting JSONL: {path}")
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
    ingest_list(recs, source)


def ingest_json(path: Path, source: str):
    print(f"[{source}] ingesting JSON: {path}")
    data = json.load(open(path))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of records")
    ingest_list(data, source)


def ingest_csv(path: Path, source: str):
    print(f"[{source}] ingesting CSV: {path}")
    recs = []
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            recs.append(row)
    ingest_list(recs, source)


def main():
    for filepath in DATA_FILES:
        p = Path(filepath)
        source = p.stem  # e.g. "seed_memory"
        if not p.exists():
            print(f"⚠️  File not found, skipping: {p}")
            continue
        ext = p.suffix.lower()
        try:
            if ext == ".jsonl":
                ingest_jsonl(p, source)
            elif ext == ".json":
                ingest_json(p, source)
            elif ext == ".csv":
                ingest_csv(p, source)
            else:
                print(f"❓  Unsupported extension {ext}, skipping {p}")
        except Exception as e:
            print(f"❌  Error ingesting {p}: {e}")
    print("✅ All ingestion tasks complete.")


if __name__ == "__main__":
    main()
