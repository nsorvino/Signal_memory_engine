#!/usr/bin/env python
# scripts/ingest_memories.py

import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

# ensure project root is on PYTHONPATH so vector_store can be imported
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv()

from vector_store.embeddings import get_embedding
from vector_store.pinecone_index import init_pinecone_index

# 1) Initialize Pinecone & grab the Index object
index = init_pinecone_index(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV", "us-east-1"),
    index_name=os.getenv("PINECONE_INDEX", "signal-engine"),
    dimension=384,
    metric="cosine",
)


def load_jsonl(path: str):
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        yield json.loads(line)


def normalize_record(rec: dict, source: str):
    """
    Map varied input schemas into a uniform Pinecone upsert payload:
      - id: unique record id
      - content: text to embed/retrieve
      - metadata: flat dict of primitives (no nested dicts)
    """
    # pick out id, content, then collect everything else as meta
    if source == "seed":
        rid = rec.get("id") or uuid.uuid4().hex
        content = rec["content"]
        raw_meta = {k: v for k, v in rec.items() if k not in ("id", "content")}
    elif source == "drift":
        rid = rec.get("timestamp") or uuid.uuid4().hex
        content = rec.get("notes", "")
        raw_meta = {
            "timestamp": rec.get("timestamp"),
            **rec.get("flags", {}),
            **rec.get("biometrics", {}),
        }
    elif source == "breakdown":
        rid = rec.get("thread_id") or uuid.uuid4().hex
        content = rec.get("excerpt", "")
        raw_meta = {k: v for k, v in rec.items() if k not in ("thread_id", "excerpt")}
    else:
        rid, content, raw_meta = uuid.uuid4().hex, "", {}

    raw_meta["source"] = source

    # flatten any nested dicts into top-level keys
    flat_meta = {}
    for k, v in raw_meta.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                # only primitives or list[str] allowed:
                if isinstance(subv, str | int | float | bool) or (
                    isinstance(subv, list) and all(isinstance(x, str) for x in subv)
                ):
                    flat_meta[subk] = subv
                else:
                    # fallback to string
                    flat_meta[subk] = str(subv)
        else:
            flat_meta[k] = v

    return rid, content, flat_meta


def upsert_file(path: str, source: str):
    records = list(load_jsonl(path))
    batch = []
    for rec in records:
        rid, text, meta = normalize_record(rec, source)
        vec = get_embedding(text)
        batch.append((rid, vec, meta))

    # Pinecone upsert takes list of (id, vector, metadata_dict)
    index.upsert(vectors=batch)
    print(f"[{source}] upserted {len(batch)} items")


if __name__ == "__main__":
    # adjust these paths if your data lives elsewhere
    upsert_file("data/seed_memory.jsonl", source="seed")
    upsert_file("data/drift_simulation.jsonl", source="drift")
    upsert_file("data/breakdown_examples.jsonl", source="breakdown")
