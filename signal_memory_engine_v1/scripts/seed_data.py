#!/usr/bin/env python
# scripts/seed_data.py

import glob
import json
import os

from dotenv import load_dotenv

load_dotenv()

from vector_store.embeddings import get_embedder
from vector_store.pinecone_index import index, init_pinecone_index

# ── CONFIG ───────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = os.getenv("PINECONE_INDEX", "signal-engine")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_DIR = "data/seed"
BATCH_SIZE = 100


# ── UTILITIES ────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def normalize_record(rec):
    """
    Returns (text, metadata, id) for each record, regardless of its schema.
    """
    text = rec.get("content") or rec.get("text") or rec.get("excerpt") or ""
    metadata = {k: v for k, v in rec.items() if k not in ("content", "text", "excerpt")}
    rid = rec.get("id") or rec.get("thread_id") or os.urandom(8).hex()
    return text, metadata, rid


# ── INGESTION FUNCTION ──────────────────────────────────────
def upsert_seed_memories(data_dir: str = DATA_DIR):
    if not (PINECONE_API_KEY and OPENAI_API_KEY):
        raise RuntimeError("Set PINECONE_API_KEY and OPENAI_API_KEY in .env")

    # 1) init Pinecone
    init_pinecone_index(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
        index_name=INDEX_NAME,
        dimension=384,
        metric="cosine",
    )
    # 2) embedder
    embedder = get_embedder(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

    # 3) discover files
    patterns = [
        "seed_memory*.jsonl",
        "drift_simulation*.jsonl",
        "breakdown_examples*.jsonl",
        "seed_memories*.json",
    ]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(data_dir, pat)))

    to_upsert = []
    for path in paths:
        print(f">>> ingesting {path}")
        records = load_jsonl(path) if path.endswith(".jsonl") else load_json(path)
        for rec in records:
            text, metadata, rid = normalize_record(rec)
            vec = embedder.embed_documents([text])[0]
            to_upsert.append((rid, vec, metadata))

            if len(to_upsert) >= BATCH_SIZE:
                index.upsert(vectors=to_upsert)
                to_upsert.clear()

    if to_upsert:
        index.upsert(vectors=to_upsert)

    print("✅ All seed data upserted to Pinecone.")


# ── ENTRYPOINT ──────────────────────────────────────────────
if __name__ == "__main__":
    upsert_seed_memories()
