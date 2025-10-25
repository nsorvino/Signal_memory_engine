# agents/__init__.py

import os

from dotenv import load_dotenv

from vector_store import init_pinecone_index

# 1) Load your Pinecone creds
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")

if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY in .env")

# 2) Define all of the indexes you want to guarantee exist
AGENT_INDEXES = {
    "signal-engine": {"dimension": 384, "metric": "cosine"},
    "axis-memory": {"dimension": 384, "metric": "cosine"},
    "oria-memory": {"dimension": 384, "metric": "cosine"},
    "sentinel-memory": {"dimension": 384, "metric": "cosine"},
}

# 3) Initialize (or recreate if mismatched) each one
for name, cfg in AGENT_INDEXES.items():
    # Only initialize if we have a real API key and we're not in test mode
    if os.getenv("PINECONE_API_KEY") and os.getenv("PINECONE_API_KEY") != "dummy":
        init_pinecone_index(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV,
            index_name=name,
            dimension=cfg["dimension"],
            metric=cfg["metric"],
        )
