import hashlib
import os
from textwrap import wrap

from langchain.embeddings.base import Embeddings

# LangChain-compatible OpenAI Embeddings Builder
from langchain_openai import OpenAIEmbeddings

# Sentence-Transformers for local embedding
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────
# Default models and keys
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Toggle source of embeddings: if set, use OpenAI; otherwise use local SentenceTransformer
USE_OPENAI = os.getenv("USE_OPENAI_EMBEDDINGS") is not None

# ── Initialize local model (384-dim) ────────────────────────
_local_model = SentenceTransformer("all-MiniLM-L6-v2")


# ── OpenAI Embedder Factory ───────────────────────────────
def get_embedder(model: str | None = None, openai_api_key: str | None = None) -> OpenAIEmbeddings:
    """
    Return a LangChain-compatible OpenAIEmbeddings instance.

    Parameters:
        model: the OpenAI embedding model name, default from env.
        openai_api_key: your OpenAI API key, default from env.
    """
    return OpenAIEmbeddings(
        model=model or DEFAULT_EMBED_MODEL,
        openai_api_key=openai_api_key or DEFAULT_API_KEY,
    )


# ── Unified embedding interface ─────────────────────────────
def get_embedding(text: str) -> list[float]:
    """
    Embed a single text string either locally (384-dim) or via OpenAI (1536-dim),
    based on the USE_OPENAI flag.

    Returns:
        A flat list of floats for Pinecone upsert.
    """
    if USE_OPENAI:
        embedder: Embeddings = get_embedder()
        vector = embedder.embed_query(text)  # type: ignore
        return list(vector)
    else:
        return _local_model.encode(text, show_progress_bar=False).tolist()


# ── Legacy local embeddings (explicit) ─────────────────────
def get_local_embedding(text: str) -> list[float]:
    """
    Embed a single text string locally using SentenceTransformer.
    """
    return _local_model.encode(text, show_progress_bar=False).tolist()


# ── Chunked local embedding ─────────────────────────────────
def process_text_to_embeddings(text: str, width: int = 512) -> list[list[float]]:
    """
    Split longer text into fixed-size chunks and embed each chunk locally.
    Returns a list of embedding vectors.
    """
    chunks = wrap(text, width)
    return [_local_model.encode(chunk, show_progress_bar=False).tolist() for chunk in chunks]


# ── Utility ID generator ───────────────────────────────────
def generate_id_from_text(text: str) -> str:
    """
    Deterministically generate a unique ID for a chunk of text.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()
