# agents/sentinel.py
import os
from functools import lru_cache

from dotenv import load_dotenv

from scripts.langchain_retrieval import build_qa_chain

load_dotenv()

ROLE_SENTINEL = "M™ Shadow Sentinel"


@lru_cache(maxsize=1)
def get_sentinel_chain():
    return build_qa_chain(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_env=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp"),
        index_name=os.getenv("PINECONE_INDEX_SENTINEL", "sentinel-memory"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        k=3,
    )
