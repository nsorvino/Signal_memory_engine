# vector_store/__init__.py

from .embeddings import get_embedding
from .pinecone_index import init_pinecone_index

__all__ = ["init_pinecone_index", "get_embedding"]
