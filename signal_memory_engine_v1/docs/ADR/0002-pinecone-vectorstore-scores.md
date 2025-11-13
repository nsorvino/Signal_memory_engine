# ADR 0002: Implement Pinecone vector store score-aware search

## Status
Accepted — 2025-11-13

## Context
Our FastAPI `/query` route relies on LangChain’s `VectorStore` interface. During a request the API calls `similarity_search_with_score` to retrieve document/score pairs. The Pinecone-backed vector store we vend in `vector_store/pinecone_vectorstore.py` inherited the base `VectorStore` default, which raises `NotImplementedError`. As soon as the endpoint attempted retrieval, the exception propagated, and the API returned `502 Vectorstore retrieval failed`.

The endpoint works in tests because we usually exercise the stubbed vector store; the regression surfaced only when running the live Pinecone-backed implementation.

## Decision
Implement true score-aware retrieval inside `PineconeVectorStore`:

- Added `_similarity_search_with_scores` helper that executes a single Pinecone query and yields `(Document, score)` tuples.
- Reimplemented `similarity_search` and `similarity_search_with_score` to reuse the helper so both methods stay consistent and we avoid duplicate queries.
- Left the public API surface unchanged for callers such as `scripts.langchain_retrieval.build_qa_chain`.

This guarantees the LangChain contract is honored when the API requests scored results.

## Consequences
- The API now returns successful responses instead of 502 errors when Pinecone is configured.
- Downstream code (e.g., scoring logic) can rely on populated `score` values for each match.
- Adds a tiny amount of code duplication mitigation by centralizing the query → document mapping.
- No behavioural change for environments still using the stubbed Pinecone client; they continue to receive empty matches with default score `0.0`.

