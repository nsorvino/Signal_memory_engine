# api/routes/search.py

from fastapi import APIRouter, HTTPException, Query

from api import deps
from api.models import MemoryMatch

router = APIRouter(tags=["search"])


@router.get("/memory/search", response_model=list[MemoryMatch])
def search_memory(
    q: str = Query(..., description="Natural-language query"),
    top_k: int = Query(3, ge=1, le=20),
):
    # 1) Embed with shared embedder
    try:
        q_vec = deps.plain_embedder.embed_query(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    # 2) Query raw Pinecone index
    try:
        resp = deps.pinecone_index.query(
            vector=q_vec,
            top_k=top_k,
            include_metadata=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query error: {e}")

    # 3) Normalize response
    out: list[MemoryMatch] = []
    matches = getattr(resp, "matches", resp.get("matches", []))
    for m in matches:
        # support SDK object or dict
        mid = m.id if hasattr(m, "id") else m["id"]
        mscore = m.score if hasattr(m, "score") else m["score"]
        meta = m.metadata if hasattr(m, "metadata") else m.get("metadata", {}) or {}
        out.append(
            MemoryMatch(
                id=mid,
                score=float(mscore),
                text=meta.get("content") or meta.get("text"),
                agent=meta.get("agent"),
                tags=meta.get("tags"),
                metadata=meta,
            )
        )
    return out


@router.get("/memory/vector_query", response_model=list[MemoryMatch])
def vector_query(
    q: str = Query(..., description="Natural-language query (low-level vector path)"),
    top_k: int = Query(3, ge=1, le=20),
):
    # identical to search_memory; kept as a separate path if different behaviors are desired later
    return search_memory(q=q, top_k=top_k)
