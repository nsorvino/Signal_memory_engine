# signal_memory_engine_v1/api/routes/score.py
from fastapi import APIRouter, HTTPException, status

from api.deps import flag_from_score, vectorstore
from api.models import QueryRequest
from coherence.commons import map_events_to_memory

router = APIRouter()


@router.post("/score")
def score(req: QueryRequest):
    try:
        raw_hits = vectorstore.similarity_search_with_score(req.query, k=req.k)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Vectorstore retrieval failed"
        ) from e
    events = map_events_to_memory(raw_hits)
    top_score = max((e.get("score", 0.0) for e in events), default=0.0)
    flag = flag_from_score(top_score)
    return {"trust_score": top_score, "flag": flag}
