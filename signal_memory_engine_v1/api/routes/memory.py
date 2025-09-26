# ============================================================================
# api/routes/memory.py  →  GET /memory_log
# ============================================================================
from fastapi import APIRouter, HTTPException

from api.models import TraceRecord
from utils.tracing import read_trace_tail

router = APIRouter()


@router.get("/memory_log", response_model=list[TraceRecord])
def memory_log(limit: int = 20):
    try:
        return read_trace_tail(limit)
    except FileNotFoundError:
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
