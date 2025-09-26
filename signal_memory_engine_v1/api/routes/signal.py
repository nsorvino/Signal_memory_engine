# ============================================================================
# api/routes/signal.py  →  /signal + /drift endpoints
# ============================================================================
import json as _json
import os
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status

from agents.router_stub import log_routing_decision, route_agent
from api.models import SignalEventIn, SignalEventOut
from storage.sqlite_store import init_db, insert_event, list_by_user
from utils.dashboard import send_to_dashboard

router = APIRouter()
DB_PATH = os.getenv("SME_DB_PATH", "./data/signal.db")
init_db(DB_PATH)


@router.post("/signal", response_model=SignalEventOut)
def log_signal(event: SignalEventIn):
    if event.agent_id:
        selected_agent = event.agent_id
        reason = "Agent override"
    else:
        decision = route_agent(
            event.user_query,
            event.emotional_tone or 0.0,
            event.signal_type,
            event.drift_score,
        )
        selected_agent = decision["selected_agent"]
        reason = decision["reason"]

    escalate = 1 if float(event.drift_score) > 0.5 else 0
    ts = datetime.utcnow().isoformat()

    stored = event.model_dump()
    stored["timestamp"] = ts
    stored["escalate_flag"] = escalate
    stored["agent_id"] = selected_agent

    try:
        event_id = insert_event(DB_PATH, stored)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to persist signal"
        ) from e
    stored["id"] = event_id

    log_routing_decision(
        {
            "selected_agent": selected_agent,
            "reason": reason,
            "user_id": event.user_id,
            "signal_type": event.signal_type,
            "drift_score": event.drift_score,
        }
    )

    send_to_dashboard({**stored})
    return stored


@router.get("/drift/{user_id}", response_model=list[SignalEventOut])
def get_drift(user_id: str, limit: int = Query(10, ge=1, le=100)):
    rows = list_by_user(DB_PATH, user_id=user_id, limit=limit)
    for r in rows:
        if isinstance(r.get("payload"), str):
            try:
                r["payload"] = _json.loads(r["payload"])
            except Exception:
                r["payload"] = None
    return rows
