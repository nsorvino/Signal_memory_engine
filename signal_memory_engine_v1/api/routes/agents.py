# signal_memory_engine_v1/api/routes/agents.py
import os

from fastapi import APIRouter

from api.deps import AGENTS

router = APIRouter()


@router.get("/agents")
def list_agents():
    enabled = set(
        r.strip().upper()
        for r in os.getenv("ENABLED_AGENTS", "AXIS,ORIA,SENTINEL").split(",")
        if r.strip()
    )
    out = []
    for role, _, _ in AGENTS:
        out.append({"role": role, "enabled": role.upper() in enabled})
    return {"agents": out}
