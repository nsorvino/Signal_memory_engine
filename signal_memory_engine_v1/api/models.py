# ============================================================================
# api/models.py (consolidated models)
# ============================================================================
from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    content: str
    score: float


class QueryRequest(BaseModel):
    query: str
    k: int = 3


class QueryResponse(BaseModel):
    answer: str
    chunks: list[Chunk]
    flag: str
    suggestion: str
    trust_score: float


class AgentResponse(BaseModel):
    answer: str
    chunks: list[Chunk]
    flag: str
    suggestion: str
    trust_score: float


class MultiQueryResponse(BaseModel):
    agents: dict[str, AgentResponse]


class TraceRecord(BaseModel):
    timestamp: str
    request_id: str
    agent: str
    query: str
    flag: str
    trust_score: float


class MemoryMatch(BaseModel):
    id: str
    score: float
    text: str | None = None
    agent: str | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class SignalEventIn(BaseModel):
    user_id: str
    user_query: str
    signal_type: str
    drift_score: float = Field(ge=0, le=1)
    emotional_tone: float | None = Field(default=0.0, ge=0, le=1)
    agent_id: str | None = None
    payload: dict | None = None
    relationship_context: str | None = None
    diagnostic_notes: str | None = None


class SignalEventOut(SignalEventIn):
    id: int
    timestamp: str
    escalate_flag: int
