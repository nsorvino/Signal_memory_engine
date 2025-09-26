# ============================================================================
# api/routes/query.py  →  POST /query
# ============================================================================
import logging
import time
import uuid

import mlflow
from fastapi import APIRouter, HTTPException, status

from api import deps
from api.models import Chunk, QueryRequest, QueryResponse
from coherence.commons import map_events_to_memory
from sensors.biometric import sample_all_signals
from utils.llm import invoke_chain
from utils.tracing import trace_log

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    request_id = uuid.uuid4().hex
    start_ts = time.time()
    logger.debug("Received /query [%s]: %s", request_id, req)

    with mlflow.start_run(run_name=f"single-{request_id}"):
        # log run params
        mlflow.log_param("query", req.query)
        mlflow.log_param("k", req.k)
        # if these exist in deps, log them like before
        if hasattr(deps, "EMBED_MODEL"):
            mlflow.log_param("embed_model", deps.EMBED_MODEL)
        if hasattr(deps, "LLM_MODEL"):
            mlflow.log_param("llm_model", deps.LLM_MODEL)

        # biometrics
        try:
            signals = sample_all_signals()
        except Exception:
            signals = {"hrv": 0.0, "temperature": 0.0, "blink_rate": 0.0}
        mlflow.log_metric("hrv", signals.get("hrv", 0))
        mlflow.log_metric("temperature", signals.get("temperature", 0))
        mlflow.log_metric("blink_rate", signals.get("blink_rate", 0))
        if "gsr" in signals:
            mlflow.log_metric("gsr", signals["gsr"])
        if "emotion_label" in signals:
            mlflow.log_param("emotion_label", signals["emotion_label"])

        # prompt
        prefix = (
            "Current biometric readings:\n"
            f"• HRV: {signals['hrv']:.1f} ms\n"
            f"• Temp: {signals['temperature']:.1f} °C\n"
            f"• Blink rate: {signals['blink_rate']:.1f} bpm\n\n"
            "Use this context to answer the question below."
        )
        full_prompt = f"{prefix}\n\nQuestion: {req.query}"
        mlflow.log_param("prompt", full_prompt)
        mlflow.log_text(full_prompt, "prompt.txt")

        # RAG (use invoke_chain; bubble HTTPExceptions unchanged)
        try:
            answer = invoke_chain(deps.qa, full_prompt)
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.exception("Error during RAG run [%s]", request_id)
            raise HTTPException(status_code=500, detail=str(e))

        # retrieval & scoring (keep original mapping + flags/suggestions)
        # retrieval & scoring
        try:
            raw_hits = deps.vectorstore.similarity_search_with_score(req.query, k=req.k)
        except Exception as e:
            logger.error("Vector retrieval error [%s]: %s", request_id, e)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail="Vectorstore retrieval failed"
            ) from e
        events = map_events_to_memory(raw_hits)
        top_score = max((e["score"] for e in events), default=0.0)
        flag = deps.flag_from_score(top_score)
        suggestion = deps.SUGGESTIONS[flag]
        chunks = [Chunk(content=e["content"], score=e["score"]) for e in events]

        # log outputs (same as before)
        mlflow.log_metric("top_score", top_score)
        mlflow.log_metric("num_chunks", len(raw_hits))
        mlflow.log_param("flag", flag)
        mlflow.log_param("suggestion", suggestion)
        mlflow.log_text(answer, "answer.txt")

        emo_rec = int(any(e.get("emotional_recursion_present") for e in events))
        reroute = int(any(e.get("rerouting_triggered") for e in events))
        mlflow.log_param("emotional_recursion_detected", emo_rec)
        mlflow.log_param("rerouting_triggered", reroute)

        mlflow.log_metric("chad_confidence_score", top_score)
        esc_level = 1 if flag == "concern" else 0
        mlflow.log_param("escalation_level", esc_level)

        elapsed_ms = (time.time() - start_ts) * 1000
        mlflow.log_metric("endpoint_latency_ms", elapsed_ms)
        mlflow.log_param("coherence_drift_detected", int(flag != "stable"))
        mlflow.log_param("escalation_triggered", 0)

    trace_log("single-agent", req.query, flag, top_score, request_id)

    return QueryResponse(
        answer=answer,
        chunks=chunks,
        flag=flag,
        suggestion=suggestion,
        trust_score=top_score,
    )
