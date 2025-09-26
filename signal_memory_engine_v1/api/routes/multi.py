# ============================================================================
# api/routes/multi.py  →  POST /multi_query
# ============================================================================
import logging
import time
import uuid

import httpx
import mlflow
from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from api import deps
from api.models import AgentResponse, Chunk, MultiQueryResponse, QueryRequest
from coherence.commons import map_events_to_memory
from sensors.biometric import sample_all_signals
from utils.llm import invoke_chain
from utils.tracing import trace_log

logger = logging.getLogger(__name__)
router = APIRouter()


def notify_human_loop(query: str, agents: list[str]) -> None:
    payload = {"query": query, "agents": agents}
    try:
        httpx.post(
            "https://your-escalation-endpoint.example/handoff",
            json=payload,
            timeout=httpx.Timeout(5.0),
        )
    except httpx.TimeoutException:
        logger.warning("Escalation endpoint timed out")
    except httpx.HTTPError as e:
        logger.error("Escalation HTTP error: %s", e)
    except Exception as e:
        logger.error("Escalation unexpected error: %s", e)


@router.post("/multi_query", response_model=MultiQueryResponse)
def multi_query(req: QueryRequest, bg: BackgroundTasks):
    request_id = uuid.uuid4().hex
    start_ts = time.time()
    logger.debug("Received /multi_query [%s]: %s", request_id, req)

    with mlflow.start_run(run_name=f"multi-{request_id}"):
        mlflow.log_param("query", req.query)
        mlflow.log_param("k", req.k)
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

        # shared prompt
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

        out: dict[str, AgentResponse] = {}
        all_events = []

        # iterate configured agents (deps.AGENTS should be tuples: (role, qa_chain, store))
        for role, qa_chain, store in getattr(deps, "AGENTS", getattr(deps, "ROLE_CHAINS", [])):
            safe = getattr(deps, "sanitize_key", lambda x: x)(role)

            try:
                ans = invoke_chain(qa_chain, full_prompt)
            except HTTPException as he:
                # If LLM quota/429 → 503, keep others running (like your original intent)
                if he.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
                    logger.warning("Agent %s quota hit [%s]: %s", role, request_id, he.detail)
                    out[role] = AgentResponse(
                        answer="(LLM temporarily unavailable: quota/rate limit)",
                        chunks=[],
                        flag="stable",
                        suggestion="No action",
                        trust_score=0.0,
                    )
                    trace_log(role, req.query, "stable", 0.0, request_id)
                    mlflow.log_param(f"{safe}_flag", "stable")
                    mlflow.log_metric(f"{safe}_top_score", 0.0)
                    continue
                # otherwise bubble up (consistent with original)
                raise he
            except Exception as e:
                logger.error("Agent %s failed [%s]: %s", role, request_id, e)
                out[role] = AgentResponse(
                    answer="(error)",
                    chunks=[],
                    flag="stable",
                    suggestion="No action",
                    trust_score=0.0,
                )
                trace_log(role, req.query, "stable", 0.0, request_id)
                mlflow.log_param(f"{safe}_flag", "stable")
                mlflow.log_metric(f"{safe}_top_score", 0.0)
                continue

            try:
                raw_hits = store.similarity_search_with_score(req.query, k=req.k)
            except Exception as e:
                logger.error("Agent %s vector retrieval error [%s]: %s", role, request_id, e)
                out[role] = AgentResponse(
                    answer=ans,  # keep the model’s answer if we got it
                    chunks=[],
                    flag="stable",
                    suggestion="No action",
                    trust_score=0.0,
                )
                trace_log(role, req.query, "stable", 0.0, request_id)
                mlflow.log_param(f"{safe}_flag", "stable")
                mlflow.log_metric(f"{safe}_top_score", 0.0)
                continue
            events = map_events_to_memory(raw_hits, source_agent=role)
            all_events.extend(events)
            top_score = max((e["score"] for e in events), default=0.0)
            flag = deps.flag_from_score(top_score)
            suggestion = deps.SUGGESTIONS[flag]
            chunks = [Chunk(content=e["content"], score=e["score"]) for e in events]

            out[role] = AgentResponse(
                answer=ans, chunks=chunks, flag=flag, suggestion=suggestion, trust_score=top_score
            )
            trace_log(role, req.query, flag, top_score, request_id)

            # per-agent logging like before
            mlflow.log_param(f"{safe}_flag", flag)
            mlflow.log_param(f"{safe}_suggestion", suggestion)
            mlflow.log_metric(f"{safe}_top_score", top_score)
            mlflow.log_metric(f"{safe}_num_chunks", len(raw_hits))
            mlflow.log_text(ans, f"{safe}_answer.txt")

        # cross-agent derived metrics (use all_events instead of an undefined 'events')
        emo_rec_multi = int(any(e.get("emotional_recursion_present") for e in all_events))
        reroute_multi = int(any(e.get("rerouting_triggered") for e in all_events))
        mlflow.log_param("emotional_recursion_detected", emo_rec_multi)
        mlflow.log_param("rerouting_triggered", reroute_multi)

        avg_conf = (sum(r.trust_score for r in out.values()) / (len(out) or 1)) if out else 0.0
        mlflow.log_metric("chad_confidence_score", avg_conf)
        esc_level = len([r for r in out.values() if r.flag == "concern"])
        mlflow.log_param("escalation_level", esc_level)

        elapsed_ms = (time.time() - start_ts) * 1000
        mlflow.log_metric("endpoint_latency_ms", elapsed_ms)
        mlflow.log_param(
            "coherence_drift_detected", int(any(r.flag != "stable" for r in out.values()))
        )
        mlflow.log_param(
            "escalation_triggered", int(any(r.flag == "concern" for r in out.values()))
        )

    high_agents = [r for r, resp in out.items() if resp.flag == "concern"]
    if high_agents:
        bg.add_task(notify_human_loop, req.query, high_agents)

    return MultiQueryResponse(agents=out)
