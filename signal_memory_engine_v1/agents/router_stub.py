import json
import os
from datetime import datetime
from typing import Any

# from storage.sqlite_store import init_db, insert_event
# from utils.dashboard import send_to_dashboard
from storage.sqlite_store import init_db, insert_event
from utils.dashboard import send_to_dashboard

DB_PATH = os.getenv("SME_DB_PATH", "./data/signal.db")
init_db(DB_PATH)


def route_and_log_event(
    user_id: str,
    user_query: str,
    emotional_tone: float,
    signal_type: str,
    drift_score: float,
    payload: dict | None = None,
    relationship_context: str | None = None,
    diagnostic_notes: str | None = None,
    logfile: str = "router_log.jsonl",
) -> dict:
    """
    Route the event, persist it to SQLite, keep JSONL log, and ping the dashboard stub.
    Returns: { selected_agent, reason, event_id }
    """
    # Use your validated router
    decision = route_agent(user_query, emotional_tone, signal_type, drift_score)

    # Simple escalate rule for v1
    try:
        escalate_flag = 1 if float(drift_score) > 0.5 else 0
    except Exception:
        escalate_flag = 0

    # Compose event for persistence
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "agent_id": decision["selected_agent"],
        "signal_type": signal_type,
        "emotional_tone": emotional_tone,
        "drift_score": drift_score,
        "escalate_flag": escalate_flag,
        "payload": {"query": user_query, **(payload or {})},
        "relationship_context": relationship_context,
        "diagnostic_notes": diagnostic_notes,
    }

    # Save to SQLite
    event_id = insert_event(DB_PATH, event)

    # Keep the JSONL log for easy tail -n
    log_routing_decision(
        {**decision, "user_id": user_id, "signal_type": signal_type, "drift_score": drift_score},
        logfile=logfile,
    )

    # Future-proof hook: console stub for dashboards
    send_to_dashboard({**event, "id": event_id})

    return {**decision, "event_id": event_id}


def route_agent(
    user_query: str, emotional_tone: Any, signal_type: Any, drift_score: Any
) -> dict[str, Any]:
    """
    Decide which agent should handle the incoming query based on emotional tone, signal type, and drift score.

    Args:
        user_query (str): The text of the user's query.
        emotional_tone (Any): A score (expected 0–1) indicating emotional intensity.
        signal_type (Any): A string indicating the signal type.
        drift_score (Any): A score (expected 0–1) indicating data/model drift.

    Returns:
        Dict[str, Any]: A decision dict containing:
            - selected_agent (str): Name of the chosen agent.
            - reason (str): Explanation for the routing decision.
    """
    # Validate and coerce numerical inputs
    try:
        tone = float(emotional_tone)
        drift = float(drift_score)
    except (TypeError, ValueError):
        return {"selected_agent": "Selah", "reason": "Invalid score input"}

    # Range checks for scores
    if tone < 0.0 or tone > 1.0:
        return {"selected_agent": "Selah", "reason": "Emotional tone out of range"}
    if drift < 0.0 or drift > 1.0:
        return {"selected_agent": "Selah", "reason": "Drift score out of range"}

    # Validate and normalize signal type
    if not isinstance(signal_type, str):
        return {"selected_agent": "Selah", "reason": "Invalid signal type"}
    signal_norm = signal_type.strip().lower()

    # Routing rules (priority order)
    if tone > 0.7:
        return {"selected_agent": "Axis", "reason": "High emotional tone"}

    if drift > 0.5:
        return {"selected_agent": "M", "reason": "High drift score"}

    if signal_norm == "compliance":
        return {"selected_agent": "M", "reason": "Compliance signal"}

    if signal_norm in ("biometric", "relational"):
        return {"selected_agent": "Oria", "reason": f"Signal type '{signal_norm}'"}

    # Fallback routing
    return {"selected_agent": "Selah", "reason": "Fallback routing"}


def log_routing_decision(decision: dict[str, Any], logfile: str = "router_log.jsonl") -> None:
    """
    Append a routing decision to a JSONL log file, including a timestamp.

    Args:
        decision (Dict[str, Any]): The routing decision dict.
        logfile (str): Path to the log file.
    """
    entry = {"timestamp": datetime.utcnow().isoformat(), **decision}
    with open(logfile, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    # Manual tests for all scenarios
    tests = [
        ("High tone", 0.9, "relational", 0.0),
        ("High drift", 0.1, "relational", 0.6),
        ("Compliance", 0.1, "compliance", 0.0),
        ("Oria default", 0.1, "biometric", 0.0),
        ("Unknown signal", 0.1, "unknown", 0.0),
        ("Invalid tone type", "bad", "relational", 0.0),
        ("Tone out of range", -0.1, "relational", 0.0),
        ("Drift out of range", 0.1, "relational", 1.5),
        ("Invalid signal type", 0.1, None, 0.0),
    ]
    for query, tone, sig, drift in tests:
        decision = route_agent(query, tone, sig, drift)
        print(f"{query!r} -> {decision}")
        log_routing_decision(decision)
    print("Manual tests completed. Logs written to router_log.jsonl.")
