# utils/tracing.py
import json
import logging
from collections import deque
from datetime import datetime
from typing import Any

TRACE_LOG_FILE = "trace.log"
_log = logging.getLogger(__name__)


def trace_log(agent: str, query: str, flag: str, score: float, request_id: str) -> None:
    try:
        rec = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "agent": agent,
            "query": query,
            "flag": flag,
            "trust_score": score,
        }
        with open(TRACE_LOG_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        _log.error("trace_log write failed: %s", e)


def read_trace_tail(limit: int = 20) -> list[dict[str, Any]]:
    buf: deque[dict[str, Any]] = deque(maxlen=limit)
    with open(TRACE_LOG_FILE) as f:
        for line in f:
            buf.append(json.loads(line))
    return list(buf)
