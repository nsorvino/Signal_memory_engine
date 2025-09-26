# utils/llm.py
import httpx
from fastapi import HTTPException, status

# OpenAI 1.x exceptions (present if openai is installed)
try:
    from openai import APIConnectionError, APITimeoutError, RateLimitError
except Exception:  # fallback if package/exceptions move

    RateLimitError = Exception

    APITimeoutError = Exception

    APIConnectionError = Exception


def _coerce_output(out):
    """Return a string answer from common LC chain outputs."""
    if isinstance(out, dict):
        for k in ("result", "output_text", "answer", "output"):
            if k in out:
                return out[k]
        return str(out)
    return out if isinstance(out, str) else str(out)


def invoke_chain(chain, prompt: str):
    """
    Safe invocation for LangChain RetrievalQA-style chains USING ONLY `.invoke`.
    - Tries input keys in this order: {"query": ...} → {"input": ...} → bare string
    - Maps common upstream failures to stable HTTP codes:
        429/quota -> 503 Service Unavailable
        timeouts  -> 504 Gateway Timeout
        connection errors -> 502 Bad Gateway
        anything else -> 500 Internal Server Error
    """
    try:
        # Prefer {"query": ...}; fall back to {"input": ...}; some chains accept a bare string
        try:
            out = chain.invoke({"query": prompt})
        except Exception:
            try:
                out = chain.invoke({"input": prompt})
            except Exception:
                out = chain.invoke(prompt)

        return _coerce_output(out)

    # Map OpenAI / network-ish errors
    except RateLimitError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM quota/rate limit"
        ) from e
    except (APITimeoutError, httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="LLM timeout"
        ) from e
    except (APIConnectionError, httpx.HTTPError) as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM upstream error"
        ) from e
    # If LangChain/OpenAI wraps errors, heuristically detect 429/quota text
    except Exception as e:
        msg = str(e).lower()
        if "insufficient_quota" in msg or "rate limit" in msg or "429" in msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM quota/rate limit"
            ) from e
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
