# Changelog — Signal Memory Engine - [v2.0.0]

All notable changes introduced in **PR A — Core/API**.

## 2025-09-21 - prA Core/API

### Added
- **Endpoints**
  - `GET /agents` — lists configured agents and whether each is enabled.
  - `POST /score` — lightweight trust score + flag for a query (no LLM call).
- **LLM invocation wrapper**
  - `utils/llm.invoke_chain` uses **`.invoke()` only** and normalizes outputs to a string.
  - Stable error mapping for upstream failures:
    - **429 / quota** → **503 Service Unavailable**
    - **timeouts** → **504 Gateway Timeout**
    - **connection / HTTP errors** → **502 Bad Gateway**
    - Unknowns → **500 Internal Server Error**
  - Pass-through of existing `HTTPException`.
- **Trace logging utilities**
  - `utils/tracing.py` with:
    - `trace_log(agent, query, flag, score, request_id)` — append a JSONL record to `trace.log` at the repo root.
    - `read_trace_tail(limit)` — read the last N records (used by `/memory_log`).

### Changed
- **Modular API structure**
  - Routes split and organized under `api/routes/`:
    - `query.py` → `POST /query`
    - `multi.py` → `POST /multi_query`
    - `memory.py` → `GET /memory_log`
    - `signal.py` → `POST /signal`, `GET /drift/{user_id}`
    - `search.py` → `GET /memory/search` (+ `GET /memory/vector_query` alias)
    - `agents.py` → `GET /agents`
- **Shared deps consolidated in `api/deps.py`**
  - Centralizes Pinecone init, embeddings, default QA chain, and agent stores.
  - Exposes `AGENTS`, `SUGGESTIONS`, `flag_from_score`, and `sanitize_key`.
  - Validates required env (`PINECONE_API_KEY`, `OPENAI_API_KEY`) at startup; fails fast with a clear message.
- **Event mapping**
- **Packaging (semantic, not Docker)**
  - **`setup.py` simplified** to only **direct runtime dependencies** with **`~=` pin ranges**.
  - Rationale: keep runtime deps lean; Docker lockfiles and image parity will be handled in a later PR.
- Running FastAPI server live reload
  - Run Uvicorn with explicit excludes so reload only happens on source changes (command included in README.md)
- Load dotenv in agent files

### Fixed
- **Multi-agent metrics**: remove reliance on an undefined `events` var; aggregate via `all_events` for cross-agent metrics.
- **Resilience**: per-agent failures (e.g., LLM 503) no longer take down the whole `/multi_query`; loop logs and continues.
- **Consistency**: imports normalized across `api/`, `agents/`, and `utils`.

### Removed
- **Duplicate `/query`** path that lived inside the old `memory.py` variant.  
  The canonical single-agent endpoint is `POST /query` in `api/routes/query.py`.
- **Inline trace logging in `api/main.py`** — replaced by centralized helpers in `utils/tracing.py`.

### Migration Notes
- **API consumers**: No breaking changes to existing public endpoints. New endpoints `/agents` and `/score` are additive. Expect more precise error codes:
  - 429/“rate limit” conditions now surface as **503**,
  - timeouts as **504**,
  - network/HTTP upstream issues as **502**.

### Semantic Versioning
- Bumped to **`2.0.0`**: new endpoints, new utils, updated core.py, stronger error handling, and packaging cleanup.


## 2025-09-21 - prB Runtime

### Added
- **Health endpoint** `/health` (`api/routes/health.py`)
  - Verifies required env vars (e.g., `OPENAI_API_KEY`, `PINECONE_API_KEY`) and a simple filesystem write to `trace.log`.
  - Returns `200 OK` JSON when healthy; `503` on failures.
- **CLI healthcheck** `scripts/sme_healthcheck.py`
  - Hits `$SME_HEALTH_URL` (defaults to `http://localhost:8000/health`) and exits `0`/`1`.
  - Example: `SME_HEALTH_URL=http://localhost:8000/health python scripts/sme_healthcheck.py && echo OK || echo FAIL`
- **Docker Compose healthchecks**
  - `backend` uses the CLI script so the container becomes **healthy** only after `/health` is good.
  - `mlflow` uses a lightweight `wget` check on `:5000`.
- **Dependency management (pip)**
  - Introduced **`requirements.in`** (human-edited, minimal top-level deps with `~=` pin ranges).
  - Introduced **`requirements-lock.txt`** (fully pinned lock used for reproducible installs and Docker).
- **Conda parity**
  - `environment.yml` mirrors `requirements.in` selections (Python 3.10 baseline) for users who prefer Conda/Mamba.
  - Backend/UI Dockerfiles continue to support the **Conda** path via Micromamba.
- **Dynamic backend URL (UI)**
  - Streamlit UI now reads `BACKEND_URL` and falls back to `http://localhost:8000` when not running under Docker. This makes the same UI work both locally and in Compose.
- **MLflow service image**
  - MLflow moved into its own image (`dockerfile.mlflow`), started by Compose, and used by backend via `MLFLOW_TRACKING_URI=http://mlflow:5000`.

### Changed
- **MLflow artifact serving**
  - The MLflow server now runs with:
  ```bash
  --serve-artifacts \
  --artifacts-destination ./mlflow_server/artifacts
  ```
  - The key component is the artifacts-destination flag, which tells the MLflow server where to store the artifacts and enables HTTP-proxied artifact uploads.

### Fixed
- **Dependency mismatch**
  - Aligned versions across FastAPI/Starlette/Pydantic, LangChain family, and client SDKs to remove import/runtime warnings.
  - Resolved differences between pip and Conda environments so the same code path runs under both.
- **Backend container file permissions**
  - Ensure the runtime user can read the app tree and import agent modules:
    ```bash
    # This resolves PermissionError: [Errno 13] Permission denied: '/app/agents/...' at startup.
    USER root
    COPY --chown=mambauser:mambauser . /app
    RUN chmod -R a+rX /app
    USER mambauser
    ```
- **MLflow PermissionError**
  - Client was incorrectly trying to write files directly to the server’s file system and caused permission errors.

### Notes
- No README/docs changes included here; documentation will land in **PR D**.


## 2025-09-23 - prC Tooling/Testing

### Added
- **CI**
  - `.github/` at repo root with empty `CODEOWNERS` and `workflows/ci.yml`.
- **Coverage**
  - `.coveragerc` at repo root for consistent coverage reporting across jobs.
- **Scripts**
  - `scripts/probe_openai.py` to quickly verify OpenAI connectivity.
- **Tests**
  - `tests/test_coherence_commons.py`
  - `tests/test_ingestion_batch_loader.py`
  - `tests/test_ingestion_ingest_memory.py`
  - `tests/test_smoke.py` (invokes the smoke script)
  - `tests/test_storage_sqlite_store.py`
  - `tests/conftest.py` with stubs/fixtures for external services.
- **Logging**
  - `utils/logging_setup.py` centralizes log formatting & level.
- **Tooling**
  - `.pre-commit-config.yaml` (ruff/black/mypy hooks).
  - `Makefile` with shortcuts for `dev`, `lint`, `type`, `test`, `build`, `black`.
  - `pyproject.toml` config for ruff / mypy / black.
  - `pytest.ini` for test discovery & flags.
  - `sitecustomize.py` to normalize import paths across local/CI/Docker.
- **Testing utilities**
  - `utils/pinecone_stub.py` — centralized, idempotent fake Pinecone client for offline/CI and local tests.  
    Supports `PINECONE_INDEXES`, preserves legacy import path (`pinecone.db_data.index.Index`),  
    and ensures `vector_store.pinecone_index.index` exists when modules import it directly.

### Changed
- **Smoke test**
  - `scripts/smoke_test.py` now covers new routes (`/agents`, `/score`) and exposes `run_smoke()` to keep `__main__` minimal.
- **Server bootstrap**
  - `api/main.py` calls `setup_logging()` before FastAPI app initialization so app (and optionally Uvicorn) logs use the same structured format.
- **Tests bootstrap**
  - `tests/conftest.py` condensed to import/install the shared `pinecone_stub` instead of embedding a duplicate stub.  
    Keeps prior behavior, reduces duplication, and sets default `PINECONE_INDEXES` (now includes `signal-engine`) for deterministic test runs.
- **Import normalization**
  - `sitecustomize.py` updated to install the Pinecone stub at process start when `SME_TEST_MODE=1` or `RUN_API_SMOKE=0`, preventing early imports from making network calls during tests/CI.
- **Type-checking**
  - Stub implementation made mypy-clean (removed unused `# type: ignore`, safer `setattr` usage).

### Notes
- No API behavior changes; additions are CI/test/dev-experience only.

## 2025-09-25 - prD Docs

### Added
- `.env.example` with sane defaults, including `LOG_LEVEL`.
- `docs/adr/ADR-0001-v2-backbone.md` — 1-pager capturing v2 architecture/testing decisions.
- PDFs and Diagrams to explain app flow to simplify onboarding

### Changed
- `README.md` updated with CI/testing flow, Makefile targets, Docker notes, helpful documents, and logging guidance.

### Impact
- Documentation-only; improves onboarding and consistency. No runtime changes.