# ADR-0001: v2 Backbone Decisions (API, Runtime, Tooling)

**Status:** Accepted  
**Date:** 2025-09-24  
**Owner:** Nate S.  
**Scope:** Signal Memory Engine v2 (PRs A, B, C, D)

---

## Context

- **v1 baseline:**  
  - **No CI** and effectively **one test**; all verification was manual.  
  - **Frequent bugs** surfaced during ad-hoc runs due to untyped paths and missing guards.  
  - **API inconsistencies** (error codes, payload shapes, overlapping routes) made integration brittle.  
  - **Low modularity**: shared logic scattered across files; bootstrapping duplicated; env handling inconsistent.  
  - **Logging** was ad hoc and not standardized across modules.

- **v2 goals:** Make the service reliable to run (locally and in CI), easy to test offline, and cleanly structured for future features.

---

## Goals / Non-Goals

**Goals**
- Deterministic builds across pip/conda/docker.
- Offline/CI testability without real Pinecone/OpenAI.
- Clear API/module boundaries and shared dependency bootstrapping.
- Baseline quality gates: lint, types, and coverage.

**Non-Goals**
- Re-architecting model logic or replacing LangChain.

---

## Decision

### 1) API structure & bootstrapping (PR **A**)
- Split routes under `api/routes/` (`query`, `multi`, `memory`, `signal`, `search`, `agents`, `score`).
- Centralize shared deps in `api/deps.py` (Pinecone init, embedder, QA chain, agents).
- Normalize LLM error mapping in `utils/llm.py`.
- Add `/agents` and `/score` endpoints.
- Add health route (`/health`) in PR **B**.
- Initialize structured logging via `utils/logging_setup.setup_logging()` at import time.

### 2) Runtime & packaging (PR **B**)
- Introduce **pip** `requirements.in` + **requirements-lock.txt`; mirror in **conda** `environment.yml`.
- Docker Compose with separate **backend** and **mlflow** services; health checks wired to `/health`.
- Keep FastAPI entrypoint lean; optional `lifespan` pre-warm kept small.

### 3) Tooling, testing, and offline stubs (PR **C**)
- **GitHub Actions CI** with 3 jobs:
  - **pip** matrix (Ubuntu + macOS / py310–py312): ruff, mypy, pytest + coverage.
  - **conda** job (Ubuntu/py310): pytest + coverage.
  - **docker** job: build backend image and run smoke tests inside the container.
- **Coverage gate:** `--cov-fail-under=80`.
- **Offline Pinecone stub**: `utils/pinecone_stub.py` + `sitecustomize.py` auto-installs a fake Pinecone client when
  `SME_TEST_MODE=1` or `PINECONE_API_KEY=dummy` (and supports legacy `pinecone.db_data.index.Index` paths).
- **Tests added:** coherence, ingestion (batch + normalize + ingest runner), storage, router stub, smoke wrapper.
- **Config files:** `.coveragerc` (root), `pytest.ini`, `pyproject.toml` (ruff/mypy/black), `.pre-commit-config.yaml`.
- **Makefile:** `dev`, `lint`, `type`, `test`, `build`, `black`.
- **`sitecustomize.py`:** ensures the Pinecone stub is installed early for tests/smoke/CI.
- **`scripts/probe_openai.py`:** quick connectivity check util.

### 4) Docs (PR **D**)
- Updated `README.md` (run modes, CI, testing, envs).
- Added `.env.example`.
- This ADR under `docs/adr/0001-v2-backbone.md`.
- PDFs and Diagrams explaining program flow.
- Final changelog entry.

---

## Trade-offs

- **Pros:** Deterministic CI, fast offline tests, consistent logs, clearer module boundaries.
- **Cons:** Added indirection (`sitecustomize.py`) and a small maintenance burden for the Pinecone stub API surface.

---

## Implications

- **Dev UX:** `make dev` and `make test` work out-of-the-box; warnings are minimized; tests do not require real cloud keys.
- **CI Reliability:** Fail-fast on lint/type/coverage; Docker smoke matches local behavior.
- **Extensibility:** New routes/agents slot in cleanly; stubs isolate external dependencies.

---

## Migration Notes from v1 → v2

- Provide `.env` (or `.env.example`) and prefer `make dev` / `make test`.
- Real Pinecone/OpenAI keys only needed for live runs; CI uses dummy keys.
- No breaking API changes; `/agents`, `/score`, and `/health` are additive.

---

## Repo Conventions

- **Root:** `.github/`, `.coveragerc`, `docs/adr/`, top-level README.
- **App dir (`signal_memory_engine_v1/`):** code, `pyproject.toml`, `pytest.ini`, `sitecustomize.py`.
- **Coverage:** run from repo root so `.coveragerc` applies to all jobs.

---

## Appendix: Key Env Vars

- `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, `PINECONE_INDEX` (runtime), `PINECONE_INDEXES` (tests)
- `OPENAI_API_KEY`, `OPENAI_MODEL`
- `SME_TEST_MODE` (enables offline stubs), `LOG_LEVEL`
- `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`