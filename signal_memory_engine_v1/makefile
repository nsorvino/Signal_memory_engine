.PHONY: dev test lint type build black black-check format

dev:    ; uvicorn api.main:app --reload --reload-dir api --reload-dir agents --reload-dir coherence --reload-dir utils --reload-dir storage --reload-dir scripts
test:   ; pytest -q
type:   ; mypy .
build:  ; docker compose up --build

# Lint-only (fails on issues, no changes)
lint:   ; ruff check . && black --check .

# Format with Black only
black:        ; black .

# Check formatting without changing files
black-check:  ; black --check .

# One-shot auto-fix: Ruff first (imports/quick fixes), then Black
format: ; ruff check --fix . && black .