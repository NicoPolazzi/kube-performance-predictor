# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Collector and predictor for Kubernetes microservice performance using ML. It collects Prometheus metrics from a Minikube cluster running Google's microservices-demo with Istio, then trains per-service ML models on the collected time-series data.

@docs/ARCHITECTURE.md

## Project Context

When working with this codebase, prioritize readability over cleverness. Ask clarifying questions before making architectural changes.


## Commands

```bash
# Install dependencies
poetry install

# Run linter
poetry run ruff check .

# Run type checker
poetry run mypy .

# Run tests
poetry run pytest
```


## Conventions

- Prefer standard library functions over external dependencies
- Do not re-implement functions already available in the standard library


## Testing

Tests live in `tests/`. Run with `poetry run pytest`.

### Conventions
- Use `pytest-mock` (`mocker` fixture) — never `unittest.mock` directly
- Instantiate dependencies inline in each test; no shared helper factories
- Test only public interfaces; do not call private methods directly
- Verify mock interactions with `assert_called_once()` or `assert_called_once_with()`
- Test names follow `test_<method>_<condition>_<expected>` (e.g. `test_get_throughput_when_response_is_empty_returns_nan`)


## Workflows

1. Before modifying code in `kpp/`:
   - Read the relevant module and understand existing patterns first, construct an implementation plan and get approval before writing any code

2. After any code change:
   - Run `poetry run ruff check .` and `poetry run mypy .` to catch lint/type errors
   - Run `poetry run pytest` to verify tests pass
