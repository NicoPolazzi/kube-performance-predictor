# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Collector and predictor for Kubernetes microservice performance using ML. It collects Prometheus metrics from a Minikube cluster running Google's microservices-demo with Istio, then trains per-service ML models on the collected time-series data.

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


## Architecture

The system has two independent phases:

### 1. Data Collection (`kpp/collector/`)
Connects to a live Kubernetes cluster, scales load via the loadgenerator deployment, polls Prometheus for Istio metrics, and writes results to timestamped CSV files (`performance_results_YYYYMMDD-HHMMSS.csv`).

- `collector.py` — Entry point; iterates user counts (from `.env`)
- `kubernetes_client.py` — Patches loadgenerator deployment env vars (`USERS`, `RATE`), waits for rollout
- `prometheus_client.py` — PromQL queries for response time, throughput, and CPU via Istio metrics
- `csv_writer.py` — Batched writes to CSV

### 2. ML Training & Prediction (`kpp/predictor/`)
Loads collected CSV data, normalizes per service, trains a linear model for each microservice, and saves models/plots.

- `main.py` — Training loop: creates sequence windows, throughput-percentile train/test split, Adam + ReduceLROnPlateau, saves best model weights to `models/`
- `pipeline.py` — `PerformanceDataPipeline`: validates CSV, rounds timestamps to 1-min, aggregates by (timestamp, service), fills gaps, splits by service, normalizes with per-service `MinMaxScaler`, stratifies split by throughput percentile
- `model.py` — `PerformanceModel`: linear (input → hidden → output) with ReLU activation; flattens the full sequence window before the linear layers
- `visualizer.py` — Runs inference on test set, inverts scaling, plots ground truth vs predictions to `plots/`
- `generate_table.py` — Reads `models/config_{service}.json` and prints RMSE markdown table

### Shared
- `kpp/config.py` — Frozen dataclasses for both phases: `CollectorConfig` (loads from `.env`) and `PredictorConfig` (loads from `predictor_config.yaml`)

### Data & Model Storage
- `dataset/` — Pre-collected CSV files (used as input to predictor)
- `models/` — Output: `{service}.pth` weights + `config_{service}.json` (hyperparams, best_test_loss)
- `plots/` — Output: `{service}_predictions.png` prediction visualizations

### Key Design Decisions
- The collector excludes non-HTTP services such as `redis-cart` from service discovery (no Istio metrics exposed)
- A 60-second warmup period matches Prometheus's `rate()` window before sampling begins
- A 180s cooldown is performed between load experiments


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
