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

# Run the data collector (requires running Kubernetes cluster)
poetry run python kpp/collector.py

# Run the predictor/trainer
poetry run python kpp/predictor/main.py

# Generate performance table from trained models
poetry run python kpp/predictor/generate_table.py
```

## Architecture

The system has two independent phases:

### 1. Data Collection (`kpp/collector/`)
Connects to a live Kubernetes cluster, scales load via the loadgenerator deployment, polls Prometheus for Istio metrics, and writes results to timestamped CSV files (`performance_results_YYYYMMDD-HHMMSS.csv`).

- `collector.py` — Entry point; iterates user counts (from `.env`), applies 180s cooldown between experiments
- `kubernetes_client.py` — Patches loadgenerator deployment env vars (`USERS`, `RATE`), waits for rollout
- `prometheus_client.py` — PromQL queries for response time, throughput, and CPU via Istio metrics
- `csv_writer.py` — Batched writes to CSV with timestamped filenames
- `config.py` — Loads `.env` into a frozen `Config` dataclass

### 2. ML Training & Prediction (`kpp/predictor/`)
Loads collected CSV data, normalizes per service, trains a GRU model for each microservice, and saves models/plots.

- `main.py` — Training loop: creates sequence windows (length 5), 80/20 train/test split, Adam + ReduceLROnPlateau, saves best model weights to `models/`
- `pipeline.py` — `PerformancesDataPipeline`: validates CSV, rounds timestamps to 1-min, aggregates by (timestamp, service), fills gaps, splits by service, normalizes with per-service `MinMaxScaler`, stratifies split by user count
- `model.py` — `PerformancesGRU`: 2-layer GRU (input=4, hidden=64) → linear output
- `visualizer.py` — Runs inference on test set, inverts scaling, plots ground truth vs predictions to `plots/`
- `generate_table.py` — Reads `models/config_{service}.json` and prints RMSE markdown table

### Data & Model Storage
- `dataset/` — Pre-collected CSV files (used as input to predictor)
- `models/` — Output: `gru_{service}.pth` weights + `config_{service}.json` (hyperparams, best_test_loss)
- `plots/` — Output: `{service}_predictions.png` prediction visualizations

### Key Design Decisions
- Models are trained **per microservice** independently
- The data pipeline stores `MinMaxScaler` instances on the pipeline object for use during inverse-transform in evaluation
- Stratified split by user count prevents train/test leakage across load levels
- The collector excludes `redis-cart` from service discovery
- A 60-second warmup period matches Prometheus's rate() window before sampling begins
- PyTorch is used for load the dataset and to train the model

## Workflows

1) Before modifying code in `kpp/predictor/`:
    - Construct an implementation plan