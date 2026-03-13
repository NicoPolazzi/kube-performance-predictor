# Architecture

## Overview

The system has two independent phases: data collection from a live Kubernetes cluster, and ML training/prediction on the collected data.

## Phase 1: Data Collection (`kpp/collector/`)

Connects to a live Kubernetes cluster, scales load via the loadgenerator deployment, polls Prometheus for Istio metrics, and writes results to timestamped CSV files (`performance_results_YYYYMMDD-HHMMSS.csv`).

CSV columns: `Timestamp`, `Service`, `User Count`, `Response Time (s)`, `Throughput (req/s)`, `CPU Usage`, `Replicas`, `CPU Request`

- `collector.py` — Entry point; for each experiment, scales replicas, ramps load via loadgenerator, waits 60s warmup, collects metrics, cools down 180s; try/finally ensures load stops and replicas reset on exit
- `sample.py` — `PerformanceSample` frozen dataclass: `service_name`, `response_time`, `throughput`, `cpu_usage`, `replicas`, `cpu_request`
- `kubernetes_client.py` — Service discovery (`get_services_names`, excludes `redis-cart`); CPU request/replica queries per service; patches loadgenerator env vars (`USERS`, `RATE`) and scales it; can scale any service deployment and wait for rollout
- `prometheus_client.py` — PromQL queries for response time, throughput, and CPU via Istio metrics; returns `math.nan` for missing data
- `csv_writer.py` — Writes timestamped CSV (`performance_results_YYYYMMDD-HHMMSS.csv`) with header; batched `write_samples()` appends rows

## Phase 2: ML Training & Prediction (`kpp/predictor/`)

Loads collected CSV data, normalizes per service, trains a linear model for each microservice, and saves models/plots.

- `main.py` — Orchestration: seeds RNG, runs the training loop, calls `evaluate()`+`plot()` per service, prints the metrics table via `generate_metrics_table()`; also owns `plot()`, `compute_metrics()`
- `pipeline.py` — `PerformanceDataPipeline`: validates CSV, rounds timestamps to 1-min, aggregates by (timestamp, service), fills gaps, splits by service, splits into train/test via interpolation or extrapolation strategy, normalizes with per-service `StandardScaler`
- `model.py` — `PerformanceModel` (multi-head trunk with GELU); `train_model()` (Adam + ReduceLROnPlateau, restores best weights in memory, returns loss history); `evaluate()` (inference on test set, inverts scaling, returns predictions/targets/user counts)

## Shared

- `kpp/config.py` — Frozen dataclasses for both phases: `CollectorConfig` (loads from `confs/experiments.yaml`) and `PredictorConfig` (loads from `confs/predictor.yaml`)

## Data & Model Storage

- `dataset/` — Pre-collected CSV files (used as input to predictor)
- `models/` — Output: `{service}.pth` weights + `config_{service}.json` (hyperparams, best_test_loss)
- `results/{experiment}/` — Output: `predictions/{service}_predictions.png`, `losses/{service}_losses.png`, and `metrics_table.md`

## Key Design Decisions

- The collector excludes non-HTTP services such as `redis-cart` from service discovery (no Istio metrics exposed)
- A 60-second warmup period matches Prometheus's `rate()` window before sampling begins
- A 180s cooldown is performed between load experiments
- Replica scaling is part of each experiment: each `(user_count, replicas)` pair is tested; all services reset to 1 replica in a try/finally cleanup
- `KubernetesClient` polls deployment status with a 180s timeout after any patch, ensuring metrics are only collected after full rollout
