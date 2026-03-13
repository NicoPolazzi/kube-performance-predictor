# Kube Performance Predictor

A Master's thesis project for predicting Kubernetes microservice performance using machine learning. It collects Prometheus metrics from a Minikube cluster running Google's microservices-demo with Istio, then trains per-service neural network models on the collected time-series data.

## Requirements

Ensure that you have the following tools installed:

- [Poetry](https://python-poetry.org/)
- [kubectl](https://kubernetes.io/releases/download/)
- [Minikube](https://minikube.sigs.k8s.io/docs/start/)
- [Istioctl](https://istio.io/latest/docs/setup/getting-started/)
- [Kustomize](https://github.com/kubernetes-sigs/kustomize)

## Installation

Install the Python dependencies using Poetry:

```shell
poetry install
```

## Architecture

### Project Structure

```
kpp/
├── collector/               # Data collection
│   ├── main.py              # Orchestrator: runs load experiments, writes CSV
│   ├── kubernetes_client.py # K8s API wrapper: scaling, service discovery
│   ├── prometheus_client.py # PromQL adapter: response time, throughput, CPU
│   ├── csv_writer.py        # Writes timestamped CSV output
│   └── sample.py            # PerformanceSample frozen dataclass
├── predictor/               # ML training and evaluation
│   ├── main.py              # Orchestrator: training loop, plotting, RMSE table
│   ├── pipeline.py          # PerformanceDataPipeline: preprocessing and splitting
│   └── model.py             # PerformanceModel: network definition and training loop
├── config.py                # Typed config dataclasses (CollectorConfig, PredictorConfig)
└── logging_config.py        # Logging setup
confs/
├── experiments.yaml         # Collector settings: load profiles and experiment list
└── predictor.yaml           # ML hyperparameters: pipeline, model, training, scheduler
dataset/                     # Pre-collected CSV files
tests/                       
```

### Package Descriptions

#### `kpp.collector`

Orchestrates load experiments against a live Kubernetes cluster. `KubernetesClient` handles service discovery (excluding `redis-cart`, which has no Istio metrics), scales service deployments, and ramps load by patching the loadgenerator's `USERS`/`RATE` environment variables. `PrometheusClient` queries Istio metrics via PromQL, returning `math.nan` for missing data. Each experiment runs for a configurable duration with a 60s warmup before sampling; a 180s cooldown separates experiments. Results are written to `dataset/performance_results_YYYYMMDD-HHMMSS.csv` by `CsvWriter`.

#### `kpp.predictor`

Loads CSVs from `dataset/` and runs them through `PerformanceDataPipeline`, which validates the schema, rounds timestamps to 1-minute intervals, aggregates by (timestamp, service), fills gaps, and splits into train/test sets using either an interpolation or extrapolation strategy. Each service is normalized independently with a `StandardScaler` plus log transform. A `PerformanceModel` is then trained per service using PyTorch (Adam optimizer with `ReduceLROnPlateau`), with best weights saved to `models/`. After training, `evaluate()` inverts the scaling to produce predictions in original units, and `plot()` generates prediction visualizations in `results/{strategy}/`.

#### `kpp.config`

Dataclasses for both phases: `CollectorConfig` (loaded from `confs/experiments.yaml`) and `PredictorConfig` (loaded from `confs/predictor.yaml`). Both use standard-library YAML parsing with `tomllib`-style field mapping to keep dependencies minimal.

## Configuration

### `confs/experiments.yaml`

Controls the collector:

| Key | Description |
|-----|-------------|
| `experiment_duration_seconds` | How long each experiment runs (default: 600) |
| `profile` | Which load profile to use (`normal` or `overload`) |
| `profiles.<name>` | List of `{users, replicas}` experiments to run |

### `confs/predictor.yaml`

Controls the predictor:

| Section | Key | Description |
|---------|-----|-------------|
| `pipeline` | `train_ratio` | Fraction of data used for training (default: 0.9) |
| `pipeline` | `split_strategy` | `interpolation` or `extrapolation` |
| `model` | `hidden_size` | Size of first shared hidden layer (default: 256) |
| `model` | `hidden_size_2` | Size of second shared hidden layer (default: 256) |
| `model` | `head_hidden_size` | Size of each output head hidden layer (default: 128) |
| `model` | `dropout` | Dropout rate (default: 0.3) |
| `training` | `epochs` | Maximum training epochs (default: 100) |
| `training` | `learning_rate` | Initial Adam learning rate (default: 0.003) |
| `training` | `batch_size` | Mini-batch size (default: 32) |
| `training` | `weight_decay` | Adam weight decay (default: 0.0001) |
| `scheduler` | `factor` | LR reduction factor on plateau (default: 0.5) |
| `scheduler` | `patience` | Epochs to wait before reducing LR (default: 20) |
| `scheduler` | `min_lr` | Minimum learning rate floor (default: 0.000001) |

## Setup

### Create the cluster

Initialize a Minikube cluster with sufficient resources:

```shell
minikube start --cpus=6 --memory 8192 --disk-size 32g
```

### Install Istio and Addons

Install the Istio minimal profile and enable sidecar injection for the default namespace:

```shell
istioctl install --set profile=minimal -y
kubectl label namespace default istio-injection=enabled
```

**Important**: The minimal profile does not include Prometheus by default. Install it manually to enable metrics collection:

```shell
kubectl apply -f https://raw.githubusercontent.com/istio/istio/master/samples/addons/prometheus.yaml
```

### Install Gateway API CRDs

Minikube may lack specific Custom Resource Definitions (CRDs) required for the Gateway API. Install them if missing:

```shell
kubectl get crd gateways.gateway.networking.k8s.io &> /dev/null || \
{ kubectl kustomize "github.com/kubernetes-sigs/gateway-api/config/crd?ref=v1.3.0" | kubectl apply -f -; }
```

### Deploy the application

Clone the demo repository and deploy the application with the Istio service mesh component:

```shell
git clone --depth 1 --branch v0 https://github.com/GoogleCloudPlatform/microservices-demo.git
cd microservices-demo/kustomize
kustomize edit add component components/service-mesh-istio
kubectl apply -k .
```

Wait for all pods to be in the `Running` state before proceeding.

## Usage

### Expose Prometheus

Before running the experiment, expose the Prometheus server so the collector can access it:

```shell
kubectl port-forward -n istio-system service/prometheus 9090:9090
```

### Run the collector

In a new terminal, execute the collector script to run load experiments and collect metrics:

```shell
poetry run python kpp/collector/main.py
```

Results are written to `dataset/performance_results_YYYYMMDD-HHMMSS.csv`.

### Train and predict

After collecting data, run the predictor to train per-service models and generate plots:

```shell
poetry run python kpp/predictor/main.py
```

Results (predictions, loss plots, metrics table) are saved to `results/{strategy}/` (e.g. `results/merged/`).

## Development

```shell
# Run linter, type checker, and unit tests in one step
poetry run poe check

# Run e2e smoke test (predictor pipeline, slower)
poetry run poe e2e

# Run individual checks
poetry run ruff check .
poetry run mypy .
poetry run pytest
```
