import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

_PREDICTOR_CONFIG_PATH = Path(__file__).parent.parent / "predictor_config.yaml"


# ── Collector ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CollectorConfig:
    prometheus_url: str
    experiment_duration: int
    query_interval: int
    warmup_period: int
    user_counts: list[int]
    service_replicas: dict[str, int]

    @classmethod
    def from_env(cls) -> "CollectorConfig":
        load_dotenv()
        start = int(os.getenv("TEST_USER_START", "10"))
        end = int(os.getenv("TEST_USER_END", "20"))
        step = int(os.getenv("TEST_USER_STEP", "5"))
        raw = os.getenv("SERVICE_REPLICAS", "{}")
        service_replicas: dict[str, int] = json.loads(raw)
        return cls(
            prometheus_url=os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
            experiment_duration=int(os.getenv("EXPERIMENT_DURATION_SECONDS", "600")),
            query_interval=int(os.getenv("QUERY_SAMPLE_DURATION_SECONDS", "60")),
            warmup_period=int(os.getenv("QUERY_SAMPLE_DURATION_SECONDS", "60")) * 2,
            user_counts=list(range(start, end + 1, step)),
            service_replicas=service_replicas,
        )


# ── Predictor ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineConfig:
    sequence_length: int
    train_lower_percentile: float
    train_upper_percentile: float


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    batch_size: int
    weight_decay: float


@dataclass(frozen=True)
class SchedulerConfig:
    factor: float
    patience: int
    min_lr: float


@dataclass(frozen=True)
class PredictorConfig:
    pipeline: PipelineConfig
    model: ModelConfig
    training: TrainingConfig
    scheduler: SchedulerConfig

    @classmethod
    def from_yaml(cls, path: Path = _PREDICTOR_CONFIG_PATH) -> "PredictorConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            pipeline=PipelineConfig(**raw["pipeline"]),
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
            scheduler=SchedulerConfig(**raw["scheduler"]),
        )
