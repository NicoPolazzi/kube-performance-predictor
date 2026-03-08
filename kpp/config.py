from dataclasses import dataclass
from pathlib import Path

import yaml

_PREDICTOR_CONFIG_PATH = Path(__file__).parent.parent / "confs" / "predictor_config.yaml"
_EXPERIMENTS_CONFIG_PATH = Path(__file__).parent.parent / "confs" / "experiments.yaml"


# ── Collector ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExperimentConfig:
    users: int
    replicas: dict[str, int]


@dataclass(frozen=True)
class CollectorConfig:
    experiment_duration: int
    query_interval: int
    warmup_period: int
    experiments: list[ExperimentConfig]

    @classmethod
    def from_yaml(cls, path: Path = _EXPERIMENTS_CONFIG_PATH) -> "CollectorConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        experiments = [ExperimentConfig(**e) for e in raw["experiments"]]
        query_interval = int(raw["query_sample_duration_seconds"])
        return cls(
            experiment_duration=int(raw["experiment_duration_seconds"]),
            query_interval=query_interval,
            warmup_period=query_interval * 2,
            experiments=experiments,
        )


# ── Predictor ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineConfig:
    sequence_length: int
    train_ratio: float
    split_strategy: str = "interpolation"


@dataclass
class ModelConfig:
    hidden_size: int
    hidden_size_2: int
    head_hidden_size: int = 32
    dropout: float = 0.0


@dataclass
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
