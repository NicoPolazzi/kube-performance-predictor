from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent.parent.parent / "predictor_config.yaml"


@dataclass(frozen=True)
class PipelineConfig:
    sequence_length: int
    train_lower_percentile: float
    train_upper_percentile: float


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_layers: int
    dropout: float
    use_attention: bool


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    batch_size: int


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


def _load_config() -> PredictorConfig:
    with open(_CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)

    return PredictorConfig(
        pipeline=PipelineConfig(**raw["pipeline"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        scheduler=SchedulerConfig(**raw["scheduler"]),
    )


config = _load_config()
