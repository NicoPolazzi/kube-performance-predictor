from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kpp.config import ModelConfig, PipelineConfig, PredictorConfig, SchedulerConfig, TrainingConfig
from kpp.predictor.main import compute_metrics
from kpp.predictor.model import PerformanceModel, evaluate, train_model
from kpp.predictor.pipeline import PerformanceDataPipeline

_DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets"
_NORMAL_CSV = _DATASET_DIR / "performance_results_normal.csv"
_OVERLOAD_CSV = _DATASET_DIR / "performance_results_overload.csv"


_CONFIG = PredictorConfig(
    pipeline=PipelineConfig(train_ratio=0.9),
    model=ModelConfig(hidden_size=128, hidden_size_2=64, head_hidden_size=64, dropout=0.2),
    training=TrainingConfig(epochs=50, learning_rate=0.001, batch_size=32, weight_decay=0.001),
    scheduler=SchedulerConfig(factor=0.5, patience=10, min_lr=1e-6),
)

_TARGET_COLS = ["Response Time (s)", "Throughput (req/s)", "CPU Usage"]


_INTERPOLATION_MAPE_CEILINGS = {
    "Response Time (s)": 25.0,
    "Throughput (req/s)": 20.0,
    "CPU Usage": 20.0,
}
_EXTRAPOLATION_MAPE_CEILINGS = {
    "Response Time (s)": 100.0,
    "Throughput (req/s)": 85.0,
    "CPU Usage": 1000.0,
}


def _train_and_assert(pipeline, datasets, mape_ceilings):
    """Train a model per service and run quality/sanity assertions."""
    assert datasets, "Pipeline returned no service datasets"

    for service_name, data_split in sorted(datasets.items()):
        train_dataset = data_split["train"]
        test_dataset = data_split["test"]

        train_loader = DataLoader(
            train_dataset, batch_size=_CONFIG.training.batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=_CONFIG.training.batch_size, shuffle=False, num_workers=0
        )

        input_size = train_dataset.tensors[0].shape[1]
        output_size = train_dataset.tensors[1].shape[1]

        torch.manual_seed(42)
        model = PerformanceModel(
            input_size=input_size,
            output_size=output_size,
            hidden_size=_CONFIG.model.hidden_size,
            hidden_size_2=_CONFIG.model.hidden_size_2,
            head_hidden_size=_CONFIG.model.head_hidden_size,
            dropout=_CONFIG.model.dropout,
        )
        train_model(
            config=_CONFIG,
            service_name=service_name,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=_CONFIG.training.epochs,
            learning_rate=_CONFIG.training.learning_rate,
        )

        scaler = pipeline.scalers[service_name]
        all_features = list(scaler.feature_names_in_)

        real_predictions, real_targets, _ = evaluate(
            model=model,
            test_loader=test_loader,
            scaler=scaler,
            target_columns=_TARGET_COLS,
            feature_names=all_features,
            x_feature_names=list(pipeline.input_columns),
            log_transform_columns=PerformanceDataPipeline.LOG_TRANSFORM_COLUMNS,
        )

        target_indices = [all_features.index(col) for col in _TARGET_COLS]
        metrics = compute_metrics(real_predictions, real_targets, _TARGET_COLS, target_indices)
        for col, col_metrics in metrics.items():
            mape = col_metrics["MAPE"]
            ceiling = mape_ceilings[col]
            assert mape < ceiling, (
                f"Quality gate failed for '{service_name}' / '{col}': MAPE={mape:.2f}% >= {ceiling}%"
            )


def test_interpolation_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    pipeline = PerformanceDataPipeline(_TARGET_COLS)
    datasets = pipeline.run(
        str(_NORMAL_CSV),
        train_ratio=_CONFIG.pipeline.train_ratio,
        split_strategy="interpolation",
    )

    _train_and_assert(pipeline, datasets, _INTERPOLATION_MAPE_CEILINGS)


def test_extrapolation_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    pipeline = PerformanceDataPipeline(_TARGET_COLS)
    datasets = pipeline.run(
        str(_NORMAL_CSV),
        split_strategy="extrapolation",
        test_csv_path=str(_OVERLOAD_CSV),
    )

    _train_and_assert(pipeline, datasets, _EXTRAPOLATION_MAPE_CEILINGS)
