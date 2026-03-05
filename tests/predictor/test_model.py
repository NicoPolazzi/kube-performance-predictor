import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from kpp.config import ModelConfig, PipelineConfig, PredictorConfig, SchedulerConfig, TrainingConfig
from kpp.predictor.model import PerformanceModel, evaluate, train_model

FEATURE_NAMES = [
    "User Count",
    "Response Time (s)",
    "Throughput (req/s)",
    "CPU Usage",
    "Replicas",
    "CPU Request",
    "Δ User Count",
    "Δ Throughput (req/s)",
]
TARGET_COLUMNS = ["Response Time (s)", "Throughput (req/s)", "CPU Usage"]
N_FEATURES = len(FEATURE_NAMES)
N_SAMPLES = 10


def test_forward_produces_correct_output_shape_dimension():
    model = PerformanceModel(input_size=5, output_size=3)
    x = torch.randn(8, 5)  # (batch, features) — flat tabular input
    output = model(x)
    assert output.shape == torch.Size([8, 3])


def _make_loader(n_samples: int, n_features: int, target_columns: list[str], feature_names: list[str], rng: np.random.Generator, num_workers: int = 1) -> tuple[DataLoader, MinMaxScaler]:
    raw_data = rng.random((n_samples, n_features)).astype(np.float32)
    scaler = MinMaxScaler()
    scaler.fit(raw_data)
    scaled = scaler.transform(raw_data).astype(np.float32)
    target_indices = [feature_names.index(c) for c in target_columns]
    X = torch.tensor(scaled)  # (N_SAMPLES, N_FEATURES) — 2D flat
    y = torch.tensor(scaled[:, target_indices])
    loader = DataLoader(TensorDataset(X, y), batch_size=4, num_workers=num_workers)
    return loader, scaler


def test_evaluate_raises_when_target_not_in_features():
    rng = np.random.default_rng(42)
    loader, scaler = _make_loader(N_SAMPLES, N_FEATURES, TARGET_COLUMNS, FEATURE_NAMES, rng)
    model = PerformanceModel(input_size=N_FEATURES, output_size=len(TARGET_COLUMNS))

    with pytest.raises(ValueError, match="not found in feature_names"):
        evaluate(
            model=model,
            test_loader=loader,
            scaler=scaler,
            target_columns=["Nonexistent Column"],
            feature_names=FEATURE_NAMES,
        )


def test_evaluate_raises_when_feature_count_mismatches_scaler():
    rng = np.random.default_rng(42)
    loader, scaler = _make_loader(N_SAMPLES, N_FEATURES, TARGET_COLUMNS, FEATURE_NAMES, rng)
    model = PerformanceModel(input_size=N_FEATURES, output_size=len(TARGET_COLUMNS))

    too_few_features = FEATURE_NAMES[:-1]  # one less than scaler was fitted on
    with pytest.raises(ValueError, match="scaler expects"):
        evaluate(
            model=model,
            test_loader=loader,
            scaler=scaler,
            target_columns=TARGET_COLUMNS,
            feature_names=too_few_features,
        )


def test_evaluate_raises_when_user_count_missing():
    rng = np.random.default_rng(42)
    loader, _ = _make_loader(N_SAMPLES, N_FEATURES, TARGET_COLUMNS, FEATURE_NAMES, rng)
    model = PerformanceModel(input_size=N_FEATURES, output_size=len(TARGET_COLUMNS))

    no_user_count = [f for f in FEATURE_NAMES if f != "User Count"]
    rng2 = np.random.default_rng(0)
    raw2 = rng2.random((N_SAMPLES, len(no_user_count))).astype(np.float32)
    small_scaler = MinMaxScaler()
    small_scaler.fit(raw2)

    with pytest.raises(ValueError, match="User Count"):
        evaluate(
            model=model,
            test_loader=loader,
            scaler=small_scaler,
            target_columns=["Response Time (s)"],
            feature_names=no_user_count,
        )


def test_evaluate_returns_predictions_in_original_scale():
    # Use values in [10, 110] so inverse_transform produces values >> 1.
    # If inverse_transform were removed, target values would remain in [0, 1].
    rng = np.random.default_rng(42)
    raw_data = (rng.random((N_SAMPLES, N_FEATURES)) * 100 + 10).astype(np.float32)
    scaler = MinMaxScaler()
    scaler.fit(raw_data)
    scaled = scaler.transform(raw_data).astype(np.float32)
    target_indices = [FEATURE_NAMES.index(c) for c in TARGET_COLUMNS]
    X = torch.tensor(scaled)
    y = torch.tensor(scaled[:, target_indices])
    loader = DataLoader(TensorDataset(X, y), batch_size=4, num_workers=1)
    model = PerformanceModel(input_size=N_FEATURES, output_size=len(TARGET_COLUMNS))

    real_predictions, real_targets, _ = evaluate(
        model=model,
        test_loader=loader,
        scaler=scaler,
        target_columns=TARGET_COLUMNS,
        feature_names=FEATURE_NAMES,
    )
    assert real_predictions.shape == real_targets.shape
    # Targets should be inverse-transformed back to the original [10, 110] range.
    # Without inverse_transform they would remain in [0, 1].
    assert real_targets[:, target_indices].max() > 5.0


def test_evaluate_returns_user_counts_as_integers():
    rng = np.random.default_rng(42)
    loader, scaler = _make_loader(N_SAMPLES, N_FEATURES, TARGET_COLUMNS, FEATURE_NAMES, rng)
    model = PerformanceModel(input_size=N_FEATURES, output_size=len(TARGET_COLUMNS))

    _, _, user_counts_int = evaluate(
        model=model,
        test_loader=loader,
        scaler=scaler,
        target_columns=TARGET_COLUMNS,
        feature_names=FEATURE_NAMES,
    )
    assert np.issubdtype(user_counts_int.dtype, np.integer)


def test_train_model_restores_best_weights_in_memory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rng = np.random.default_rng(42)
    loader, _ = _make_loader(N_SAMPLES, N_FEATURES, TARGET_COLUMNS, FEATURE_NAMES, rng, num_workers=0)
    target_indices = [FEATURE_NAMES.index(c) for c in TARGET_COLUMNS]
    raw_data = rng.random((N_SAMPLES, N_FEATURES)).astype(np.float32)
    scaler = MinMaxScaler()
    scaler.fit(raw_data)
    scaled = scaler.transform(raw_data).astype(np.float32)
    X = torch.tensor(scaled)
    y = torch.tensor(scaled[:, target_indices])
    train_loader = DataLoader(TensorDataset(X, y), batch_size=4, num_workers=0)
    test_loader = DataLoader(TensorDataset(X, y), batch_size=4, num_workers=0)

    model = PerformanceModel(input_size=N_FEATURES, output_size=len(TARGET_COLUMNS))
    config = PredictorConfig(
        pipeline=PipelineConfig(sequence_length=5, train_ratio=0.8),
        model=ModelConfig(hidden_size=64),
        training=TrainingConfig(epochs=1, learning_rate=0.001, batch_size=4, weight_decay=0.003),
        scheduler=SchedulerConfig(factor=0.5, patience=10, min_lr=1e-6),
    )

    train_model(
        config=config,
        service_name="frontend",
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=1,
    )

    assert not (tmp_path / "models").exists()
    assert model.training is False
