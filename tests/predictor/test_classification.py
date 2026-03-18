from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from kpp.config import ClassificationConfig, ModelConfig, PipelineConfig, PredictorConfig, SchedulerConfig, TrainingConfig
from kpp.predictor.classification import (
    _extract_input_features,
    _run_classification_experiment,
    _run_regression_classification_experiment,
    compute_classification_metrics,
    cpu_to_label,
    regression_to_classes,
    validate_csv_path,
)

_FIXTURE_CSV = Path(__file__).parent / "fixtures/small_sample.csv"


@pytest.mark.parametrize(
    "values, expected",
    [
        # Good: below good_upper
        (np.array([0.0, 10.0, 39.9]), [0, 0, 0]),
        # Danger: between good_upper and danger_upper
        (np.array([40.0, 50.0, 59.9]), [1, 1, 1]),
        # Bottleneck: at or above danger_upper
        (np.array([60.0, 80.0, 100.0]), [2, 2, 2]),
        # Exact boundaries
        (np.array([40.0, 60.0]), [1, 2]),
        # Empty array
        (np.array([]), []),
    ],
    ids=["good", "danger", "bottleneck", "exact_boundaries", "empty"],
)
def test_cpu_to_label_assigns_correct_classes(values, expected):
    labels = cpu_to_label(values, thresholds=[40.0, 60.0])
    np.testing.assert_array_equal(labels, expected)


def test_cpu_to_label_handles_nan_input():
    values = np.array([10.0, np.nan, 70.0])
    labels = cpu_to_label(values, thresholds=[40.0, 60.0])
    assert labels[0] == 0
    assert labels[2] == 2


def test_compute_classification_metrics_returns_per_class_and_macro():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    class_names = ["good", "danger", "bottleneck"]

    metrics = compute_classification_metrics(y_true, y_pred, class_names)

    assert set(metrics.keys()) == {"good", "danger", "bottleneck", "weighted"}
    for name in class_names:
        assert metrics[name]["precision"] == 1.0
        assert metrics[name]["recall"] == 1.0
        assert metrics[name]["f1"] == 1.0
    assert metrics["weighted"]["f1"] == 1.0


def test_regression_to_classes_thresholds_predictions():
    # cpu_pct = cpu_pred / (cpu_request * replicas) * 100
    # 0.02 / (0.1 * 1) * 100 = 20% → good (0)
    # 0.05 / (0.1 * 1) * 100 = 50% → danger (1)
    # 0.08 / (0.1 * 1) * 100 = 80% → bottleneck (2)
    cpu_preds = np.array([0.02, 0.05, 0.08])
    cpu_requests = np.array([0.1, 0.1, 0.1])
    replicas = np.array([1.0, 1.0, 1.0])
    thresholds = [40.0, 60.0]

    labels = regression_to_classes(cpu_preds, cpu_requests, replicas, thresholds)
    np.testing.assert_array_equal(labels, [0, 1, 2])


def test_classification_config_properties_derive_from_thresholds():
    cfg = ClassificationConfig(good_upper=30.0, danger_upper=70.0)
    assert cfg.thresholds == [30.0, 70.0]
    assert cfg.class_names == ["good", "danger", "bottleneck"]
    assert cfg.num_classes == 3


def test_validate_csv_path_raises_for_missing_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        validate_csv_path("/nonexistent/path.csv")


def test_validate_csv_path_succeeds_for_existing_file(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("a,b\n1,2\n")
    validate_csv_path(str(csv))


def test_run_classification_experiment_produces_metrics_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Build a minimal dataset: 3 unique user counts, enough rows for interpolation split
    df = pd.read_csv(_FIXTURE_CSV)
    normal_csv = tmp_path / "normal.csv"
    df.to_csv(normal_csv, index=False)

    config = PredictorConfig(
        pipeline=PipelineConfig(train_ratio=0.7),
        model=ModelConfig(hidden_size=16, hidden_size_2=8, head_hidden_size=8, dropout=0.0),
        training=TrainingConfig(epochs=2, learning_rate=0.01, batch_size=8, weight_decay=0.0),
        scheduler=SchedulerConfig(factor=0.5, patience=5, min_lr=1e-6),
    )
    cls_config = ClassificationConfig(good_upper=40.0, danger_upper=60.0)
    output_dir = tmp_path / "results"

    _run_classification_experiment(
        config=config,
        cls_config=cls_config,
        csv_path=str(normal_csv),
        overload_csv_path=None,
        split_strategy="interpolation",
        output_dir=output_dir,
        table_title="Test Experiment",
    )

    md_files = list(output_dir.rglob("*.md"))
    assert md_files, "Expected at least one Markdown metrics file to be written"


def test_extract_input_features_returns_original_scale():
    # Build a known scaler fitted on [CPU Request, Replicas] columns (in a 3-feature space:
    # User Count, CPU Request, Replicas). We put CPU Request at index 1, Replicas at index 2.
    feature_names = ["User Count", "CPU Request", "Replicas"]
    input_columns = ["User Count", "CPU Request", "Replicas"]

    cpu_requests_orig = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    replicas_orig = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    user_counts = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    data = np.column_stack([user_counts, cpu_requests_orig, replicas_orig])
    scaler = StandardScaler()
    scaler.fit(data)
    normalized = scaler.transform(data).astype(np.float32)

    X = torch.tensor(normalized, dtype=torch.float32)
    y = torch.zeros(len(X), 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False)

    cpu_req_out, replicas_out = _extract_input_features(loader, input_columns, feature_names, scaler)

    np.testing.assert_allclose(cpu_req_out, cpu_requests_orig, rtol=1e-4)
    np.testing.assert_allclose(replicas_out, replicas_orig, rtol=1e-4)


def test_run_regression_classification_experiment_produces_metrics_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    df = pd.read_csv(_FIXTURE_CSV)
    normal_csv = tmp_path / "normal.csv"
    df.to_csv(normal_csv, index=False)

    config = PredictorConfig(
        pipeline=PipelineConfig(train_ratio=0.7),
        model=ModelConfig(hidden_size=16, hidden_size_2=8, head_hidden_size=8, dropout=0.0),
        training=TrainingConfig(epochs=2, learning_rate=0.01, batch_size=8, weight_decay=0.0),
        scheduler=SchedulerConfig(factor=0.5, patience=5, min_lr=1e-6),
    )
    cls_config = ClassificationConfig(good_upper=40.0, danger_upper=60.0)
    output_dir = tmp_path / "results"

    _run_regression_classification_experiment(
        config=config,
        cls_config=cls_config,
        csv_path=str(normal_csv),
        overload_csv_path=None,
        split_strategy="interpolation",
        output_dir=output_dir,
        table_title="Regression Test",
    )

    md_files = list(output_dir.rglob("*.md"))
    assert md_files, "Expected at least one Markdown metrics file to be written"
    png_files = list((output_dir / "confusion").rglob("*.png"))
    assert png_files, "Expected at least one confusion matrix PNG to be written"
