import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from kpp.predictor.model import PerformanceModel, evaluate

FEATURE_NAMES = [
    "User Count",
    "Response Time (s)",
    "Throughput (req/s)",
    "CPU Usage",
    "Δ User Count",
    "Δ Throughput (req/s)",
]
TARGET_COLUMNS = ["Response Time (s)", "Throughput (req/s)", "CPU Usage"]
N_FEATURES = len(FEATURE_NAMES)
SEQ_LEN = 4
N_SAMPLES = 10


def _make_loader_and_model() -> tuple[DataLoader, PerformanceModel, MinMaxScaler]:
    rng = np.random.default_rng(42)
    raw_data = rng.random((N_SAMPLES + SEQ_LEN, N_FEATURES)).astype(np.float32)

    scaler = MinMaxScaler()
    scaler.fit(raw_data)
    scaled = scaler.transform(raw_data).astype(np.float32)

    target_indices = [FEATURE_NAMES.index(c) for c in TARGET_COLUMNS]
    x_list, y_list = [], []
    for i in range(N_SAMPLES):
        x_list.append(scaled[i : i + SEQ_LEN])
        y_list.append(scaled[i + SEQ_LEN, target_indices])

    X = torch.tensor(np.array(x_list))
    y = torch.tensor(np.array(y_list))
    loader = DataLoader(TensorDataset(X, y), batch_size=4, num_workers=1)

    model = PerformanceModel(input_size=SEQ_LEN * N_FEATURES, output_size=len(TARGET_COLUMNS))
    return loader, model, scaler


def test_forward_produces_correct_output_shape_dimension():
    model = PerformanceModel(input_size=5 * 4, output_size=3)
    x = torch.randn(8, 4, 5)
    output = model(x)
    assert output.shape == torch.Size([8, 3])


def test_evaluate_raises_when_target_not_in_features():
    loader, model, scaler = _make_loader_and_model()
    with pytest.raises(ValueError, match="not found in feature_names"):
        evaluate(
            model=model,
            test_loader=loader,
            scaler=scaler,
            target_columns=["Nonexistent Column"],
            feature_names=FEATURE_NAMES,
        )


def test_evaluate_raises_when_feature_count_mismatches_scaler():
    loader, model, scaler = _make_loader_and_model()
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
    loader, model, _ = _make_loader_and_model()
    no_user_count = [f for f in FEATURE_NAMES if f != "User Count"]

    # Refit a scaler with one fewer column so the count matches
    rng = np.random.default_rng(0)
    raw = rng.random((N_SAMPLES + SEQ_LEN, len(no_user_count))).astype(np.float32)
    small_scaler = MinMaxScaler()
    small_scaler.fit(raw)

    with pytest.raises(ValueError, match="User Count"):
        evaluate(
            model=model,
            test_loader=loader,
            scaler=small_scaler,
            target_columns=["Response Time (s)"],
            feature_names=no_user_count,
        )


def test_evaluate_returns_predictions_in_original_scale():
    loader, model, scaler = _make_loader_and_model()
    real_predictions, real_targets, _ = evaluate(
        model=model,
        test_loader=loader,
        scaler=scaler,
        target_columns=TARGET_COLUMNS,
        feature_names=FEATURE_NAMES,
    )
    assert real_predictions.shape == real_targets.shape
    assert real_predictions.dtype in (np.float32, np.float64)


def test_evaluate_returns_user_counts_as_integers():
    loader, model, scaler = _make_loader_and_model()
    _, _, user_counts_int = evaluate(
        model=model,
        test_loader=loader,
        scaler=scaler,
        target_columns=TARGET_COLUMNS,
        feature_names=FEATURE_NAMES,
    )
    assert np.issubdtype(user_counts_int.dtype, np.integer)
