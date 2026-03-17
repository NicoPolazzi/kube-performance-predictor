from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from kpp.predictor.model import evaluate
from kpp.predictor.pipeline import PerformanceDataPipeline

FIXTURE_CSV = Path(__file__).parent / "fixtures/small_sample.csv"

TARGET_COLUMNS = ["Response Time (s)", "Throughput (req/s)", "CPU Usage"]

_TRAIN_RATIO = 0.7


def test_run_returns_dict_keyed_by_service():
    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    result = pipeline.run(str(FIXTURE_CSV), train_ratio=_TRAIN_RATIO)
    assert set(result.keys()) == {"adservice", "frontend"}


def test_run_output_x_shape_is_samples_features():
    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    result = pipeline.run(str(FIXTURE_CSV), train_ratio=_TRAIN_RATIO)
    for service_data in result.values():
        assert isinstance(service_data["train"], TensorDataset)
        X_train = service_data["train"].tensors[0]
        assert X_train.ndim == 2
        assert X_train.shape[1] == 6  # 9 total features (incl. "CPU Usage %" extra col) minus 3 targets


def test_run_output_y_shape_is_samples_targets():
    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    result = pipeline.run(str(FIXTURE_CSV), train_ratio=_TRAIN_RATIO)
    for service_data in result.values():
        assert isinstance(service_data["train"], TensorDataset)
        y_train = service_data["train"].tensors[1]
        assert y_train.ndim == 2
        assert y_train.shape[1] == len(TARGET_COLUMNS)


def test_run_raises_when_required_column_missing(tmp_path):
    df = pd.read_csv(FIXTURE_CSV).drop(columns=["CPU Usage"])
    bad_csv = tmp_path / "bad.csv"
    df.to_csv(bad_csv, index=False)

    pipeline = PerformanceDataPipeline(target_columns=["Response Time (s)"])
    with pytest.raises(ValueError, match="CPU Usage"):
        pipeline.run(str(bad_csv))


def test_run_succeeds_with_minimal_rows(tmp_path):
    """With tabular sampling, any number of rows > 0 produces samples (no window size constraint)."""
    df = pd.read_csv(FIXTURE_CSV)
    # Take one row per unique user count to get exactly 3 unique counts (minimum for interpolation)
    small_df = df[df["Service"] == "adservice"].drop_duplicates(subset=["User Count"]).head(3)
    small_csv = tmp_path / "tiny.csv"
    small_df.to_csv(small_csv, index=False)

    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    result = pipeline.run(str(small_csv))
    assert "adservice" in result
    X_train = result["adservice"]["train"].tensors[0]
    assert X_train.ndim == 2


def test_run_stores_scalers_for_each_service():
    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    result = pipeline.run(str(FIXTURE_CSV), train_ratio=_TRAIN_RATIO)
    assert set(pipeline.scalers.keys()) == set(result.keys())


def test_run_normalizes_training_data_with_standard_scaler():
    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    result = pipeline.run(str(FIXTURE_CSV), train_ratio=_TRAIN_RATIO)
    for service_data in result.values():
        X_train = service_data["train"].tensors[0]
        y_train = service_data["train"].tensors[1]
        # StandardScaler: values centered near 0, std ≈ 1; most values within ~5 stds
        assert float(X_train.min()) > -5.0
        assert float(X_train.max()) < 5.0
        assert float(y_train.min()) > -5.0
        assert float(y_train.max()) < 5.0


def test_run_interpolation_split_test_size_matches_held_out_rows():
    # Fixture: 3 unique user counts [4, 6, 8], 11 rows each per service.
    # train_ratio=0.9 → n_holdout=max(1, round(3*0.1))=1 → holdout=[6].
    # Test split: 11 rows; with tabular sampling → 11 samples directly.
    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    result = pipeline.run(str(FIXTURE_CSV), train_ratio=0.9, split_strategy="interpolation")
    for service_data in result.values():
        n_test = service_data["test"].tensors[0].shape[0]
        assert n_test == 11  # 11 rows for the held-out user count → 11 tabular samples


def test_run_interpolation_split_raises_when_too_few_user_counts(tmp_path):
    df = pd.read_csv(FIXTURE_CSV)
    # Keep only rows with a single user count so there are < 3 unique values
    small_df = df[df["User Count"] == 4]
    small_csv = tmp_path / "single_count.csv"
    small_df.to_csv(small_csv, index=False)

    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    with pytest.raises(ValueError, match="at least 3 unique user counts"):
        pipeline.run(str(small_csv), split_strategy="interpolation")


def test_run_extrapolation_split_returns_all_normal_rows_as_train(tmp_path):
    df = pd.read_csv(FIXTURE_CSV)
    normal_csv = tmp_path / "normal.csv"
    overload_csv = tmp_path / "overload.csv"
    df.to_csv(normal_csv, index=False)
    # Overload CSV: shift user counts up so they are outside training range
    overload_df = df.copy()
    overload_df["User Count"] = overload_df["User Count"] * 10
    overload_df.to_csv(overload_csv, index=False)

    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    result = pipeline.run(
        str(normal_csv),
        split_strategy="extrapolation",
        test_csv_path=str(overload_csv),
    )

    assert set(result.keys()) == {"adservice", "frontend"}
    for service_data in result.values():
        assert isinstance(service_data["train"], TensorDataset)
        assert isinstance(service_data["test"], TensorDataset)


def test_run_extrapolation_split_drops_service_missing_from_test(tmp_path):
    df = pd.read_csv(FIXTURE_CSV)
    normal_csv = tmp_path / "normal.csv"
    overload_csv = tmp_path / "overload_partial.csv"
    df.to_csv(normal_csv, index=False)
    # Overload CSV has only adservice, not frontend
    overload_df = df[df["Service"] == "adservice"].copy()
    overload_df.to_csv(overload_csv, index=False)

    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    result = pipeline.run(
        str(normal_csv),
        split_strategy="extrapolation",
        test_csv_path=str(overload_csv),
    )

    assert set(result.keys()) == {"adservice"}


def test_run_extrapolation_split_raises_when_test_csv_path_is_none():
    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    with pytest.raises(ValueError, match="test_csv_path is required"):
        pipeline.run(str(FIXTURE_CSV), split_strategy="extrapolation")


def test_run_merged_split_returns_correct_services(tmp_path):
    df = pd.read_csv(FIXTURE_CSV)
    normal_csv = tmp_path / "normal.csv"
    overload_csv = tmp_path / "overload.csv"
    df.to_csv(normal_csv, index=False)
    overload_df = df.copy()
    overload_df["User Count"] = overload_df["User Count"] * 10
    overload_df.to_csv(overload_csv, index=False)

    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    result = pipeline.run(
        str(normal_csv),
        split_strategy="merged",
        test_csv_path=str(overload_csv),
    )

    assert set(result.keys()) == {"adservice", "frontend"}


def test_run_merged_split_test_size_matches_held_out_rows(tmp_path):
    # Fixture: 3 unique user counts [4, 6, 8], 11 rows each per service.
    # Overload multiplies by 10 → [40, 60, 80]. Combined: 6 unique counts.
    # train_ratio=0.9 → n_holdout=max(1, round(6*0.1))=1 → holdout middle value.
    # Sorted counts: [4, 6, 8, 40, 60, 80], start=(6-1)//2=2, holdout=[8].
    # Test split: 11 rows for user count 8.
    df = pd.read_csv(FIXTURE_CSV)
    normal_csv = tmp_path / "normal.csv"
    overload_csv = tmp_path / "overload.csv"
    df.to_csv(normal_csv, index=False)
    overload_df = df.copy()
    overload_df["User Count"] = overload_df["User Count"] * 10
    overload_df.to_csv(overload_csv, index=False)

    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    result = pipeline.run(
        str(normal_csv),
        train_ratio=0.9,
        split_strategy="merged",
        test_csv_path=str(overload_csv),
    )

    for service_data in result.values():
        n_test = service_data["test"].tensors[0].shape[0]
        assert n_test == 11  # 11 rows for the held-out user count


def test_run_merged_split_raises_when_test_csv_path_is_none():
    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    with pytest.raises(ValueError, match="test_csv_path is required"):
        pipeline.run(str(FIXTURE_CSV), split_strategy="merged")


def test_run_aggregates_rows_with_same_rounded_timestamp(tmp_path):
    df = pd.read_csv(FIXTURE_CSV)
    svc_df = df[df["Service"] == "adservice"].copy()

    # Create two rows with timestamps that round to the same minute
    base_ts = 1771846210.0
    row1 = svc_df.iloc[0].copy()
    row2 = svc_df.iloc[0].copy()
    row1["Timestamp"] = base_ts
    row2["Timestamp"] = base_ts + 10  # same minute after rounding

    dup_df = pd.concat([pd.DataFrame([row1, row2]), svc_df.iloc[2:]], ignore_index=True)
    dup_csv = tmp_path / "dup.csv"
    dup_df.to_csv(dup_csv, index=False)

    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    result = pipeline.run(str(dup_csv), train_ratio=_TRAIN_RATIO)

    # dup_df has len(svc_df) input rows for adservice; after merging the two same-minute
    # rows into one, there are len(svc_df)-1 rows → len(svc_df)-1 tabular samples total.
    n_train = result["adservice"]["train"].tensors[0].shape[0]
    n_test = result["adservice"]["test"].tensors[0].shape[0]
    assert n_train + n_test == len(svc_df) - 1


def test_normalize_service_fits_scaler_on_log_response_time(tmp_path):
    # Build 10 rows with distinct user counts [1..10].
    # Interpolation split with train_ratio=0.7 → n_holdout=max(1, round(10*0.3))=3
    # Holdout user counts are the middle 3: [4, 5, 6]
    # Training rows are user counts [1, 2, 3, 7, 8, 9, 10]
    n = 10
    max_rt = 100.0
    rt_values = [max_rt, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 5.0]
    df = pd.DataFrame({
        "Timestamp": [1000.0 + i * 60 for i in range(n)],
        "Service": ["svc"] * n,
        "User Count": list(range(1, n + 1)),
        "Response Time (s)": rt_values,
        "Throughput (req/s)": [100.0] * n,
        "CPU Usage": [0.5] * n,
        "Replicas": [1.0] * n,
        "CPU Request": [0.5] * n,
    })
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    pipeline.run(str(csv_path), train_ratio=0.7, split_strategy="interpolation")

    scaler = pipeline.scalers["svc"]
    rt_col_idx = list(scaler.feature_names_in_).index("Response Time (s)")
    # Training rows are user counts [1,2,3,7,8,9,10] → indices [0,1,2,6,7,8,9]
    train_indices = [0, 1, 2, 6, 7, 8, 9]
    train_rt = [rt_values[i] for i in train_indices]
    expected_mean = float(np.mean(np.log10(np.array(train_rt) + 1e-9)))
    assert abs(scaler.mean_[rt_col_idx] - expected_mean) < 1e-3


def test_run_classification_returns_long_labels():
    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    result = pipeline.run_classification(
        str(FIXTURE_CSV), thresholds=[40.0, 60.0], train_ratio=_TRAIN_RATIO
    )
    assert set(result.keys()) == {"adservice", "frontend"}
    for service_data in result.values():
        labels = service_data["train"].tensors[1]
        assert labels.dtype == torch.long
        assert labels.ndim == 1
        assert set(labels.tolist()).issubset({0, 1, 2})


def test_run_classification_x_shape_matches_regression():
    reg_pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    reg_result = reg_pipeline.run(str(FIXTURE_CSV), train_ratio=_TRAIN_RATIO)

    cls_pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    cls_result = cls_pipeline.run_classification(
        str(FIXTURE_CSV), thresholds=[40.0, 60.0], train_ratio=_TRAIN_RATIO
    )

    for service in reg_result:
        # Classification X has all features (no target split), regression X excludes targets
        reg_total_features = (
            reg_result[service]["train"].tensors[0].shape[1]
            + reg_result[service]["train"].tensors[1].shape[1]
        )
        cls_features = cls_result[service]["train"].tensors[0].shape[1]
        assert cls_features == reg_total_features


def test_run_classification_stratified_all_classes_in_test(tmp_path):
    # Build a dataset where cpu_pct spans all three classes:
    #   good (<40%):       cpu_usage=0.02  → pct=20%
    #   danger (40-60%):   cpu_usage=0.05  → pct=50%
    #   bottleneck (≥60%): cpu_usage=0.08  → pct=80%
    # (cpu_request=0.1, replicas=1 → pct = cpu_usage / 0.1 * 100)
    # Each class uses a non-overlapping timestamp range so the pipeline does not
    # average rows from different classes together during its groupby aggregation.
    n = 30

    def _make_rows(cpu_usage_val: float, ts_start: float) -> pd.DataFrame:
        return pd.DataFrame({
            "Timestamp": np.arange(ts_start, ts_start + n * 60, 60, dtype=float),
            "Service": ["svc"] * n,
            "User Count": [50.0] * n,
            "Response Time (s)": [0.01] * n,
            "Throughput (req/s)": [10.0] * n,
            "CPU Usage": [cpu_usage_val] * n,
            "Replicas": [1.0] * n,
            "CPU Request": [0.1] * n,
        })

    normal_csv = tmp_path / "normal.csv"
    overload_csv = tmp_path / "overload.csv"
    # Each class occupies a distinct time window (n * 60 seconds apart)
    _make_rows(0.02, 1_000_000.0).to_csv(normal_csv, index=False)
    pd.concat(
        [_make_rows(0.05, 1_000_000.0 + n * 60), _make_rows(0.08, 1_000_000.0 + 2 * n * 60)],
        ignore_index=True,
    ).to_csv(overload_csv, index=False)

    pipeline = PerformanceDataPipeline(target_columns=TARGET_COLUMNS)
    result = pipeline.run_classification(
        str(normal_csv),
        thresholds=[40.0, 60.0],
        train_ratio=0.9,
        split_strategy="stratified",
        test_csv_path=str(overload_csv),
    )

    assert "svc" in result
    test_labels = result["svc"]["test"].tensors[1].tolist()
    # All three classes must be present in the test set
    assert set(test_labels) == {0, 1, 2}
    train_labels = result["svc"]["train"].tensors[1].tolist()
    assert set(train_labels) == {0, 1, 2}


def test_evaluate_inverts_log_transform_for_response_time():
    # Set up a scaler fitted on log10-space values for two features: User Count + Response Time.
    feature_names = ["User Count", "Response Time (s)"]
    target_columns = ["Response Time (s)"]
    original_rt = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    log_rt = np.log10(original_rt + 1e-9)
    user_counts = np.array([10.0, 20.0, 30.0])

    train_data = np.column_stack([user_counts, log_rt]).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(train_data)
    normalized = scaler.transform(train_data)

    # X: (n, features=2) — flat 2D tabular input
    X = torch.tensor(normalized, dtype=torch.float32)
    y = torch.tensor(normalized[:, 1:2], dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, y), batch_size=3, shuffle=False)

    class ZeroModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.size(0), 1)

    _, real_targets, _ = evaluate(
        model=ZeroModel(),
        test_loader=loader,
        scaler=scaler,
        target_columns=target_columns,
        feature_names=feature_names,
        log_transform_columns=["Response Time (s)"],
    )

    rt_idx = feature_names.index("Response Time (s)")
    np.testing.assert_allclose(real_targets[:, rt_idx], original_rt, rtol=1e-5)
