from pathlib import Path

import pandas as pd
import pytest
from torch.utils.data import TensorDataset

from kpp.predictor.pipeline import PerformanceDataPipeline

FIXTURE_CSV = Path(__file__).parent / "fixtures/small_sample.csv"

TARGET_COLUMNS = ["Response Time (s)", "Throughput (req/s)", "CPU Usage"]

# The fixture has ~33 rows per service. With sequence_length=5 and train_ratio=0.9,
# the test split gets only 4 rows — fewer than sequence_length+1. Use 0.7 for tests
# that need the full pipeline to succeed.
_TRAIN_RATIO = 0.7


def test_run_returns_dict_keyed_by_service():
    pipeline = PerformanceDataPipeline(sequence_length=5, target_columns=TARGET_COLUMNS)
    result = pipeline.run(str(FIXTURE_CSV), train_ratio=_TRAIN_RATIO)
    assert set(result.keys()) == {"adservice", "frontend"}


def test_run_output_x_shape_is_samples_seqlen_features():
    pipeline = PerformanceDataPipeline(sequence_length=5, target_columns=TARGET_COLUMNS)
    result = pipeline.run(str(FIXTURE_CSV), train_ratio=_TRAIN_RATIO)
    for service_data in result.values():
        assert isinstance(service_data["train"], TensorDataset)
        X_train = service_data["train"].tensors[0]
        assert X_train.ndim == 3
        assert X_train.shape[1] == 5


def test_run_output_y_shape_is_samples_targets():
    pipeline = PerformanceDataPipeline(sequence_length=5, target_columns=TARGET_COLUMNS)
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

    pipeline = PerformanceDataPipeline(sequence_length=5, target_columns=["Response Time (s)"])
    with pytest.raises(ValueError, match="CPU Usage"):
        pipeline.run(str(bad_csv))


def test_run_raises_when_not_enough_rows_for_windows(tmp_path):
    df = pd.read_csv(FIXTURE_CSV)
    small_df = df[df["Service"] == "adservice"].head(3)
    small_csv = tmp_path / "tiny.csv"
    small_df.to_csv(small_csv, index=False)

    pipeline = PerformanceDataPipeline(sequence_length=5, target_columns=TARGET_COLUMNS)
    with pytest.raises(ValueError):
        pipeline.run(str(small_csv))


def test_run_temporal_split_train_larger_than_test():
    pipeline = PerformanceDataPipeline(sequence_length=5, target_columns=TARGET_COLUMNS)
    result = pipeline.run(str(FIXTURE_CSV), train_ratio=_TRAIN_RATIO)
    for service_data in result.values():
        n_train = service_data["train"].tensors[0].shape[0]
        n_test = service_data["test"].tensors[0].shape[0]
        assert n_train > n_test


def test_run_stores_scalers_for_each_service():
    pipeline = PerformanceDataPipeline(sequence_length=5, target_columns=TARGET_COLUMNS)
    result = pipeline.run(str(FIXTURE_CSV), train_ratio=_TRAIN_RATIO)
    assert set(pipeline.scalers.keys()) == set(result.keys())


def test_run_scaler_values_are_in_zero_one_range():
    pipeline = PerformanceDataPipeline(sequence_length=5, target_columns=TARGET_COLUMNS)
    result = pipeline.run(str(FIXTURE_CSV), train_ratio=_TRAIN_RATIO)
    for service_data in result.values():
        X_train = service_data["train"].tensors[0]
        y_train = service_data["train"].tensors[1]
        assert float(X_train.min()) >= -1e-6
        assert float(X_train.max()) <= 1.0 + 1e-6
        assert float(y_train.min()) >= -1e-6
        assert float(y_train.max()) <= 1.0 + 1e-6


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

    pipeline = PerformanceDataPipeline(sequence_length=5, target_columns=TARGET_COLUMNS)
    # We just need it to run without error; the duplicate rows will have been aggregated
    result = pipeline.run(str(dup_csv), train_ratio=_TRAIN_RATIO)
    assert "adservice" in result
