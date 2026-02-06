from pathlib import Path

import pytest
from pandas import DataFrame

from kpp.predictor.data_loader import REQUIRED_COLUMNS, load_data


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"


def test_load_data_when_csv_is_valid(test_data_dir):
    csv_path = test_data_dir / "correct_metrics.csv"
    df = load_data(str(csv_path))
    assert isinstance(df, DataFrame)
    assert list(df.columns) == REQUIRED_COLUMNS


def test_load_data_when_csv_has_missing_metrics(test_data_dir):
    csv_path = test_data_dir / "incorrect_metrics.csv"

    with pytest.raises(ValueError, match="Missing required metrics"):
        load_data(str(csv_path))


def test_load_data_validate_timestamps(test_data_dir):
    csv_path = test_data_dir / "correct_metrics.csv"
    df = load_data(str(csv_path))
    service_df = df[df["Service"] == "recommendationservice"].copy()
    time_diffs = service_df["Timestamp"].diff().mean().total_seconds()  # type: ignore
    assert time_diffs == 60
