import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

# We assume these functions will exist in 'kpp.predictor.data_processor'
from kpp.predictor.data_processor import create_sliding_windows, normalize_data


@pytest.fixture
def test_time_series_data():
    return pd.DataFrame(
        {
            "user_count": [10, 20, 30, 40, 50, 60],
            "cpu": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "memory": [100, 200, 300, 400, 500, 600],
            "requests": [10, 10, 10, 10, 10, 10],
        }
    )


def test_normalize_data_scales_correctly(test_time_series_data):
    scaled_df, scaler = normalize_data(test_time_series_data)

    assert isinstance(scaled_df, pd.DataFrame)
    assert isinstance(scaler, MinMaxScaler)

    assert scaled_df.max(axis=None) <= 1.0
    assert scaled_df.min(axis=None) >= 0.0
    assert scaled_df["memory"].iloc[-1] == 1.0
    assert scaled_df["memory"].iloc[0] == 0.0


def test_create_sequences_correctly_creates_windows(test_time_series_data):
    sequence_length = 3
    expected_samples = 3
    target_columns = ["cpu", "memory", "requests"]

    X, y = create_sliding_windows(test_time_series_data, sequence_length, target_columns)

    assert X.dtype == np.float32, f"X should be float32, got {X.dtype}"
    assert y.dtype == np.float32, f"y should be float32, got {y.dtype}"

    assert X.shape == (expected_samples, sequence_length, 4)
    assert y.shape == (expected_samples, len(target_columns))

    # Checking the CPU
    assert np.allclose(X[0, :, 1], [0.1, 0.2, 0.3])
    assert np.allclose(X[-1, :, 1], [0.3, 0.4, 0.5])
    assert y[0, 0] == 0.4
    assert y[-1, 0] == 0.6
