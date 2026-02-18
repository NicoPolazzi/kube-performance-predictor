import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalize the data contained in the dataframe using a MinMaxScaler. It returns the fitted scaler
    and the normalized dataframe.
    """
    scaler = MinMaxScaler()
    scaler.fit(dataframe)
    scaled_df = pd.DataFrame(scaler.transform(dataframe), columns=dataframe.columns)

    return scaled_df, scaler


def create_sliding_windows(
    dataframe: pd.DataFrame, sequence_length: int, target_columns: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows of sequence_length from dataframe.

    X: History of all features.
    y: Future of only target features.
    """
    target_indices = dataframe.columns.get_indexer(pd.Index(target_columns))
    data = dataframe.to_numpy(dtype=np.float32)

    X_samples = []
    y_samples = []
    num_samples = len(data) - sequence_length

    for i in range(num_samples):
        X_sample = data[i : i + sequence_length]
        y_sample = data[i + sequence_length, target_indices]
        X_samples.append(X_sample)
        y_samples.append(y_sample)

    return np.array(X_samples), np.array(y_samples)
