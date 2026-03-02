import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

type ServiceDatasets = Dict[str, Dict[str, TensorDataset]]


class PerformanceDataPipeline:
    """
    A robust pipeline for processing Kubernetes telemetry data.

    Flow:
    1. Load CSV & Fix Timestamps (Aggregation)
    2. Split by Microservice
    3. Temporal Split -> Train: first 90% of time-ordered rows; Test: last 10%
    4. Normalize (Per Service, fit on train only) -> Stores Scalers for later inversion
    5. Windowing -> Creates (X, y) tensors for ML training
    """

    THROUGHPUT_COL = "Throughput (req/s)"

    REQUIRED_COLUMNS = [
        "Timestamp",
        "Service",
        "User Count",
        "Response Time (s)",
        THROUGHPUT_COL,
        "CPU Usage",
        "Replicas",
        "CPU Request",
    ]

    DELTA_COLUMNS = ["Δ User Count", "Δ Throughput (req/s)"]

    def __init__(self, sequence_length: int, target_columns: List[str]):
        self.sequence_length = sequence_length
        self.target_columns = target_columns
        self.scalers: Dict[str, MinMaxScaler] = {}  # Used to invert the prediction

    def run(self, csv_path: str, train_ratio: float = 0.9) -> ServiceDatasets:
        """
        Returns a nested dictionary where we save the splits of the dataset for each microservice:
            {
                "frontend": {
                    "train": TensorDataset(X_train, y_train),
                    "test":  TensorDataset(X_test, y_test)
                },
                "backend": ...
            }

        Split strategy: 90/10 temporal split per service. The data is already time-ordered
        from aggregation; the first floor(len * train_ratio) rows go to train, the rest to
        test. This supports an interpolation experiment where the model trains on a
        representative slice of the normal-workload dataset and evaluates on the held-out
        portion.
        """
        df = self._load_data(csv_path)
        service_dfs = self._split_by_service(df)
        processed_datasets = {}

        for service_name, service_df in service_dfs.items():
            service_df = self._add_delta_features(service_df)
            train_raw, test_raw = self._temporal_split(service_df, train_ratio, service_name)
            train_df, test_df = self._normalize_service(train_raw, test_raw, service_name)
            X_train, y_train = self._create_windows(train_df)
            X_test, y_test = self._create_windows(test_df)
            processed_datasets[service_name] = {
                "train": TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                "test": TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
            }

        return processed_datasets

    def _load_data(self, path: str) -> pd.DataFrame:
        """
        Loads CSV, validates headers, and handles timestamp aggregation.
        The timestamps are used for giving the data an order.
        """
        df = pd.read_csv(path)
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required metrics: {missing}")

        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")

        df["Timestamp"] = df["Timestamp"].dt.round("1min")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df.groupby(["Timestamp", "Service"], as_index=False)[numeric_cols].mean()

        missing_before = df[numeric_cols].isna().sum().sum()
        df[numeric_cols] = df.groupby("Service")[numeric_cols].transform(
            lambda x: x.ffill().bfill()
        )
        filled_count = missing_before - df[numeric_cols].isna().sum().sum()
        if filled_count > 0:
            logger.warning(
                f"Filled {filled_count} missing values via ffill/bfill. Large counts may indicate data quality issues."
            )

        return df

    def _split_by_service(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        return {str(service): group.copy() for service, group in df.groupby("Service")}

    def _add_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Δ User Count"] = df["User Count"].diff().fillna(0)
        df["Δ Throughput (req/s)"] = df[self.THROUGHPUT_COL].diff().fillna(0)
        return df

    def _temporal_split(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        service_name: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the raw (un-normalized) dataframe by time order.

        The dataframe is already time-ordered from _load_data aggregation. The first
        floor(len(df) * train_ratio) rows go to train; the remaining rows go to test.
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        logger.info(
            f"[{service_name}] Temporal split (ratio={train_ratio:.2f}): "
            f"{len(train_df)} train, {len(test_df)} test rows."
        )

        return train_df, test_df

    def _normalize_service(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, service_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fits a MinMaxScaler on the training split only, then transforms both splits.
        This prevents test-set information from leaking into the scaler.
        """
        numeric_train = train_df.select_dtypes(include=[np.number])
        numeric_test = test_df.select_dtypes(include=[np.number])

        scaler = MinMaxScaler()
        scaler.fit(numeric_train)
        self.scalers[service_name] = scaler

        train_scaled = scaler.transform(numeric_train)
        test_scaled = scaler.transform(numeric_test)

        train_normalized = pd.DataFrame(
            train_scaled, columns=numeric_train.columns, index=train_df.index
        )
        test_normalized = pd.DataFrame(
            test_scaled, columns=numeric_test.columns, index=test_df.index
        )
        return train_normalized, test_normalized

    def _create_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        target_indices = df.columns.get_indexer(pd.Index(self.target_columns))
        missing_cols = [
            col for col, idx in zip(self.target_columns, target_indices, strict=True) if idx == -1
        ]
        if missing_cols:
            raise ValueError(
                f"Target columns not found in data: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        data = df.to_numpy(dtype=np.float32)
        num_samples = len(data) - self.sequence_length
        if num_samples <= 0:
            raise ValueError(
                f"Not enough data to create windows: {len(data)} rows with "
                f"sequence_length={self.sequence_length}. Need at least {self.sequence_length + 1} rows."
            )

        x_samples = []
        y_samples = []

        for i in range(num_samples):
            x_sample = data[i : i + self.sequence_length]
            y_sample = data[i + self.sequence_length, target_indices]
            x_samples.append(x_sample)
            y_samples.append(y_sample)

        return np.array(x_samples), np.array(y_samples)
