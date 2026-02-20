import logging
from typing import Dict, List, Tuple, TypeAlias

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("predictor.pipeline")

ServiceDatasets: TypeAlias = Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]


class PerformancesDataPipeline:
    """
    A robust pipeline for processing Kubernetes telemetry data.

    Flow:
    1. Load CSV & Fix Timestamps (Aggregation)
    2. Split by Microservice
    3. Normalize (Per Service) -> Stores Scalers for later inversion
    4. Stratified Split (Per User Load) -> Prevents extrapolation errors
    5. Windowing -> Creates (X, y) tensors for ML training
    """

    REQUIRED_COLUMNS = [
        "Timestamp",
        "Service",
        "User Count",
        "Response Time (s)",
        "Throughput (req/s)",
        "CPU Usage",
    ]

    def __init__(self, sequence_length: int, target_columns: List[str]):
        self.sequence_length = sequence_length
        self.target_columns = target_columns
        self.scalers: Dict[str, MinMaxScaler] = {}  # Used to invert the prediction

    def run(
        self, csv_path: str, split_ratio: float = 0.8
    ) -> ServiceDatasets:
        """
        Returns a nested dictionary where we save the splits of the dataset for each microservice:
            {
                "frontend": {
                    "train": (X_train, y_train),
                    "test":  (X_test, y_test)
                },
                "backend": ...
            }
        """
        df = self._load_data(csv_path)
        service_dfs = self._split_by_service(df)
        processed_datasets = {}

        for service_name, service_df in service_dfs.items():
            if "User Count" not in service_df.columns:
                raise ValueError(
                    f"Column 'User Count' required for stratified splitting (service: {service_name})."
                )
            normalized_df = self._normalize_service(service_df, service_name)
            train_df, test_df = self._stratified_split(normalized_df, split_ratio)
            X_train, y_train = self._create_windows(train_df)
            X_test, y_test = self._create_windows(test_df)
            processed_datasets[service_name] = {
                "train": (X_train, y_train),
                "test": (X_test, y_test),
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
            logger.warning(f"Filled {filled_count} missing values via ffill/bfill. Large counts may indicate data quality issues.")

        return df

    def _split_by_service(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        return {str(service): group.copy() for service, group in df.groupby("Service")}

    def _normalize_service(self, df: pd.DataFrame, service_name: str) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include=[np.number])
        scaler = MinMaxScaler()
        scaler.fit(numeric_df)
        scaled_values = scaler.transform(numeric_df)

        self.scalers[service_name] = scaler
        return pd.DataFrame(scaled_values, columns=numeric_df.columns, index=df.index)

    def _stratified_split(
        self, df: pd.DataFrame, ratio: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data while preserving the 'User Count' distribution. So, for each user count,
        the function creates a train-test split with the specified ratio.
        """
        train_dfs = []
        test_dfs = []

        for _, group in df.groupby("User Count"):
            group = group.sort_index()
            split_idx = int(len(group) * ratio)
            test_slice = group.iloc[split_idx:]
            if len(test_slice) == 0:
                logger.warning(
                    f"A user count group of size {len(group)} produced no test samples "
                    f"with ratio={ratio}. Consider collecting more data."
                )
            train_dfs.append(group.iloc[:split_idx])
            test_dfs.append(test_slice)

        train = pd.concat(train_dfs).sort_index()
        test = pd.concat(test_dfs).sort_index()
        return train, test

    def _create_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        target_indices = df.columns.get_indexer(pd.Index(self.target_columns))
        missing_cols = [col for col, idx in zip(self.target_columns, target_indices, strict=True) if idx == -1]
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

        X_samples = []
        y_samples = []

        for i in range(num_samples):
            X_sample = data[i : i + self.sequence_length]
            y_sample = data[i + self.sequence_length, target_indices]
            X_samples.append(X_sample)
            y_samples.append(y_sample)

        return np.array(X_samples), np.array(y_samples)
