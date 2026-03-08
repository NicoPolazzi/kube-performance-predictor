import logging
from typing import ClassVar, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
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

    LOG_TRANSFORM_COLUMNS: ClassVar[List[str]] = ["Response Time (s)"]

    def __init__(self, sequence_length: int, target_columns: List[str]):
        self.sequence_length = sequence_length
        self.target_columns = target_columns
        self.scalers: Dict[str, StandardScaler] = {}  # Used to invert the prediction
        self.input_columns: List[str] = []  # Non-target feature names, set by _create_samples
        self.feature_names: List[str] = []  # All feature names as seen by the scaler

    def run(
        self,
        csv_path: str,
        train_ratio: float = 0.9,
        split_strategy: str = "temporal",
        test_csv_path: str | None = None,
    ) -> ServiceDatasets:
        """
        Returns a nested dictionary where we save the splits of the dataset for each microservice:
            {
                "frontend": {
                    "train": TensorDataset(X_train, y_train),
                    "test":  TensorDataset(X_test, y_test)
                },
                "backend": ...
            }

        Split strategies:
        - "temporal": first floor(len * train_ratio) rows → train; rest → test.
        - "interpolation": middle user-count value(s) held out as test set.
        - "extrapolation": all rows from csv_path → train; all rows from test_csv_path → test.
          Requires test_csv_path to be set.
        """
        df = self._load_data(csv_path)
        service_dfs = self._split_by_service(df)
        processed_datasets = {}

        if split_strategy == "extrapolation":
            if test_csv_path is None:
                raise ValueError("test_csv_path is required when split_strategy='extrapolation'")
            test_df = self._load_data(test_csv_path)
            test_service_dfs = self._split_by_service(test_df)
            paired = self._extrapolation_split(service_dfs, test_service_dfs)
            for service_name, (train_raw, test_raw) in paired.items():
                train_norm, test_norm = self._normalize_service(
                train_raw, test_raw, service_name, fit_on_combined=True
            )
                X_train, y_train = self._create_samples(train_norm)
                X_test, y_test = self._create_samples(test_norm)
                processed_datasets[service_name] = {
                    "train": TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                    "test": TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                }
            return processed_datasets

        for service_name, service_df in service_dfs.items():
            service_df = self._add_ratio_features(service_df)
            if split_strategy == "interpolation":
                train_raw, test_raw = self._interpolation_split(
                    service_df, train_ratio, service_name
                )
            else:
                train_raw, test_raw = self._temporal_split(service_df, train_ratio, service_name)
            train_df, test_df = self._normalize_service(train_raw, test_raw, service_name)
            X_train, y_train = self._create_samples(train_df)
            X_test, y_test = self._create_samples(test_df)
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

    def _add_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Load per Replica"] = df["User Count"] / df["Replicas"]
        df["CPU per User"] = (df["CPU Request"] * df["Replicas"]) / df["User Count"]
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

    def _interpolation_split(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        service_name: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Holds out the middle user-count value(s) as the test set to measure interpolation ability.

        Sorted unique user counts are computed; the middle n_holdout values are withheld.
        This guarantees test samples fall strictly within the training range (interpolation,
        not extrapolation). Requires at least 3 unique user counts.
        """
        sorted_counts = sorted(df["User Count"].unique())
        n_unique = len(sorted_counts)
        if n_unique < 3:
            raise ValueError(
                f"[{service_name}] Interpolation split requires at least 3 unique user counts, "
                f"got {n_unique}: {sorted_counts}"
            )
        n_holdout = max(1, round(n_unique * (1 - train_ratio)))
        start = (n_unique - n_holdout) // 2
        holdout = set(sorted_counts[start : start + n_holdout])

        train_df = df[~df["User Count"].isin(holdout)].copy()
        test_df = df[df["User Count"].isin(holdout)].copy()

        logger.info(
            f"[{service_name}] Interpolation split: holdout user counts={sorted(holdout)}, "
            f"{len(train_df)} train rows, {len(test_df)} test rows."
        )
        return train_df, test_df

    def _extrapolation_split(
        self,
        train_service_dfs: Dict[str, pd.DataFrame],
        test_service_dfs: Dict[str, pd.DataFrame],
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Pairs per-service dataframes from two separate datasets (normal → train, overload → test).

        Only services present in both datasets are included. Services missing from either side
        are logged as warnings and dropped.
        """
        train_services = set(train_service_dfs.keys())
        test_services = set(test_service_dfs.keys())
        common = train_services & test_services
        dropped = (train_services | test_services) - common
        if dropped:
            logger.warning(
                f"Extrapolation split: dropping {len(dropped)} service(s) absent from one dataset: "
                f"{sorted(dropped)}"
            )
        result: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        for service_name in sorted(common):
            train_df = train_service_dfs[service_name].copy()
            test_df = test_service_dfs[service_name].copy()
            logger.info(
                f"[{service_name}] Extrapolation split: {len(train_df)} train rows (normal), "
                f"{len(test_df)} test rows (overload)."
            )
            result[service_name] = (train_df, test_df)
        return result

    def _normalize_service(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        service_name: str,
        fit_on_combined: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fits a StandardScaler then transforms both splits.

        By default fits on training data only (prevents test-set leakage).
        When fit_on_combined=True, fits on the union of train and test — used
        for extrapolation splits where the test distribution is out-of-range of
        the training data and the scaler must cover the full value range.
        """
        numeric_train = train_df.select_dtypes(include=[np.number])
        numeric_test = test_df.select_dtypes(include=[np.number])

        for col in self.LOG_TRANSFORM_COLUMNS:
            if col in numeric_train.columns:
                numeric_train = numeric_train.copy()
                numeric_test = numeric_test.copy()
                numeric_train[col] = np.log10(numeric_train[col] + 1e-9)
                numeric_test[col] = np.log10(numeric_test[col] + 1e-9)

        self.feature_names = numeric_train.columns.tolist()

        scaler = StandardScaler()
        if fit_on_combined:
            scaler.fit(pd.concat([numeric_train, numeric_test]))
        else:
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

    def _create_samples(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        target_indices = df.columns.get_indexer(pd.Index(self.target_columns))
        missing_cols = [
            col for col, idx in zip(self.target_columns, target_indices, strict=True) if idx == -1
        ]
        if missing_cols:
            raise ValueError(f"Target columns not found in data: {missing_cols}.")

        input_indices = [i for i in range(len(df.columns)) if i not in set(target_indices)]
        self.input_columns = [df.columns[i] for i in input_indices]

        data = df.to_numpy(dtype=np.float32)
        if len(data) == 0:
            raise ValueError("Not enough data to create samples.")

        return data[:, input_indices], data[:, target_indices]
