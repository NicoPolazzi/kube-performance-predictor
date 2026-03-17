import logging
from typing import ClassVar, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

type ServiceDatasets = Dict[str, Dict[str, TensorDataset]]

CPU_PERCENTAGE_COL = "CPU Percentage"


class PerformanceDataPipeline:
    """
    A robust pipeline for processing Kubernetes telemetry data.

    Flow:
    1. Load CSV & Fix Timestamps (Aggregation)
    2. Split by Microservice
    3. Train/Test Split (interpolation or extrapolation strategy)
    4. Normalize (Per Service, fit on train only) -> Stores Scalers for later inversion
    5. Create (X, y) tensors for ML training
    """

    TIMESTAMP_COL = "Timestamp"
    SERVICE_COL = "Service"
    USER_COUNT_COL = "User Count"
    RESPONSE_TIME_COL = "Response Time (s)"
    THROUGHPUT_COL = "Throughput (req/s)"
    CPU_USAGE_COL = "CPU Usage"
    REPLICAS_COL = "Replicas"
    CPU_REQUEST_COL = "CPU Request"
    LOAD_PER_REPLICA_COL = "Load per Replica"
    CPU_PER_USER_COL = "CPU per User"

    REQUIRED_COLUMNS = [
        TIMESTAMP_COL,
        SERVICE_COL,
        USER_COUNT_COL,
        RESPONSE_TIME_COL,
        THROUGHPUT_COL,
        CPU_USAGE_COL,
        REPLICAS_COL,
        CPU_REQUEST_COL,
    ]

    LOG_TRANSFORM_COLUMNS: ClassVar[List[str]] = [RESPONSE_TIME_COL]

    def __init__(self, target_columns: List[str]):
        self.target_columns = target_columns
        self.scalers: Dict[str, StandardScaler] = {}  # Used to invert the prediction
        self.input_columns: List[str] = []  # Non-target feature names, set by _create_samples
        self.feature_names: List[str] = []  # All feature names as seen by the scaler

    def _load_and_split(
        self,
        csv_path: str,
        train_ratio: float,
        split_strategy: str,
        test_csv_path: str | None,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Returns {service: (train_df, test_df)} with ratio features added."""
        df = self._load_data(csv_path)
        service_dfs = self._split_by_service(df)

        if split_strategy in ("extrapolation", "merged"):
            if test_csv_path is None:
                raise ValueError(
                    f"test_csv_path is required when split_strategy='{split_strategy}'"
                )
            test_df = self._load_data(test_csv_path)
            test_service_dfs = self._split_by_service(test_df)

            if split_strategy == "extrapolation":
                return self._extrapolation_split(service_dfs, test_service_dfs)
            else:
                return self._merged_split_all(
                    service_dfs, test_service_dfs, train_ratio
                )

        result: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        for service_name, service_df in service_dfs.items():
            service_df = self._add_ratio_features(service_df)
            if split_strategy == "interpolation":
                train_raw, test_raw = self._interpolation_split(
                    service_df, train_ratio, service_name
                )
            else:
                raise ValueError(
                    f"Unknown split_strategy '{split_strategy}'. "
                    f"Valid options: 'interpolation', 'extrapolation', 'merged'."
                )
            result[service_name] = (train_raw, test_raw)
        return result

    def run(
        self,
        csv_path: str,
        train_ratio: float = 0.9,
        split_strategy: str = "interpolation",
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
        - "interpolation": middle user-count value(s) held out as test set.
        - "extrapolation": all rows from csv_path → train; all rows from test_csv_path → test.
          Requires test_csv_path to be set.
        - "merged": concatenates csv_path and test_csv_path, then splits using middle
          user-count holdout (same as interpolation). Requires test_csv_path to be set.
        """
        paired = self._load_and_split(csv_path, train_ratio, split_strategy, test_csv_path)
        fit_on_combined = split_strategy in ("extrapolation", "merged")
        processed_datasets: ServiceDatasets = {}

        for service_name, (train_raw, test_raw) in paired.items():
            train_norm, test_norm = self._normalize_service(
                train_raw, test_raw, service_name, fit_on_combined=fit_on_combined
            )
            X_train, y_train = self._create_samples(train_norm)
            X_test, y_test = self._create_samples(test_norm)
            processed_datasets[service_name] = {
                "train": TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                "test": TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
            }

        return processed_datasets

    def run_classification(
        self,
        csv_path: str,
        thresholds: list[float],
        train_ratio: float = 0.9,
        split_strategy: str = "interpolation",
        test_csv_path: str | None = None,
    ) -> ServiceDatasets:
        """
        Like run(), but produces (X, labels) where labels are integer class indices
        derived from CPU percentage thresholds.

        Labels are extracted from raw CPU percentage *before* normalization so that
        the classification targets are based on un-scaled physical values.

        Split strategies:
        - "interpolation" / "extrapolation" / "merged": same semantics as run().
        - "stratified": merges csv_path + test_csv_path, then splits each class
          independently so that every class contributes (1 - train_ratio) of its
          rows to the test set.  Requires test_csv_path.  Unlike "merged", the
          test set is guaranteed to contain all classes present in the data.
        """
        if split_strategy == "stratified":
            return self._run_stratified_classification(
                csv_path, test_csv_path, thresholds, train_ratio
            )

        paired = self._load_and_split(csv_path, train_ratio, split_strategy, test_csv_path)
        fit_on_combined = split_strategy in ("extrapolation", "merged")
        processed_datasets: ServiceDatasets = {}

        for service_name, (train_raw, test_raw) in paired.items():
            train_cpu_pct = self._compute_cpu_percentage(train_raw)
            test_cpu_pct = self._compute_cpu_percentage(test_raw)

            train_labels = self._apply_thresholds(train_cpu_pct, thresholds)
            test_labels = self._apply_thresholds(test_cpu_pct, thresholds)

            train_norm, test_norm = self._normalize_service(
                train_raw, test_raw, service_name, fit_on_combined=fit_on_combined
            )

            X_train = train_norm.to_numpy(dtype=np.float32)
            X_test = test_norm.to_numpy(dtype=np.float32)

            processed_datasets[service_name] = {
                "train": TensorDataset(
                    torch.from_numpy(X_train),
                    torch.from_numpy(train_labels).long(),
                ),
                "test": TensorDataset(
                    torch.from_numpy(X_test),
                    torch.from_numpy(test_labels).long(),
                ),
            }

        return processed_datasets

    def _run_stratified_classification(
        self,
        csv_path: str,
        test_csv_path: str | None,
        thresholds: list[float],
        train_ratio: float,
    ) -> ServiceDatasets:
        """Merges both CSVs, stratifies the split by class label, returns (X, label) datasets."""
        if test_csv_path is None:
            raise ValueError("test_csv_path is required when split_strategy='stratified'")

        train_df = self._load_data(csv_path)
        overload_df = self._load_data(test_csv_path)
        combined_df = pd.concat([train_df, overload_df], ignore_index=True)
        service_dfs = self._split_by_service(combined_df)

        processed_datasets: ServiceDatasets = {}
        for service_name, service_df in sorted(service_dfs.items()):
            service_df = self._add_ratio_features(service_df)

            cpu_pct = self._compute_cpu_percentage(service_df)
            labels = self._apply_thresholds(cpu_pct, thresholds)

            train_raw, test_raw, train_labels, test_labels = self._stratified_cls_split(
                service_df, labels, train_ratio, service_name
            )

            train_norm, test_norm = self._normalize_service(
                train_raw, test_raw, service_name, fit_on_combined=False
            )

            X_train = train_norm.to_numpy(dtype=np.float32)
            X_test = test_norm.to_numpy(dtype=np.float32)

            processed_datasets[service_name] = {
                "train": TensorDataset(
                    torch.from_numpy(X_train),
                    torch.from_numpy(train_labels).long(),
                ),
                "test": TensorDataset(
                    torch.from_numpy(X_test),
                    torch.from_numpy(test_labels).long(),
                ),
            }

        return processed_datasets

    @staticmethod
    def _stratified_cls_split(
        df: pd.DataFrame,
        labels: np.ndarray,
        train_ratio: float,
        service_name: str,
        rng_seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Stratified random split: each class contributes (1 - train_ratio) rows to the test set.

        Returns (train_df, test_df, train_labels, test_labels).
        Each class with ≥2 samples sends at least 1 to the test set.
        Single-sample classes are kept in training only.
        """
        rng = np.random.default_rng(rng_seed)
        train_idx: list[int] = []
        test_idx: list[int] = []

        for cls in np.unique(labels):
            cls_indices = np.where(labels == cls)[0]
            n_test = max(1, round(len(cls_indices) * (1 - train_ratio)))
            if len(cls_indices) < 2:
                train_idx.extend(cls_indices.tolist())
                continue
            n_test = min(n_test, len(cls_indices) - 1)
            sampled = rng.choice(cls_indices, size=n_test, replace=False)
            test_set = set(sampled.tolist())
            test_idx.extend(sorted(test_set))
            train_idx.extend([i for i in cls_indices.tolist() if i not in test_set])

        train_idx_sorted = sorted(train_idx)
        test_idx_sorted = sorted(test_idx)

        unique_vals, counts = np.unique(labels, return_counts=True)
        class_counts = {int(c): int(n) for c, n in zip(unique_vals, counts, strict=True)}
        logger.info(
            f"[{service_name}] Stratified split: {len(train_idx_sorted)} train, "
            f"{len(test_idx_sorted)} test rows. Class distribution: {class_counts}"
        )

        return (
            df.iloc[train_idx_sorted].copy(),
            df.iloc[test_idx_sorted].copy(),
            labels[train_idx_sorted],
            labels[test_idx_sorted],
        )

    @staticmethod
    def _compute_cpu_percentage(df: pd.DataFrame) -> np.ndarray:
        """Returns cpu_usage / (cpu_request * replicas) * 100."""
        cpu_usage = df[PerformanceDataPipeline.CPU_USAGE_COL].to_numpy(dtype=np.float64)
        cpu_request = df[PerformanceDataPipeline.CPU_REQUEST_COL].to_numpy(dtype=np.float64)
        replicas = df[PerformanceDataPipeline.REPLICAS_COL].to_numpy(dtype=np.float64)
        result: np.ndarray = cpu_usage / (cpu_request * replicas) * 100
        return result

    @staticmethod
    def _apply_thresholds(values: np.ndarray, thresholds: list[float]) -> np.ndarray:
        """Assigns integer class labels: 0 below thresholds[0], 1 between, 2 above thresholds[1]."""
        labels = np.zeros(len(values), dtype=np.int64)
        labels[values >= thresholds[0]] = 1
        labels[values >= thresholds[1]] = 2
        return labels

    def _load_data(self, path: str) -> pd.DataFrame:
        """
        Loads CSV, validates headers, and handles timestamp aggregation.
        The timestamps are used for giving the data an order.
        """
        df = pd.read_csv(path)
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required metrics: {missing}")

        if self.TIMESTAMP_COL in df.columns:
            df[self.TIMESTAMP_COL] = pd.to_datetime(df[self.TIMESTAMP_COL], unit="s")

        df[self.TIMESTAMP_COL] = df[self.TIMESTAMP_COL].dt.round("1min")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df.groupby([self.TIMESTAMP_COL, self.SERVICE_COL], as_index=False)[numeric_cols].mean()

        missing_before = df[numeric_cols].isna().sum().sum()
        df[numeric_cols] = df.groupby(self.SERVICE_COL)[numeric_cols].transform(
            lambda x: x.ffill().bfill()
        )
        filled_count = missing_before - df[numeric_cols].isna().sum().sum()
        if filled_count > 0:
            logger.warning(
                f"Filled {filled_count} missing values via ffill/bfill. Large counts may indicate data quality issues."
            )

        return df

    def _split_by_service(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        return {str(service): group.copy() for service, group in df.groupby(self.SERVICE_COL)}

    def _add_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.LOAD_PER_REPLICA_COL] = df[self.USER_COUNT_COL] / df[self.REPLICAS_COL]
        df[self.CPU_PER_USER_COL] = (df[self.CPU_REQUEST_COL] * df[self.REPLICAS_COL]) / df[self.USER_COUNT_COL]
        return df

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
        sorted_counts = sorted(df[self.USER_COUNT_COL].unique())
        n_unique = len(sorted_counts)
        if n_unique < 3:
            raise ValueError(
                f"[{service_name}] Interpolation split requires at least 3 unique user counts, "
                f"got {n_unique}: {[float(x) for x in sorted_counts]}"
            )
        n_holdout = max(1, round(n_unique * (1 - train_ratio)))
        start = (n_unique - n_holdout) // 2
        holdout = set(sorted_counts[start : start + n_holdout])

        train_df = df[~df[self.USER_COUNT_COL].isin(holdout)].copy()
        test_df = df[df[self.USER_COUNT_COL].isin(holdout)].copy()

        logger.info(
            f"[{service_name}] Interpolation split: holdout user counts={[float(x) for x in sorted(holdout)]}, "
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

    def _merged_split_all(
        self,
        train_service_dfs: Dict[str, pd.DataFrame],
        test_service_dfs: Dict[str, pd.DataFrame],
        train_ratio: float,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Concatenates per-service dataframes from two datasets, then splits using
        middle user-count holdout (same as interpolation).

        Only services present in both datasets are included. By using the same
        split logic as interpolation but with overload data in the training set,
        merged acts as a direct control for the extrapolation experiment.
        """
        train_services = set(train_service_dfs.keys())
        test_services = set(test_service_dfs.keys())
        common = train_services & test_services
        dropped = (train_services | test_services) - common
        if dropped:
            logger.warning(
                f"Merged split: dropping {len(dropped)} service(s) absent from one dataset: "
                f"{sorted(dropped)}"
            )
        result: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        for service_name in sorted(common):
            combined = pd.concat(
                [train_service_dfs[service_name], test_service_dfs[service_name]],
                ignore_index=True,
            )
            combined = self._add_ratio_features(combined)
            train_df, test_df = self._interpolation_split(combined, train_ratio, service_name)
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
