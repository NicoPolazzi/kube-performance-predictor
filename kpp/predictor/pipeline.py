import logging
from typing import Dict, List, Tuple, TypeAlias

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

ServiceDatasets: TypeAlias = Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]


class PerformancesDataPipeline:
    """
    A robust pipeline for processing Kubernetes telemetry data.

    Flow:
    1. Load CSV & Fix Timestamps (Aggregation)
    2. Split by Microservice
    3. Throughput-Percentile Split -> Train: [min, max] req/s; Test: outside that range
    4. Normalize (Per Service, fit on train only) -> Stores Scalers for later inversion
    5. Windowing -> Creates (X, y) tensors for ML training
    """

    REQUIRED_COLUMNS = [
        "Timestamp",
        "Service",
        "User Count",
        "Response Time (s)",
        "Throughput (req/s)",
        "CPU Usage",
        "CPU Usage %",
    ]

    DELTA_COLUMNS = ["Δ CPU Usage %", "Δ User Count", "Δ Throughput (req/s)"]

    def __init__(self, sequence_length: int, target_columns: List[str]):
        self.sequence_length = sequence_length
        self.target_columns = target_columns
        self.scalers: Dict[str, MinMaxScaler] = {}  # Used to invert the prediction

    def run(
        self,
        csv_path: str,
        train_lower_percentile: float = 0.15,
        train_upper_percentile: float = 0.85,
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

        Split strategy: samples within the per-service throughput percentile range
        [train_lower_percentile, train_upper_percentile] go to train; samples outside
        that range go to test. Percentiles are computed per service, so each service gets a
        balanced split regardless of its absolute throughput level. The scaler is fit exclusively
        on the training split to avoid data leakage.
        """
        df = self._load_data(csv_path)
        service_dfs = self._split_by_service(df)
        processed_datasets = {}

        for service_name, service_df in service_dfs.items():
            service_df = self._add_delta_features(service_df)
            train_raw, test_raw = self._throughput_percentile_split(
                service_df, train_lower_percentile, train_upper_percentile, service_name
            )
            train_df, test_df = self._normalize_service(train_raw, test_raw, service_name)
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
            logger.warning(
                f"Filled {filled_count} missing values via ffill/bfill. Large counts may indicate data quality issues."
            )

        return df

    def _split_by_service(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        return {str(service): group.copy() for service, group in df.groupby("Service")}

    def _add_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Δ CPU Usage %"] = df["CPU Usage %"].diff().fillna(0)
        df["Δ User Count"] = df["User Count"].diff().fillna(0)
        df["Δ Throughput (req/s)"] = df["Throughput (req/s)"].diff().fillna(0)
        return df

    def _throughput_percentile_split(
        self,
        df: pd.DataFrame,
        lower_percentile: float,
        upper_percentile: float,
        service_name: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the raw (un-normalized) dataframe by per-service throughput percentile boundaries.

        The lower and upper throughput thresholds are computed from this service's own distribution,
        so each service gets a balanced split regardless of its absolute throughput level. Samples
        within [p_lower, p_upper] go to train; samples outside go to test.
        """
        throughput = df["Throughput (req/s)"]
        min_val = throughput.quantile(lower_percentile)
        max_val = throughput.quantile(upper_percentile)
        train_mask = (throughput >= min_val) & (throughput <= max_val)

        train_df = df[train_mask].copy()
        test_df = df[~train_mask].copy()

        logger.info(
            f"[{service_name}] Throughput-percentile split "
            f"(p{lower_percentile * 100:.0f}–p{upper_percentile * 100:.0f}): "
            f"{train_mask.sum()} train ({min_val:.2f}–{max_val:.2f} req/s), "
            f"{(~train_mask).sum()} test."
        )

        if len(train_df) == 0:
            raise ValueError(
                f"[{service_name}] No training samples found in the throughput percentile range "
                f"[p{lower_percentile * 100:.0f}, p{upper_percentile * 100:.0f}]. "
                f"Adjust train_lower_percentile / train_upper_percentile."
            )
        if len(test_df) == 0:
            logger.warning(
                f"[{service_name}] No test samples outside the throughput percentile range. "
                "The evaluation will not measure out-of-distribution generalisation."
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

        scaler = MinMaxScaler(clip=True)
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

        X_samples = []
        y_samples = []

        for i in range(num_samples):
            X_sample = data[i : i + self.sequence_length]
            y_sample = data[i + self.sequence_length, target_indices]
            X_samples.append(X_sample)
            y_samples.append(y_sample)

        return np.array(X_samples), np.array(y_samples)
