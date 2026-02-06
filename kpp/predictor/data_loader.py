import pandas as pd

REQUIRED_COLUMNS = [
    "Timestamp",
    "Service",
    "User Count",
    "Response Time (s)",
    "Throughput (req/s)",
    "CPU Usage",
]


def load_data(path: str) -> pd.DataFrame:
    """Loads CSV and verify the correct header"""
    df = pd.read_csv(path)

    missing_column = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_column:
        raise ValueError(f"Missing required metrics: {missing_column}")
    return df
