import csv
import logging
import time
from pathlib import Path

from kpp.collector.sample import PerformanceSample

logger = logging.getLogger(__name__)

HEADERS = [
    "Timestamp",
    "Service",
    "User Count",
    "Response Time (s)",
    "Throughput (req/s)",
    "CPU Usage",
    "CPU Usage %",
]

_DEFAULT_DATASET_DIR = Path(__file__).parent.parent.parent / "dataset"


class CsvWriter:
    filename: Path

    def __init__(self, dataset_dir: Path = _DEFAULT_DATASET_DIR) -> None:
        """Creates the file using a timestamp in the name and writes the header."""
        dataset_dir.mkdir(parents=True, exist_ok=True)
        self.filename = dataset_dir / f"performance_results_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        self._initialize_file()

    def _initialize_file(self) -> None:
        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)

    def write_samples(
        self, samples: list[PerformanceSample], user_count: int, timestamp: float
    ) -> None:
        """write_samples writes a batch of performance samples to the CSV."""
        if not samples:
            logger.error("empty samples batch!")
            return

        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            rows = [
                [
                    timestamp,
                    sample.service_name,
                    user_count,
                    sample.response_time,
                    sample.throughput,
                    sample.cpu_usage,
                    sample.cpu_usage / sample.cpu_request,
                ]
                for sample in samples
            ]
            writer.writerows(rows)
            logger.info(f"wrote samples on CSV for {user_count} users.")
