import csv
import logging
import time

from sample import PerformanceSample

logger = logging.getLogger(__name__)


class CsvWriter:
    filename: str

    def __init__(self) -> None:
        """Creates the file using a timestamp in the name and writes the header."""
        self.filename = f"performance_results_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        self._initialize_file()

    def _initialize_file(self) -> None:
        headers = ["Service", "User Count", "Response Time (s)", "Throughput (req/s)", "CPU Usage"]

        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def write_samples(self, samples: list[PerformanceSample], user_count: int) -> None:
        """write_samples writes a batch of performance samples to the CSV."""
        if not samples:
            logger.error("empty samples batch!")
            return

        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            rows = [
                [
                    sample.service_name,
                    user_count,
                    sample.response_time,
                    sample.throughput,
                    sample.cpu_usage,
                ]
                for sample in samples
            ]
            writer.writerows(rows)
            logger.info(f"wrote samples on CSV for {user_count} users.")
