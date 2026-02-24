import csv
from pathlib import Path

import pytest

from kpp.collector.csv_writer import HEADERS, CsvWriter
from kpp.collector.sample import PerformanceSample


@pytest.fixture()
def writer(tmp_path):
    return CsvWriter(dataset_dir=tmp_path)


def _read_csv(path: Path) -> list[list[str]]:
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_write_samples_when_samples_are_not_empty_correctly_write_rows(writer):
    samples = [
        PerformanceSample("svc-a", 0.1, 5.0, 0.2, 0.5),
        PerformanceSample("svc-b", 0.2, 10.0, 0.4, 1.0),
        PerformanceSample("svc-c", 0.3, 15.0, 0.6, 2.0),
    ]
    writer.write_samples(samples, user_count=10, timestamp=1000.0)
    rows = _read_csv(writer.filename)
    assert len(rows) == 4
    assert rows[0] == HEADERS
    assert float(rows[1][0]) == pytest.approx(1000.0)
    assert rows[1][1] == "svc-a"
    assert int(rows[1][2]) == 10
    assert float(rows[1][3]) == pytest.approx(0.1)
    assert float(rows[1][4]) == pytest.approx(5.0)
    assert float(rows[1][5]) == pytest.approx(0.2)
    assert float(rows[1][6]) == pytest.approx(0.2 / 0.5)


def test_write_samples_when_empty_samples_writes_only_header(writer):
    writer.write_samples([], user_count=10, timestamp=1000.0)
    rows = _read_csv(writer.filename)
    assert len(rows) == 1
    assert rows[0] == HEADERS
