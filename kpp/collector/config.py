import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _load_user_counts() -> List[int]:
    """
    Generates a list of user counts.
    """
    start = os.getenv("TEST_USER_START", "10")
    end = os.getenv("TEST_USER_END", "20")
    step = os.getenv("TEST_USER_STEP", "5")

    return list(range(int(start), int(end) + 1, int(step)))


@dataclass(frozen=True)
class Config:
    prometheus_url: str = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
    experiment_duration: int = int(os.getenv("EXPERIMENT_DURATION_SECONDS", "600"))
    query_interval: int = int(os.getenv("QUERY_SAMPLE_DURATION_SECONDS", "60"))
    warmup_period: int = int(os.getenv("QUERY_SAMPLE_DURATION_SECONDS", "60"))
    user_counts: list[int] = field(default_factory=_load_user_counts)


config = Config()
