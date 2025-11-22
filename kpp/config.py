import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration container.
    """

    prometheus_url: str = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
    experiment_duration: int = int(os.getenv("EXPERIMENT_DURATION_SECONDS", "30"))
    query_interval: int = int(os.getenv("QUERY_SAMPLE_DURATION_SECONDS", "5"))
    warmup_period: int = query_interval * 2
    user_counts: list[int] = field(
        default_factory=lambda: [int(x) for x in os.getenv("USER_COUNTS_TO_TEST", "10").split(",")]
    )


config = Config()
