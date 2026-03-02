from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceSample:
    service_name: str
    response_time: float
    throughput: float
    cpu_usage: float
    replicas: int
    cpu_request: float
