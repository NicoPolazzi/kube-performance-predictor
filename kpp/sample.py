from dataclasses import dataclass


@dataclass
class PerformanceSample:
    service_name: str
    response_time: float
    throughput: float
    cpu_usage: float
