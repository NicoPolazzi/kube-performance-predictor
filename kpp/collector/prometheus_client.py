import math
from typing import Any

from prometheus_api_client import PrometheusConnect


class PrometheusClient:
    """The idea here is to create an adapter for the external dependency."""

    prom: PrometheusConnect

    def __init__(self, prom: PrometheusConnect) -> None:
        self.prom = prom

    def get_average_response_time(self, service_name: str) -> float:
        """Returns the response time in seconds for service_name. Assumes services are in the default namespace."""
        return self._query(
            f'sum(rate(istio_request_duration_milliseconds_sum{{destination_workload=~"{service_name}",destination_workload_namespace="default"}}[1m]))'
            f'/sum(rate(istio_request_duration_milliseconds_count{{destination_workload=~"{service_name}",destination_workload_namespace="default"}}[1m])) / 1000'
        )

    def get_throughput(self, service_name: str) -> float:
        """Returns the requests per second for service_name. Assumes services are in the default namespace."""
        return self._query(
            f'sum(rate(istio_requests_total{{destination_workload=~"{service_name}", destination_workload_namespace="default"}}[1m]))'
        )

    def get_cpu_usage(self, service_name: str) -> float:
        """Returns the CPU usage for pods running service_name. Assumes services are in the default namespace."""
        return self._query(
            f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{service_name}-.*", namespace="default"}}[1m]))'
        )

    def _query(self, promql: str) -> float:
        response = self.prom.custom_query(query=promql)
        return self._extract_metric_value(response)

    def _extract_metric_value(self, response: Any) -> float:
        if not response:
            return math.nan

        return float(response[0]["value"][1])
