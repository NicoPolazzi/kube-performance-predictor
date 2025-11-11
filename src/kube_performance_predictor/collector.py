import math
from prometheus_api_client import PrometheusConnect

prom = PrometheusConnect("http://localhost:9090")
VALUE_KEY = "value"


def get_average_response_time(service_name: str) -> float:
    """
    get_average_response_time returns the response time in seconds of a service_name quering the Prometheus server.

    We are assuming that all our services resides in the default namespace.
    """
    result = prom.custom_query(
        query=f'sum(rate(istio_request_duration_milliseconds_sum{{destination_workload=~"{service_name}",destination_workload_namespace="default"}}[1m]))/sum(rate(istio_request_duration_milliseconds_count{{destination_workload=~"{service_name}",destination_workload_namespace="default"}}[1m])) / 1000'
    )

    if not result:
        return math.nan

    value_str = result[0][VALUE_KEY][1]
    return float(value_str)


def get_throughtput(service_name: str) -> float:
    """
    get_throughput returns the throughtput of a service_name quering the Prometheus server.

    We are assuming that all our services resides in the default namespace.
    """

    result = prom.custom_query(
        query=f'sum(rate(istio_requests_total{{destination_workload=~"{service_name}", destination_workload_namespace="default"}}[5m]))'
    )

    if not result:
        return math.nan

    value_str = result[0][VALUE_KEY][1]
    return float(value_str)


def get_cpu_usage(service_name: str) -> float:
    """
    get_cpu_usage returns the cpu usage of the pods running a service_name quering the Prometheus server.

    We are assuming that all our services resides in the default namespace.
    """

    result = prom.custom_query(
        query=f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{service_name}-.*", destination_workload_namespace="default"}}[5m]))'
    )

    if not result:
        return math.nan

    value_str = result[0][VALUE_KEY][1]
    return float(value_str)
