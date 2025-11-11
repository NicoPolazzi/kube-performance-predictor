from prometheus_api_client import PrometheusConnect


def get_average_response_time(prometheus_url: str) -> float:
    prom = PrometheusConnect(url=prometheus_url)

    data = prom.custom_query(
        query='sum(rate(istio_request_duration_milliseconds_sum{destination_workload=~"frontend",destination_workload_namespace="default"}[1m]))/sum(rate(istio_request_duration_milliseconds_count{destination_workload=~"frontend",destination_workload_namespace="default"}[1m])) / 1000'
    )

    value_str = data[0]["value"][1]
    return float(value_str)
