from prometheus_api_client import PrometheusConnect


def get_average_response_time(service_name: str) -> float:
    prom = PrometheusConnect("http://localhost:9090")

    data = prom.custom_query(
        query=f'sum(rate(istio_request_duration_milliseconds_sum{{destination_workload=~"{service_name}",destination_workload_namespace="default"}}[1m]))/sum(rate(istio_request_duration_milliseconds_count{{destination_workload=~"{service_name}",destination_workload_namespace="default"}}[1m])) / 1000'
    )

    value_str = data[0]["value"][1]
    return float(value_str)
