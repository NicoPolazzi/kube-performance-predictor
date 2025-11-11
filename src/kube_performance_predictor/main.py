from prometheus_api_client import PrometheusConnect

PROMETHEUS_SERVER_URL = "http://localhost:9090"


def main():
    prom = PrometheusConnect(PROMETHEUS_SERVER_URL, disable_ssl=True)
    print(
        prom.custom_query(
            'sum(rate(istio_request_duration_milliseconds_sum{destination_workload=~"frontend",destination_workload_namespace="default"}[1m]))/sum(rate(istio_request_duration_milliseconds_count{destination_workload=~"frontend",destination_workload_namespace="default"}[1m])) / 1000'
        )
    )

    print(
        prom.custom_query(
            'sum(rate(istio_requests_total{destination_workload!~"unknown"}[5m])) by (destination_workload)'
        )
    )

    print(prom.get_current_metric_value(metric_name="up", label_config={"namespace": "default"}))

    print(
        prom.custom_query(
            'sum(rate(container_cpu_usage_seconds_total{pod=~"emailservice-.*", namespace="default"}[5m])) by (pod)'
        )
    )

    print(
        prom.custom_query(
            'sum by (destination_workload) (istio_requests_total{destination_workload_namespace="default"})'
        )
    )


if __name__ == "__main__":
    main()
