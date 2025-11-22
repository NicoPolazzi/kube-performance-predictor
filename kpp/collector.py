import logging
import time
from dataclasses import dataclass

from kubernetes_client import KubernetesClient
from prometheus_client import PrometheusClient

# TODO: use a configuration module instead of hardcoding configuration
PROMETHEUS_URL = "http://localhost:9090"
EXPERIMENT_DURATION_SECONDS = 30
QUERY_SAMPLE_DURATION_SECONDS = 5
WARMUP_PERIOD_SECONDS = QUERY_SAMPLE_DURATION_SECONDS * 2
USER_COUNTS_TO_TEST = [100]


@dataclass
class PerformanceSample:
    service_name: str
    response_time: float
    throughput: float
    cpu_usage: float


def main():
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(filename="app.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    prom_client = PrometheusClient(PROMETHEUS_URL)
    kube_client = KubernetesClient()
    service_names = kube_client.get_services_names()

    try:
        for user_count in USER_COUNTS_TO_TEST:
            logger.info(f"Starting test for {user_count} users...")
            kube_client.change_performance_test_load(str(user_count))
            time.sleep(WARMUP_PERIOD_SECONDS)  # We skip the first performance sample
            _collect_data_samples(logger, service_names=service_names, client=prom_client)
            logger.info(f"Test for {user_count} users ended with success")
    finally:
        kube_client.stop_loadgenerator()

    logger.info("Experiment ended with success.")


def _collect_data_samples(logger, service_names: set[str], client: PrometheusClient) -> None:
    current_experiment_duration = WARMUP_PERIOD_SECONDS
    while current_experiment_duration <= EXPERIMENT_DURATION_SECONDS:
        for service_name in service_names:
            sample = PerformanceSample(
                service_name=service_name,
                response_time=client.get_average_response_time(service_name),
                throughput=client.get_throughput(service_name),
                cpu_usage=client.get_cpu_usage(service_name),
            )
            logger.info(sample)

        time.sleep(QUERY_SAMPLE_DURATION_SECONDS)
        current_experiment_duration += QUERY_SAMPLE_DURATION_SECONDS


if __name__ == "__main__":
    main()
