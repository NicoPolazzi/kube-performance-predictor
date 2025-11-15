import logging
import time

from dataclasses import dataclass

from prometheus_client import PrometheusClient
from kubernetes_client import KubernetesClient

PROMETHEUS_URL = "http://localhost:9090"
EXPERIMENT_DURATION_SECONDS = 60
QUERY_SAMPLE_DURATION_SECONDS = 10
USER_COUNTS_TO_TEST = [10, 50, 100]


@dataclass
class PerformanceSample:
    service_name: str
    response_time: float
    throughput: float
    cpu_usage: float


def main():
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(filename="app.log", mode="w"), logging.StreamHandler()],
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    prom_client = PrometheusClient(PROMETHEUS_URL)
    kube_client = KubernetesClient()

    service_names = kube_client.get_services_names()

    for user_count in USER_COUNTS_TO_TEST:
        logger.info(f"Starting test for {user_count} users...")
        kube_client.change_performance_test_load(str(user_count))

        current_experiment_duration = 0
        while current_experiment_duration <= EXPERIMENT_DURATION_SECONDS:
            for serive_name in service_names:
                sample = PerformanceSample(
                    service_name=serive_name,
                    response_time=prom_client.get_average_response_time(serive_name),
                    throughput=prom_client.get_throughtput(serive_name),
                    cpu_usage=prom_client.get_cpu_usage(serive_name),
                )
                logger.info(sample)

            time.sleep(QUERY_SAMPLE_DURATION_SECONDS)
            current_experiment_duration += QUERY_SAMPLE_DURATION_SECONDS

        logger.info(f"Test for {user_count} users ended with success")

    logger.info("Experiment ended with success.")


if __name__ == "__main__":
    main()
