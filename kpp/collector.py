import logging
import time

from dataclasses import dataclass

from prometheus_client import PrometheusClient
from kubernetes_client import KubernetesClient

PROMETHEUS_URL = "http://localhost:9090"
DEPLOYMENT_NAME = "loadgenerator"
NAMESPACE = "default"
TEST_DURATION_SECONDS = 300
USER_COUNTS_TO_TEST = [10]  # TODO: test with different loads


@dataclass
class PerformanceSample:
    service_name: str
    response_time: float
    throughput: float
    cpu_usage: float


def main():
    logging.basicConfig(format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    prom_client = PrometheusClient(PROMETHEUS_URL)
    kube_client = KubernetesClient()

    service_names = kube_client.get_services_names()

    time_interval = 15

    try:
        while True:
            for serive_name in service_names:
                sample = PerformanceSample(
                    service_name=serive_name,
                    response_time=prom_client.get_average_response_time(serive_name),
                    throughput=prom_client.get_throughtput(serive_name),
                    cpu_usage=prom_client.get_cpu_usage(serive_name),
                )
                logger.info(sample)

            time.sleep(time_interval)

    except KeyboardInterrupt:
        logger.info("program interrupted by the User")


if __name__ == "__main__":
    main()
