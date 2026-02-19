import time

from config import config
from csv_writer import CsvWriter
from kubernetes_client import KubernetesClient
from kpp.logging_config import setup_logging
from prometheus_client import PrometheusClient
from sample import PerformanceSample

COOLDOWN_SECONDS = 180


def main():
    logger = setup_logging("collector", log_file="app.log")

    writer = CsvWriter()
    prom_client = PrometheusClient(config.prometheus_url)
    kube_client = KubernetesClient()
    service_names = kube_client.get_services_names()

    try:
        for user_count in config.user_counts:
            logger.info(f"Starting test for {user_count} users...")
            kube_client.change_performance_test_load(str(user_count))
            _collect_data_samples(
                service_names=service_names,
                client=prom_client,
                writer=writer,
                user_count=user_count,
            )
            logger.info(f"Test for {user_count} users ended with success")
            logger.info(f"waiting for {COOLDOWN_SECONDS} seconds...")
            kube_client.stop_loadgenerator()
            time.sleep(COOLDOWN_SECONDS)

    finally:
        kube_client.stop_loadgenerator()

    logger.info("Experiment ended with success!")


def _collect_data_samples(
    service_names: set[str], client: PrometheusClient, writer: CsvWriter, user_count: int
) -> None:
    time.sleep(config.warmup_period)  # We skip the first performance sample
    current_experiment_duration = config.warmup_period

    while current_experiment_duration <= config.experiment_duration:
        samples_batch = []
        current_timestamp = time.time()

        for service_name in service_names:
            sample = PerformanceSample(
                service_name=service_name,
                response_time=client.get_average_response_time(service_name),
                throughput=client.get_throughput(service_name),
                cpu_usage=client.get_cpu_usage(service_name),
            )
            samples_batch.append(sample)

        writer.write_samples(samples_batch, user_count, current_timestamp)
        time.sleep(config.query_interval)
        current_experiment_duration += config.query_interval


if __name__ == "__main__":
    main()
