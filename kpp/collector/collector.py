import time

from kubernetes import client
from kubernetes import config as kube_config
from prometheus_api_client import PrometheusConnect

from kpp.collector.csv_writer import CsvWriter
from kpp.collector.kubernetes_client import KubernetesClient
from kpp.collector.prometheus_client import PrometheusClient
from kpp.collector.sample import PerformanceSample
from kpp.config import CollectorConfig
from kpp.logging_config import setup_logging

COOLDOWN_SECONDS = 180

logger = setup_logging("collector", log_file="app.log")


def main():
    config = CollectorConfig.from_env()
    writer = CsvWriter()
    prom_client = PrometheusClient(PrometheusConnect(url=config.prometheus_url, disable_ssl=True))
    kube_config.load_kube_config()
    kube_client = KubernetesClient(core_api=client.CoreV1Api(), apps_api=client.AppsV1Api())
    service_names = kube_client.get_services_names()

    for service_name, replicas in config.service_replicas.items():
        kube_client.scale_service_deployment(service_name, replicas)
    cpu_requests = kube_client.get_cpu_requests()

    try:
        for user_count in config.user_counts:
            logger.info(f"Starting test for {user_count} users...")
            kube_client.change_performance_test_load(str(user_count))
            time.sleep(config.warmup_period)  # We skip the first performance sample
            _collect_data_samples(
                config=config,
                service_names=service_names,
                client=prom_client,
                writer=writer,
                user_count=user_count,
                cpu_requests=cpu_requests,
            )
            logger.info(f"Test for {user_count} users ended with success")
            logger.info(f"waiting for {COOLDOWN_SECONDS} seconds...")
            kube_client.stop_load_generation()
            time.sleep(COOLDOWN_SECONDS)

    finally:
        kube_client.stop_load_generation()
        for service_name in config.service_replicas:
            kube_client.scale_service_deployment(service_name, 1)

    logger.info("Experiment ended with success!")


def _collect_data_samples(
    config: CollectorConfig,
    service_names: set[str],
    client: PrometheusClient,
    writer: CsvWriter,
    user_count: int,
    cpu_requests: dict[str, float],
) -> None:
    for service in service_names:
        cpu_request = cpu_requests.get(service, 0.0)
        if cpu_request > 0:
            cpu_pct = client.get_cpu_usage(service) / cpu_request
            if not (0.10 <= cpu_pct <= 0.40):
                logger.warning(
                    f"[{user_count} users] {service}: CPU% {cpu_pct:.1%} is outside 10-40% range"
                )

    current_experiment_duration = 0

    while current_experiment_duration <= config.experiment_duration:
        samples_batch = []
        current_timestamp = time.time()

        for service_name in service_names:
            sample = PerformanceSample(
                service_name=service_name,
                response_time=client.get_average_response_time(service_name),
                throughput=client.get_throughput(service_name),
                cpu_usage=client.get_cpu_usage(service_name),
                cpu_request=cpu_requests.get(service_name, 0.0),
            )
            samples_batch.append(sample)

        writer.write_samples(samples_batch, user_count, current_timestamp)
        for sample in samples_batch:
            logger.info(f"[{user_count} users] {sample.service_name}: wrote sample on CSV")
        time.sleep(config.query_interval)
        current_experiment_duration += config.query_interval


if __name__ == "__main__":
    main()
