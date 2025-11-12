import logging
import time

from dataclasses import dataclass

from kpp.collector import Provider
from kpp.pod_info_extractor import get_services_names

PROMETHEUS_SERVER_URL = "http://localhost:9090"


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

    provider = Provider(PROMETHEUS_SERVER_URL)
    service_names = get_services_names()

    time_interval = 15

    try:
        while True:
            for serive_name in service_names:
                sample = PerformanceSample(
                    service_name=serive_name,
                    response_time=provider.get_average_response_time(serive_name),
                    throughput=provider.get_throughtput(serive_name),
                    cpu_usage=provider.get_cpu_usage(serive_name),
                )
                logger.info(sample)

            time.sleep(time_interval)

    except KeyboardInterrupt:
        logger.info("program interrupted by the User")


if __name__ == "__main__":
    main()
