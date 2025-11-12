from collector import Provider
import logging

PROMETHEUS_SERVER_URL = "http://localhost:9090"


def main():
    logging.basicConfig(format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    provider = Provider(PROMETHEUS_SERVER_URL)

    logger.info("ART: %f", provider.get_average_response_time("frontend"))
    logger.info("throughput: %f", provider.get_throughtput("frontend"))
    logger.info("CPU usage: %f", provider.get_cpu_usage("frontend"))


if __name__ == "__main__":
    main()
