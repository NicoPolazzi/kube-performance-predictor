import logging
import time
from typing import cast

from kubernetes import client, config

DEFAULT_NAMESPACE = "default"
APP_LABEL = "app"
LOADGENERATOR_NAME = "loadgenerator"
PATCH_TIMEOUT_SECONDS = 180
PATCH_POLL_INTERVAL_SECONDS = 5

logger = logging.getLogger(__name__)


class KubernetesClient:
    """KubernetesClient is a client that is responsible for interacting with a k8s cluster."""

    api_instance: client.CoreV1Api
    apps_api_istance: client.AppsV1Api

    def __init__(self) -> None:
        config.load_kube_config()
        self.api_instance = client.CoreV1Api()
        self.apps_api_istance = client.AppsV1Api()

    def get_services_names(self) -> set[str]:
        """
        get_service_names recover a list of services' names from a k8s cluster.

        We assume that all target services reside in the default namespace and have an app label.
        """

        pods = self.api_instance.list_namespaced_pod(
            namespace=DEFAULT_NAMESPACE, label_selector=APP_LABEL, watch=False
        )

        service_names = {pod.metadata.labels["app"] for pod in pods.items}
        # This is needed because the redis-cart communicates via TCP.
        service_names.discard("redis-cart")

        logger.debug(f"Service names found in the cluster: {service_names}")
        return service_names

    def change_performance_test_load(self, user_count: str) -> None:
        """
        change_performance_test_load patches the existing loadgenerator deployement.

        This is needed because in this way we can control the performance test.
        To perform the patch we need firstly to get the deployment from K8s, then we apply the patch and then we need
        to wait for the deployment of the new service.

        We aim to generate after one second 'user_count' concurrent users.

        We are patching the container named main and we check multiple criteria in order to check if the
        patch is correctly applied.
        """

        logger.info(f"Changing the number of concurrent users to {user_count}")
        patched_deployment = _patch_deployment(
            name=LOADGENERATOR_NAME, user_count=user_count, api=self.apps_api_istance
        )
        _wait_for_patch_completion(patched_deployment, self.apps_api_istance)

    def get_cpu_requests(self) -> dict[str, float]:
        """
        get_cpu_requests retrieves the CPU request (in cores) for each deployment in the default namespace.

        Returns a dict mapping service name to CPU request in cores.
        Values like "500m" are converted to 0.5; values like "1" are returned as 1.0.
        """
        deployments = self.apps_api_istance.list_namespaced_deployment(
            namespace=DEFAULT_NAMESPACE
        )
        cpu_requests: dict[str, float] = {}

        for deployment in deployments.items:
            name = deployment.metadata.labels.get(APP_LABEL)
            if name is None or name == LOADGENERATOR_NAME:
                continue

            containers = deployment.spec.template.spec.containers
            total_cpu = 0.0
            for container in containers:
                if container.resources and container.resources.requests:
                    raw = container.resources.requests.get("cpu", "0")
                    total_cpu += _parse_cpu(raw)

            if total_cpu > 0:
                cpu_requests[name] = total_cpu

        logger.debug(f"CPU requests per service: {cpu_requests}")
        return cpu_requests

    def stop_loadgenerator(self) -> None:
        """
        Scales the loadgenerator deployment to 0 replicas to stop traffic generation.
        """

        body = {"spec": {"replicas": 0}}

        try:
            self.apps_api_istance.patch_namespaced_deployment(
                name=LOADGENERATOR_NAME, namespace=DEFAULT_NAMESPACE, body=body
            )
            logger.info("Load generator stopped successfully.")
        except client.ApiException as e:
            logger.error(f"Failed to stop load generator: {e}")


def _parse_cpu(raw: str) -> float:
    """Converts a Kubernetes CPU string to float cores. E.g. '500m' -> 0.5, '1' -> 1.0."""
    raw = raw.strip()
    if raw.endswith("m"):
        return int(raw[:-1]) / 1000.0
    return float(raw)


def _patch_deployment(name: str, user_count: str, api: client.AppsV1Api) -> client.V1Deployment:
    deployment = cast(
        client.V1Deployment, api.read_namespaced_deployment(name=name, namespace="default")
    )

    if deployment.spec is None:
        raise ValueError(f"Deployment '{name}' has no spec.")

    deployment.spec.replicas = 1

    container = deployment.spec.template.spec.containers[0]

    for env_var in container.env:
        if env_var.name == "USERS":
            logger.debug(
                f"Found '{env_var.name}'. Changing value from '{env_var.value}' to '{user_count}'"
            )
            env_var.value = user_count

        if env_var.name == "RATE":
            logger.debug(
                f"Found '{env_var.name}'. Changing value from '{env_var.value}' to '{user_count}'"
            )
            env_var.value = user_count

    patched_deployment = api.patch_namespaced_deployment(
        name=name, namespace="default", body=deployment
    )

    return cast(client.V1Deployment, patched_deployment)


def _wait_for_patch_completion(
    patched_deployment: client.V1Deployment, api: client.AppsV1Api
) -> None:
    if patched_deployment is None or patched_deployment.metadata is None:
        logger.error(f"Invalid patched_deployment passed: {patched_deployment}")
        raise ValueError("patched_deployment or its metadata is None")

    target_generation = patched_deployment.metadata.generation

    timeout_seconds = PATCH_TIMEOUT_SECONDS
    sleep_interval = PATCH_POLL_INTERVAL_SECONDS
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        try:
            deployment = cast(
                client.V1Deployment,
                api.read_namespaced_deployment(name=LOADGENERATOR_NAME, namespace="default"),
            )
        except client.ApiException as e:
            logger.error(f"Error reading deployment status: {e}")
            time.sleep(sleep_interval)
            continue

        if deployment.status is None:
            raise ValueError(f"Deployment {deployment} has no status.")

        status = deployment.status

        if deployment.spec is None:
            raise ValueError(f"Deployment '{deployment}' has no spec.")

        spec = deployment.spec

        observed_generation = status.observed_generation or 0
        updated_replicas = status.updated_replicas or 0
        available_replicas = status.available_replicas or 0

        desired_replicas = spec.replicas or 0

        if (
            observed_generation >= target_generation
            and updated_replicas == desired_replicas
            and available_replicas == desired_replicas
        ):
            logger.info("Deployment loadgenerator rollout is complete.")
            break

        time.sleep(sleep_interval)
