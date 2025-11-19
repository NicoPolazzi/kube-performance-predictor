import logging
import time

from kubernetes import client, config

DEFAULT_NAMESPACE = "default"
APP_LABEL = "app"

logger = logging.getLogger(__name__)


# TODO: probably a method to stop the loadgenerator at the end of the experiments is needed. At the current state, the generation runs infinitely.
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
            name="loadgenerator", user_count=user_count, api=self.apps_api_istance
        )
        _wait_for_patch_completition(patched_deployment, self.apps_api_istance)


def _patch_deployment(name: str, user_count: str, api: client.AppsV1Api) -> client.V1Deployment:
    deployment = api.read_namespaced_deployment(name=name, namespace="default")

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

    return patched_deployment


def _wait_for_patch_completition(
    patched_deployment: client.V1Deployment, api: client.AppsV1Api
) -> None:
    if patched_deployment is None or patched_deployment.metadata is None:
        logger.error(f"Invalid patched_deployment passed: {patched_deployment}")
        raise ValueError("patched_deployment or its metadata is None")

    target_generation = patched_deployment.metadata.generation

    timeout_seconds = 180
    sleep_interval = 5
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        try:
            deployment = api.read_namespaced_deployment(name="loadgenerator", namespace="default")
        except client.ApiException as e:
            logger.error(f"Error reading deployment status: {e}")
            time.sleep(sleep_interval)
            continue

        status = deployment.status
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
