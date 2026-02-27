import logging
import time
from typing import cast

from kubernetes import client

DEFAULT_NAMESPACE = "default"
APP_LABEL = "app"
LOADGENERATOR_NAME = "loadgenerator"
PATCH_TIMEOUT_SECONDS = 180
PATCH_POLL_INTERVAL_SECONDS = 5

logger = logging.getLogger(__name__)


class KubernetesClient:
    """Client for interacting with a Kubernetes cluster during performance tests.

    Two separate API groups are required because Kubernetes splits its API surface by resource kind:

    - CoreV1Api  — pod-level resources (used by get_services_names to list running pods)
    - AppsV1Api  — deployment-level resources (used by get_cpu_requests, stop_load_generation,
                   and change_performance_test_load to read and patch Deployments)
    """

    api_instance: client.CoreV1Api
    apps_api_istance: client.AppsV1Api

    def __init__(self, core_api: client.CoreV1Api, apps_api: client.AppsV1Api) -> None:
        self.api_instance = core_api
        self.apps_api_istance = apps_api

    def get_services_names(self) -> set[str]:
        """
        get_service_names recover a list of services' names from a k8s cluster.

        We assume that all target services reside in the default namespace and have an app label.

        NOTE: redis-cart is a Redis instance that communicates over raw TCP, not HTTP.
        Istio's Envoy proxy does not emit HTTP-level metrics (request count, latency) for
        TCP-only workloads, so redis-cart would always appear as zero-traffic in Prometheus.
        Excluding it avoids misleading gaps in the collected dataset.
        """
        pods = self.api_instance.list_namespaced_pod(
            namespace=DEFAULT_NAMESPACE, label_selector=APP_LABEL, watch=False
        )
        service_names = {pod.metadata.labels[APP_LABEL] for pod in pods.items}
        service_names.discard("redis-cart")
        logger.debug(f"Service names found in the cluster: {service_names}")
        return service_names

    def change_performance_test_load(self, user_count: str) -> None:
        """
        Patches the loadgenerator deployment to run with user_count concurrent users.

        The flow is:
        1. Read the current loadgenerator deployment from Kubernetes.
        2. Set replicas to 1 — the deployment may have been scaled to 0 by stop_loadgenerator()
           between experiments, so this ensures the pod is actually running.
        3. Set the USERS and RATE env vars on every container to user_count. The loadgenerator
           has a single app container, but we search all containers by env-var name rather than
           hardcoding an index so that Istio sidecar injection does not break the patch.
        4. Apply the patch and wait for the rollout to complete before returning.
        """

        logger.info(f"Changing the number of concurrent users to {user_count}")
        patched_deployment = self._patch_loadgenerator_deployment(user_count=user_count)
        self._wait_for_patch_completion(patched_deployment, self.apps_api_istance)

    def _wait_for_patch_completion(
        self, patched_deployment: client.V1Deployment, api: client.AppsV1Api
    ) -> None:
        deployment_name = patched_deployment.metadata.name
        target_generation = patched_deployment.metadata.generation
        timeout_seconds = PATCH_TIMEOUT_SECONDS
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            try:
                deployment = cast(
                    client.V1Deployment,
                    api.read_namespaced_deployment(name=deployment_name, namespace="default"),
                )
            except client.ApiException as e:
                logger.error(f"Error reading deployment status: {e}")
                time.sleep(PATCH_POLL_INTERVAL_SECONDS)
                continue

            observed_generation = deployment.status.observed_generation or 0
            updated_replicas = deployment.status.updated_replicas or 0
            available_replicas = deployment.status.available_replicas or 0
            desired_replicas = deployment.spec.replicas or 0

            if (
                observed_generation >= target_generation
                and updated_replicas == desired_replicas
                and available_replicas == desired_replicas
            ):
                logger.info(f"Deployment {deployment_name} rollout is complete.")
                break

            time.sleep(PATCH_POLL_INTERVAL_SECONDS)
        else:
            raise TimeoutError(
                f"Deployment '{deployment_name}' did not complete within {timeout_seconds} seconds."
            )

    def _patch_loadgenerator_deployment(self, user_count: str) -> client.V1Deployment:
        deployment = cast(
            client.V1Deployment,
            self.apps_api_istance.read_namespaced_deployment(
                name=LOADGENERATOR_NAME, namespace=DEFAULT_NAMESPACE
            ),
        )

        deployment.spec.replicas = 1
        for container in deployment.spec.template.spec.containers:
            for env_var in container.env:
                if env_var.name in ("USERS", "RATE"):
                    logger.debug(
                        f"Found '{env_var.name}'. Changing value from '{env_var.value}' to '{user_count}'"
                    )
                    env_var.value = user_count

        patched_deployment = self.apps_api_istance.patch_namespaced_deployment(
            name=LOADGENERATOR_NAME, namespace=DEFAULT_NAMESPACE, body=deployment
        )

        return cast(client.V1Deployment, patched_deployment)

    def get_cpu_requests(self) -> dict[str, float]:
        """
        get_cpu_requests retrieves the CPU request (in cores) for each deployment in the default namespace.

        Returns a dict mapping service name to CPU request in cores.
        Values like "500m" are converted to 0.5; values like "1" are returned as 1.0.
        """
        deployments = self.apps_api_istance.list_namespaced_deployment(namespace=DEFAULT_NAMESPACE)
        cpu_requests: dict[str, float] = {}

        for deployment in deployments.items:
            name = deployment.metadata.labels.get(APP_LABEL)
            if name is None or name == LOADGENERATOR_NAME:
                continue

            containers = deployment.spec.template.spec.containers
            total_cpu = 0.0
            for container in containers:
                if container.resources and container.resources.requests:
                    cpu = container.resources.requests.get("cpu", "0")
                    total_cpu += self._convert_millicores_to_quantity(cpu)

            if total_cpu > 0:
                replicas = deployment.spec.replicas or 1
                cpu_requests[name] = total_cpu * replicas

        logger.debug(f"CPU requests per service: {cpu_requests}")
        return cpu_requests

    def _convert_millicores_to_quantity(self, raw: str) -> float:
        raw = raw.strip()
        if raw.endswith("m"):
            return int(raw[:-1]) / 1000.0
        return float(raw)

    def scale_service_deployment(self, service_name: str, replicas: int) -> None:
        """
        Scales a named service deployment to the specified replica count and waits for rollout.
        """
        body = {"spec": {"replicas": replicas}}
        patched = self.apps_api_istance.patch_namespaced_deployment(
            name=service_name, namespace=DEFAULT_NAMESPACE, body=body
        )
        self._wait_for_patch_completion(cast(client.V1Deployment, patched), self.apps_api_istance)

    def stop_load_generation(self) -> None:
        """Scales the loadgenerator deployment to 0 replicas, stopping traffic generation."""

        body = {"spec": {"replicas": 0}}

        try:
            self.apps_api_istance.patch_namespaced_deployment(
                name=LOADGENERATOR_NAME, namespace=DEFAULT_NAMESPACE, body=body
            )
            logger.info("Load generator stopped successfully.")
        except client.ApiException as e:
            logger.error(f"Failed to stop load generator: {e}")
