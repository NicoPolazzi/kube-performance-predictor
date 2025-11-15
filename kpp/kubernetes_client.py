import time
from kubernetes import client, config


class KubernetesClient:
    """KubernetesClient is a client that is responsible for interacting with a k8s cluster."""

    api_instance: client.CoreV1Api

    def __init__(self) -> None:
        config.load_kube_config()
        self.api_instance = client.CoreV1Api()

    def get_services_names(self) -> set[str]:
        """
        get_service_names recover a list of services' names from a k8s cluster.

        We assume that all target services reside in the default namespace and have an app label.
        """

        pods = self.api_instance.list_namespaced_pod(
            namespace="default", label_selector="app", watch=False
        )

        service_names = {pod.metadata.labels["app"] for pod in pods.items}

        return service_names

    def apply_loadgenerator_patch(self, user_count: str) -> None:
        """
        apply_loadgenerator_patch patches the existing loadgenerator deployement.

        This is needed because in this way we can control the performance test.
        To perform the patch we need firstly to get the deployment from K8s, then we apply the patch and then we need
        to wait for the deployment of the new service.

        We are patching the container named main
        """

        # Apply the patch
        kube_app_api = client.AppsV1Api()
        deployment = kube_app_api.read_namespaced_deployment(
            name="loadgenerator", namespace="default"
        )

        container = deployment.spec.template.spec.containers[0]  # type: ignore

        for env_var in container.env:
            if env_var.name == "USERS":
                print(
                    f"Found '{env_var.name}'. Changing value from '{env_var.value}' to '{user_count}'"
                )
                env_var.value = user_count
                break

        patched_deployment = kube_app_api.patch_namespaced_deployment(
            name="loadgenerator", namespace="default", body=deployment
        )

        target_generation = patched_deployment.metadata.generation  # type: ignore

        # Wait for the patch to be applied
        print(
            f"\nWaiting for deployment loadgenerator (generation {target_generation}) to complete..."
        )
        timeout_seconds = 180
        sleep_interval = 5
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            try:
                deployment = kube_app_api.read_namespaced_deployment(
                    name="loadgenerator", namespace="default"
                )
            except client.ApiException as e:
                print(f"Error reading deployment status: {e}")
                time.sleep(sleep_interval)
                continue

            status = deployment.status  # type: ignore
            spec = deployment.spec  # type: ignore

            observed_generation = status.observed_generation or 0
            updated_replicas = status.updated_replicas or 0
            available_replicas = status.available_replicas or 0
            desired_replicas = spec.replicas

            if (
                observed_generation >= target_generation
                and updated_replicas == desired_replicas
                and available_replicas == desired_replicas
            ):

                print("✅ Deployment loadgenerator rollout is complete.")
                break

            time.sleep(sleep_interval)

        print("Ready to start the experiment!")
