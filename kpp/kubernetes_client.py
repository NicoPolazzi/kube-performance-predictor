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
