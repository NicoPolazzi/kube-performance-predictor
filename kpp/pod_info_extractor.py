from kubernetes import client, config


def get_services_list() -> set[str]:
    """
    Connects to Kubernetes and retrieves a set of services' names.

    We assume that all the services reside in the default namespace and have an app label.
    """
    config.load_kube_config()
    api_instance = client.CoreV1Api()
    pods = api_instance.list_namespaced_pod(namespace="default", label_selector="app", watch=False)

    service_names = {pod.metadata.labels["app"] for pod in pods.items}

    return service_names
