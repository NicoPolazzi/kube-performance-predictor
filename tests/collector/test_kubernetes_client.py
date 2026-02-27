from types import SimpleNamespace

import pytest

from kpp.collector.kubernetes_client import PATCH_TIMEOUT_SECONDS, KubernetesClient


def _make_pod(app_label: str) -> SimpleNamespace:
    return SimpleNamespace(metadata=SimpleNamespace(labels={"app": app_label}))


def _make_deployment(app_label: str, cpu_raw: str, replicas: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(labels={"app": app_label}, name=app_label),
        spec=SimpleNamespace(
            replicas=replicas,
            template=SimpleNamespace(
                spec=SimpleNamespace(
                    containers=[
                        SimpleNamespace(resources=SimpleNamespace(requests={"cpu": cpu_raw}))
                    ]
                )
            )
        ),
    )


def _make_ready_deployment(generation: int = 1, replicas: int = 1, name: str = "loadgenerator") -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(generation=generation, name=name),
        status=SimpleNamespace(
            observed_generation=generation, updated_replicas=replicas, available_replicas=replicas
        ),
        spec=SimpleNamespace(replicas=replicas),
    )


def _make_not_ready_deployment(generation: int = 1, name: str = "loadgenerator") -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(generation=generation, name=name),
        status=SimpleNamespace(observed_generation=0, updated_replicas=0, available_replicas=0),
        spec=SimpleNamespace(replicas=1),
    )


def test_get_services_names_returns_HTTP_service_names(mocker):
    pods = SimpleNamespace(
        items=[
            _make_pod("frontend"),
            _make_pod("redis-cart"),
            _make_pod("cartservice"),
        ]
    )
    core_api = mocker.MagicMock()
    core_api.list_namespaced_pod.return_value = pods
    client = KubernetesClient(core_api=core_api, apps_api=mocker.MagicMock())
    result = client.get_services_names()
    assert result == {"frontend", "cartservice"}


def test_get_cpu_requests_when_deployments_available_excludes_loadgenerator(mocker):
    apps_api = mocker.MagicMock()
    apps_api.list_namespaced_deployment.return_value = SimpleNamespace(
        items=[
            _make_deployment("frontend", "500m"),
            _make_deployment("loadgenerator", "200m"),
            _make_deployment("cartservice", "1"),
        ]
    )
    client = KubernetesClient(core_api=mocker.MagicMock(), apps_api=apps_api)
    result = client.get_cpu_requests()
    assert len(result) == 2
    assert result["frontend"] == pytest.approx(0.5)
    assert result["cartservice"] == pytest.approx(1)


def test_stop_load_generation_patches_replicas_to_zero(mocker):
    apps_api = mocker.MagicMock()
    client = KubernetesClient(core_api=mocker.MagicMock(), apps_api=apps_api)
    client.stop_load_generation()
    _, kwargs = apps_api.patch_namespaced_deployment.call_args
    assert kwargs["body"]["spec"]["replicas"] == 0


def test_change_performance_test_load_sets_users_and_rate(mocker):
    apps_api = mocker.MagicMock()

    env_users = SimpleNamespace(name="USERS", value="0")
    env_rate = SimpleNamespace(name="RATE", value="0")
    source = SimpleNamespace(
        spec=SimpleNamespace(
            replicas=0,
            template=SimpleNamespace(
                spec=SimpleNamespace(containers=[SimpleNamespace(env=[env_users, env_rate])])
            ),
        )
    )
    ready = _make_ready_deployment()
    apps_api.read_namespaced_deployment.side_effect = [source, ready]
    apps_api.patch_namespaced_deployment.return_value = ready

    mocker.patch("kpp.collector.kubernetes_client.time.time", return_value=0)
    mocker.patch("kpp.collector.kubernetes_client.time.sleep")

    client = KubernetesClient(core_api=mocker.MagicMock(), apps_api=apps_api)
    client.change_performance_test_load("42")

    assert env_users.value == "42"
    assert env_rate.value == "42"


def test_change_performance_test_load_when_deployment_never_ready_raises_timeout(mocker):
    apps_api = mocker.MagicMock()

    env_users = SimpleNamespace(name="USERS", value="0")
    env_rate = SimpleNamespace(name="RATE", value="0")
    source = SimpleNamespace(
        spec=SimpleNamespace(
            replicas=0,
            template=SimpleNamespace(
                spec=SimpleNamespace(containers=[SimpleNamespace(env=[env_users, env_rate])])
            ),
        )
    )
    not_ready = _make_not_ready_deployment(generation=2)
    apps_api.read_namespaced_deployment.side_effect = [source, not_ready]
    apps_api.patch_namespaced_deployment.return_value = not_ready

    mocker.patch(
        "kpp.collector.kubernetes_client.time.time", side_effect=[0, 0, PATCH_TIMEOUT_SECONDS + 1]
    )
    mocker.patch("kpp.collector.kubernetes_client.time.sleep")

    client = KubernetesClient(core_api=mocker.MagicMock(), apps_api=apps_api)
    with pytest.raises(TimeoutError):
        client.change_performance_test_load("10")


def test_get_cpu_requests_multiplies_by_replica_count(mocker):
    apps_api = mocker.MagicMock()
    apps_api.list_namespaced_deployment.return_value = SimpleNamespace(
        items=[
            _make_deployment("frontend", "100m", replicas=2),
            _make_deployment("cartservice", "500m", replicas=3),
        ]
    )
    client = KubernetesClient(core_api=mocker.MagicMock(), apps_api=apps_api)
    result = client.get_cpu_requests()
    assert result["frontend"] == pytest.approx(0.2)   # 100m * 2 replicas
    assert result["cartservice"] == pytest.approx(1.5)  # 500m * 3 replicas


def test_scale_service_deployment_patches_replicas_and_waits(mocker):
    apps_api = mocker.MagicMock()
    ready = _make_ready_deployment(name="currencyservice", replicas=3)
    apps_api.patch_namespaced_deployment.return_value = ready
    apps_api.read_namespaced_deployment.return_value = ready

    mocker.patch("kpp.collector.kubernetes_client.time.time", return_value=0)
    mocker.patch("kpp.collector.kubernetes_client.time.sleep")

    client = KubernetesClient(core_api=mocker.MagicMock(), apps_api=apps_api)
    client.scale_service_deployment("currencyservice", 3)

    apps_api.patch_namespaced_deployment.assert_called_once_with(
        name="currencyservice", namespace="default", body={"spec": {"replicas": 3}}
    )
    apps_api.read_namespaced_deployment.assert_called_once_with(
        name="currencyservice", namespace="default"
    )


def test_wait_for_patch_completion_uses_deployment_name_from_metadata(mocker):
    apps_api = mocker.MagicMock()
    ready = _make_ready_deployment(name="myservice", replicas=2)
    apps_api.read_namespaced_deployment.return_value = ready

    mocker.patch("kpp.collector.kubernetes_client.time.time", return_value=0)
    mocker.patch("kpp.collector.kubernetes_client.time.sleep")

    client = KubernetesClient(core_api=mocker.MagicMock(), apps_api=apps_api)
    client._wait_for_patch_completion(ready, apps_api)

    apps_api.read_namespaced_deployment.assert_called_once_with(
        name="myservice", namespace="default"
    )
