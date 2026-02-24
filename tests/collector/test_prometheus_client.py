import math

import pytest

from kpp.collector.prometheus_client import PrometheusClient


def test_get_average_response_time_returns_float(mocker):
    mock_prom = mocker.MagicMock()
    mock_prom.custom_query.return_value = [{"value": ["1234567890", "0.042"]}]
    client = PrometheusClient(mock_prom)
    result = client.get_average_response_time("frontend")
    assert result == pytest.approx(0.042)
    mock_prom.custom_query.assert_called_once()


def test_get_throughput_returns_float(mocker):
    mock_prom = mocker.MagicMock()
    mock_prom.custom_query.return_value = [{"value": ["1234567890", "12.5"]}]
    client = PrometheusClient(mock_prom)
    result = client.get_throughput("frontend")
    assert result == pytest.approx(12.5)
    mock_prom.custom_query.assert_called_once()


def test_get_cpu_usage_returns_float(mocker):
    mock_prom = mocker.MagicMock()
    mock_prom.custom_query.return_value = [{"value": ["1234567890", "0.35"]}]
    client = PrometheusClient(mock_prom)
    result = client.get_cpu_usage("frontend")
    assert result == pytest.approx(0.35)
    mock_prom.custom_query.assert_called_once()


def test_get_throughput_when_response_is_empty_returns_nan(mocker):
    mock_prom = mocker.MagicMock()
    mock_prom.custom_query.return_value = []
    client = PrometheusClient(mock_prom)
    result = client.get_throughput("frontend")
    assert math.isnan(result)
    mock_prom.custom_query.assert_called_once()


def test_get_throughput_when_response_is_none_returns_nan(mocker):
    mock_prom = mocker.MagicMock()
    mock_prom.custom_query.return_value = None
    client = PrometheusClient(mock_prom)
    result = client.get_throughput("frontend")
    assert math.isnan(result)
    mock_prom.custom_query.assert_called_once()
