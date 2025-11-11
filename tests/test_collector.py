from kube_performance_predictor.collector import get_average_response_time


def test_get_response_time(mocker):
    "Test that querying the response time from Promethus successfully return the response time"
    mock_prometheus_response = [{"metric": {}, "value": [1678886400, "0.75"]}]
    mock_prom_connect_class = mocker.patch("kube_performance_predictor.collector.PrometheusConnect")
    mock_prom_instance = mock_prom_connect_class.return_value
    mock_prom_instance.custom_query.return_value = mock_prometheus_response

    result = get_average_response_time(prometheus_url="http://fake-url")

    assert result == 0.75
