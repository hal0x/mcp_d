"""Тесты для эндпоинта метрик."""

import pytest
from fastapi.testclient import TestClient
from web.dashboard.server import app


def test_metrics_endpoint_ok():
    """Тест что эндпоинт /metrics отвечает и содержит основные серии."""
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "llm_latency_seconds" in body
    assert "errors_total" in body


def test_health_endpoint_ok():
    """Тест что эндпоинт /health отвечает."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True


@pytest.mark.parametrize("route", ["/", "/dashboard"])
def test_dashboard_page_ok(route):
    """Тест что HTML-дашборд отвечает и содержит основные элементы."""
    client = TestClient(app)
    response = client.get(route)
    assert response.status_code == 200
    body = response.text
    assert "HALv1 Monitoring Dashboard" in body
    assert "llm_latency_seconds" in body
    assert "Prometheus metrics" in body
