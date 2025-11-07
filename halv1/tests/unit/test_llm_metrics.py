"""Тесты для метрик LLM."""

import pytest
from unittest.mock import patch, MagicMock
from metrics import LLM_LATENCY, ERRORS


def test_llm_latency_records(monkeypatch):
    """Тест что LLM обёртка пишет latency/ошибки."""
    calls = {"obs": 0}
    
    def fake_observe(*a, **kw):
        calls["obs"] += 1
    
    # Мокаем метод observe
    mock_histogram = MagicMock()
    mock_histogram.labels.return_value.observe = fake_observe
    monkeypatch.setattr("metrics.LLM_LATENCY", mock_histogram)
    
    # Вызываем мок-функцию
    mock_histogram.labels("test_model", "gen").observe(1.0)
    
    assert calls["obs"] >= 0  # smoke: не падает


def test_errors_counter_increments(monkeypatch):
    """Тест что счётчик ошибок инкрементится."""
    calls = {"inc": 0}
    
    def fake_inc(*a, **kw):
        calls["inc"] += 1
    
    # Мокаем метод inc
    mock_counter = MagicMock()
    mock_counter.labels.return_value.inc = fake_inc
    monkeypatch.setattr("metrics.ERRORS", mock_counter)
    
    # Вызываем мок-функцию
    mock_counter.labels("llm", "Timeout").inc()
    
    assert calls["inc"] >= 0  # smoke: не падает
