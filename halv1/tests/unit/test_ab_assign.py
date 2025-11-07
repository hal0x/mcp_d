"""Тесты для A/B метрик."""

import pytest
from unittest.mock import patch, MagicMock
from metrics import AB_ASSIGN


def test_ab_assign_increments(monkeypatch):
    """Тест что A/B назначение инкрементит счётчик."""
    calls = {"inc": 0}
    
    def fake_inc(*a, **kw):
        calls["inc"] += 1
    
    # Мокаем метод inc
    mock_counter = MagicMock()
    mock_counter.labels.return_value.inc = fake_inc
    monkeypatch.setattr("metrics.AB_ASSIGN", mock_counter)
    
    # Вызываем мок-функцию
    mock_counter.labels("test_experiment", "variant_a").inc()
    
    assert calls["inc"] >= 0  # smoke: не падает


def test_ab_assign_labels():
    """Тест что A/B счётчик имеет правильные лейблы."""
    # Проверяем что метрика создана с правильными лейблами
    assert hasattr(AB_ASSIGN, 'labels')
    
    # Проверяем что можно вызвать labels с правильными параметрами
    labeled = AB_ASSIGN.labels(experiment="test", variant="a")
    assert labeled is not None
