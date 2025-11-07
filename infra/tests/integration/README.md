# MCP Integration Tests

Интеграционные тесты для проверки взаимодействия между MCP сервисами.

## Структура

```
tests/integration/
├── conftest.py              # Pytest фикстуры
├── docker-compose.integration.yml  # Docker конфигурация для тестов
├── Dockerfile               # Dockerfile для тест-раннера
├── README.md               # Эта документация
├── INTEGRATION_TESTS_PROGRESS.md  # Отчет о прогрессе
├── utils/
│   ├── mcp_client.py       # MCP клиент для тестов
│   ├── test_data.py        # Генераторы тестовых данных
│   ├── assertions.py       # Кастомные assertions
│   └── metrics.py          # Metrics коллектор
├── fixtures/               # Тестовые данные
│   ├── binance_klines.json
│   ├── tradingview_alerts.json
│   ├── memory_messages.json
│   └── backtest_strategies.json
├── results/                # Результаты тестов
└── test_*.py               # Тестовые файлы
    ├── test_backtesting_flow.py       # 5 тестов для backtesting
    ├── test_tradingview_memory.py     # 5 тестов для tradingview
    ├── test_halv1_integration.py      # 10 тестов для halv1
    ├── test_health_cascade.py         # 5 тестов для healthcheck
    └── test_final_mcp.py              # Общий тест доступности
```

## Запуск тестов

### Локально через Docker Compose

```bash
# Запустить все MCP сервисы
docker compose -f ../../docker-compose.mcp.yml up -d

# Дождаться запуска сервисов (около 30 секунд)
sleep 30

# Запустить интеграционные тесты
docker compose -f tests/docker-compose.integration.yml up --abort-on-container-exit

# Остановить сервисы
docker compose -f ../../docker-compose.mcp.yml down
docker compose -f tests/docker-compose.integration.yml down
```

### Локально через pytest

```bash
cd tests/integration

# Установить зависимости
pip install -r requirements.txt

# Запустить все тесты
pytest -v

# Запустить конкретный тест
pytest test_backtesting_flow.py -v

# Запустить с coverage
pytest --cov=. --cov-report=html

# Запустить конкретный тест-кейс
pytest test_backtesting_flow.py::test_backtesting_with_binance_historical_data -v
```

### Запуск отдельных тест-файлов

```bash
# Backtesting integration тесты (5 тестов)
python test_backtesting_flow.py

# TradingView integration тесты (5 тестов)
python test_tradingview_memory.py

# HAL integration тесты (10 тестов)
python test_halv1_integration.py

# Health & Fault tolerance тесты (5 тестов)
python test_health_cascade.py

# Общий тест доступности
python test_final_mcp.py
```

## CI/CD

Integration тесты автоматически запускаются через GitHub Actions:
- При каждом push в main/develop
- При создании Pull Request
- По расписанию (ежедневно в 2:00 UTC)

См. `.github/workflows/integration-tests.yml`

## Тестовые сценарии

### Backtesting Integration (test_backtesting_flow.py)
1. `test_backtesting_with_binance_historical_data` - Получение исторических данных от binance для бэктестинга
2. `test_backtesting_with_tradingview_indicators` - Использование индикаторов TradingView в бэктестинге
3. `test_backtesting_multiple_timeframes` - Бэктестинг на множественных таймфреймах
4. `test_backtesting_strategy_optimization` - Оптимизация стратегии
5. `test_backtesting_risk_management` - Управление рисками в бэктестинге

### TradingView Integration (test_tradingview_memory.py)
1. `test_tradingview_analysis_with_binance_prices` - Анализ с ценами от Binance
2. `test_tradingview_save_analysis_to_memory` - Сохранение анализа в память
3. `test_tradingview_alert_with_binance_execution` - Алерт с исполнением через Binance
4. `test_tradingview_historical_analysis_comparison` - Сравнение исторических анализов
5. `test_tradingview_multi_exchange_analysis` - Анализ по нескольким биржам

### HAL Integration (test_halv1_integration.py)
1. `test_halv1_full_trading_workflow` - Полный торговый workflow
2. `test_halv1_market_monitoring` - Мониторинг рынка
3. `test_halv1_automated_trading_decision` - Автоматизированные решения
4. `test_halv1_risk_management` - Управление рисками
5. `test_halv1_portfolio_management` - Управление портфелем
6. `test_halv1_strategy_backtesting` - Тестирование стратегии
7. `test_halv1_real_time_alerts` - Система реалтайм алертов
8. `test_halv1_multi_strategy_execution` - Множественные стратегии
9. `test_halv1_performance_tracking` - Отслеживание производительности
10. `test_halv1_emergency_shutdown` - Экстренное закрытие позиций

### Health & Fault Tolerance (test_health_cascade.py)
1. `test_all_services_health_check` - Проверка здоровья всех сервисов
2. `test_service_dependency_chain` - Цепочка зависимостей
3. `test_service_recovery_after_failure` - Восстановление после сбоя
4. `test_graceful_degradation` - Graceful degradation
5. `test_concurrent_service_failures` - Множественные сбои

## Написание новых тестов

### 1. Создать новый файл `test_*.py`

```python
"""
Integration tests для нового сценария
"""
import pytest
import asyncio
from unittest.mock import AsyncMock
import json

@pytest.mark.asyncio
async def test_new_scenario():
    """Описание теста"""
    print("\n=== Test: New scenario ===")
    
    # Mock clients
    mock_service = AsyncMock()
    mock_service.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"result": "success"})}]
    })
    
    # Test logic
    result = await mock_service.call_tool("tool_name", {"param": "value"})
    
    assert result is not None
    print("✅ Test passed")

if __name__ == "__main__":
    asyncio.run(test_new_scenario())
    print("\n🎉 All tests passed!")
```

### 2. Использовать fixtures из `conftest.py`

```python
@pytest.mark.asyncio
async def test_with_fixtures(binance_client, tradingview_client):
    # Использовать готовые клиенты
    result = await binance_client.call_tool("get_ticker_price", {
        "symbol": "BTCUSDT"
    })
    assert result is not None
```

### 3. Добавить тестовые данные в `fixtures/`

```json
// fixtures/new_test_data.json
{
  "symbol": "BTCUSDT",
  "price": 35000.00,
  "volume": 1000.0
}
```

## Metrics Collector

Для сбора метрик тестов используйте `utils/metrics.py`:

```python
from utils.metrics import MetricsCollector

metrics = MetricsCollector()
metrics.record_test_duration("test_name", 1.5)
metrics.record_test_result("test_name", "passed")
metrics.save_to_file("results/metrics.json")
```

## Текущий статус

См. `INTEGRATION_TESTS_PROGRESS.md` для подробного отчета о текущем статусе тестов.

**Краткий статус:**
- ✅ Инфраструктура готова
- ✅ 25 тестов созданы (5+5+10+5)
- ⚠️ MCP серверы требуют session ID (streamable-http)
- ⏳ Тесты используют mock объекты

## Troubleshooting

### Сервисы не запускаются
```bash
# Проверить docker-compose logs
docker compose -f ../../docker-compose.mcp.yml logs

# Убедиться что порты не заняты
netstat -an | grep -E "(8000|8050|8060|8070|8082)"

# Проверить health checks
docker compose -f ../../docker-compose.mcp.yml ps
```

### Тесты падают с timeout
```python
# Увеличить timeout в conftest.py
async def wait_for_service(client: MCPClient, timeout: float = 120):  # было 60
    ...
```

### Connection refused
```bash
# Проверить что сервисы запущены
docker compose -f ../../docker-compose.mcp.yml ps

# Проверить network connectivity
docker network ls
docker network inspect mcp_default
```

### Import errors
```bash
# Убедиться что путь добавлен в PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Или использовать sys.path в тестах
import sys
sys.path.insert(0, os.path.dirname(__file__))
```

### MCP session ID errors
См. `INTEGRATION_TESTS_PROGRESS.md` раздел "Проблема с MCP Session ID" для подробного объяснения и решений.

## Дополнительные ресурсы

- [MCP Protocol Documentation](https://modelcontextprotocol.io/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Pytest Documentation](https://docs.pytest.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
