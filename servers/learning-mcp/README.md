# Learning MCP Server

Система offline обучения и генерации профилей решений на основе данных supervisor-mcp.

## Возможности

- **Offline обучение** - анализ агрегатов из supervisor-mcp
- **Генерация DecisionProfile** - создание оптимизированных профилей решений
- **Синхронизация с Policy MCP** - автоматическое размещение и активация профилей
- **Health-check Policy MCP** - быстрый контроль доступности и количества активных профилей
- **Паттерн-анализ** - выявление успешных и неуспешных паттернов
- **Адаптивные рекомендации** - предложения по улучшению стратегий
- **A/B тестирование** - сравнение различных профилей

## Архитектура

```
Learning MCP
    ↓ читает агрегаты
Supervisor MCP (7d/30d data)
    ↓ генерирует DecisionProfile
Policy MCP (хранит профили)
    ↓ используется
HALv1 (применяет профили)
```

## Quick Start

```bash
# Stdio mode
python -m learning_mcp.server --stdio

# HTTP mode
python -m learning_mcp.server --host 0.0.0.0 --port 8003
```

## API

### Training
- `POST /train/offline` → обучение на исторических данных
- `POST /train/online` → инкрементальное обучение

### Profiles
- `POST /propose` → предложить новый DecisionProfile
- `GET /profiles/compare` → сравнить профили
- `POST /profiles/validate` → валидировать профиль

### Analysis
- `GET /patterns/successful` → успешные паттерны
- `GET /patterns/failed` → проблемные паттерны
- `POST /analyze/correlation` → корреляционный анализ

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LEARNING_SUPERVISOR_URL` | Supervisor MCP URL | `http://localhost:8001` |
| `LEARNING_POLICY_URL` | Policy MCP REST API URL | `http://localhost:8002` |
| `LEARNING_POLICY_TIMEOUT` | Timeout для запросов к Policy MCP | `10.0` |
| `LEARNING_ONLINE_LEARNING_WINDOW` | Окно для инкрементального обучения | `1d` |
| `LEARNING_ONLINE_LEARNING_INTERVAL_SECONDS` | Интервал цикла online learning | `900` |
| `LEARNING_AB_TEST_THRESHOLD` | Минимальная дельта успеха для активации | `0.01` |
| `LEARNING_MIN_SAMPLES` | Минимум образцов для обучения | `100` |
| `LEARNING_CONFIDENCE_THRESHOLD` | Порог уверенности | `0.7` |
| `LEARNING_POLICY_TIMEOUT` | Таймаут Policy MCP запросов | `10.0` |

## Development

```bash
# Tests
pytest tests/

# Benchmark training pipelines
python -m learning_mcp.scripts.benchmark --windows 7d 30d --metrics success_rate

# Linting
ruff check .
black .
mypy .
```

## License

MIT License
