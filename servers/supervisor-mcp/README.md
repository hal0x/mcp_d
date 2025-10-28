# Supervisor MCP Server

Центральный надсмотрщик экосистемы MCP - реестр, health checks, метрики, факты и алерты.

## Features

- **Discovery & Registry**: список MCP, их версии, доступные инструменты
- **Health & Status**: периодические health-пробы, SLA/latency/error-rate
- **Metrics & Facts Store**: сбор и хранение техметрик и бизнес-фактов
- **Aggregation & Query**: агрегаты за 7/30 дней, KPI для принятия решений
- **Alerts & Actions**: правила алертов и рекомендации действий

## Quick Start

```bash
# Подготовка окружения
cp .env.example .env  # при необходимости обновите креды БД/Redis

# Stdio mode
python -m supervisor_mcp.server --stdio

# HTTP mode
python -m supervisor_mcp.server --host 0.0.0.0 --port 8000
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `SUPERVISOR_DB_URL` | PostgreSQL connection URL | `postgresql+asyncpg://supervisor:supervisor@localhost:5432/supervisor` |
| `SUPERVISOR_DB_POOL_SIZE` | (Optional) sqlalchemy pool size | `None` |
| `SUPERVISOR_DB_MAX_OVERFLOW` | (Optional) pool overflow | `None` |
| `SUPERVISOR_DB_ECHO` | SQLAlchemy echo | `false` |
| `SUPERVISOR_REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `SUPERVISOR_REDIS_CACHE_TTL_SECONDS` | TTL для агрегатов в секундах | `60` |
| `SUPERVISOR_AGGREGATION_REFRESH_SECONDS` | Период пересчёта агрегатов | `300` |
| `SUPERVISOR_METRICS_RETENTION_DAYS` | Срок хранения метрик/фактов | `30` |
| `SUPERVISOR_LOG_LEVEL` | Logging level | `INFO` |

## API

### Discovery
- `GET /mcp/registry` → список MCP, версии, эндпоинты, протоколы
- `GET /mcp/{name}/capabilities` → инструменты, схемы, ограничения, квоты

### Health
- `GET /health` → сводный статус
- `GET /health/{name}` → метрики конкретного MCP

### Ingest
- `POST /ingest/metric` → `{name, ts, value, tags}` (батч)
- `POST /ingest/event` → `Fact:*` события

### Query
- `GET /query/metrics?name=…&from=…&to=…`
- `GET /query/agg?kind=business&window=7d`
- `GET /query/facts?kind=Fact:Trade&from=…`

### Alerts
- `POST /alerts/rules` (CRUD)
- `GET /alerts/active`

## Development

```bash
# Tests
pytest tests/

# Linting
ruff check .
black .
mypy .
```

## License

MIT License
