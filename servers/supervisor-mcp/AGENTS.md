# Supervisor MCP — Архитектура и интеграции

## Назначение

Supervisor MCP централизует информацию об экосистеме: регистрирует MCP-сервера, собирает технические/бизнес метрики, хранит факты, агрегирует KPI и управляет алертами. Сервис предоставляет MCP-инструменты и REST API, на которые опираются learning/policy/orchestrator MCP.

## Общая схема

```
┌─────────────────────────┐       ┌──────────────────────────┐
│ FastMCP (supervisor)    │       │ FastAPI HTTP API         │
│  ├─ MCP Tools           │       │  ├─ /query/agg           │
│  └─ /stdio|/http        │       │  └─ /query/facts         │
└─────────┬───────────────┘       └─────────┬────────────────┘
          │                                   │
          ▼                                   ▼
┌─────────────────────────┐       ┌──────────────────────────┐
│ MetricsService (async)  │◄─────►│ PostgreSQL (metrics,     │
│  ├─ ingest/query        │        │ facts, aggregates)       │
│  └─ Redis cache         │        └──────────────────────────┘
│ Registry/Health/Alerts  │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│ Downstream MCP clients  │
│ (learning, policy, HAL) │
└─────────────────────────┘
```

## Основные компоненты

- **MetricsService** — пишет метрики/факты в PostgreSQL, кеширует агрегаты в Redis, отдает REST/MCP-запросы.
- **RegistryService** — хранение метаданных MCP (поддерживает Keep-Alive обновление).
- **HealthService** — хранение статусов и SLA.
- **AlertsService** — правила и активные алерты.

## Хранилище

### PostgreSQL
- `metrics` — временные ряды тех.метрик.
- `facts` — бизнес-факты (`Fact:*`).
- `aggregates` — снапшоты агрегатов (7d/30d).
- `mcp_registry`, `health_status`, `alert_rules`, `active_alerts`.

### Redis
- Кеш агрегатов (`supervisor:agg:{kind}:{window}`) с TTL (по умолчанию 60 секунд).

## Интеграции

- **Learning MCP** — считывает `/query/agg` и `/query/facts`.
- **Policy MCP** — получает факты для управления профилями (roadmap).
- **HALv1** — отправляет факты/метрики, использует агрегаты для принятия решений.

## Потоки данных

1. Клиенты вызывают MCP tools (`ingest_metric`, `ingest_fact`) — данные пишутся в PostgreSQL и инвалидируют кеш.
2. Запросы на `/query/agg` → MetricsService проверяет Redis → при отсутствии пересчитывает и обновляет кеш + таблицу `aggregates`.
3. Learning MCP периодически забирает агрегаты 7d/30d через REST и обучает профили.

## TODO / Roadmap

- Планировщик фоновых агрегатов (cron).
- Retention-очистка таблиц.
- Расширенная авторизация (API keys/JWT).

