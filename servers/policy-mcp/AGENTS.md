# Policy MCP Server — Архитектура и Руководство

## Обзор

Policy MCP обеспечивает централизованное управление профилями решений и политиками для всей MCP-экосистемы. Сервер предоставляет инструменты FastMCP, которые позволяют создавать, обновлять и активировать профили, а также оценивать входящие запросы в соответствии с правилами. Дополнительно доступен REST API для интеграции с learning-mcp и другими сервисами.

## Архитектура

```
┌────────────────────────┐
│ FastMCP (policy-mcp)   │
│  ├─ MCP Tools          │
│  └─ REST API (/profiles)│
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐        ┌────────────────────────┐
│ ProfileService         │◄──────►│ PostgreSQL (asyncpg)   │
│  ├─ CRUD + версии      │        │  └─ decision_profiles   │
│  └─ Decision evaluate  │        │  └─ profile_versions    │
└────────────┬───────────┘        └────────────────────────┘
             │
             ▼
┌────────────────────────┐
│ Pydantic модели        │
│  └─ DecisionProfile    │
│  └─ DecisionRequest    │
│  └─ DecisionResponse   │
└────────────────────────┘
```

### Основные компоненты

- **FastMCP сервер** — объявляет и обслуживает инструменты (`create_profile`, `list_profiles`, `evaluate_decision` и др.).
- **ProfileService** — инкапсулирует бизнес-логику работы с профилями и хранением версий.
- **PostgreSQL** — основное хранилище для профилей и их версий. Доступ осуществляется через SQLAlchemy async engine и драйвер `asyncpg`.
- **Pydantic модели** — представляют доменные сущности (DecisionProfile, DecisionRequest, DecisionResponse) и скрывают детали хранения.

## Поток данных

1. Клиент FastMCP вызывает инструмент.
2. ProfileService открывает сессию PostgreSQL через асинхронный session factory.
3. Выполняется операция (создание/обновление/чтение), при необходимости фиксируется новая версия профиля.
4. Результат сериализуется в Pydantic модель и возвращается клиенту.

## Хранилище

- `decision_profiles` — актуальные состояния профилей; содержит JSONB-поля для правил (`rules`) и произвольных метаданных (`metadata_json`).
- `profile_versions` — журнал версий профилей, полезный для аудита и будущих механизмов отката.

Инициализация схемы происходит автоматически при старте сервера через `policy_mcp.db.init_db()`. Для продакшена рекомендуется использовать Alembic-миграции на основе тех же моделей.

## Инструменты

- `create_profile`, `update_profile`, `delete_profile` — управление жизненным циклом профиля.
- `list_profiles`, `get_profile`, `get_active_profile` — чтение состояний.
- `list_profile_versions`, `rollback_profile` — чтение и восстановление исторических версий.
- `configure_profile_experiment`, `list_profile_experiments` — настройка A/B экспериментов и веса профилей.
- `activate_profile` — отметка нужного профиля активным.
- `evaluate_decision` — проверка входящего контекста по активному профилю.

## Конфигурация

Основные параметры задаются через переменные окружения (`POLICY_DB_URL`, `POLICY_DB_POOL_SIZE`, `POLICY_DB_MAX_OVERFLOW`, `POLICY_DB_ECHO`, `POLICY_LOG_LEVEL`). Пример — в `.env.example`.

## Планы развития

- Расширить систему feature flags и audit trail.
- Интеграция с learning-mcp для автоматического деплоя и rollout-флагов.
