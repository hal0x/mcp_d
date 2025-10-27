# MCP Environment Variable Template

Единый шаблон и соглашение об именовании переменных окружения для всей MCP-инфраструктуры. Используйте этот документ как справочник при настройке `.env` файлов, `infra/docker-compose.mcp.yml` и сервисных конфигураций (`config.py`, Pydantic settings).

## 1. Базовые инфраструктурные переменные (без префикса)

| Переменная        | Значение по умолчанию | Назначение                         |
|-------------------|-----------------------|------------------------------------|
| `REDIS_HOST`      | `redis`               | Хост Redis в docker-сети           |
| `REDIS_PORT`      | `6379`                | Порт Redis                         |
| `REDIS_DB`        | `0`                   | Номер базы Redis                   |
| `POSTGRES_HOST`   | `postgres`            | Хост PostgreSQL                    |
| `POSTGRES_PORT`   | `5432`                | Порт PostgreSQL                    |
| `POSTGRES_DB`     | `tradingview_scanners`| Имя БД по умолчанию (`POSTGRES_DATABASE` допускается как алиас во время миграции) |
| `POSTGRES_USER`   | `tradingview`         | Пользователь PostgreSQL            |
| `POSTGRES_PASSWORD` | `tradingview`       | Пароль PostgreSQL                  |

> Эти переменные доступны всем сервисам. При необходимости каждый сервис может переопределить их собственными значениями через префиксы.

## 2. Общие правила для MCP сервисов

Для каждого сервиса используйте префикс вида `{SERVICE}` (в верхнем регистре, дефис заменяется на `_`). Ниже ─ обязательные ключи:

| Шаблон переменной             | Пример для `tradingview-mcp` | Назначение                                                          |
|------------------------------|------------------------------|----------------------------------------------------------------------|
| `{SERVICE}_HOST`             | `TRADINGVIEW_MCP_HOST`       | Биндинг сервера (`0.0.0.0` по умолчанию)                            |
| `{SERVICE}_PORT`             | `TRADINGVIEW_MCP_PORT`       | Внутренний порт сервиса                                             |
| `{SERVICE}_LOG_LEVEL`        | `TRADINGVIEW_MCP_LOG_LEVEL`  | Глобальный уровень логирования                                      |
| `{SERVICE}_DEFAULT_TRANSPORT`| `TRADINGVIEW_MCP_DEFAULT_TRANSPORT` | Транспорт (`stdio` или `streamable-http`)                     |
| `{SERVICE}_URL`              | `TRADINGVIEW_MCP_URL`        | HTTP URL для межсервисной коммуникации                              |

Дополнительные опции описываются тем же префиксом: `TRADINGVIEW_MCP_REDIS_HOST`, `BACKTESTING_MCP_OPTUNA_DB_HOST` и т.д. Для вложенных структур допускается двойное подчёркивание: `BACKTESTING_MCP__OPTUNA__DB_HOST`.

### Принятые префиксы для текущих сервисов

| Сервис             | Префикс            |
|--------------------|--------------------|
| `binance-mcp`      | `BINANCE_MCP_`     |
| `tradingview-mcp`  | `TRADINGVIEW_MCP_` |
| `memory-mcp`       | `MEMORY_MCP_`*     |
| `shell-mcp`        | `SHELL_MCP_`       |
| `backtesting-mcp`  | `BACKTESTING_MCP_` |

\* Исторически проект использует префикс `TG_DUMP_`. План миграции: обновить Pydantic-настройки и документацию на `MEMORY_MCP_`, сохраняя обратную совместимость через алиасы.

## 3. Производные URL-адреса (универсальные)

| Переменная              | Значение по умолчанию                    |
|-------------------------|------------------------------------------|
| `BINANCE_MCP_URL`       | `http://binance-mcp:8000`                |
| `TRADINGVIEW_MCP_URL`   | `http://tradingview-mcp:8060`            |
| `MEMORY_MCP_URL`        | `http://memory-mcp:8050`                 |
| `SHELL_MCP_URL`         | `http://shell-mcp:8070`                  |
| `BACKTESTING_MCP_URL`   | `http://backtesting-mcp:8082`            |

Эти переменные используются как консистентная точка входа для других сервисов (включая `halv1`).

## 4. Пример `.env` файла

```env
# Инфраструктура
REDIS_HOST=redis
REDIS_PORT=6379
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Binance MCP
BINANCE_MCP_HOST=0.0.0.0
BINANCE_MCP_PORT=8000
BINANCE_MCP_LOG_LEVEL=INFO
BINANCE_MCP_DEFAULT_TRANSPORT=stdio

# TradingView MCP
TRADINGVIEW_MCP_HOST=0.0.0.0
TRADINGVIEW_MCP_PORT=8060
TRADINGVIEW_MCP_LOG_LEVEL=INFO
TRADINGVIEW_MCP_DEFAULT_TRANSPORT=streamable-http

# Backtesting MCP
BACKTESTING_MCP_HOST=0.0.0.0
BACKTESTING_MCP_PORT=8082
BACKTESTING_MCP_LOG_LEVEL=INFO
BACKTESTING_MCP_DEFAULT_TRANSPORT=stdio

# Производные ссылки
BINANCE_MCP_URL=http://binance-mcp:8000
TRADINGVIEW_MCP_URL=http://tradingview-mcp:8060
```

## 5. Чеклист миграции сервиса на стандарт

1. Переименуйте текущие переменные в соответствии с префиксом `{SERVICE}_`.
2. Проверьте, что `infra/docker-compose.mcp.yml` и код сервиса читают одинаковые имена.
3. Обновите `AGENTS.md` и `README.md`, указав новые переменные.
4. Добавьте значения по умолчанию в `.env` или профильный `.env.example`.
5. Убедитесь, что в проверках (`tests/config` и т.п.) используются новые ключи.

## 6. Добавление нового MCP сервиса

1. Выберите уникальный порт и добавьте строку в секцию «Производные URL-адреса».
2. Создайте префикс `{SERVICE}_` и используйте его во всех конфигурационных файлах.
3. Обновите `config/mcp-services.yaml` (см. приоритет 2.2 плана).
4. Дополните общий чеклист в `templates/new-mcp-service-checklist.md` (когда появится).

Следование этому шаблону обеспечивает единообразие конфигурации, упрощает автоматическую проверку (`scripts/test_consistency.py`) и ускоряет онбординг новых сервисов.
