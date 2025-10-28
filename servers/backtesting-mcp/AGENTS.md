# Backtesting MCP Server – Архитектура и Руководство

## Обзор проекта

**Backtesting MCP** — это Model Context Protocol сервер, предоставляющий агентам инструменты для бэктестинга торговых стратегий, оптимизации параметров и сравнения готовых результатов. Сервер использует реальные рыночные данные из подключённых MCP-сервисов биржевых данных (Binance, TradingView).

## Архитектура

### Структура проекта

```
backtesting-mcp/
├── README.md
├── AGENTS.md
├── pyproject.toml
└── src/
    └── backtesting_mcp/
        ├── __init__.py           # Точка входа для скрипта backtesting-mcp
        ├── config.py             # Настройки и переменные окружения
        ├── datasource.py         # Загрузка свечей из внешних MCP серверов
        ├── server.py             # MCP сервер и объявление инструментов
        ├── services/             # Бизнес-логика без зависимостей от MCP
        │   ├── __init__.py
        │   ├── backtesting.py    # Основная логика бэктестинга и метрики
        │   └── health.py         # Сервис проверки здоровья системы
        └── tools/                # MCP инструменты и модели
            ├── __init__.py
            ├── backtesting_tools.py  # Регистрация MCP инструментов
            └── models.py             # Pydantic модели для валидации
```

### Ключевые компоненты

#### 1. MCP Server (`server.py`)
- Настраивает `FastMCP`, регистрирует инструменты
- Преобразует ISO8601 строки в `datetime`, нормализует метрики
- Поддерживает stdio и HTTP режимы

#### 2. Backtesting Service (`services/backtesting.py`)
- Загружает реальные OHLC данные из внешних источников
- Реализует Moving Average Crossover стратегию
- Собирает метрики: winrate, profit factor, Sharpe, drawdown
- Оптимизирует параметры через Optuna (TPE по умолчанию, поддержка Random/CMA-ES/Grid)

#### 3. Health Service (`services/health.py`)
- Проверяет состояние сервера и внешних источников данных
- Возвращает статусы: "healthy", "degraded", "unhealthy"

#### 4. Market Data Source (`datasource.py`)
- Загружает свечи из Binance/TradingView MCP серверов
- Возвращает ошибку при недоступности источников данных
- Приоритет источников: `BACKTEST_PREFERRED_SOURCE`

#### 5. MCP Tools (`tools/`)
- `backtesting_tools.py` — регистрация инструментов с `@mcp.tool()`
- `models.py` — Pydantic модели для валидации параметров

#### 6. Конфигурация (`config.py`)
- Загрузка настроек через `pydantic-settings` с префиксом `BACKTEST_`
- Динамическое чтение переменных окружения без кэширования

#### 7. Пакет (`__init__.py`)
- Экспортирует `main` для CLI-скрипта `backtesting-mcp`

## Поток данных

1. Агент → MCP инструмент
2. `FastMCP` → `tools/backtesting_tools.py`
3. `tools/` → `services/`
4. `services/` → `MarketDataSource` (или синтетические данные)
5. Результат → агент

## Инструменты (Tools)

### Стандарт описания инструментов
- **Первое предложение**: короткое действие на английском (≤90 символов), начинается с глагола.
- **Источник**: используем `description=` декоратора `@mcp.tool`; docstring расширяет детали и примеры.
- **Единообразие**: одинаковый стиль для всех инструментов, чтобы `list_tools` выглядел предсказуемо.
- **Практика**: дополнительные подсказки и форматы аргументов документируем после первой строки.

### `run_backtest`

**Описание**: Запускает бэктест торговой стратегии на исторических данных с использованием пересечения скользящих средних.

**Параметры**:
- `strategy` (string, обязательный): Название торговой стратегии (например, "ma_crossover")
- `symbol` (string, обязательный): Торговый символ (например, "BTCUSDT")
- `timeframe` (string, по умолчанию "1h"): Таймфрейм свечей (1m, 5m, 15m, 1h, 4h, 1d)
- `start` (string, по умолчанию "2025-01-01T00:00:00"): Дата начала в ISO формате
- `end` (string, по умолчанию "2025-12-31T23:59:59"): Дата окончания в ISO формате
- `parameters` (object, по умолчанию {}): Параметры стратегии (например, {"fast": 10, "slow": 20})

**Возвращает**:
```json
{
  "strategy": "ma_crossover",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "period": ["2025-01-01T00:00:00", "2025-12-31T23:59:59"],
  "parameters": {"fast": 10, "slow": 20},
  "metrics": {
    "total_return": 0.15,
    "annualized_return": 0.32,
    "sharpe": 1.45,
    "max_drawdown": -0.08,
    "trades": 45,
    "winrate": 0.62,
    "profit_factor": 1.8
  },
  "equity_curve": [1000, 1015, 1023, ...],
  "trade_log": [
    {
      "entry_time": "2025-03-15T10:00:00",
      "exit_time": "2025-03-16T14:00:00",
      "side": "long",
      "entry_price": 21000,
      "exit_price": 21500,
      "pnl": 2.38
    }
  ]
}
```

**Особенности**: Использует только реальные рыночные данные. Требует доступности внешних источников данных.

### `optimize_parameters`

**Описание**: Оптимизирует параметры торговой стратегии с помощью Optuna (TPE по умолчанию) и сохраняет историю в PostgreSQL storage.

**Параметры**:
- `strategy` (string, обязательный): Название торговой стратегии
- `symbol` (string, обязательный): Торговый символ
- `timeframe` (string, по умолчанию "1h"): Таймфрейм свечей
- `start` (string, по умолчанию "2025-01-01T00:00:00"): Дата начала в ISO формате
- `end` (string, по умолчанию "2025-12-31T23:59:59"): Дата окончания в ISO формате
- `parameter_space` (object, по умолчанию {}): Пространство параметров для оптимизации (например, {"fast": [5, 10, 15], "slow": [20, 30, 40]})
- `objective` (string, по умолчанию "return"): Целевая функция оптимизации ("return" или "sharpe", без учёта регистра)
- `trials` (integer, по умолчанию 20, диапазон 1-1000): Количество Optuna-trial'ов

**Возвращает**:
```json
{
  "strategy": "ma_crossover",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "objective": "return",
  "best_params": {"fast": 12, "slow": 26},
  "best_score": 0.18,
  "trials_completed": 20,
  "evaluations": [
    {
      "trial_number": 0,
      "fast": 5,
      "slow": 20,
      "score": 0.12,
      "total_return": 0.12,
      "sharpe": 1.15,
      "max_drawdown": -0.08,
      "trades": 38,
      "state": "COMPLETE",
      "error": null
    }
  ],
  "optuna": {
    "study_name": "ma_crossover_BTCUSDT_1h_2025-01-01_2025-12-31",
    "sampler": "TPE",
    "pruner": "median",
    "storage": "postgresql://tradingview:***@postgres:5432/tradingview_scanners",
    "n_trials": 20,
    "best_trial_number": 17,
    "datetime_start": "2025-01-01T00:00:00",
    "datetime_complete": "2025-01-01T00:05:00"
  }
}
```

**Особенности**: Использует Optuna (поддержка TPE, Random, CMA-ES и Grid samplers), хранит исследования в PostgreSQL через `BACKTEST_OPTUNA_*` настройки, возвращает расширенную историю и метаданные исследования.

### `compare_strategies`

**Описание**: Сравнивает результаты нескольких бэктестов и создаёт рейтинг стратегий по выбранным метрикам.

**Параметры**:
- `results` (array, обязательный): Список результатов бэктестов для сравнения

**Возвращает**:
```json
{
  "leaderboard": [
    {
      "rank": 1,
      "strategy": "ma_crossover",
      "symbol": "BTCUSDT",
      "timeframe": "1h",
      "total_return": 0.18,
      "sharpe": 1.45,
      "max_drawdown": -0.08,
      "trades": 45
    },
    {
      "rank": 2,
      "strategy": "rsi_strategy",
      "symbol": "ETHUSDT",
      "timeframe": "1h",
      "total_return": 0.15,
      "sharpe": 1.32,
      "max_drawdown": -0.12,
      "trades": 38
    }
  ],
  "winner": {
    "rank": 1,
    "strategy": "ma_crossover",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "total_return": 0.18,
    "sharpe": 1.45,
    "max_drawdown": -0.08,
    "trades": 45
  },
  "comparison_metrics": ["total_return", "sharpe", "max_drawdown"],
  "total_strategies": 2
}
```

**Особенности**: Обрабатывает результаты без некоторых метрик (значения по умолчанию). Поддержка внешних систем.

### `health`

**Описание**: Проверяет состояние сервера и доступность внешних источников данных.

**Параметры**:
- `check_data_sources` (boolean, по умолчанию true): Проверять ли доступность источников данных

**Возвращает**:
```json
{
  "status": "healthy",
  "timestamp": 1703123456.789,
  "uptime": 3600.5,
  "data_sources": {
    "binance": {
      "available": true,
      "response_time": 0.15,
      "url": "http://binance-mcp:8000"
    },
    "tradingview": {
      "available": false,
      "error": "Connection timeout",
      "url": "http://tradingview-mcp:8060"
    }
  },
  "server_info": {
    "version": "0.2.0",
    "preferred_source": "binance",
    "max_candles": 10000
  }
}
```

**Особенности**: Статусы "healthy", "degraded", "unhealthy". Включает время ответа и детали ошибок.

### `version`

**Описание**: Возвращает информацию о версии сервера, конфигурации и доступных возможностях.

**Параметры**: Отсутствуют

**Возвращает**:
```json
{
  "version": "0.2.0",
  "package": "backtesting-mcp",
  "description": "Model Context Protocol server for running strategy backtests and optimizations",
  "config": {
    "default_timeframe": "1h",
    "preferred_source": "binance",
    "optuna_sampler": "TPE",
    "optuna_pruner": "median",
    "optuna_storage": "postgresql://tradingview:***@postgres:5432/tradingview_scanners"
  },
  "features": [
    "real_market_data_integration",
    "optuna_parameter_optimization",
    "persistent_optuna_storage",
    "strategy_comparison",
    "multiple_data_sources"
  ]
}
```

**Особенности**: Полная информация о конфигурации и возможностях сервера.

## Конфигурация и переменные окружения

| Переменная | Назначение | Значение по умолчанию |
|------------|------------|------------------------|
| `BACKTEST_DEFAULT_TIMEFRAME` | Таймфрейм по умолчанию | `1h` |
| `BACKTEST_PREFERRED_SOURCE` | Источник свечей (`binance`, `tradingview`, `both`) | `binance` |
| `BACKTEST_BINANCE_URL` | Базовый URL `binance-mcp` | `http://localhost:8000` |
| `BACKTEST_TRADINGVIEW_URL` | Базовый URL `tradingview-mcp` | `http://localhost:8060` |
| `BACKTEST_REQUEST_TIMEOUT` | Тайм-аут HTTP запросов в секундах | `15.0` |
| `BACKTEST_DATA_ROOT` | Локальный каталог для кеша (зарезервировано) | `data/` |
| `BACKTEST_OPTUNA_DB_HOST` | Хост PostgreSQL для Optuna storage | `postgres` |
| `BACKTEST_OPTUNA_DB_PORT` | Порт PostgreSQL для Optuna storage | `5432` |
| `BACKTEST_OPTUNA_DB_NAME` | Имя базы данных для Optuna исследований | `tradingview_scanners` |
| `BACKTEST_OPTUNA_DB_USER` | Пользователь PostgreSQL для Optuna | `tradingview` |
| `BACKTEST_OPTUNA_DB_PASSWORD` | Пароль PostgreSQL для Optuna | `tradingview` |
| `BACKTEST_OPTUNA_SAMPLER` | Алгоритм семплинга Optuna (TPE|Random|Grid|CmaEs) | `TPE` |
| `BACKTEST_OPTUNA_PRUNER` | Алгоритм прунинга Optuna (median|percentile|none) | `median` |
| `BACKTEST_OPTUNA_N_JOBS` | Количество параллельных Optuna job'ов | `1` |

Все настройки читаются из ENV с префиксом `BACKTEST_`. При изменении переменных окружения перезапустите процесс, чтобы применить новые значения.
Optuna storage URL вычисляется методом `BacktestingSettings.get_optuna_storage_url()` из параметров подключения.

## Принципы разработки

- **Типизация**: Строгие аннотации типов, Pydantic модели, dataclass `BacktestResult`
- **Реальные данные**: Использование только реальных рыночных данных из внешних источников
- **Чистая архитектура**: Разделение на `datasource`, `services`, `tools`, `server`
- **Конфигурация**: Все настройки через ENV с префиксом `BACKTEST_`
- **Документирование MCP**: Обязательные `name`, `description`, `args_schema` в `@mcp.tool()`

## Локальный запуск и проверки

```bash
# Установка зависимостей (используем uv)
uv sync --project backtesting-mcp

# Запуск MCP сервера по stdio
uv run --project backtesting-mcp backtesting-mcp --stdio

# HTTP режим для локальной разработки
uv run --project backtesting-mcp backtesting-mcp --host 0.0.0.0 --port 8082

# Форматирование и линтинг
uv run --project backtesting-mcp black src
uv run --project backtesting-mcp isort src

# Тесты (при появлении тестовой матрицы)
uv run --project backtesting-mcp pytest
```

При работе с внешними MCP сервисами убедитесь, что `binance-mcp` и/или `tradingview-mcp` подняты и доступны по указанным URL.

### Docker Compose развертывание

#### Пересборка и перезапуск после изменений кода

**⚠️ ВАЖНО**: После внесения изменений в код необходимо пересобрать контейнер и перезапустить сервис:

```bash
# Пересборка контейнера с изменениями кода
docker-compose -f ../../infra/docker-compose.mcp.yml build backtesting-mcp

# Перезапуск сервиса с новым образом
docker-compose -f ../../infra/docker-compose.mcp.yml up -d backtesting-mcp

# Проверка статуса сервиса
docker-compose -f ../../infra/docker-compose.mcp.yml ps backtesting-mcp

# Просмотр логов для диагностики
docker-compose -f ../../infra/docker-compose.mcp.yml logs -f backtesting-mcp
```

#### Полная пересборка всех сервисов

```bash
# Пересборка всех MCP сервисов
docker-compose -f ../../infra/docker-compose.mcp.yml build

# Перезапуск всех сервисов
docker-compose -f ../../infra/docker-compose.mcp.yml up -d

# Проверка статуса всех сервисов
docker-compose -f ../../infra/docker-compose.mcp.yml ps
```

#### Отладка и диагностика

```bash
# Вход в контейнер для отладки
docker-compose -f ../../infra/docker-compose.mcp.yml exec backtesting-mcp bash

# Проверка переменных окружения в контейнере
docker-compose -f ../../infra/docker-compose.mcp.yml exec backtesting-mcp env

# Проверка доступности сервиса
curl http://localhost:8082/health
```

#### Очистка и пересборка с нуля

```bash
# Остановка и удаление контейнеров
docker-compose -f ../../infra/docker-compose.mcp.yml down

# Удаление образов (принудительная пересборка)
docker-compose -f ../../infra/docker-compose.mcp.yml build --no-cache backtesting-mcp

# Запуск с чистого листа
docker-compose -f ../../infra/docker-compose.mcp.yml up -d backtesting-mcp
```

## Качество кода и стандарты

### Инструменты качества
```bash
uv run --project backtesting-mcp black src
uv run --project backtesting-mcp ruff check src
uv run --project backtesting-mcp mypy src
uv run --project backtesting-mcp pytest tests/ --cov=src --cov-report=html
```

### Стандарты кодирования
- **PEP 8**, **Black** (88 символов), **Ruff**, **MyPy** (strict mode)
- **Type hints** для всех функций, **Docstrings** для публичных методов
- **SOLID**, **DRY**, **KISS**, **YAGNI** принципы
- **Conventional Commits**: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`

### Документирование MCP инструментов
- **Обязательные поля**: `name`, `description`, `args_schema` в `@mcp.tool()`
- **Pydantic модели**: Детальные описания с примерами и ограничениями
- **Консистентность**: Соответствие между кодом и `AGENTS.md`

## Мониторинг и логирование

- **Логирование**: Структурированные логи с контекстом операций
- **Health check**: Инструмент `health` для проверки состояния сервера
- **Статусы**: "healthy", "degraded", "unhealthy"
- **Отладка**: Детальные логи процесса бэктестинга и fallback операций

## Документация и отчёты

### ⚠️ Важно: правила ведения документации

**НЕ создавайте новые .md файлы для отчётов при каждом изменении.**

- **Не создавать** файлы вида `REPORT_2024.md`, `CHANGES.md`, `UPDATE.md`.
- **Не дублировать** информацию, уже присутствующую в существующих документах.

### Обновление документации

1. **`CHANGELOG.md`** — фиксируем релизные изменения (создать, если потребуется).
2. **`README.md`** — обновляем инструкции по запуску и использованию.
3. **`AGENTS.md`** — поддерживаем в актуальном состоянии информацию об архитектуре.
4. **Существующие .md файлы** — дополняем, вместо создания новых.

### Формат changelog

```markdown
## [Версия] - YYYY-MM-DD

### Добавлено
- Новая функциональность

### Изменено
- Изменения в существующей функциональности

### Исправлено
- Исправления ошибок

### Удалено
- Удалённая функциональность
```

Принципы: консолидация информации, единый формат, отслеживаемость изменений.

## Развитие проекта

### Приоритеты развития
1. **Стабильность**: Надёжная работа с внешними источниками данных
2. **Производительность**: Оптимизация алгоритмов бэктестинга и оптимизации
3. **Функциональность**: Новые стратегии и метрики анализа
4. **Интеграция**: Расширение поддержки источников данных

### Roadmap идей

#### Краткосрочные цели (1-3 месяца)
- [ ] Расширить список стратегий (RSI, Bollinger Bands, MACD)
- [ ] Добавить дополнительные метрики (Sortino ratio, Calmar ratio)
- [ ] Улучшить обработку ошибок и логирование
- [ ] Добавить поддержку пользовательских стратегий

#### Среднесрочные цели (3-6 месяцев)
- [ ] Интегрировать дополнительные источники данных (Bybit, Coinbase)
- [ ] Добавить майкросервис кеширования исторических свечей
- [ ] Внедрить тестовую матрицу с синтетическими сценариями (bull/bear/sideways)
- [ ] Добавить веб-интерфейс для мониторинга

#### Долгосрочные цели (6+ месяцев)
- [ ] Поддержка машинного обучения для оптимизации стратегий
- [ ] Интеграция с системами риск-менеджмента
- [ ] Поддержка мульти-активных стратегий
- [ ] API для внешних систем и плагинов

### Технические улучшения
- **Производительность**: Оптимизация алгоритмов загрузки данных
- **Масштабируемость**: Поддержка больших объёмов исторических данных
- **Надёжность**: Улучшение обработки ошибок и восстановления
- **Безопасность**: Аудит безопасности и валидации данных
