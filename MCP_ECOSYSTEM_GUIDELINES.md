# MCP Ecosystem Guidelines
# Руководство по разработке MCP серверов и автономных AI агентов

**Версия:** 1.0  
**Дата:** 7 ноября 2025  
**Статус:** Действующий стандарт

---

## Содержание

1. [Введение](#введение)
2. [Архитектурные принципы](#архитектурные-принципы)
3. [Стандарт MCP инструментов](#стандарт-mcp-инструментов)
4. [Принципы разработки](#принципы-разработки)
5. [Стандарты кодирования](#стандарты-кодирования)
6. [Тестирование](#тестирование)
7. [Конфигурация](#конфигурация)
8. [Безопасность](#безопасность)
9. [Мониторинг и логирование](#мониторинг-и-логирование)
10. [Документация](#документация)
11. [Специфика AI агентов](#специфика-ai-агентов)
12. [Специфика MCP серверов](#специфика-mcp-серверов)
13. [Шаблоны и примеры](#шаблоны-и-примеры)

---

## Введение

### Назначение экосистемы

Экосистема MCP (Model Context Protocol) — это распределенная система микросервисов и автономных AI агентов, предназначенная для:

- **Торговый анализ**: Анализ криптовалютного рынка (TradingView MCP)
- **Исполнение сделок**: Управление торговыми операциями (Binance MCP)
- **Тестирование стратегий**: Бэктестинг и оптимизация (Backtesting MCP)
- **Выполнение кода**: Безопасное исполнение в изолированной среде (Shell MCP)
- **Память и контекст**: Индексация и поиск информации (Memory MCP)
- **Автономные агенты**: Интеллектуальные агенты для Telegram (HALv1)

### Ключевые компоненты

```
mcp/
├── servers/                    # MCP серверы
│   ├── backtesting-mcp/       # Бэктестинг торговых стратегий
│   ├── binance-mcp/           # Интеграция с Binance API
│   ├── learning-mcp/         # Система обучения и адаптации
│   ├── memory-mcp/            # Система памяти и поиска
│   ├── shell-mcp/             # Выполнение кода в Docker
│   ├── supervisor-mcp/        # Управление и мониторинг серверов
│   └── tradingview-mcp/       # Анализ рынка через TradingView
├── infra/                      # Инфраструктура
│   ├── postgres/              # Конфигурация PostgreSQL
│   ├── templates/             # Шаблоны для новых проектов
│   ├── tests/                 # Интеграционные тесты
│   └── wheels/                # Python wheels для зависимостей
├── impl_plan/                  # Планы реализации HAL агента
│   ├── hal_ai_agent_plan.md
│   ├── hal_autonomous_agent_overview.md
│   ├── hal_extras_addendum.md
│   ├── HAL_IMPLEMENTATION_CHECKLIST.md
│   └── hal_implementation_instructions.md
├── halv1/                     # Автономный AI агент для Telegram
├── config/                    # Общие конфигурации
└── mcp-web-ui/                # Веб-интерфейс для MCP
```

### Взаимодействие компонентов

- **MCP серверы** предоставляют инструменты через Model Context Protocol
- **AI агенты** используют MCP инструменты для выполнения задач
- **Общая память** (Memory MCP) обеспечивает контекст и историю
- **Docker Compose** координирует развертывание всех компонентов

---

## Архитектурные принципы

### 1. Model Context Protocol (MCP)

**MCP** — это стандартизированный протокол для взаимодействия между AI агентами и инструментами.

Основные концепции:
- **Tools**: Функции, доступные агентам
- **Resources**: Данные и контекст
- **Prompts**: Шаблоны для взаимодействия
- **Transports**: stdio, HTTP, WebSocket

### 2. FastMCP как базовый фреймворк

Все MCP серверы используют **FastMCP** — Python-фреймворк для быстрой разработки MCP серверов.

```python
from fastmcp import FastMCP

mcp = FastMCP("server-name")

@mcp.tool()
def example_tool(param: str) -> dict:
    """Short description (≤90 chars) starting with a verb."""
    # Implementation
    return {"result": "value"}
```

### 3. Модульная структура проектов

**Стандартная структура MCP сервера:**

```
project-mcp/
├── src/
│   └── project_mcp/
│       ├── __init__.py          # Точка входа
│       ├── server.py            # MCP сервер
│       ├── config.py            # Конфигурация (Pydantic Settings)
│       ├── services/            # Бизнес-логика
│       │   ├── __init__.py
│       │   └── core_service.py
│       └── tools/               # MCP инструменты
│           ├── __init__.py
│           ├── models.py        # Pydantic модели
│           └── tool_handlers.py
├── tests/                       # Тесты
├── examples/                    # Примеры использования
├── docs/                        # Документация
├── pyproject.toml              # Конфигурация проекта
├── README.md
├── AGENTS.md                   # Архитектурное руководство
└── CHANGELOG.md                # История изменений
```

### 4. Разделение ответственности

**Чистая архитектура:**

- `server.py` — Настройка FastMCP, регистрация инструментов
- `config.py` — Загрузка настроек через Pydantic Settings
- `services/` — Бизнес-логика без зависимостей от MCP
- `tools/` — MCP инструменты, тонкая обертка над сервисами
- `models.py` — Pydantic модели для валидации данных

**Пример потока данных:**

```
Агент → MCP Tool → Service Layer → External API/Database
       ↓
    Validation (Pydantic)
       ↓
    Business Logic
       ↓
    Response → Агент
```

### 5. Поддержка транспортов

**Stdio транспорт** (для локальной интеграции):
```bash
python -m project_mcp.server --stdio
```

**HTTP транспорт** (для распределенной архитектуры):
```bash
python -m project_mcp.server --host 0.0.0.0 --port 8000
```

**Docker Compose:**
```yaml
services:
  project-mcp:
    build: ./project-mcp
    ports:
      - "8000:8000"
    environment:
      - PROJECT_API_KEY=${API_KEY}
```

---

## Стандарт MCP инструментов

### 1. Формат описания инструментов

**Обязательные требования:**

✅ **Первая строка ≤90 символов**  
✅ **Глагол в настоящем времени** (Executes, Returns, Fetches, Analyzes)  
✅ **Английский язык** (для единообразия в list_tools)  
✅ **Детали во втором предложении** или docstring

**Примеры:**

✅ **ПРАВИЛЬНО:**
```python
@mcp.tool()
def run_backtest(symbol: str, strategy: str) -> dict:
    """Runs a backtest of trading strategy on historical market data."""
    # Implementation
```

❌ **НЕПРАВИЛЬНО:**
```python
@mcp.tool()
def run_backtest(symbol: str, strategy: str) -> dict:
    """Эта функция запускает бэктест торговой стратегии на исторических данных с использованием пересечения скользящих средних и возвращает детальные метрики производительности включая Sharpe ratio и максимальную просадку."""
```

### 2. Использование декораторов

**Базовая регистрация:**
```python
@mcp.tool()
def tool_name(param: str) -> dict:
    """Description."""
    return {"status": "success"}
```

**С кастомным описанием:**
```python
@mcp.tool(description="Custom short description")
def tool_name(param: str) -> dict:
    """Extended documentation here."""
    return {"status": "success"}
```

**С логированием:**
```python
@mcp.tool()
@log_tool  # Кастомный декоратор для логирования
def tool_name(param: str) -> dict:
    """Description."""
    return {"status": "success"}
```

### 3. Pydantic модели для валидации

**Определение моделей:**
```python
from pydantic import BaseModel, Field

class BacktestRequest(BaseModel):
    """Request model for backtesting."""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    strategy: str = Field(..., description="Strategy name")
    start: str = Field(default="2025-01-01T00:00:00", description="Start date (ISO8601)")
    end: str = Field(default="2025-12-31T23:59:59", description="End date (ISO8601)")
    parameters: dict = Field(default_factory=dict, description="Strategy parameters")

class BacktestResult(BaseModel):
    """Result model for backtesting."""
    strategy: str
    symbol: str
    metrics: dict
    equity_curve: list[float]
    trade_log: list[dict]
```

**Использование в инструментах:**
```python
@mcp.tool()
def run_backtest(request: BacktestRequest) -> BacktestResult:
    """Runs a backtest of trading strategy on historical market data."""
    # Валидация происходит автоматически
    result = backtesting_service.execute(request)
    return BacktestResult(**result)
```

### 4. Документирование параметров

**Best practices:**

1. **Обязательные vs опциональные параметры**
   - Явно указывайте обязательные поля через `Field(...)`
   - Опциональные с `Field(default=...)`

2. **Взаимоисключающие параметры**
   ```python
   # Документируйте: code XOR script_path (один обязателен)
   code: str | None = Field(default=None)
   script_path: str | None = Field(default=None)
   ```

3. **Формат сложных полей**
   ```python
   env: list[str] = Field(
       default_factory=list,
       description="Environment variables as KEY=VALUE strings",
       examples=[["API_KEY=secret", "DEBUG=true"]]
   )
   ```

4. **Диапазоны и ограничения**
   ```python
   limit: int = Field(
       default=50,
       ge=1,
       le=200,
       description="Maximum number of results (1-200)"
   )
   ```

### 5. Примеры вызова инструментов

**Включайте примеры в docstring:**

```python
@mcp.tool()
def coin_analysis(symbol: str, exchange: str = "KUCOIN") -> dict:
    """Analyzes a specific cryptocurrency with technical indicators.
    
    Examples:
        >>> coin_analysis(symbol="BTCUSDT", exchange="KUCOIN")
        >>> coin_analysis(symbol="ETHUSDT", exchange="BINANCE")
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT, ETHUSDT)
        exchange: Exchange name (KUCOIN, BINANCE, etc.)
    
    Returns:
        Analysis with RSI, MACD, Bollinger Bands and trading signals
    """
    return analyzer.analyze(symbol, exchange)
```

---

## Принципы разработки

### 1. Типизация

**Строгая типизация всех функций:**

```python
from typing import Any, Literal

def process_data(
    input_data: list[dict[str, Any]],
    mode: Literal["fast", "accurate"] = "fast",
    timeout: float = 30.0
) -> dict[str, Any]:
    """Process data with specified mode."""
    ...
```

**Использование Pydantic:**
```python
from pydantic import BaseModel, validator

class Config(BaseModel):
    api_key: str
    timeout: float = 30.0
    max_retries: int = 3
    
    @validator("timeout")
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError("timeout must be positive")
        return v
```

### 2. Чистая архитектура

**Принципы:**

- **Dependency Inversion**: Services зависят от абстракций, не от деталей
- **Single Responsibility**: Каждый модуль имеет одну причину для изменения
- **Interface Segregation**: Клиенты не зависят от неиспользуемых интерфейсов

**Пример:**

```python
# services/base.py
from abc import ABC, abstractmethod

class MarketDataProvider(ABC):
    @abstractmethod
    async def get_candles(self, symbol: str, interval: str) -> list[dict]:
        pass

# services/binance_provider.py
class BinanceProvider(MarketDataProvider):
    async def get_candles(self, symbol: str, interval: str) -> list[dict]:
        # Implementation
        pass

# services/backtesting.py
class BacktestingService:
    def __init__(self, data_provider: MarketDataProvider):
        self.provider = data_provider
    
    async def run(self, symbol: str):
        candles = await self.provider.get_candles(symbol, "1h")
        # Backtest logic
```

### 3. Конфигурация через ENV

**Использование Pydantic Settings:**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PROJECT_",  # Префикс для переменных
        env_file=".env",
        case_sensitive=False
    )
    
    api_key: str
    api_secret: str
    default_timeout: float = 30.0
    max_retries: int = 3
    debug: bool = False

# Использование
settings = Settings()
```

**Префиксы для каждого сервиса:**
- `BACKTEST_` — Backtesting MCP
- `BINANCE_` — Binance MCP
- `TRADINGVIEW_` — TradingView MCP
- `SHELL_MCP_` — Shell MCP
- `MEMORY_` — Memory MCP

### 4. SOLID, DRY, KISS, YAGNI

**SOLID:**
- **S**ingle Responsibility: Один класс = одна ответственность
- **O**pen/Closed: Открыт для расширения, закрыт для модификации
- **L**iskov Substitution: Подклассы заменяемы базовыми классами
- **I**nterface Segregation: Много специфичных интерфейсов
- **D**ependency Inversion: Зависимость от абстракций

**DRY** (Don't Repeat Yourself):
```python
# ❌ Плохо
def get_btc_price(): ...
def get_eth_price(): ...
def get_xrp_price(): ...

# ✅ Хорошо
def get_price(symbol: str): ...
```

**KISS** (Keep It Simple, Stupid):
```python
# ❌ Сложно
result = [x for x in data if condition(x)] if data else []

# ✅ Просто
if not data:
    return []
return [x for x in data if condition(x)]
```

**YAGNI** (You Aren't Gonna Need It):
- Не добавляйте функциональность "на всякий случай"
- Реализуйте только то, что нужно сейчас

### 5. Graceful Error Handling

**Структурированная обработка ошибок:**

```python
from typing import Any
import logging

logger = logging.getLogger(__name__)

@mcp.tool()
def risky_operation(param: str) -> dict[str, Any]:
    """Performs a risky operation with proper error handling."""
    try:
        result = perform_operation(param)
        return {
            "success": True,
            "data": result
        }
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return {
            "success": False,
            "error": "invalid_input",
            "message": str(e)
        }
    except ExternalAPIError as e:
        logger.error(f"API error: {e}")
        return {
            "success": False,
            "error": "api_error",
            "message": "External service unavailable"
        }
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return {
            "success": False,
            "error": "internal_error",
            "message": "An unexpected error occurred"
        }
```

---

## Стандарты кодирования

### 1. PEP 8

Следуйте [PEP 8](https://peps.python.org/pep-0008/) — официальному стандарту Python.

**Основные правила:**
- Отступы: 4 пробела
- Максимальная длина строки: 88 символов (Black)
- Именование: `snake_case` для функций/переменных, `PascalCase` для классов
- Константы: `UPPER_CASE`
- Приватные методы: `_private_method`

### 2. Black (88 символов)

**Автоматическое форматирование:**

```bash
# Форматирование всего проекта
black .

# Проверка без изменений
black --check .

# Конфигурация в pyproject.toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
```

### 3. Ruff линтинг

**Быстрый и современный линтер:**

```bash
# Проверка кода
ruff check .

# Автоматическое исправление
ruff check --fix .

# Конфигурация в pyproject.toml
[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W"]
ignore = []
```

**Основные правила:**
- `E` — PEP 8 ошибки
- `F` — Ошибки pyflakes
- `I` — Сортировка импортов (isort)
- `N` — Именование (pep8-naming)
- `W` — Предупреждения

### 4. MyPy (strict mode)

**Проверка типов:**

```bash
# Проверка типов
mypy src/

# Конфигурация в pyproject.toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

**Примеры:**

```python
# ✅ Правильно
def calculate(value: float) -> float:
    return value * 2

# ❌ Неправильно (отсутствуют типы)
def calculate(value):
    return value * 2
```

### 5. Conventional Commits

**Формат коммита:**

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Типы:**
- `feat:` — Новая функциональность
- `fix:` — Исправление ошибки
- `docs:` — Изменения в документации
- `test:` — Добавление/изменение тестов
- `refactor:` — Рефакторинг кода
- `perf:` — Улучшение производительности
- `style:` — Форматирование (без изменения логики)
- `chore:` — Обслуживание (зависимости, конфиг)

**Для AI агентов:**
- `ai:` — Изменения в логике агента
- `memory:` — Изменения в системе памяти

**Для MCP серверов:**
- `security:` — Изменения безопасности
- `analytics:` — Изменения в аналитике

**Примеры:**

```bash
feat(indicators): добавить новый технический индикатор ADX
fix(screener): исправить расчет Bollinger Bands
docs(api): обновить документацию по стратегиям
test(executor): добавить тесты для Docker исполнителя
refactor(llm): упростить интерфейс LLM клиентов
perf(api): оптимизировать батчевые запросы к TradingView
security(docker): ужесточить политики безопасности
ai(planning): улучшить алгоритм планирования задач
memory(graph): реализовать графовую память
```

---

## Тестирование

### 1. Unit тесты (80%+ покрытие)

**Структура тестов:**

```
tests/
├── __init__.py
├── conftest.py              # Фикстуры pytest
├── test_services/
│   ├── test_backtesting.py
│   └── test_indicators.py
├── test_tools/
│   └── test_mcp_tools.py
└── test_integration/
    └── test_end_to_end.py
```

**Пример unit теста:**

```python
import pytest
from project_mcp.services import BacktestingService

@pytest.fixture
def service():
    return BacktestingService()

def test_backtest_calculates_metrics(service):
    """Test that backtest calculates all required metrics."""
    result = service.run(
        symbol="BTCUSDT",
        strategy="ma_crossover",
        parameters={"fast": 10, "slow": 20}
    )
    
    assert "total_return" in result.metrics
    assert "sharpe" in result.metrics
    assert "max_drawdown" in result.metrics
    assert result.metrics["trades"] > 0

def test_backtest_handles_invalid_symbol(service):
    """Test that backtest handles invalid symbols gracefully."""
    with pytest.raises(ValueError, match="Invalid symbol"):
        service.run(symbol="INVALID", strategy="ma_crossover")
```

**Цель: минимум 80% покрытие:**

```bash
# Запуск тестов с покрытием
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Просмотр отчета
open htmlcov/index.html
```

### 2. Интеграционные тесты

**Тестирование взаимодействия компонентов:**

```python
import pytest
from fastmcp.testing import MCPTestClient

@pytest.mark.integration
async def test_backtest_tool_integration():
    """Test full backtest flow through MCP tool."""
    async with MCPTestClient(mcp) as client:
        response = await client.call_tool(
            "run_backtest",
            {
                "symbol": "BTCUSDT",
                "strategy": "ma_crossover",
                "parameters": {"fast": 10, "slow": 20}
            }
        )
        
        assert response["success"] is True
        assert "metrics" in response["data"]
        assert response["data"]["metrics"]["total_return"] != 0
```

### 3. Mock объекты

**Мокирование внешних зависимостей:**

```python
from unittest.mock import Mock, patch
import pytest

@pytest.fixture
def mock_binance_client():
    """Mock Binance API client."""
    client = Mock()
    client.get_klines.return_value = [
        [1609459200000, "29000", "30000", "28000", "29500", "1000"],
        # ... more data
    ]
    return client

def test_backtest_with_mocked_data(mock_binance_client):
    """Test backtest with mocked market data."""
    with patch("project_mcp.services.BinanceClient", return_value=mock_binance_client):
        service = BacktestingService()
        result = service.run("BTCUSDT", "ma_crossover")
        
        assert result is not None
        mock_binance_client.get_klines.assert_called_once()
```

### 4. Структура тестов

**Паттерн AAA (Arrange-Act-Assert):**

```python
def test_indicator_calculation():
    # Arrange: подготовка данных
    prices = [100, 102, 101, 103, 105]
    period = 3
    
    # Act: выполнение действия
    result = calculate_sma(prices, period)
    
    # Assert: проверка результата
    assert len(result) == len(prices) - period + 1
    assert result[0] == pytest.approx(101.0)
```

**Параметризованные тесты:**

```python
@pytest.mark.parametrize("symbol,expected_exchange", [
    ("BTCUSDT", "BINANCE"),
    ("BTC-USDT", "KUCOIN"),
    ("BTC/USDT", "GENERAL"),
])
def test_symbol_parsing(symbol, expected_exchange):
    """Test symbol parsing for different exchanges."""
    result = parse_symbol(symbol)
    assert result.exchange == expected_exchange
```

---

## Конфигурация

### 1. Переменные окружения

**Структура .env файла:**

```bash
# Binance MCP
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_DEMO_TRADING=false

# TradingView MCP
TRADINGVIEW_API_KEY=your_api_key
DEFAULT_EXCHANGE=KUCOIN
DEFAULT_TIMEFRAME=15m

# Backtesting MCP
BACKTEST_PREFERRED_SOURCE=binance
BACKTEST_DEFAULT_TIMEFRAME=1h

# Shell MCP
SHELL_MCP_DEFAULT_IMAGE=python:3.11
SHELL_MCP_TIMEOUT_SECONDS=120
SHELL_MCP_MAX_CONTAINERS=10

# Memory MCP
MEMORY_DB_PATH=/data/memory_graph.db
QDRANT_URL=http://qdrant:6333
EMBEDDINGS_URL=http://embeddings:80

# Общие настройки
LOG_LEVEL=INFO
DEBUG=false
```

### 2. Префиксы для каждого сервиса

**Стандартные префиксы:**

| Сервис | Префикс | Примеры |
|--------|---------|---------|
| Backtesting MCP | `BACKTEST_` | `BACKTEST_DEFAULT_TIMEFRAME` |
| Binance MCP | `BINANCE_` | `BINANCE_API_KEY` |
| TradingView MCP | `TRADINGVIEW_` | `TRADINGVIEW_API_KEY` |
| Shell MCP | `SHELL_MCP_` | `SHELL_MCP_DEFAULT_IMAGE` |
| Memory MCP | `MEMORY_` | `MEMORY_DB_PATH` |

**Избегайте конфликтов:** Используйте уникальные префиксы для каждого сервиса.

### 3. Pydantic Settings

**Базовый паттерн:**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SERVICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API настройки
    api_key: str
    api_secret: str
    
    # Настройки по умолчанию
    default_timeout: float = 30.0
    max_retries: int = 3
    
    # Режим работы
    debug: bool = False
    demo_mode: bool = False

@lru_cache()
def get_settings() -> ServiceSettings:
    """Get cached settings instance."""
    return ServiceSettings()

# Использование
settings = get_settings()
```

### 4. Docker Compose интеграция

**Пример docker-compose.yml:**

```yaml
version: '3.8'

services:
  binance-mcp:
    build: ./servers/binance-mcp
    ports:
      - "8000:8000"
    environment:
      - BINANCE_API_KEY=${BINANCE_API_KEY}
      - BINANCE_API_SECRET=${BINANCE_API_SECRET}
    env_file:
      - .env
    restart: unless-stopped

  tradingview-mcp:
    build: ./servers/tradingview-mcp
    ports:
      - "8060:8060"
    environment:
      - TRADINGVIEW_API_KEY=${TRADINGVIEW_API_KEY}
      - DEFAULT_EXCHANGE=KUCOIN
    depends_on:
      - postgres
    restart: unless-stopped

  backtesting-mcp:
    build: ./servers/backtesting-mcp
    ports:
      - "8082:8082"
    environment:
      - BACKTEST_BINANCE_URL=http://binance-mcp:8000
      - BACKTEST_TRADINGVIEW_URL=http://tradingview-mcp:8060
    depends_on:
      - binance-mcp
      - tradingview-mcp
    restart: unless-stopped

  shell-mcp:
    build: ./servers/shell-mcp
    ports:
      - "8070:8070"
    environment:
      - SHELL_MCP_DEBUG=false
    restart: unless-stopped

  memory-mcp:
    build: ./servers/memory-mcp
    ports:
      - "8050:8050"
    environment:
      - MEMORY_MCP_DEBUG=false
    depends_on:
      - redis
    restart: unless-stopped

  supervisor-mcp:
    build: ./servers/supervisor-mcp
    ports:
      - "8001:8001"
    environment:
      - SUPERVISOR_MCP_DEBUG=false
    restart: unless-stopped

  bright-data-mcp:
    image: node:18-alpine
    ports:
      - "8083:8083"
    command: ["npx", "@brightdata/mcp"]
    environment:
      - API_TOKEN=${BRIGHTDATA_API_TOKEN}
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=tradingview_scanners
      - POSTGRES_USER=tradingview
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./infra/postgres/initdb.d:/docker-entrypoint-initdb.d:ro
    restart: unless-stopped

volumes:
  postgres_data:
```

### 5. JSON конфигурация для MCP клиента

**Пример конфигурации для MCP клиента:**

```json
{
  "mcpServers": {
    "Bright Data": {
      "command": "npx",
      "args": ["@brightdata/mcp"],
      "env": {
        "API_TOKEN": "${BRIGHTDATA_API_TOKEN}"
      }
    },
    "tradingview-mcp": {
      "url": "http://127.0.0.1:8060/mcp"
    },
    "binance-mcp-demo": {
      "url": "http://127.0.0.1:8000/mcp"
    },
    "shell-mcp": {
      "url": "http://127.0.0.1:8070/mcp"
    },
    "memory-mcp": {
      "url": "http://127.0.0.1:8050/mcp"
    },
    "backtesting-mcp": {
      "url": "http://127.0.0.1:8082/mcp"
    },
    "supervisor-mcp": {
      "url": "http://127.0.0.1:8001/mcp"
    }
  }
}
```

**Описание серверов:**
- **Bright Data**: Внешний MCP сервер для работы с данными
- **tradingview-mcp**: Анализ рынка через TradingView API
- **binance-mcp-demo**: Демо-режим торговли на Binance
- **shell-mcp**: Выполнение команд в Docker контейнерах
- **memory-mcp**: Система памяти и поиска
- **backtesting-mcp**: Бэктестинг торговых стратегий
- **supervisor-mcp**: Управление и мониторинг MCP серверов

---

## Безопасность

### 1. API ключи в ENV

**❌ НИКОГДА:**
- Не коммитьте API ключи в Git
- Не храните секреты в коде
- Не логируйте секретные данные

**✅ ВСЕГДА:**
- Используйте переменные окружения
- Добавьте `.env` в `.gitignore`
- Используйте секретные менеджеры в продакшене

**Пример безопасной конфигурации:**

```python
from pydantic import SecretStr

class Settings(BaseSettings):
    api_key: SecretStr  # Автоматически маскируется при логировании
    api_secret: SecretStr
    
    def get_api_key(self) -> str:
        """Get API key value."""
        return self.api_key.get_secret_value()
```

### 2. Валидация входных данных

**Всегда валидируйте входные данные:**

```python
from pydantic import BaseModel, validator, Field

class OrderRequest(BaseModel):
    symbol: str = Field(..., regex=r"^[A-Z]{2,10}USDT$")
    side: Literal["buy", "sell"]
    quantity: float = Field(..., gt=0)
    price: float | None = Field(None, gt=0)
    
    @validator("symbol")
    def validate_symbol(cls, v):
        """Validate symbol format."""
        if not v.endswith("USDT"):
            raise ValueError("Only USDT pairs supported")
        return v.upper()
    
    @validator("quantity")
    def validate_quantity(cls, v):
        """Validate quantity is positive."""
        if v <= 0:
            raise ValueError("Quantity must be positive")
        return v
```

### 3. Изоляция выполнения (Docker)

**Shell MCP паттерн:**

```python
def execute_code(code: str, language: str) -> dict:
    """Execute code in isolated Docker container."""
    container = docker_client.containers.run(
        image=get_image_for_language(language),
        command=["python", "-c", code],
        detach=True,
        mem_limit="512m",       # Ограничение памяти
        cpu_quota=50000,        # Ограничение CPU
        network_disabled=False,  # Контроль сети
        remove=True,            # Автоудаление
        read_only=True,         # Только чтение ФС
        security_opt=["no-new-privileges"],  # Безопасность
    )
    
    try:
        container.wait(timeout=120)
        return {
            "stdout": container.logs(stdout=True).decode(),
            "stderr": container.logs(stderr=True).decode(),
        }
    finally:
        container.remove(force=True)
```

### 4. Rate Limiting

**Защита от злоупотреблений:**

```python
from time import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = defaultdict(list)
    
    def check_limit(self, key: str) -> bool:
        """Check if rate limit is exceeded."""
        now = time()
        self.calls[key] = [
            t for t in self.calls[key]
            if now - t < self.period
        ]
        
        if len(self.calls[key]) >= self.max_calls:
            return False
        
        self.calls[key].append(now)
        return True

# Использование
limiter = RateLimiter(max_calls=100, period=60.0)

@mcp.tool()
def rate_limited_tool(param: str) -> dict:
    """Tool with rate limiting."""
    if not limiter.check_limit("global"):
        return {"error": "Rate limit exceeded"}
    
    return perform_operation(param)
```

---

## Мониторинг и логирование

### 1. Структурированные логи

**Настройка логирования:**

```python
import logging
import sys

def setup_logging(level: str = "INFO"):
    """Setup structured logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/app.log"),
        ],
    )

# Использование
logger = logging.getLogger(__name__)

@mcp.tool()
def monitored_tool(param: str) -> dict:
    """Tool with structured logging."""
    logger.info(f"Tool called with param={param}")
    
    try:
        result = perform_operation(param)
        logger.info(f"Tool succeeded, result_size={len(result)}")
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Tool failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
```

**Структурированное логирование (JSON):**

```python
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger()
logger.info("backtest.started", symbol="BTCUSDT", strategy="ma_crossover")
```

### 2. Health Check инструменты

**Стандартный health check:**

```python
import time
from datetime import datetime

start_time = time.time()

@mcp.tool()
def health(check_dependencies: bool = True) -> dict:
    """Checks the health status of the server and its dependencies."""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time() - start_time,
        "server_info": {
            "version": "1.0.0",
            "name": "project-mcp",
        },
    }
    
    if check_dependencies:
        dependencies = {}
        
        # Проверка базы данных
        try:
            db.ping()
            dependencies["database"] = {"status": "healthy", "response_time": 0.05}
        except Exception as e:
            dependencies["database"] = {"status": "unhealthy", "error": str(e)}
            status["status"] = "degraded"
        
        # Проверка внешнего API
        try:
            response = requests.get(external_api_url, timeout=5)
            dependencies["external_api"] = {
                "status": "healthy" if response.ok else "unhealthy",
                "response_time": response.elapsed.total_seconds(),
            }
        except Exception as e:
            dependencies["external_api"] = {"status": "unhealthy", "error": str(e)}
            status["status"] = "degraded"
        
        status["dependencies"] = dependencies
    
    return status
```

### 3. Метрики производительности

**Сбор метрик:**

```python
from time import time
from functools import wraps

def track_performance(func):
    """Decorator to track function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        try:
            result = func(*args, **kwargs)
            duration = time() - start
            logger.info(
                f"Function {func.__name__} completed",
                extra={
                    "function": func.__name__,
                    "duration": duration,
                    "success": True,
                }
            )
            return result
        except Exception as e:
            duration = time() - start
            logger.error(
                f"Function {func.__name__} failed",
                extra={
                    "function": func.__name__,
                    "duration": duration,
                    "success": False,
                    "error": str(e),
                }
            )
            raise
    return wrapper

@mcp.tool()
@track_performance
def expensive_operation(param: str) -> dict:
    """Operation with performance tracking."""
    return heavy_computation(param)
```

### 4. Алерты

**Telegram уведомления:**

```python
import requests

class TelegramAlerter:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    def send_alert(self, message: str, level: str = "INFO"):
        """Send alert to Telegram."""
        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "🚨"}
        formatted_message = f"{emoji.get(level, '')} {level}: {message}"
        
        try:
            requests.post(
                self.api_url,
                json={"chat_id": self.chat_id, "text": formatted_message},
                timeout=5,
            )
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

# Использование
alerter = TelegramAlerter(
    bot_token=settings.telegram_bot_token,
    chat_id=settings.telegram_chat_id,
)

def critical_operation():
    try:
        result = perform_operation()
    except Exception as e:
        alerter.send_alert(f"Critical operation failed: {e}", level="ERROR")
        raise
```

---

## Документация

### ⚠️ ВАЖНО: Правила работы с документацией

**НЕ СОЗДАВАЙТЕ .md файлы отчёты каждый раз!**

❌ **НЕ создавайте:**
- Новые .md файлы для каждого изменения или отчёта
- Файлы типа `REPORT_2024.md`, `CHANGES.md`, `UPDATE.md`
- Дубликаты информации в разных файлах

✅ **ВСЕГДА обновляйте существующие файлы:**

1. **`CHANGELOG.md`** — основной файл для записи изменений
2. **`README.md`** — обновляйте при значительных изменениях
3. **`AGENTS.md`** — обновляйте при изменениях архитектуры
4. **Существующие .md файлы** — дополняйте, не создавайте новые

### Формат CHANGELOG.md

**Следуйте [Keep a Changelog](https://keepachangelog.com/):**

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Новая функциональность в разработке

### Changed
- Изменения в существующей функциональности

## [1.2.0] - 2025-01-15

### Added
- Добавлена поддержка Optuna для оптимизации параметров
- Новый инструмент `optimize_parameters` с TPE sampler

### Changed
- Улучшена производительность бэктестинга на 30%
- Обновлены зависимости до последних версий

### Fixed
- Исправлена утечка памяти в векторном поиске
- Исправлен расчет Sharpe ratio для коротких периодов

### Deprecated
- Метод `old_backtest()` будет удален в версии 2.0.0

### Removed
- Удалена поддержка Python 3.9

### Security
- Обновлен requests для устранения CVE-2024-XXXXX

## [1.1.0] - 2024-12-01

...
```

### Структура README.md

**Базовая структура:**

```markdown
# Project Name

Brief description of what this MCP server does.

## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

```bash
# Using pip
pip install -e .

# Using uv
uv sync
```

## Quick Start

```bash
# Stdio mode
python -m project_mcp.server --stdio

# HTTP mode
python -m project_mcp.server --host 0.0.0.0 --port 8000
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PROJECT_API_KEY` | API key | Required |
| `PROJECT_TIMEOUT` | Request timeout | `30.0` |

## Tools

### tool_name

Description of the tool.

**Parameters:**
- `param1` (string, required): Description
- `param2` (int, optional): Description

**Example:**
```json
{
  "param1": "value",
  "param2": 42
}
```

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
```

### Структура AGENTS.md

**Руководство по архитектуре для AI агентов:**

```markdown
# Project MCP Server - Архитектура и Руководство

## Обзор проекта

Brief description and purpose.

## Архитектура

### Структура проекта

```
project-mcp/
├── src/
│   └── project_mcp/
│       ├── server.py
│       ├── config.py
│       ├── services/
│       └── tools/
```

### Компоненты системы

#### 1. Component Name
- **Назначение**: What it does
- **Функции**: List of functions
- **Паттерны**: Design patterns used

## Принципы разработки

### 1. Принцип
Description and examples

## Инструменты (Tools)

### Стандарт описания инструментов
- Rules for tool descriptions
- Format requirements

### tool_name

**Описание**: What the tool does

**Параметры**:
- `param` (type, required/optional): Description

**Возвращает**:
```json
{
  "result": "value"
}
```

## Конфигурация и запуск

### Переменные окружения
...

### Запуск
...

## Качество кода и стандарты

### Инструменты качества
...

### Примеры коммитов
...
```

---

## Специфика AI агентов

### 1. Событийная архитектура

**Event Bus паттерн:**

```python
from typing import Callable, Any
from collections import defaultdict

class EventBus:
    """Centralized event bus for agent communication."""
    
    def __init__(self):
        self.subscribers: dict[str, list[Callable]] = defaultdict(list)
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to event type."""
        self.subscribers[event_type].append(handler)
    
    def publish(self, event_type: str, data: Any):
        """Publish event to subscribers."""
        for handler in self.subscribers[event_type]:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler failed: {e}")

# Использование
event_bus = EventBus()

# Подписка
event_bus.subscribe("task.completed", on_task_completed)
event_bus.subscribe("error.occurred", on_error)

# Публикация
event_bus.publish("task.completed", {"task_id": "123", "result": "success"})
```

### 2. Система памяти

**Эпизодическая и семантическая память:**

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MemoryRecord:
    """Memory record structure."""
    record_id: str
    content: str
    source: str
    timestamp: datetime
    tags: list[str]
    metadata: dict
    embedding: list[float] | None = None

class MemorySystem:
    """Unified memory system for AI agents."""
    
    def __init__(self, db_path: str, vector_store_url: str):
        self.graph = TypedGraphMemory(db_path)
        self.vector_store = QdrantVectorStore(vector_store_url)
    
    async def ingest(self, record: MemoryRecord):
        """Ingest memory record."""
        # Добавить в граф (FTS)
        await self.graph.add_node(record)
        
        # Добавить вектор (ANN)
        if record.embedding:
            await self.vector_store.upsert(record.record_id, record.embedding)
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None
    ) -> list[MemoryRecord]:
        """Hybrid search: FTS + vector."""
        # FTS поиск
        fts_results = await self.graph.search_text(query, limit=top_k)
        
        # Векторный поиск
        vector_results = await self.vector_store.search(
            query_embedding=embed(query),
            limit=top_k
        )
        
        # Объединение результатов
        return merge_results(fts_results, vector_results, top_k)
```

### 3. Адаптивность и обучение

**Адаптация поведения агента:**

```python
class AdaptiveAgent:
    """Agent that learns from experience."""
    
    def __init__(self):
        self.experience = []
        self.performance_metrics = {}
    
    def execute_task(self, task: Task) -> Result:
        """Execute task and learn from outcome."""
        start_time = time.time()
        
        try:
            # Выбор стратегии на основе опыта
            strategy = self.select_strategy(task)
            
            # Выполнение
            result = strategy.execute(task)
            
            # Сохранение успешного опыта
            self.record_experience(
                task=task,
                strategy=strategy,
                result=result,
                duration=time.time() - start_time,
                success=True
            )
            
            return result
        
        except Exception as e:
            # Сохранение неудачного опыта
            self.record_experience(
                task=task,
                strategy=strategy,
                error=str(e),
                duration=time.time() - start_time,
                success=False
            )
            raise
    
    def select_strategy(self, task: Task) -> Strategy:
        """Select best strategy based on past experience."""
        # Анализ прошлого опыта
        similar_tasks = self.find_similar_tasks(task)
        
        # Выбор стратегии с лучшей производительностью
        if similar_tasks:
            return self.best_performing_strategy(similar_tasks)
        
        # Дефолтная стратегия для новых типов задач
        return DefaultStrategy()
```

### 4. Планирование и выполнение

**Multi-step планирование:**

```python
from typing import Protocol

class Step(Protocol):
    """Interface for execution step."""
    def execute(self) -> Any: ...
    def rollback(self) -> None: ...

class Plan:
    """Execution plan with rollback support."""
    
    def __init__(self, steps: list[Step]):
        self.steps = steps
        self.executed_steps: list[Step] = []
    
    async def execute(self) -> Any:
        """Execute plan with automatic rollback on failure."""
        try:
            for step in self.steps:
                logger.info(f"Executing step: {step.__class__.__name__}")
                result = await step.execute()
                self.executed_steps.append(step)
                logger.info(f"Step completed: {step.__class__.__name__}")
            
            return result
        
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            await self.rollback()
            raise
    
    async def rollback(self):
        """Rollback executed steps in reverse order."""
        for step in reversed(self.executed_steps):
            try:
                logger.info(f"Rolling back step: {step.__class__.__name__}")
                await step.rollback()
            except Exception as e:
                logger.error(f"Rollback failed for {step.__class__.__name__}: {e}")

# Использование
plan = Plan([
    FetchMarketDataStep(),
    AnalyzeDataStep(),
    GenerateSignalsStep(),
    ExecuteTradesStep(),
])

result = await plan.execute()
```

---

## Интеграция Bright Data MCP

### 1. Архитектура интеграции

Bright Data MCP интегрирован в экосистему как универсальный сервис скрапинга, управляемый через `supervisor-mcp`. Все компоненты экосистемы могут делать синхронные запросы на скрапинг через централизованный оркестратор.

**Архитектура взаимодействия:**
```
┌─────────────────┐
│   HAL Agent     │───┐
└─────────────────┘   │
                      │
┌─────────────────┐   │    ┌──────────────────┐
│ TradingView MCP │───┼───→│ Supervisor MCP   │
└─────────────────┘   │    │  (Orchestrator)  │
                      │    └──────────────────┘
┌─────────────────┐   │             │
│ Backtesting MCP │───┘             │
└─────────────────┘                 │
                                    ↓
                          ┌──────────────────┐
                          │ Bright Data MCP  │
                          │  (NPX Runtime)   │
                          └──────────────────┘
                                    │
                          ┌──────────────────┐
                          │   Memory MCP     │
                          │  (Storage)       │
                          └──────────────────┘
```

### 2. Доступные инструменты скрапинга

**Через supervisor-mcp доступны следующие инструменты:**

- `scrape_url(url, options)` - скрапинг одной URL
- `scrape_urls_batch(urls, options)` - пакетный скрапинг нескольких URLs
- `scrape_search_results(query, search_engine, limit, options)` - скрапинг результатов поиска
- `get_scraping_status(task_id)` - проверка статуса задачи
- `get_scraping_history(limit, status_filter)` - история скрапинга
- `clear_scraping_cache()` - очистка кеша

### 3. Примеры использования

**Скрапинг новостей для анализа сентимента:**
```python
# Через supervisor-mcp
result = await supervisor.scrape_url(
    url="https://coinmarketcap.com/currencies/bitcoin/",
    options={"extract": ["price", "volume", "market_cap"]}
)

# Сохранение в memory-mcp
await memory.ingest_scraped_content(
    url=result["url"],
    title=result["title"],
    content=result["content"],
    metadata=result["metadata"]
)
```

**Пакетный скрапинг для мониторинга конкурентов:**
```python
# Скрапинг цен с разных бирж
urls = [
    "https://coinmarketcap.com/currencies/bitcoin/",
    "https://coinbase.com/price/bitcoin",
    "https://binance.com/en/price/bitcoin"
]

results = await supervisor.scrape_urls_batch(
    urls=urls,
    options={"extract": ["price", "change_24h"]}
)

# Сохранение всех результатов
for result in results:
    if result.get("status") != "failed":
        await memory.ingest_scraped_content(
            url=result["url"],
            title=result.get("title", ""),
            content=result["content"],
            metadata=result["metadata"],
            tags=["price_monitoring", "bitcoin"]
        )
```

**Поиск и скрапинг новостей:**
```python
# Поиск новостей о Bitcoin
search_results = await supervisor.scrape_search_results(
    query="bitcoin news today",
    search_engine="google",
    limit=10,
    options={"extract": ["title", "snippet", "url"]}
)

# Скрапинг найденных страниц
for result in search_results:
    if result.get("url"):
        content = await supervisor.scrape_url(
            url=result["url"],
            options={"extract": ["title", "content", "publish_date"]}
        )
        
        await memory.ingest_scraped_content(
            url=content["url"],
            title=content["title"],
            content=content["content"],
            metadata={
                "search_query": "bitcoin news today",
                "search_engine": "google",
                **content["metadata"]
            },
            tags=["news", "bitcoin", "sentiment_analysis"]
        )
```

### 4. Интеграция с торговыми сервисами

**Обогащение данных для TradingView MCP:**
```python
# Получение дополнительных данных о монете
coin_data = await supervisor.scrape_url(
    url=f"https://coinmarketcap.com/currencies/{symbol.lower()}/",
    options={
        "extract": [
            "market_cap", "volume_24h", "circulating_supply",
            "total_supply", "max_supply", "price_change_24h"
        ]
    }
)

# Использование в анализе
analysis = await tradingview.coin_analysis(
    symbol=symbol,
    exchange=exchange,
    timeframe=timeframe
)

# Обогащение анализа скрапленными данными
enriched_analysis = {
    **analysis,
    "market_data": coin_data["metadata"],
    "scraped_at": coin_data["timestamp"]
}
```

**Исторические данные для бэктестинга:**
```python
# Скрапинг исторических новостей
historical_news = await supervisor.scrape_search_results(
    query=f"{symbol} news 2024",
    search_engine="google",
    limit=50,
    options={
        "extract": ["title", "snippet", "url", "date"],
        "date_range": "2024-01-01:2024-12-31"
    }
)

# Сохранение для корреляции с ценами
for news in historical_news:
    await memory.ingest_scraped_content(
        url=news["url"],
        title=news["title"],
        content=news["snippet"],
        metadata={
            "symbol": symbol,
            "date": news["date"],
            "type": "historical_news"
        },
        tags=["historical", "news", symbol.lower()]
    )
```

### 5. Конфигурация и переменные окружения

**Основные переменные:**
```bash
# Bright Data API токен
BRIGHT_DATA_API_TOKEN=your_api_token_here

# URL Bright Data MCP сервера
BRIGHT_DATA_MCP_URL=http://bright-data-mcp:8083

# Настройки кеширования
BRIGHT_DATA_CACHE_TTL=3600  # 1 час
BRIGHT_DATA_MAX_CONCURRENT=5  # Максимум одновременных запросов
```

**Docker Compose конфигурация:**
```yaml
bright-data-mcp:
  image: node:18-alpine
  container_name: mcp-bright-data-mcp-1
  restart: unless-stopped
  working_dir: /app
  command: ["npx", "-y", "@brightdata/mcp"]
  environment:
    - API_TOKEN=${BRIGHT_DATA_API_TOKEN}
    - PORT=8083
    - NODE_ENV=production
  ports:
    - "8083:8083"
  depends_on:
    redis:
      condition: service_healthy
```

### 6. Мониторинг и отладка

**Проверка статуса скрапинга:**
```python
# Общий статус
status = supervisor.get_scraping_status()
print(f"Всего задач: {status['total_tasks']}")
print(f"Успешных: {status['completed_tasks']}")
print(f"Неудачных: {status['failed_tasks']}")
print(f"Процент успеха: {status['success_rate']:.2%}")

# История задач
history = supervisor.get_scraping_history(limit=10)
for task in history:
    print(f"Задача {task['task_id']}: {task['status']}")
    if task.get('error'):
        print(f"Ошибка: {task['error']}")
```

**Очистка кеша:**
```python
# Очистка кеша скрапинга
result = supervisor.clear_scraping_cache()
print(f"Очищено элементов: {result['cleared_items']}")
```

---

## Специфика MCP серверов

### 1. Интеграция с внешними API

**Паттерн HTTP клиента:**

```python
import httpx
from typing import Any

class ExternalAPIClient:
    """Base client for external API integration."""
    
    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"}
        )
    
    async def request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        json: dict | None = None
    ) -> Any:
        """Make HTTP request with error handling."""
        try:
            response = await self.client.request(
                method=method,
                url=endpoint,
                params=params,
                json=json
            )
            response.raise_for_status()
            return response.json()
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e}")
            raise APIError(f"API returned {e.response.status_code}")
        
        except httpx.TimeoutException:
            logger.error("Request timeout")
            raise APIError("Request timeout")
        
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            raise APIError(f"Unexpected error: {e}")
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

# Использование
client = ExternalAPIClient(
    base_url="https://api.binance.com",
    api_key=settings.binance_api_key
)

async with client:
    data = await client.request("GET", "/api/v3/ticker/price", params={"symbol": "BTCUSDT"})
```

### 2. Обработка рыночных данных

**Нормализация свечей:**

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Candle:
    """Normalized candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @classmethod
    def from_binance(cls, data: list) -> "Candle":
        """Parse Binance candle format."""
        return cls(
            timestamp=datetime.fromtimestamp(data[0] / 1000),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5])
        )
    
    @classmethod
    def from_tradingview(cls, data: dict) -> "Candle":
        """Parse TradingView candle format."""
        return cls(
            timestamp=datetime.fromtimestamp(data["time"]),
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"]
        )

def normalize_candles(data: list, source: str) -> list[Candle]:
    """Normalize candles from different sources."""
    if source == "binance":
        return [Candle.from_binance(c) for c in data]
    elif source == "tradingview":
        return [Candle.from_tradingview(c) for c in data]
    else:
        raise ValueError(f"Unknown source: {source}")
```

### 3. Выполнение торговых операций

**Безопасное размещение ордеров:**

```python
from enum import Enum

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"

@dataclass
class Order:
    """Order representation."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None

class TradingService:
    """Safe trading operations."""
    
    def __init__(self, client: ExchangeClient, demo_mode: bool = True):
        self.client = client
        self.demo_mode = demo_mode
    
    async def place_order(self, order: Order) -> dict:
        """Place order with validation and safety checks."""
        # Валидация ордера
        self.validate_order(order)
        
        # Проверка баланса
        balance = await self.client.get_balance(order.symbol.split("USDT")[0])
        if order.side == OrderSide.SELL and balance < order.quantity:
            raise InsufficientBalanceError(f"Insufficient balance: {balance}")
        
        # Демо режим
        if self.demo_mode:
            logger.info(f"[DEMO] Would place order: {order}")
            return {"order_id": "demo_123", "status": "filled"}
        
        # Реальное размещение
        logger.warning(f"[LIVE] Placing order: {order}")
        result = await self.client.place_order(
            symbol=order.symbol,
            side=order.side.value,
            type=order.order_type.value,
            quantity=order.quantity,
            price=order.price,
        )
        
        # Логирование
        logger.info(f"Order placed: {result['order_id']}")
        
        return result
    
    def validate_order(self, order: Order):
        """Validate order parameters."""
        if order.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if order.order_type == OrderType.LIMIT and order.price is None:
            raise ValueError("Limit order requires price")
        
        if order.order_type == OrderType.STOP_LOSS and order.stop_price is None:
            raise ValueError("Stop loss order requires stop_price")
```

### 4. Бэктестинг и оптимизация

**Бэктестинг фреймворк:**

```python
@dataclass
class BacktestResult:
    """Backtest result metrics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: list[float]
    trades: list[dict]

class Strategy(Protocol):
    """Strategy interface."""
    def on_candle(self, candle: Candle) -> Signal | None: ...
    def on_trade_closed(self, trade: Trade): ...

class BacktestEngine:
    """Backtesting engine."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.equity_curve = [initial_capital]
        self.trades = []
        self.position = None
    
    def run(
        self,
        strategy: Strategy,
        candles: list[Candle],
        commission: float = 0.001
    ) -> BacktestResult:
        """Run backtest."""
        for candle in candles:
            # Генерация сигнала
            signal = strategy.on_candle(candle)
            
            # Открытие позиции
            if signal and not self.position:
                self.position = self.open_position(candle, signal)
            
            # Закрытие позиции
            elif self.position and self.should_close(candle, self.position):
                trade = self.close_position(candle, commission)
                self.trades.append(trade)
                strategy.on_trade_closed(trade)
            
            # Обновление equity
            equity = self.calculate_equity(candle)
            self.equity_curve.append(equity)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> BacktestResult:
        """Calculate performance metrics."""
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        winning_trades = [t for t in self.trades if t["pnl"] > 0]
        losing_trades = [t for t in self.trades if t["pnl"] <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        gross_profit = sum(t["pnl"] for t in winning_trades)
        gross_loss = abs(sum(t["pnl"] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        sharpe_ratio = self.calculate_sharpe()
        max_drawdown = self.calculate_max_drawdown()
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=self.annualize_return(total_return),
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            equity_curve=self.equity_curve,
            trades=self.trades
        )
```

---

## Шаблоны и примеры

### 1. Шаблон структуры проекта

**Минимальный MCP сервер:**

```
servers/project-mcp/
├── src/
│   └── project_mcp/
│       ├── __init__.py
│       ├── server.py           # FastMCP сервер
│       ├── config.py           # Pydantic Settings
│       ├── services/
│       │   ├── __init__.py
│       │   └── core.py         # Бизнес-логика
│       └── tools/
│           ├── __init__.py
│           ├── models.py       # Pydantic модели
│           └── handlers.py     # Обработчики инструментов
├── tests/
│   ├── conftest.py
│   ├── test_services/
│   └── test_tools/
├── examples/
│   └── example_usage.py
├── docs/
│   └── api.md
├── pyproject.toml
├── README.md
├── AGENTS.md
├── CHANGELOG.md
├── Dockerfile
├── .env.example
└── .gitignore
```

### 2. Примеры инструментов

**Базовый инструмент:**

```python
from fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("example-server")

class ExampleRequest(BaseModel):
    """Request model."""
    param: str = Field(..., description="Parameter description")
    limit: int = Field(default=10, ge=1, le=100)

@mcp.tool()
def example_tool(request: ExampleRequest) -> dict:
    """Performs an example operation with validation."""
    result = service.process(request.param, request.limit)
    return {
        "success": True,
        "data": result,
        "metadata": {
            "param": request.param,
            "limit": request.limit
        }
    }
```

**Health check:**

```python
import time

start_time = time.time()

@mcp.tool()
def health() -> dict:
    """Checks server health status."""
    return {
        "status": "healthy",
        "uptime": time.time() - start_time,
        "version": "1.0.0"
    }
```

**Version info:**

```python
@mcp.tool()
def version() -> dict:
    """Returns server version and configuration."""
    return {
        "version": "1.0.0",
        "name": "example-mcp",
        "config": {
            "timeout": settings.timeout,
            "max_retries": settings.max_retries
        },
        "features": [
            "feature_1",
            "feature_2"
        ]
    }
```

### 3. Примеры конфигурации

**pyproject.toml:**

```toml
[project]
name = "project-mcp"
version = "1.0.0"
description = "MCP server for X"
authors = [{name = "Your Name", email = "you@example.com"}]
requires-python = ">=3.11"
dependencies = [
    "fastmcp>=0.1.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "httpx>=0.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.scripts]
project-mcp = "project_mcp.server:main"

[tool.black]
line-length = 88
target-version = ['py311']

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
python_version = "3.11"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**.env.example:**

```bash
# API Configuration
PROJECT_API_KEY=your_api_key_here
PROJECT_API_SECRET=your_api_secret_here

# Server Configuration
PROJECT_HOST=0.0.0.0
PROJECT_PORT=8000
PROJECT_DEBUG=false

# Timeouts and Limits
PROJECT_TIMEOUT=30.0
PROJECT_MAX_RETRIES=3
PROJECT_RATE_LIMIT=100

# Logging
PROJECT_LOG_LEVEL=INFO
```

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy application
COPY src/ src/

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "-m", "project_mcp.server", "--host", "0.0.0.0", "--port", "8000"]
```

### 4. Примеры тестов

**Базовый тест:**

```python
import pytest
from project_mcp.services import CoreService

@pytest.fixture
def service():
    """Service fixture."""
    return CoreService()

def test_service_processes_data(service):
    """Test that service processes data correctly."""
    result = service.process("test", 10)
    
    assert result is not None
    assert len(result) <= 10
    assert all(isinstance(item, dict) for item in result)
```

**Асинхронный тест:**

```python
import pytest

@pytest.mark.asyncio
async def test_async_operation():
    """Test async operation."""
    result = await async_service.fetch_data("param")
    assert result["status"] == "success"
```

**MCP инструмент тест:**

```python
from fastmcp.testing import MCPTestClient

@pytest.mark.asyncio
async def test_mcp_tool():
    """Test MCP tool through client."""
    async with MCPTestClient(mcp) as client:
        response = await client.call_tool(
            "example_tool",
            {"param": "value", "limit": 5}
        )
        
        assert response["success"] is True
        assert "data" in response
        assert len(response["data"]) <= 5
```

---

## Заключение

Этот документ является живым стандартом и должен обновляться по мере развития экосистемы MCP.

**Ключевые принципы:**

1. **Единообразие**: Все MCP серверы следуют одним стандартам
2. **Качество**: Высокие требования к коду и тестированию
3. **Документация**: Актуальная и полная документация
4. **Безопасность**: Безопасность на всех уровнях
5. **Производительность**: Оптимизация и мониторинг

**Для вопросов и предложений:**

- Создавайте issues в репозитории
- Обновляйте AGENTS.md в своем проекте
- Следуйте Conventional Commits
- Поддерживайте CHANGELOG.md актуальным

---

**Версия:** 1.0  
**Последнее обновление:** 7 ноября 2025  
**Авторы:** HAL Team

