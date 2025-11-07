# Детальный анализ проблем MCP сервисов

## Дата анализа: 2025-10-22

## Статус сервисов

### ✅ binance-mcp (Полностью работает)
- **Статус**: Здоров
- **Проблем**: Нет
- **Конфигурация**: DEMO режим, API ключи настроены
- **MCP сервер**: `/app/mcp_server.py` существует и работает

### ❌ tradingview-mcp (Проблемы с конфигурацией)

**Проблема 1: Отсутствует mcp_server.py в корне**
```bash
# В /app/ нет mcp_server.py
ls /app/
# Есть только: README.md, build/, pyproject.toml, src/
```

**Проблема 2: Неправильная структура импортов**
```python
# Ошибка при импорте конфигурации
from tradingview_mcp.config import get_config
# ImportError: cannot import name 'get_config'
```

**Проблема 3: MCP сервер в неправильном месте**
```bash
# Сервер находится в build/lib/, а не в корне
find /app -name "server.py"
# /app/build/lib/tradingview_mcp/server.py
```

**Решение для tradingview-mcp:**
1. Создать `/app/mcp_server.py` как точку входа
2. Исправить импорты конфигурации
3. Обновить Dockerfile для правильной структуры

### ❌ memory-mcp (Circular import)

**Проблема 1: Circular import в модуле mcp**
```python
# Ошибка circular import
from memory_mcp.mcp.server import main
# ImportError: cannot import name 'bind' from partially initialized module
```

**Проблема 2: Отсутствует база данных**
```bash
# База данных не создана
ls /data/memory_graph.db
# File not found
```

**Проблема 3: Неправильная структура пакета**
```
memory_mcp/mcp/__init__.py пытается импортировать из server.py
server.py пытается импортировать из __init__.py
```

**Решение для memory-mcp:**
1. Исправить circular import в `memory_mcp/mcp/__init__.py`
2. Создать базу данных при запуске
3. Реструктурировать импорты

### ❌ shell-mcp (Отсутствует MCP сервер)

**Проблема 1: Нет mcp_server.py**
```bash
ls /app/
# Нет файла mcp_server.py
```

**Проблема 2: Неизвестная структура проекта**
- Сервис запущен, но структура неясна
- Healthcheck ищет процесс 'shell-mcp'

**Решение для shell-mcp:**
1. Найти правильную точку входа
2. Создать mcp_server.py если нужно
3. Обновить healthcheck

### ❌ backtesting-mcp (Отсутствует MCP сервер)

**Проблема 1: Нет mcp_server.py**
```bash
ls /app/
# Нет файла mcp_server.py
```

**Проблема 2: Неизвестная структура проекта**
- Сервис запущен, но структура неясна
- Healthcheck ищет процесс 'backtesting-mcp'

**Решение для backtesting-mcp:**
1. Найти правильную точку входа
2. Создать mcp_server.py если нужно
3. Обновить healthcheck

## Общие проблемы

### 1. Несогласованность структуры проектов
- **binance-mcp**: Имеет `/app/mcp_server.py` ✅
- **Остальные**: Нет единой точки входа ❌

### 2. Различные транспорты
```bash
# Из анализа интеграции:
DEFAULT_TRANSPORT: binance-mcp=stdio, tradingview-mcp=streamable-http
```

### 3. Проблемы с healthcheck
- Некоторые healthcheck ищут процессы, которых нет
- Некоторые проверяют порты, но сервисы в stdio режиме

## Приоритеты исправления

### Высокий приоритет
1. **memory-mcp**: Исправить circular import
2. **tradingview-mcp**: Создать правильную точку входа

### Средний приоритет  
3. **shell-mcp**: Найти/создать MCP сервер
4. **backtesting-mcp**: Найти/создать MCP сервер

### Низкий приоритет
5. Стандартизировать структуру всех проектов
6. Унифицировать healthcheck'и

## Конкретные команды для исправления

### Для tradingview-mcp:
```bash
# 1. Создать точку входа
docker exec tradingview-mcp-1 bash -c "
cat > /app/mcp_server.py << 'EOF'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app/build/lib')
from tradingview_mcp.server import main
if __name__ == '__main__':
    main()
EOF
chmod +x /app/mcp_server.py
"
```

### Для memory-mcp:
```bash
# 1. Исправить circular import
# Нужно отредактировать /usr/local/lib/python3.11/site-packages/memory_mcp/mcp/__init__.py
# Убрать импорт bind из __init__.py
```

### Для shell-mcp и backtesting-mcp:
```bash
# 1. Найти структуру проекта
docker exec shell-mcp-1 find /app -name "*.py" | head -10
docker exec backtesting-mcp-1 find /app -name "*.py" | head -10
```

## Тестирование исправлений

После каждого исправления запускать:
```bash
cd tests/integration
python test_simple_real.py
```

## Ожидаемый результат

После исправления всех проблем:
```json
{
  "binance-mcp": {"healthy": true},
  "tradingview-mcp": {"healthy": true},
  "memory-mcp": {"healthy": true}, 
  "shell-mcp": {"healthy": true},
  "backtesting-mcp": {"healthy": true}
}
```

Здоровых сервисов: 5/5 ✅
