# Руководство по запуску TradingView MCP Server

## Проблема
Сервер tradingview-mcp не запускался из-за отсутствующего файла `config.py` и проблем с относительными импортами.

## Решение
1. **Создан недостающий файл `config.py`** с классом `Settings` и функцией `get_settings`
2. **Исправлены импорты** в `server.py` - заменены относительные импорты на абсолютные
3. **Создан скрипт запуска** `run_server.py` для удобного запуска сервера

## Способы запуска

### 1. Через uv (рекомендуемый)
```bash
cd /Users/hal/projects/mcp/tradingview-mcp
uv run tradingview-mcp
```

### 2. Как модуль Python
```bash
cd /Users/hal/projects/mcp/tradingview-mcp
uv run python -m tradingview_mcp.server
```

### 3. Через созданный скрипт
```bash
cd /Users/hal/projects/mcp/tradingview-mcp
uv run python run_server.py
```

### 4. С MCP Inspector (для разработки)
```bash
cd /Users/hal/projects/mcp/tradingview-mcp
uv run mcp dev run_server.py
```

## Что было исправлено

1. **Отсутствующий файл `config.py`**:
   - Создан класс `Settings` с настройками сервера
   - Добавлена функция `get_settings()` для получения глобального экземпляра настроек

2. **Проблемы с импортами**:
   - Заменен `from .config import Settings, get_settings` на `from tradingview_mcp.config import Settings, get_settings`
   - Это позволяет запускать сервер как модуль

3. **Скрипт запуска**:
   - Создан `run_server.py` который правильно настраивает Python path
   - Импортирует объект `mcp` напрямую для совместимости с MCP Inspector

## Статус
✅ Сервер теперь запускается успешно всеми способами
✅ MCP Inspector работает корректно
✅ Все зависимости установлены и работают
