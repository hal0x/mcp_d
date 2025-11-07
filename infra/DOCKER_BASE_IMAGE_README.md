# MCP Docker Base Image

Этот проект содержит оптимизированную Docker инфраструктуру для MCP (Model Context Protocol) сервисов с использованием общего базового образа.

## Преимущества

- **Быстрая сборка**: Общие зависимости устанавливаются один раз в базовом образе
- **Меньший размер**: Избегаем дублирования зависимостей между сервисами
- **Консистентность**: Все сервисы используют одинаковую базовую среду
- **Кэширование**: Docker слои кэшируются между сборками разных сервисов

## Структура

```
├── Dockerfile.base              # Базовый образ
├── requirements-base.txt        # Python зависимости
├── docker-compose.mcp.yml  # Основной docker-compose с оптимизированными Dockerfile
└── scripts/
    ├── build-base-image.sh     # Скрипт сборки базового образа
    ├── update-dockerfiles.sh    # Скрипт обновления Dockerfile сервисов
    ├── migrate-to-base-image.sh # Полный скрипт миграции
    └── switch-dockerfiles.sh   # Скрипт переключения между Dockerfile
```

## Использование

### Быстрый старт

```bash
# Полная миграция одной командой
./scripts/migrate-to-base-image.sh
```

### Пошаговая миграция

#### 1. Сборка базового образа

```bash
# Собрать базовый образ
./scripts/build-base-image.sh
```

Этот скрипт:
- Собирает базовый образ `mcp-base:latest`
- Устанавливает общие системные пакеты и Python зависимости
- Создает тег с датой для версионирования

#### 2. Обновление Dockerfile сервисов

```bash
# Автоматически обновить все Dockerfile для использования базового образа
./scripts/update-dockerfiles.sh
```

Этот скрипт:
- Создает резервные копии оригинальных Dockerfile
- Генерирует новые Dockerfile.new файлы
- Извлекает только специфичные зависимости каждого сервиса

#### 3. Переключение между Dockerfile

```bash
# Показать текущий статус
./scripts/switch-dockerfiles.sh status

# Переключиться на новые Dockerfile (с базовым образом)
./scripts/switch-dockerfiles.sh new

# Переключиться обратно на старые Dockerfile
./scripts/switch-dockerfiles.sh old

# Очистить временные файлы
./scripts/switch-dockerfiles.sh cleanup
```

#### 4. Запуск сервисов

```bash
# Использовать основной docker-compose
docker-compose -f docker-compose.mcp.yml up -d

# Или собрать только базовый образ
docker-compose -f docker-compose.mcp.yml --profile build-only up mcp-base
```

## Общие зависимости

Базовый образ включает:

### Системные пакеты
- `build-essential` - компиляторы C/C++
- `curl` - HTTP клиент
- `libpq-dev` - PostgreSQL клиент
- `git` - система контроля версий
- `ca-certificates` - SSL сертификаты

### Python пакеты
- **Web framework**: fastapi, uvicorn
- **MCP protocol**: mcp, fastapi-mcp
- **Data validation**: pydantic, pydantic-settings
- **Database**: asyncpg, sqlalchemy, alembic
- **Caching**: redis
- **HTTP clients**: httpx, aiohttp, requests
- **Logging**: structlog
- **Configuration**: python-dotenv, PyYAML
- **Utilities**: click, tqdm, aiofiles
- **Data processing**: numpy, pandas

## Специфичные зависимости сервисов

Каждый сервис добавляет только свои уникальные зависимости:

### binance-mcp
- `python-binance==1.0.19`

### memory-mcp
- `chromadb>=0.4.0`
- `ollama>=0.1.0`
- `networkx>=3.0`
- `rank-bm25>=0.2.2`
- `scikit-learn>=1.3.0`
- `hdbscan==0.8.40`
- `natasha>=1.6.0`
- `qdrant-client>=1.7.0`

### tradingview-mcp
- Специфичные зависимости для TradingView API

## Миграция существующих сервисов

1. **Соберите базовый образ**:
   ```bash
   ./scripts/build-base-image.sh
   ```

2. **Обновите Dockerfile сервисов**:
   ```bash
   ./scripts/update-dockerfiles.sh
   ```

3. **Проверьте новые Dockerfile.new файлы** и замените оригинальные при необходимости

4. **Обновите docker-compose.yml** для использования новых Dockerfile

## Мониторинг и отладка

### Размер образов
```bash
# Проверить размер базового образа
docker images mcp-base

# Сравнить размеры до и после оптимизации
docker images | grep -E "(binance-mcp|memory-mcp|tradingview-mcp)"
```

### Логи сборки
```bash
# Подробные логи сборки базового образа
docker build -f infra/Dockerfile.base -t mcp-base:latest . --progress=plain

# Логи сборки конкретного сервиса
docker-compose -f docker-compose.mcp.yml build binance-mcp
```

## Troubleshooting

### Проблема: Базовый образ не найден
```bash
# Убедитесь, что базовый образ собран
docker images | grep mcp-base

# Если нет, соберите его
./scripts/build-base-image.sh
```

### Проблема: Ошибки в новых Dockerfile
```bash
# Восстановите оригинальные Dockerfile из резервных копий
find servers/ -name "Dockerfile.backup" -exec sh -c 'mv "$1" "${1%.backup}"' _ {} \;
```

### Проблема: Конфликты зависимостей
Проверьте файл `requirements-base.txt` и убедитесь, что версии пакетов совместимы с требованиями сервисов.

## Дальнейшие улучшения

1. **Многоэтапная сборка**: Разделить образы на runtime и build-time
2. **Distroless образы**: Использовать distroless для production
3. **Архитектурная поддержка**: Добавить поддержку ARM64
4. **Автоматические обновления**: CI/CD для обновления базового образа
5. **Мониторинг**: Интеграция с системами мониторинга Docker образов
