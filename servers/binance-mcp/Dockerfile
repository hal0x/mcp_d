# syntax=docker/dockerfile:1.6
# Обновленный Dockerfile для binance-mcp с использованием базового образа
FROM mcp-base:latest

# Переключение на root для установки пакетов
USER root

# Копируем специфичные зависимости сервиса
COPY pyproject.toml uv.lock ./
COPY docs/README.md docs/README.md

# Установка специфичных зависимостей сервиса через uv
RUN uv sync --frozen --no-dev

# Копируем исходный код сервиса
COPY . .

# Установка переменных окружения
ENV PATH="/app/.venv/bin:${PATH}"

# Переключение обратно на непривилегированного пользователя
USER mcp

# Экспорт порта сервиса
EXPOSE 8000

# Команда запуска сервиса
CMD ["python", "mcp_server.py", "--host", "0.0.0.0", "--port", "8000"]
