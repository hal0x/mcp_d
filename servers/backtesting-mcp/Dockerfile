# syntax=docker/dockerfile:1.6
# Обновленный Dockerfile для backtesting-mcp с использованием базового образа
FROM mcp-base:latest

# Переключение на root для установки пакетов
USER root

# Установка uv если не установлен в базовом образе
RUN pip install uv

# Копируем все файлы проекта
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY mcp_server.py ./

# Установка зависимостей через uv (кэшируется при неизменных pyproject.toml)
RUN uv sync --frozen --no-dev && \
    chmod -R 755 /app/.venv

# Установка переменных окружения
ENV PATH="/app/.venv/bin:${PATH}"

# Переключение обратно на непривилегированного пользователя
USER mcp

# Экспорт порта сервиса
EXPOSE 8082

# Команда запуска сервиса
CMD ["python", "mcp_server.py", "--host", "0.0.0.0", "--port", "8082"]
