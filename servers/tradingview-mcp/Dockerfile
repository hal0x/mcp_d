# syntax=docker/dockerfile:1.6
# Оптимизированный Dockerfile для tradingview-mcp
FROM mcp-base:latest

# Переключение на root для установки пакетов
USER root

# Установка uv если не установлен в базовом образе
RUN pip install uv

# Копируем все файлы проекта
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY run_server.py ./
COPY mcp_server.py ./
RUN mkdir -p configs
COPY configs/scanner_scheduler.yaml ./configs/

# Установка зависимостей через uv (кэшируется при неизменных pyproject.toml/uv.lock)
# Примечание: Docker будет использовать кэш для этого слоя если pyproject.toml/uv.lock не изменились
RUN uv sync --frozen --no-dev && \
    chmod -R 755 /app/.venv

# Установка переменных окружения
ENV PATH="/app/.venv/bin:${PATH}"

# Копируем файл конфигурации планировщика в финальное место
RUN mkdir -p /usr/local/lib/python3.11/site-packages/tradingview_mcp/configs && \
    cp configs/scanner_scheduler.yaml /usr/local/lib/python3.11/site-packages/tradingview_mcp/configs/scanner_scheduler.yaml

# Переключение обратно на непривилегированного пользователя
USER mcp

# Экспорт порта сервиса
EXPOSE 8000

# Команда запуска сервиса
CMD ["python", "run_server.py", "--host", "0.0.0.0", "--port", "8000"]
