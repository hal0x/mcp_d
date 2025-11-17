# syntax=docker/dockerfile:1.6

FROM mcp-base:latest as deps

# Переключение на root для установки пакетов
USER root

# Рабочая директория
WORKDIR /app

# Настройки окружения для оптимизации uv
ENV UV_HTTP_TIMEOUT=300 \
    UV_NO_CACHE=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# При изменении pyproject.toml или uv.lock этот слой пересоберется
COPY pyproject.toml uv.lock ./

RUN mkdir -p ./wheels ./src
COPY wheels/ ./wheels/

# Устанавливаем зависимости сервиса через uv (используем кэш Docker)
# Этот слой будет кэшироваться до изменения pyproject.toml или uv.lock
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -d "./wheels" ] && [ "$(ls -A ./wheels 2>/dev/null)" ]; then \
    echo "Используем локальные wheels для ускорения сборки"; \
    uv sync --frozen --no-dev --find-links ./wheels && \
    uv pip install setuptools>=61.0; \
    else \
    echo "Локальные wheels не найдены, используем стандартную установку"; \
    uv sync --frozen --no-dev && \
    uv pip install setuptools>=61.0; \
    fi

# Этап 2: Финальный образ с кодом
FROM deps as final

ENV PATH="/app/.venv/bin:${PATH}"

# Копируем все файлы кода в одном слое для уменьшения количества слоев
# Этот слой пересобирается при изменении любого файла кода
COPY src/ ./src/
COPY run_server.py mcp_server.py README.md ./

# Создаём необходимые директории с правильными правами доступа
# Переключаемся на root для создания директорий
USER root
RUN mkdir -p /app/artifacts/reports /app/artifacts/sessions /app/config/entity_dictionaries && \
    chown -R mcp:mcp /app/artifacts /app/config && \
    chmod -R 755 /app/artifacts /app/config

# Переключение обратно на непривилегированного пользователя
USER mcp

# Экспорт порта сервиса
EXPOSE 8050

# Healthcheck для мониторинга состояния контейнера
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('localhost', 8050), 3).close()" || exit 1

# Команда запуска сервиса (используем новый HTTP сервер)
CMD ["python", "run_server.py"]
