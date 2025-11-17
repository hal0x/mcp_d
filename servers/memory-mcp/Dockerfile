# syntax=docker/dockerfile:1.6

FROM mcp-base:latest as deps

USER root
WORKDIR /app

ENV UV_HTTP_TIMEOUT=300 \
    UV_NO_CACHE=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml uv.lock ./

RUN mkdir -p ./wheels ./src
COPY wheels/ ./wheels/

# Кэш зависимостей: слой пересобирается только при изменении pyproject.toml/uv.lock
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -d "./wheels" ] && [ "$(ls -A ./wheels 2>/dev/null)" ]; then \
    uv sync --frozen --no-dev --find-links ./wheels && \
    uv pip install setuptools>=61.0; \
    else \
    uv sync --frozen --no-dev && \
    uv pip install setuptools>=61.0; \
    fi

FROM deps as final

ENV PATH="/app/.venv/bin:${PATH}"

# Код копируется отдельным слоем для оптимизации кэширования
COPY src/ ./src/
COPY run_server.py mcp_server.py README.md ./

USER root
RUN mkdir -p /app/artifacts/reports /app/artifacts/sessions /app/config/entity_dictionaries && \
    chown -R mcp:mcp /app/artifacts /app/config && \
    chmod -R 755 /app/artifacts /app/config

USER mcp

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('localhost', 8050), 3).close()" || exit 1

CMD ["python", "run_server.py"]
