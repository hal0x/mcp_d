# syntax=docker/dockerfile:1.6
# Обновленный Dockerfile для supervisor-mcp с использованием базового образа
FROM mcp-base:latest

# Переключение на root для установки пакетов
USER root

# Копируем специфичные зависимости сервиса
COPY src/ src/
COPY pyproject.toml .

# Установка пакета
RUN pip install --no-cache-dir .

# Переключение обратно на непривилегированного пользователя
USER mcp

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "-m", "supervisor_mcp.server", "--host", "0.0.0.0", "--port", "8000"]
