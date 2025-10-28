# syntax=docker/dockerfile:1.6
# Обновленный Dockerfile для memory-mcp с использованием базового образа
FROM mcp-base:latest

# Переключение на root для установки пакетов
USER root

# Копируем специфичные зависимости сервиса
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts

# Установка пакета
RUN pip install --no-cache-dir -e .

# Переключение обратно на непривилегированного пользователя
USER mcp

# Экспорт порта сервиса
EXPOSE 8050

# Команда запуска сервиса
CMD ["python", "scripts/mcp_server.py", "--host", "0.0.0.0", "--port", "8050"]
