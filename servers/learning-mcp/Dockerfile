# syntax=docker/dockerfile:1.6
# Обновленный Dockerfile для learning-mcp с использованием базового образа
FROM mcp-base:latest

# Переключение на root для установки пакетов
USER root

# Копируем специфичные зависимости сервиса
COPY pyproject.toml .

# Установка пакета
RUN pip install --no-cache-dir -e .

# Copy application
COPY src/ src/

# Переключение обратно на непривилегированного пользователя
USER mcp

# Expose port
EXPOSE 8003

# Run server
CMD ["python", "-m", "learning_mcp.server", "--host", "0.0.0.0", "--port", "8003"]
