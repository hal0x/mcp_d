# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY pyproject.toml ./
COPY requirements.txt ./

# Устанавливаем Python зависимости
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements.txt

# Копируем исходный код
COPY . .

# Создаем необходимые директории
RUN mkdir -p db/artifacts db/index db/memory db/raw db/session

# Устанавливаем права доступа
RUN chmod -R 755 db/

# Создаем непривилегированного пользователя
RUN useradd -m -u 1000 halv1 && chown -R halv1:halv1 /app

# Переключаемся на непривилегированного пользователя
USER halv1

# Открываем порт
EXPOSE 8000

# Команда по умолчанию
CMD ["python", "main.py"]
