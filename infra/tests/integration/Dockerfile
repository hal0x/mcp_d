FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка Python зависимостей
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Создание директории для результатов тестов
RUN mkdir -p /tests/results

# Рабочая директория
WORKDIR /tests

# Команда по умолчанию
CMD ["pytest", "/tests", "-v", "--tb=short", "--junitxml=/tests/results/results.xml"]
