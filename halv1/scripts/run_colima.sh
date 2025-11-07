#!/bin/bash

# Скрипт для запуска HALv1 с Colima (оптимизированная версия)

set -e

echo "🚀 Запуск HALv1 с Colima..."

# Проверяем наличие Colima
if ! command -v colima &> /dev/null; then
    echo "❌ Colima не установлена. Установите: brew install colima"
    exit 1
fi

# Проверяем статус Colima
if ! colima status &> /dev/null; then
    echo "🔄 Colima не запущена. Запускаем..."
    
    # Определяем оптимальные настройки для Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        echo "🍎 Обнаружен Apple Silicon, используем оптимизированные настройки..."
        colima start --cpu 4 --memory 8 --disk 100 --arch aarch64 --mount-type virtiofs
    else
        echo "💻 Intel Mac, используем стандартные настройки..."
        colima start --cpu 2 --memory 4 --disk 50
    fi
    
    echo "✅ Colima запущена"
else
    echo "✅ Colima уже запущена"
fi

# Проверяем Docker
if ! docker info &> /dev/null; then
    echo "❌ Docker не доступен через Colima"
    exit 1
fi

echo "🐳 Docker доступен через Colima"

# Создаем .env файл, если его нет
if [ ! -f .env ]; then
    echo "📝 Создаем .env файл..."
    cat > .env << EOF
# Colima настройки
DOCKER_NETWORK_MODE=host
DOCKER_ALLOW_INTERNET=true

# Telegram настройки (заполните своими данными)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_SUMMARY_CHAT_ID=134432210
TELEGRAM_GOAL_CHAT_ID=134432210

# LLM настройки
LLM_PROVIDER=lmstudio
LLM_HOST=127.0.0.1
LLM_PORT=1234
LLM_API_KEY=your_api_key_here

# Executor настройки
EXECUTOR_PROVIDER=docker

# Интернет настройки
INTERNET_USER_AGENT=halv1-bot/1.0
INTERNET_MAX_RETRIES=3
EOF
    echo "⚠️  Отредактируйте .env файл с вашими настройками перед запуском!"
    echo "🔑 Обязательно заполните TELEGRAM_BOT_TOKEN!"
    exit 1
fi

# Проверяем наличие необходимых переменных
if [ -z "$TELEGRAM_BOT_TOKEN" ] && ! grep -q "TELEGRAM_BOT_TOKEN" .env; then
    echo "❌ TELEGRAM_BOT_TOKEN не установлен. Добавьте его в .env файл."
    exit 1
fi

# Тестируем сеть
echo "🧪 Тестируем сетевой доступ..."
if python test_docker_network.py > /dev/null 2>&1; then
    echo "✅ Сеть работает корректно"
else
    echo "⚠️  Проблемы с сетью, но продолжаем..."
fi

# Сборка и запуск
echo "🔨 Собираем Docker образ..."
docker compose build

echo "🚀 Запускаем HALv1..."
docker compose up

echo "✅ HALv1 запущен с Colima!"
echo "📱 Telegram бот должен быть доступен"
echo "🌐 Интернет-функции включены"
echo "🔍 Для остановки: docker compose down"
echo "🛑 Для остановки Colima: colima stop"
