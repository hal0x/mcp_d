#!/bin/bash

echo "🚀 Установка MCP сервера для Telegram дампов..."

# Проверяем наличие Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Пожалуйста, установите Python 3.8+"
    exit 1
fi

# Проверяем версию Python
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Требуется Python 3.8+, найден $python_version"
    exit 1
fi

echo "✅ Python $python_version найден"

# Проверяем наличие Ollama
if ! command -v ollama &> /dev/null; then
    echo "⚠️ Ollama не найден. Установите Ollama с https://ollama.ai"
    echo "После установки выполните:"
    echo "  ollama serve"
    echo "  ollama pull dengcao/Qwen3-Embedding-4B:Q5_K_M"
    echo ""
    read -p "Продолжить установку Python зависимостей? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Ollama найден"

    # Проверяем, запущен ли Ollama сервер
    if ! curl -s http://localhost:11434/api/version &> /dev/null; then
        echo "⚠️ Ollama сервер не запущен. Запустите: ollama serve"
    else
        echo "✅ Ollama сервер запущен"
    fi
fi

# Создаем виртуальное окружение
echo "📦 Создание виртуального окружения..."
python3 -m venv venv
source venv/bin/activate

# Устанавливаем зависимости
echo "📥 Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Установка завершена!"
echo ""
echo "Следующие шаги:"
echo "1. Убедитесь, что Ollama запущен: ollama serve"
echo "2. Загрузите модель: ollama pull dengcao/Qwen3-Embedding-4B:Q5_K_M"
echo "3. Проверьте подключение: python check_ollama.py"
echo "4. Запустите сервер: python run_server.py"
echo ""
echo "Для тестирования выполните:"
echo "  python test_server.py"
