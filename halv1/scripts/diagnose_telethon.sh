#!/bin/bash
# Диагностика проблем с Telethon

echo "🔍 Диагностика Telethon"
echo "======================="

# Активируем виртуальное окружение
source venv/bin/activate

echo "1. Проверка файла сессии..."
if [ -f "db/session/user.session" ]; then
    echo "✅ Файл сессии существует"
    ls -la db/session/user.session
else
    echo "❌ Файл сессии не найден"
    exit 1
fi

echo ""
echo "2. Проверка блокировки файла сессии..."
if lsof db/session/user.session 2>/dev/null; then
    echo "⚠️  Файл сессии заблокирован процессами:"
    lsof db/session/user.session
    echo ""
    echo "Процессы, использующие файл сессии:"
    ps aux | grep -E "(python.*main\.py|telethon)" | grep -v grep
else
    echo "✅ Файл сессии не заблокирован"
fi

echo ""
echo "3. Проверка переменных окружения..."
if [ -f ".env" ]; then
    echo "✅ Файл .env найден"
    echo "TELETHON_API_ID: $(grep TELETHON_API_ID .env | cut -d'=' -f2)"
    echo "TELETHON_API_HASH: $(grep TELETHON_API_HASH .env | cut -d'=' -f2 | cut -c1-8)..."
else
    echo "❌ Файл .env не найден"
fi

echo ""
echo "4. Тест авторизации..."
python scripts/telethon_auth_simple.py

echo ""
echo "5. Рекомендации:"
if lsof db/session/user.session 2>/dev/null; then
    echo "  - Остановите бот: pkill -f 'python.*main.py'"
    echo "  - Или используйте: ./scripts/restart_bot_safe.sh"
else
    echo "  - Сессия готова к использованию"
    echo "  - Запустите бот: python main.py"
fi
