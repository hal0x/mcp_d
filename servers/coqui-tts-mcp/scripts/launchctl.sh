#!/bin/bash
# Скрипт для управления coqui-tts-mcp через launchctl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PLIST_NAME="com.hal.coqui-tts-mcp"
PLIST_FILE="$PROJECT_DIR/$PLIST_NAME.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
LAUNCH_AGENTS_PLIST="$LAUNCH_AGENTS_DIR/$PLIST_NAME.plist"

# Создаём директорию для логов
mkdir -p "$PROJECT_DIR/logs"

# Функция для установки
install() {
    echo "📦 Установка launchd service для coqui-tts-mcp..."
    
    # Проверяем существование виртуального окружения
    if [ ! -d "$PROJECT_DIR/.venv" ]; then
        echo "❌ Виртуальное окружение не найдено. Создайте его сначала:"
        echo "   cd $PROJECT_DIR && python3 -m venv .venv"
        exit 1
    fi
    
    # Проверяем установку пакета
    if ! "$PROJECT_DIR/.venv/bin/python" -c "import coqui_tts_mcp" 2>/dev/null; then
        echo "⚠️  Пакет не установлен. Устанавливаю..."
        "$PROJECT_DIR/.venv/bin/pip" install -e "$PROJECT_DIR"
    fi
    
    # Копируем plist файл
    if [ ! -f "$PLIST_FILE" ]; then
        echo "❌ Plist файл не найден: $PLIST_FILE"
        exit 1
    fi
    
    # Обновляем пути в plist файле (на случай если запускаем из другого места)
    sed "s|/Users/hal/projects/mcp|$(dirname "$(dirname "$PROJECT_DIR")")|g" "$PLIST_FILE" > "$LAUNCH_AGENTS_PLIST"
    
    # Загружаем сервис
    if launchctl list | grep -q "$PLIST_NAME"; then
        echo "⚠️  Сервис уже загружен. Перезагружаю..."
        launchctl unload "$LAUNCH_AGENTS_PLIST" 2>/dev/null || true
    fi
    
    launchctl load "$LAUNCH_AGENTS_PLIST"
    echo "✅ Сервис установлен и запущен!"
    echo "   Проверьте статус: $0 status"
}

# Функция для удаления
uninstall() {
    echo "🗑️  Удаление launchd service для coqui-tts-mcp..."
    
    if [ -f "$LAUNCH_AGENTS_PLIST" ]; then
        if launchctl list | grep -q "$PLIST_NAME"; then
            launchctl unload "$LAUNCH_AGENTS_PLIST" 2>/dev/null || true
        fi
        rm -f "$LAUNCH_AGENTS_PLIST"
        echo "✅ Сервис удалён!"
    else
        echo "⚠️  Сервис не найден в $LAUNCH_AGENTS_DIR"
    fi
}

# Функция для проверки статуса
status() {
    echo "📊 Статус сервиса coqui-tts-mcp:"
    echo ""
    
    if launchctl list | grep -q "$PLIST_NAME"; then
        echo "✅ Сервис загружен"
        launchctl list | grep "$PLIST_NAME"
        echo ""
        echo "📋 Логи:"
        echo "   stdout: $PROJECT_DIR/logs/stdout.log"
        echo "   stderr: $PROJECT_DIR/logs/stderr.log"
        echo ""
        echo "🔍 Последние строки из логов:"
        echo "--- stdout ---"
        tail -n 5 "$PROJECT_DIR/logs/stdout.log" 2>/dev/null || echo "   (лог пуст)"
        echo "--- stderr ---"
        tail -n 5 "$PROJECT_DIR/logs/stderr.log" 2>/dev/null || echo "   (лог пуст)"
    else
        echo "❌ Сервис не загружен"
    fi
}

# Функция для перезапуска
restart() {
    echo "🔄 Перезапуск сервиса..."
    if [ -f "$LAUNCH_AGENTS_PLIST" ]; then
        launchctl unload "$LAUNCH_AGENTS_PLIST" 2>/dev/null || true
        sleep 1
        launchctl load "$LAUNCH_AGENTS_PLIST"
        echo "✅ Сервис перезапущен!"
    else
        echo "❌ Сервис не установлен. Используйте: $0 install"
        exit 1
    fi
}

# Функция для остановки
stop() {
    echo "⏹️  Остановка сервиса..."
    if [ -f "$LAUNCH_AGENTS_PLIST" ]; then
        launchctl unload "$LAUNCH_AGENTS_PLIST" 2>/dev/null || true
        echo "✅ Сервис остановлен!"
    else
        echo "❌ Сервис не установлен"
        exit 1
    fi
}

# Функция для запуска
start() {
    echo "▶️  Запуск сервиса..."
    if [ -f "$LAUNCH_AGENTS_PLIST" ]; then
        launchctl load "$LAUNCH_AGENTS_PLIST" 2>/dev/null || true
        echo "✅ Сервис запущен!"
    else
        echo "❌ Сервис не установлен. Используйте: $0 install"
        exit 1
    fi
}

# Функция для просмотра логов
logs() {
    echo "📋 Логи сервиса coqui-tts-mcp:"
    echo ""
    if [ -f "$PROJECT_DIR/logs/stdout.log" ]; then
        echo "=== stdout.log ==="
        tail -n 50 "$PROJECT_DIR/logs/stdout.log"
    else
        echo "stdout.log не найден"
    fi
    echo ""
    if [ -f "$PROJECT_DIR/logs/stderr.log" ]; then
        echo "=== stderr.log ==="
        tail -n 50 "$PROJECT_DIR/logs/stderr.log"
    else
        echo "stderr.log не найден"
    fi
}

# Главная функция
case "${1:-}" in
    install)
        install
        ;;
    uninstall)
        uninstall
        ;;
    status)
        status
        ;;
    restart)
        restart
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    logs)
        logs
        ;;
    *)
        echo "Использование: $0 {install|uninstall|start|stop|restart|status|logs}"
        echo ""
        echo "Команды:"
        echo "  install   - Установить и запустить сервис"
        echo "  uninstall - Удалить сервис"
        echo "  start     - Запустить сервис"
        echo "  stop      - Остановить сервис"
        echo "  restart   - Перезапустить сервис"
        echo "  status    - Показать статус сервиса"
        echo "  logs      - Показать логи"
        exit 1
        ;;
esac

