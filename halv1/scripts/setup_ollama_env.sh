#!/bin/bash
# Скрипт для настройки переменных окружения Ollama
# Решает проблемы с выгрузкой раннера между вызовами

echo "Настройка переменных окружения Ollama..."

# Экспортируем переменные для текущей сессии
export OLLAMA_KEEP_ALIVE=300
export OLLAMA_USE_MMAP=1

# Добавляем в ~/.bashrc и ~/.zshrc для постоянного использования
for shell_file in ~/.bashrc ~/.zshrc; do
    if [ -f "$shell_file" ]; then
        # Проверяем, не добавлены ли уже эти переменные
        if ! grep -q "OLLAMA_KEEP_ALIVE" "$shell_file"; then
            echo "" >> "$shell_file"
            echo "# Ollama optimization settings" >> "$shell_file"
            echo "export OLLAMA_KEEP_ALIVE=300" >> "$shell_file"
            echo "export OLLAMA_USE_MMAP=1" >> "$shell_file"
            echo "Добавлено в $shell_file"
        else
            echo "Переменные уже настроены в $shell_file"
        fi
    fi
done

echo "Переменные окружения настроены:"
echo "  OLLAMA_KEEP_ALIVE=300 (держать модель в памяти 5 минут)"
echo "  OLLAMA_USE_MMAP=1 (использовать mmap для экономии памяти)"
echo ""
echo "Для применения изменений перезапустите терминал или выполните:"
echo "  source ~/.zshrc  # или source ~/.bashrc"
