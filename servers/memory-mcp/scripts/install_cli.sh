#!/bin/bash
"""
Скрипт установки memory_mcp CLI
"""

echo "🚀 Установка memory_mcp CLI..."

# Проверяем Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите Python 3.8+"
    exit 1
fi

# Проверяем pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 не найден. Установите pip"
    exit 1
fi

# Устанавливаем зависимости
echo "📦 Установка зависимостей..."
pip3 install -r requirements.txt

# Устанавливаем CLI
echo "🔧 Установка CLI..."
pip3 install -e .

# Проверяем установку
if command -v memory_mcp &> /dev/null; then
    echo "✅ memory_mcp CLI установлен успешно!"
    echo ""
    echo "🎉 Использование:"
    echo "  memory_mcp --help"
    echo "  memory_mcp check"
    echo "  memory_mcp mcp serve"
    echo ""
    echo "📖 Документация: README_UNIFIED.md"
else
    echo "❌ Ошибка при установке CLI"
    exit 1
fi
