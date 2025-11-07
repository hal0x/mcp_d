#!/bin/bash
# Скрипт для сборки базового Docker образа MCP

set -e

# Конфигурация
BASE_IMAGE_NAME="mcp-base"
BASE_IMAGE_TAG="latest"
DOCKERFILE_PATH="Dockerfile.base"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функция для логирования
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Проверка наличия Docker
if ! command -v docker &> /dev/null; then
    error "Docker не установлен или недоступен"
fi

# Проверка наличия Dockerfile
if [ ! -f "$DOCKERFILE_PATH" ]; then
    error "Dockerfile.base не найден в текущей директории"
fi

# Проверка наличия requirements-base.txt
if [ ! -f "requirements-base.txt" ]; then
    error "requirements-base.txt не найден в текущей директории"
fi

log "Начинаем сборку базового образа MCP..."

# Сборка базового образа
log "Сборка базового образа ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}"
if docker build -f "$DOCKERFILE_PATH" -t "${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}" .; then
    log "Базовый образ успешно собран!"
else
    error "Ошибка при сборке базового образа"
fi

# Проверка размера образа
IMAGE_SIZE=$(docker images --format "table {{.Size}}" "${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}" | tail -n 1)
log "Размер образа: $IMAGE_SIZE"

# Создание тега с датой для версионирования
DATE_TAG=$(date +'%Y%m%d-%H%M%S')
docker tag "${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}" "${BASE_IMAGE_NAME}:${DATE_TAG}"
log "Создан тег с датой: ${BASE_IMAGE_NAME}:${DATE_TAG}"

# Показать информацию об образе
log "Информация об образе:"
docker images "${BASE_IMAGE_NAME}"

log "Базовый образ готов к использованию!"
log "Использование: 'FROM ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}'"
