#!/bin/bash
# Скрипт для сборки базового Docker образа MCP

set -e

# Конфигурация
BASE_IMAGE_NAME="mcp-base"
BASE_IMAGE_TAG="latest"
DOCKERFILE_PATH="Dockerfile.base.prod"
DEV_DOCKERFILE_PATH="Dockerfile.base.dev"

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
    error "Dockerfile.base.prod не найден в текущей директории"
fi

if [ ! -f "$DEV_DOCKERFILE_PATH" ]; then
    error "Dockerfile.base.dev не найден в текущей директории"
fi

# Проверка наличия requirements-base-prod.txt
if [ ! -f "requirements-base-prod.txt" ]; then
    error "requirements-base-prod.txt не найден в текущей директории"
fi

# Проверка наличия requirements-base-dev.txt
if [ ! -f "requirements-base-dev.txt" ]; then
    error "requirements-base-dev.txt не найден в текущей директории"
fi

log "Начинаем сборку базовых образов MCP..."

# Сборка production образа
log "Сборка production образа ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}"
if docker build -f "$DOCKERFILE_PATH" -t "${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}" .; then
    log "Production базовый образ успешно собран!"
else
    error "Ошибка при сборке production базового образа"
fi

# Сборка dev образа
log "Сборка dev образа ${BASE_IMAGE_NAME}:dev"
if docker build -f "$DEV_DOCKERFILE_PATH" -t "${BASE_IMAGE_NAME}:dev" .; then
    log "Dev базовый образ успешно собран!"
else
    error "Ошибка при сборке dev базового образа"
fi

# Проверка размера образов
PROD_IMAGE_SIZE=$(docker images --format "table {{.Size}}" "${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}" | tail -n 1)
DEV_IMAGE_SIZE=$(docker images --format "table {{.Size}}" "${BASE_IMAGE_NAME}:dev" | tail -n 1)
log "Размер production образа: $PROD_IMAGE_SIZE"
log "Размер dev образа: $DEV_IMAGE_SIZE"

# Создание тегов с датой для версионирования
DATE_TAG=$(date +'%Y%m%d-%H%M%S')
docker tag "${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}" "${BASE_IMAGE_NAME}:${DATE_TAG}"
docker tag "${BASE_IMAGE_NAME}:dev" "${BASE_IMAGE_NAME}:dev-${DATE_TAG}"
log "Созданы теги с датой: ${BASE_IMAGE_NAME}:${DATE_TAG} и ${BASE_IMAGE_NAME}:dev-${DATE_TAG}"

# Показать информацию об образах
log "Информация об образах:"
docker images "${BASE_IMAGE_NAME}"

log "Базовые образы готовы к использованию!"
log "Production: 'FROM ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}'"
log "Development: 'FROM ${BASE_IMAGE_NAME}:dev'"
