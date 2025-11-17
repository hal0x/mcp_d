# Coqui TTS MCP Server

MCP сервер для синтеза речи с использованием **Coqui TTS** с поддержкой **Metal (MPS)** ускорения на Mac M-серии (включая M4).

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Создаём и активируем виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# Обновляем менеджер пакетов
pip install -U pip

# Устанавливаем зависимости
pip install -e .

# Или используя uv
uv sync
```

### 2. Проверка Metal (MPS)

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
# → True  ✅
```

### 3. Запуск сервера

```bash
# HTTP режим (по умолчанию)
uv run coqui-tts-mcp

# Или с явным указанием транспорта
uv run coqui-tts-mcp --transport http --host 0.0.0.0 --port 5002

# Stdio режим (для MCP клиентов)
uv run coqui-tts-mcp --transport stdio
```

Сервер будет доступен по адресу: `http://127.0.0.1:5002`

## 📡 API Endpoints

### HTTP Endpoints

#### `GET /tts?text=Hello%20World`
Синтезирует речь из текста и возвращает WAV файл.

**Пример:**
```bash
curl "http://127.0.0.1:5002/tts?text=Hello%20from%20Coqui!" -o output.wav
```

#### `GET /health`
Проверка состояния сервера.

**Ответ:**
```json
{
  "status": "ok",
  "model": "tts_models/en/ljspeech/tacotron2-DDC",
  "device": "mps",
  "tts_initialized": true
}
```

#### `GET /models`
Список доступных моделей TTS.

#### `GET /device`
Информация о доступных устройствах (CPU, CUDA, MPS).

## 🔧 MCP Tools

Сервер предоставляет следующие MCP инструменты:

### `synthesize_speech`
Синтезирует речь из текста.

**Параметры:**
- `text` (обязательный): Текст для синтеза
- `model_name` (опциональный): Имя модели TTS (используется модель по умолчанию, если не указана)

**Пример использования:**
```json
{
  "name": "synthesize_speech",
  "arguments": {
    "text": "Hello, this is a test of text-to-speech synthesis."
  }
}
```

### `list_available_models`
Возвращает список доступных моделей TTS.

### `get_device_info`
Возвращает информацию о доступных устройствах для вычислений.

## ⚙️ Конфигурация

Настройки можно задать через переменные окружения с префиксом `COQUI_TTS_MCP_`:

```bash
# Хост и порт
export COQUI_TTS_MCP_HOST=0.0.0.0
export COQUI_TTS_MCP_PORT=5002

# Модель TTS
export COQUI_TTS_MCP_MODEL_NAME=tts_models/en/ljspeech/tacotron2-DDC

# Устройство (cpu, cuda, mps, или auto)
export COQUI_TTS_MCP_DEVICE=mps

# Директория для выходных файлов
export COQUI_TTS_MCP_OUTPUT_DIR=/tmp/coqui_tts_output

# Время жизни файлов (секунды)
export COQUI_TTS_MCP_CLEANUP_AFTER_SECONDS=300

# Использовать MPS (Metal)
export COQUI_TTS_MCP_USE_MPS=true

# Уровень логирования
export COQUI_TTS_MCP_LOG_LEVEL=INFO
```

Или через аргументы командной строки:

```bash
uv run coqui-tts-mcp \
  --host 0.0.0.0 \
  --port 5002 \
  --model tts_models/en/ljspeech/tacotron2-DDC \
  --device mps \
  --log-level INFO
```

## 🌍 Мультиязычные модели

Для русского или мультиязычного синтеза можно использовать другие модели:

```bash
# Мультиязычная модель XTTS v2
uv run coqui-tts-mcp --model tts_models/multilingual/multi-dataset/xtts_v2

# Или через переменную окружения
export COQUI_TTS_MCP_MODEL_NAME=tts_models/multilingual/multi-dataset/xtts_v2
uv run coqui-tts-mcp
```

## 🔌 Интеграция с MCP клиентами

### Claude Desktop

Добавьте в `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "coqui-tts": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/mcp/servers/coqui-tts-mcp",
        "coqui-tts-mcp",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

### Другие MCP клиенты

Используйте команду запуска:
```bash
uv run --directory /path/to/coqui-tts-mcp coqui-tts-mcp --transport stdio
```

## 🐳 Docker (опционально)

Для запуска в Docker можно создать `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Копирование кода
COPY src/ ./src/

# Запуск сервера
CMD ["coqui-tts-mcp", "--host", "0.0.0.0", "--port", "5002"]
```

**Примечание:** Docker образы не поддерживают Metal (MPS), поэтому будет использоваться CPU.

## 📝 Примеры использования

### Python клиент

```python
import requests

# Синтез речи
response = requests.get(
    "http://127.0.0.1:5002/tts",
    params={"text": "Hello from Coqui TTS!"}
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### cURL

```bash
# Синтез речи
curl "http://127.0.0.1:5002/tts?text=Hello%20World" -o speech.wav

# Проверка здоровья
curl http://127.0.0.1:5002/health

# Список моделей
curl http://127.0.0.1:5002/models

# Информация об устройстве
curl http://127.0.0.1:5002/device
```

## 🔍 Troubleshooting

### Metal (MPS) не доступен

Если `torch.backends.mps.is_available()` возвращает `False`:
- Убедитесь, что используете Mac с Apple Silicon (M1/M2/M3/M4)
- Проверьте версию PyTorch (должна быть >= 2.0.0)
- Переустановите PyTorch: `pip install --upgrade torch torchvision torchaudio`

### Модель не загружается

- Проверьте интернет-соединение (модели загружаются автоматически при первом использовании)
- Убедитесь, что достаточно места на диске
- Проверьте логи сервера для деталей ошибки

### Проблемы с производительностью

- Используйте Metal (MPS) на Mac M-серии для лучшей производительности
- Для больших текстов рассмотрите разбиение на части
- Увеличьте `cleanup_after_seconds` если генерируете много файлов

## 📚 Дополнительные ресурсы

- [Coqui TTS Documentation](https://github.com/coqui-ai/TTS)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [MCP Protocol Specification](https://modelcontextprotocol.io)

## 🚀 Запуск через launchctl (macOS)

Для автоматического запуска сервера при старте системы и управления через launchctl:

### Установка

```bash
cd servers/coqui-tts-mcp
./scripts/launchctl.sh install
```

Это:
- Проверит виртуальное окружение и установит зависимости при необходимости
- Скопирует plist файл в `~/Library/LaunchAgents/`
- Загрузит и запустит сервис

### Управление сервисом

```bash
# Проверить статус
./scripts/launchctl.sh status

# Перезапустить
./scripts/launchctl.sh restart

# Остановить
./scripts/launchctl.sh stop

# Запустить
./scripts/launchctl.sh start

# Просмотр логов
./scripts/launchctl.sh logs

# Удалить сервис
./scripts/launchctl.sh uninstall
```

### Ручная установка

Если хотите установить вручную:

```bash
# 1. Скопировать plist файл
cp com.hal.coqui-tts-mcp.plist ~/Library/LaunchAgents/

# 2. Загрузить сервис
launchctl load ~/Library/LaunchAgents/com.hal.coqui-tts-mcp.plist

# 3. Проверить статус
launchctl list | grep coqui-tts-mcp
```

### Логи

Логи сохраняются в:
- `logs/stdout.log` - стандартный вывод
- `logs/stderr.log` - ошибки

Для просмотра в реальном времени:
```bash
tail -f logs/stdout.log
tail -f logs/stderr.log
```

### Настройка plist файла

Вы можете отредактировать `com.hal.coqui-tts-mcp.plist` для изменения:
- Порта (по умолчанию 5002)
- Модели TTS
- Переменных окружения
- Параметров запуска

После изменений перезагрузите сервис:
```bash
./scripts/launchctl.sh restart
```

## 📄 Лицензия

См. LICENSE файл в репозитории.

