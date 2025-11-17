# Быстрый старт Coqui TTS MCP

## ✅ Локальный запуск (для отладки)

```bash
cd servers/coqui-tts-mcp
source .venv/bin/activate
python -m coqui_tts_mcp.server --transport streamable-http --host 127.0.0.1 --port 5002
```

Сервер будет доступен на `http://127.0.0.1:5002`

## 🚀 Запуск через launchctl (автозапуск)

```bash
cd servers/coqui-tts-mcp
./scripts/launchctl.sh install
```

Управление:
```bash
./scripts/launchctl.sh status   # Проверить статус
./scripts/launchctl.sh restart  # Перезапустить
./scripts/launchctl.sh stop     # Остановить
./scripts/launchctl.sh logs     # Просмотр логов
```

## 📡 Тестирование endpoints

```bash
# Health check
curl http://127.0.0.1:5002/health

# Синтез речи
curl "http://127.0.0.1:5002/tts?text=Hello%20World" -o output.wav

# Информация об устройстве
curl http://127.0.0.1:5002/device

# Список моделей
curl http://127.0.0.1:5002/models
```

## 🔌 Интеграция с AnythingLLM

Конфигурация уже добавлена в `anythingllm_mcp_servers.json`:

```json
{
  "mcpServers": {
    "coqui-tts-mcp": {
      "command": "/Users/hal/projects/mcp/servers/coqui-tts-mcp/.venv/bin/python",
      "args": ["-m", "coqui_tts_mcp.server", "--transport", "stdio"],
      "cwd": "/Users/hal/projects/mcp/servers/coqui-tts-mcp",
      "env": {
        "COQUI_TTS_MCP_USE_MPS": "true",
        "COQUI_TTS_MCP_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

Перезапустите AnythingLLM Desktop для применения изменений.

## 📝 Доступные MCP инструменты

- `synthesize_speech` - синтез речи из текста
- `list_available_models` - список доступных моделей TTS
- `get_device_info` - информация об устройствах (CPU, CUDA, MPS)

## 🎯 HTTP Endpoints

- `GET /health` - проверка состояния
- `GET /tts?text=<text>` - синтез речи (возвращает WAV файл)
- `GET /device` - информация об устройствах
- `GET /models` - список доступных моделей


