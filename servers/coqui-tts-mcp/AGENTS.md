# Coqui TTS MCP - Agent Integration Guide

## Overview

Coqui TTS MCP сервер предоставляет инструменты для синтеза речи из текста с использованием Coqui TTS библиотеки. Сервер поддерживает Metal (MPS) ускорение на Mac M-серии для оптимальной производительности.

## Available Tools

### 1. `synthesize_speech`

Синтезирует речь из текста и возвращает путь к WAV файлу.

**Parameters:**
- `text` (string, required): Текст для синтеза речи
- `model_name` (string, optional): Имя модели TTS (по умолчанию используется модель из конфигурации)

**Returns:**
```json
{
  "file_path": "/tmp/coqui_tts_output/tts_abc123.wav",
  "text": "Hello, world!",
  "model": "tts_models/en/ljspeech/tacotron2-DDC",
  "device": "mps"
}
```

**Example Usage:**
```json
{
  "name": "synthesize_speech",
  "arguments": {
    "text": "Hello, this is a test of text-to-speech synthesis."
  }
}
```

### 2. `list_available_models`

Возвращает список всех доступных моделей TTS.

**Parameters:** None

**Returns:**
```json
{
  "available_models": [
    "tts_models/en/ljspeech/tacotron2-DDC",
    "tts_models/multilingual/multi-dataset/xtts_v2",
    ...
  ],
  "current_model": "tts_models/en/ljspeech/tacotron2-DDC"
}
```

### 3. `get_device_info`

Возвращает информацию о доступных вычислительных устройствах.

**Parameters:** None

**Returns:**
```json
{
  "cpu_available": true,
  "cuda_available": false,
  "mps_available": true,
  "current_device": "mps",
  "cuda_device_count": 0
}
```

## HTTP Endpoints

Сервер также предоставляет прямые HTTP endpoints для интеграции с другими системами:

- `GET /tts?text=<text>` - Синтез речи, возвращает WAV файл
- `GET /health` - Проверка состояния сервера
- `GET /models` - Список доступных моделей
- `GET /device` - Информация об устройствах

## Configuration

### Environment Variables

- `COQUI_TTS_MCP_HOST` - Хост для HTTP сервера (default: `0.0.0.0`)
- `COQUI_TTS_MCP_PORT` - Порт для HTTP сервера (default: `5002`)
- `COQUI_TTS_MCP_MODEL_NAME` - Имя модели TTS (default: `tts_models/en/ljspeech/tacotron2-DDC`)
- `COQUI_TTS_MCP_DEVICE` - Устройство: `cpu`, `cuda`, `mps`, или `auto` (default: auto-detect)
- `COQUI_TTS_MCP_USE_MPS` - Использовать Metal Performance Shaders (default: `true`)
- `COQUI_TTS_MCP_OUTPUT_DIR` - Директория для выходных файлов (default: `/tmp/coqui_tts_output`)
- `COQUI_TTS_MCP_CLEANUP_AFTER_SECONDS` - Время жизни файлов в секундах (default: `300`)

### Command Line Arguments

```bash
coqui-tts-mcp \
  --host 0.0.0.0 \
  --port 5002 \
  --model tts_models/en/ljspeech/tacotron2-DDC \
  --device mps \
  --transport http
```

## Use Cases

1. **Voice Assistant Integration**: Синтез ответов голосового ассистента
2. **Accessibility**: Преобразование текста в речь для пользователей с нарушениями зрения
3. **Content Creation**: Генерация аудио для подкастов, видео и других медиа
4. **Multilingual Support**: Использование мультиязычных моделей для поддержки разных языков

## Performance Notes

- На Mac M-серии Metal (MPS) обеспечивает значительное ускорение по сравнению с CPU
- Первая загрузка модели может занять время (модели загружаются автоматически)
- Генерируемые файлы автоматически очищаются через заданное время
- Для больших текстов рекомендуется разбивать на части

## Integration Examples

### With HAL Agent

```python
# В коде агента
result = await mcp_client.call_tool(
    "coqui-tts-mcp",
    "synthesize_speech",
    {"text": "Hello from HAL agent!"}
)
audio_file = result["file_path"]
# Использовать audio_file для воспроизведения
```

### Direct HTTP Call

```python
import requests

response = requests.get(
    "http://127.0.0.1:5002/tts",
    params={"text": "Hello, world!"}
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## Troubleshooting

- Если MPS недоступен, проверьте что используете Mac с Apple Silicon
- Убедитесь что PyTorch >= 2.0.0 установлен
- Проверьте логи сервера для детальной информации об ошибках




