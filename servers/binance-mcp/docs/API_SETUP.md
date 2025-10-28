# Инструкция по настройке API ключей

## Проблема
API ключи не имеют прав доступа для торговых операций. Ошибка:
```
APIError(code=-2015): Invalid API-key, IP, or permissions for action
```

## Решение

### 1. Проверьте права API ключей в Binance
- Зайдите в Binance → Account → API Management
- Убедитесь что API ключи имеют права:
  - ✅ **Spot & Margin Trading** (для создания ордеров)
  - ✅ **Read Info** (для просмотра баланса и истории)

### 2. Настройте файл .env
Создайте файл `.env` в корне проекта:

```env
# Live Trading API Keys
BINANCE_API_KEY=your_actual_api_key_here
BINANCE_API_SECRET=your_actual_api_secret_here

# Demo Trading API Keys (рекомендуется для тестирования)
DEMO_BINANCE_API_KEY=your_demo_api_key_here
DEMO_BINANCE_API_SECRET=your_demo_api_secret_here

# Configuration
BINANCE_DEMO_TRADING=True

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

### 3. Получите демо API ключи (рекомендуется)
1. Зайдите на https://demo-fapi.binance.com/ (фьючерсы) или https://demo-dapi.binance.com/ (деривативы)
2. Зарегистрируйтесь или войдите
3. Создайте API ключи в разделе API Management
4. Используйте эти ключи как DEMO_BINANCE_API_KEY и DEMO_BINANCE_API_SECRET

### 4. Перезапустите MCP сервер
После настройки .env файла перезапустите MCP сервер в Cursor.

## Безопасность
- ⚠️ **НИКОГДА** не коммитьте .env файлы
- ✅ Используйте демо режим для тестирования
- ✅ Ограничьте права API ключей только необходимыми
- ✅ Регулярно обновляйте API ключи

## Проверка настроек
После настройки запустите:
```bash
python check_permissions.py
```

Это покажет какие права доступны для ваших API ключей.
