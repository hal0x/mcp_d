# Авторизация Telethon

## Проблема
Telethon требует однократной авторизации для работы с Telegram API. Без авторизации бот не может получать список чатов и сообщения.

## Решение

### Вариант 1: Интерактивная авторизация
```bash
cd /Users/hal/projects/halv1
source venv/bin/activate
python scripts/telethon_auth.py
```
Скрипт запросит номер телефона и при необходимости пароль 2FA.

### Вариант 2: Авторизация через переменные окружения
```bash
cd /Users/hal/projects/halv1
source venv/bin/activate
export TELETHON_PHONE='+ваш_номер_телефона'
export TELETHON_PASSWORD='ваш_пароль_2fa'  # если используется 2FA
python scripts/telethon_auth_final.py
```

### Вариант 3: Добавление в .env файл
Добавьте в файл `.env`:
```
TELETHON_PHONE=+ваш_номер_телефона
TELETHON_PASSWORD=ваш_пароль_2fa  # если используется 2FA
```

Затем запустите:
```bash
cd /Users/hal/projects/halv1
source venv/bin/activate
python scripts/telethon_auth_final.py
```

## Проверка авторизации
```bash
cd /Users/hal/projects/halv1
source venv/bin/activate
python scripts/telethon_auth_simple.py
```

## Файлы сессии
После успешной авторизации файл сессии сохраняется в `db/session/user.session`. Этот файл содержит данные авторизации и не требует повторной авторизации.

## Устранение проблем
- Если сессия не работает, удалите файл `db/session/user.session` и выполните авторизацию заново
- Убедитесь, что переменные `TELETHON_API_ID` и `TELETHON_API_HASH` установлены в `.env` файле
- Номер телефона должен быть в международном формате (например, +1234567890)
