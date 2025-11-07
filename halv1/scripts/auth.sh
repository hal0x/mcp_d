read -r -p "Телефон (в международном формате): " PHONE
read -r -p "Код из Telegram: " CODE
read -r -s -p "2FA пароль (если есть, иначе Enter): " PWD; echo

PHONE="$PHONE" CODE="$CODE" PWD="$PWD" python - <<'PY'
import os, asyncio
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError, PhoneCodeInvalidError, PhoneNumberInvalidError, FloodWaitError

API_ID   = 23854887
API_HASH = "723ae75a75b451c72d8d105b091c77f3"
SESSION  = "db/session/user"

async def main():
    import os
    os.makedirs(os.path.dirname(SESSION), exist_ok=True)
    phone = os.environ["PHONE"]
    code  = os.environ["CODE"]
    pwd   = os.environ.get("PWD") or None

    client = TelegramClient(SESSION, API_ID, API_HASH)
    await client.connect()
    try:
        if await client.is_user_authorized():
            me = await client.get_me()
            print(f"✅ Уже авторизованы как {me.first_name} (id={me.id}). Сессия: {SESSION}")
            return

        sent = await client.send_code_request(phone)
        await client.sign_in(phone=phone, code=code, phone_code_hash=sent.phone_code_hash)

    except SessionPasswordNeededError:
        if not pwd:
            raise RuntimeError("Требуется 2FA пароль: перезапустите и задайте PWD.")
        await client.sign_in(password=pwd)
    except PhoneCodeInvalidError:
        raise SystemExit("⛔ Неверный код.")
    except PhoneNumberInvalidError:
        raise SystemExit("❌ Неверный номер телефона.")
    except FloodWaitError as e:
        raise SystemExit(f"⏳ Flood wait: подождите {e.seconds} сек.")
    finally:
        if client.is_connected():
            try:
                if await client.is_user_authorized():
                    me = await client.get_me()
                    print(f"✅ Успешно авторизованы как {me.first_name} (id={me.id}). Сессия сохранена: {SESSION}")
            finally:
                await client.disconnect()

asyncio.run(main())
PY
