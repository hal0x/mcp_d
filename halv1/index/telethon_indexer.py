"""Telethon indexer for fetching chat messages."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, List, Tuple

try:  # Telethon is optional for this skeleton
    from telethon import TelegramClient  # type: ignore[import-untyped]
    from telethon.tl.types import Message  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - dependency may be missing
    TelegramClient = Message = object

logger = logging.getLogger(__name__)


class TelethonIndexer:
    """Read user messages from Telegram using Telethon."""

    def __init__(self, api_id: int, api_hash: str, session: str = "user") -> None:
        if TelegramClient is object:  # pragma: no cover - informative error
            raise RuntimeError("Telethon must be installed to use TelethonIndexer")
        # Store session under writable `db/session/` unless an explicit path is provided
        if "/" in session or session.startswith("."):
            session_name = session
        else:
            base = Path("db/session")
            base.mkdir(parents=True, exist_ok=True)
            session_name = str(base / session)
        # Use the provided session name to persist authentication between runs
        # Добавляем настройки таймаута и повторных попыток
        # Use Telethon 1.x compatible kwargs: request_retries (not 'retries')
        self.client = TelegramClient(
            session_name,
            api_id,
            api_hash,
            timeout=30,          # 30 секунд таймаут
            retry_delay=2,       # 2 секунды между попытками
            request_retries=3,   # 3 попытки повторов запросов
        )
        self._message_ids_cache = {}  # Кэш существующих ID сообщений по чатам
        
        # Настройки для изолированных ретраев
        self._max_connection_retries = 3
        self._connection_retry_delays = [1.0, 2.0, 4.0]  # Экспоненциальная задержка
        self._is_connection_healthy = False

    async def ensure_connected(self) -> None:
        """Ensure TCP connection exists (no auth requirement) with isolated retries."""
        if self.client.is_connected() and self._is_connection_healthy:
            return
            
        logger.debug("Telethon: Attempting to establish connection...")
        
        for attempt in range(self._max_connection_retries):
            try:
                if not self.client.is_connected():
                    await self.client.connect()
                
                # Проверяем здоровье соединения простым запросом
                await self.client.get_me()
                self._is_connection_healthy = True
                logger.debug("Telethon: Connection established successfully")
                return
                
            except Exception as e:
                logger.warning(f"Telethon connection attempt {attempt + 1}/{self._max_connection_retries} failed: {e}")
                
                # Если это не последняя попытка, ждём перед повтором
                if attempt < self._max_connection_retries - 1:
                    delay = self._connection_retry_delays[min(attempt, len(self._connection_retry_delays) - 1)]
                    logger.debug(f"Telethon: Retrying connection in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Telethon: All connection attempts failed, continuing without Telegram")
                    self._is_connection_healthy = False
                    # Не поднимаем исключение, чтобы система могла работать без Telegram

    def is_connection_healthy(self) -> bool:
        """Проверка здоровья соединения Telethon."""
        return self._is_connection_healthy and self.client.is_connected()

    async def health_check(self) -> bool:
        """Проверка доступности Telegram API."""
        try:
            if not self.client.is_connected():
                return False
            await self.client.get_me()
            self._is_connection_healthy = True
            return True
        except Exception as e:
            logger.debug(f"Telethon health check failed: {e}")
            self._is_connection_healthy = False
            return False

    async def assert_authorized(self) -> None:
        """Ensure the session is authorized (raises with instructions otherwise)."""
        await self.ensure_connected()
        if not await self.client.is_user_authorized():
            raise RuntimeError(
                "Telethon не авторизован. Выполните однократную авторизацию: \n"
                "python - <<'PY'\n"
                "from telethon import TelegramClient\n"
                "api_id=%s; api_hash='%s'\n"
                "client=TelegramClient('%s', api_id, api_hash); client.start(); print('OK'); client.disconnect()\n"
                "PY" % ("<API_ID>", "<API_HASH>", str(self.client.session.filename))
            )

    async def request_code(self, phone: str) -> None:
        # Code request only requires TCP connection
        await self.ensure_connected()
        await self.client.send_code_request(phone)

    async def sign_in(
        self, phone: str, code: str, password: str | None = None
    ) -> dict[str, str]:
        await self.ensure_connected()
        try:
            if password:
                await self.client.sign_in(password=password)
            else:
                await self.client.sign_in(phone=phone, code=code)
            ok = await self.client.is_user_authorized()
            return {"ok": "true" if ok else "false"}
        except Exception as exc:  # pragma: no cover
            s = str(exc).lower()
            if "session password needed" in s or "password" in s:
                return {"ok": "false", "need_password": "true"}
            return {"ok": "false", "error": str(exc)}

    async def iter_messages(self) -> AsyncGenerator[Message, None]:
        """Iterate over all messages in the user's dialogs."""
        await self.assert_authorized()
        async for dialog in self.client.iter_dialogs():
            async for message in self.client.iter_messages(dialog.entity):
                yield message

    async def list_dialogs(self) -> List[Tuple[str, object]]:
        """Return a list of (title, entity) for the user's dialogs."""
        result: List[Tuple[str, object]] = []
        await self.assert_authorized()
        async for dialog in self.client.iter_dialogs():
            title = getattr(dialog, "name", None) or getattr(
                dialog.entity, "title", None
            )
            if not title:
                # Fallback to stringified id
                title = str(getattr(dialog.entity, "id", "unknown"))
            result.append((title, dialog.entity))
        return result

    async def iter_last_messages(
        self, entity: object, limit: int
    ) -> AsyncGenerator[Message, None]:
        """Yield last ``limit`` messages from a given dialog entity."""
        await self.assert_authorized()
        async for message in self.client.iter_messages(entity, limit=limit):
            yield message

    async def iter_messages_since(
        self, entity: object, since: datetime
    ) -> AsyncGenerator[Message, None]:
        """Yield messages from a dialog entity that are newer than ``since``.

        Iterates from newest to oldest and stops once messages are older than the cutoff
        to avoid traversing the whole history unnecessarily.
        """
        await self.assert_authorized()
        # Ensure both datetimes are timezone-naive for comparison
        if since.tzinfo is not None:
            since_naive = since.replace(tzinfo=None)
        else:
            since_naive = since

        async for message in self.client.iter_messages(entity):
            msg_dt = getattr(message, "date", None)
            if msg_dt is None:
                continue
            # Ensure message datetime is also timezone-naive
            if msg_dt.tzinfo is not None:
                cut = msg_dt.replace(tzinfo=None)
            else:
                cut = msg_dt
            if cut < since_naive:
                break
            yield message

    async def iter_messages_before(
        self, entity: object, before: datetime
    ) -> AsyncGenerator[Message, None]:
        """Yield messages from a dialog entity that are older than ``before``.
        
        Iterates from newest to oldest and stops when we have enough messages.
        Ограничение: не индексируем сообщения старше 1 года.
        """
        await self.assert_authorized()
        from datetime import timedelta, timezone
        
        # Ограничение по давности - не старше 1 года
        one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
        
        # Ensure both datetimes are timezone-naive for comparison
        if before.tzinfo is not None:
            before_naive = before.replace(tzinfo=None)
        else:
            before_naive = before
        
        if one_year_ago.tzinfo is not None:
            one_year_ago_naive = one_year_ago.replace(tzinfo=None)
        else:
            one_year_ago_naive = one_year_ago

        async for message in self.client.iter_messages(entity):
            msg_dt = getattr(message, "date", None)
            if msg_dt is None:
                continue
            # Ensure message datetime is also timezone-naive
            if msg_dt.tzinfo is not None:
                cut = msg_dt.replace(tzinfo=None)
            else:
                cut = msg_dt
            
            # Пропускаем сообщения старше 1 года
            if cut < one_year_ago_naive:
                break
            
            if cut < before_naive:
                yield message

    def _get_latest_indexed_date(self, chat_name: str = None) -> datetime:
        """Определяет последнюю дату индексации из db/raw для конкретного чата.
        
        Ограничение: не индексируем сообщения старше 1 года.
        """
        from pathlib import Path
        import json
        from datetime import timedelta, timezone
        
        # Максимальная давность сообщений - 1 год
        one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
        
        raw_dir = Path("db/raw")
        if not raw_dir.exists():
            # Если папки нет, начинаем с 1 месяца назад
            return datetime.now(timezone.utc) - timedelta(days=30)
        
        # Если указан конкретный чат, ищем в его папке
        if chat_name:
            # ИСПРАВЛЕНИЕ: ищем папку по sanitized имени, но проверяем содержимое по оригинальному имени
            sanitized_name = self._sanitize_chat_name(chat_name)
            chat_dir = raw_dir / sanitized_name
            if chat_dir.exists():
                latest_date = self._get_chat_latest_date(chat_dir, chat_name)
                # Не индексируем старше 1 года
                if latest_date < one_year_ago:
                    return one_year_ago
                return latest_date
            else:
                # Если чат не существует в raw, начинаем с 1 месяца назад (но не старше года)
                default_date = datetime.now(timezone.utc) - timedelta(days=30)
                return max(default_date, one_year_ago)
        
        # Иначе ищем по всем чатам
        latest_date = None
        for chat_dir in raw_dir.iterdir():
            if chat_dir.is_dir():
                chat_latest = self._get_chat_latest_date(chat_dir, chat_dir.name)
                if latest_date is None or (chat_latest and chat_latest > latest_date):
                    latest_date = chat_latest
        
        if latest_date is None:
            # Если не удалось найти даты, начинаем с 1 месяца назад (но не старше года)
            default_date = datetime.now(timezone.utc) - timedelta(days=30)
            return max(default_date, one_year_ago)
        
        # Не индексируем старше 1 года
        if latest_date < one_year_ago:
            return one_year_ago
        
        return latest_date

    def _get_chat_latest_date(self, chat_dir: Path, chat_name: str) -> datetime:
        """Получает последнюю дату для конкретного чата и определяет период индексации."""
        import json
        from datetime import timedelta
        
        latest_date = None
        latest_message_id = None
        message_count = 0
        
        # Считаем общее количество сообщений в чате и находим самое последнее
        for json_file in chat_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            message_count += 1
                            
                            # ИСПРАВЛЕНИЕ: используем date_utc вместо date
                            date_str = data.get('date_utc', '') or data.get('date', '')
                            message_id = data.get('id')
                            
                            if date_str and message_id:
                                # Парсим дату
                                if '+' in date_str or 'Z' in date_str:
                                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                                else:
                                    dt = datetime.fromisoformat(date_str)
                                
                                # Находим самое последнее сообщение по дате и ID
                                if latest_date is None or dt > latest_date or (dt == latest_date and int(message_id) > (latest_message_id or 0)):
                                    latest_date = dt
                                    latest_message_id = int(message_id)
                        except (json.JSONDecodeError, ValueError, TypeError):
                            continue
            except Exception:
                continue
        
        # Возвращаем дату начала индексации
        if latest_date:
            # Добавляем 1 минуту к дате последнего сообщения для надежности
            # Это предотвращает дубликаты из-за неточности временных меток Telegram
            from datetime import timedelta, timezone
            start_date = latest_date + timedelta(minutes=1)
            
            # Не индексируем старше 1 года
            one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
            if start_date < one_year_ago:
                return one_year_ago
            
            logger.debug(f"📅 Чат '{chat_name}': последнее сообщение {latest_message_id} от {latest_date}, начинаем с {start_date}")
            return start_date
        else:
            # Если сообщений нет, начинаем с 30 дней назад для полной индексации (но не старше года)
            from datetime import timezone
            default_date = datetime.now(timezone.utc) - timedelta(days=30)
            one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
            return max(default_date, one_year_ago)

    def _load_allowed_chats(self) -> set[str]:
        """Загружает список разрешенных чатов из файла."""
        try:
            with open("allowed_chats.txt", "r", encoding="utf-8") as f:
                return {line.strip() for line in f if line.strip()}
        except FileNotFoundError:
            logger.warning("Файл allowed_chats.txt не найден, индексируем все чаты")
            return set()
        except Exception as e:
            logger.error(f"Ошибка загрузки allowed_chats.txt: {e}")
            return set()

    def _load_message_ids_cache(self, chat_name: str) -> set[int]:
        """Загружает кэш ID сообщений для конкретного чата."""
        if chat_name in self._message_ids_cache:
            return self._message_ids_cache[chat_name]
        
        from pathlib import Path
        import json
        
        raw_dir = Path("db/raw")
        chat_dir = raw_dir / self._sanitize_chat_name(chat_name)
        
        message_ids = set()
        if chat_dir.exists():
            for json_file in chat_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                message_id = data.get('id')
                                if message_id:
                                    message_ids.add(int(message_id))
                            except (json.JSONDecodeError, ValueError, TypeError):
                                continue
                except Exception:
                    continue
        
        self._message_ids_cache[chat_name] = message_ids
        return message_ids

    def _sanitize_chat_name(self, chat_name: str) -> str:
        """Очищает имя чата для использования в качестве имени папки."""
        # ИСПРАВЛЕНИЕ: используем ту же логику, что и в raw_storage.py
        name = chat_name.strip().replace("\\", "/")
        # Keep alphanumerics and a small safe set, replace others
        safe = "".join(ch if ch.isalnum() or ch in " _-." else "_" for ch in name)
        # Collapse slashes to underscores to avoid nested/absolute paths
        safe = safe.replace("/", "_")
        if not safe or safe == ".":
            safe = "unknown"
        if safe.startswith("."):
            safe = "_" + safe.lstrip(".")
        return safe

    def _message_exists(self, message_id: int, chat_name: str) -> bool:
        """Проверяет, существует ли сообщение уже в базе данных (быстрая версия с кэшем)."""
        message_ids = self._load_message_ids_cache(chat_name)
        exists = message_id in message_ids
        
        # Логируем для отладки первые несколько проверок
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        if self._debug_counter < 5:
            logger.debug(f"🔍 Проверка дубликата: сообщение {message_id} в чате '{chat_name}' - {'найдено' if exists else 'новое'}")
            self._debug_counter += 1
        
        return exists

    def _build_message_ids_cache(self, allowed_chats: set[str]) -> None:
        """Предварительно строит кэш ID сообщений для всех разрешенных чатов."""
        logger.info("🔨 Строим кэш ID сообщений для всех чатов...")
        
        for i, chat_name in enumerate(allowed_chats, 1):
            logger.info(f"📦 [{i}/{len(allowed_chats)}] Загружаем кэш для чата: {chat_name}")
            self._load_message_ids_cache(chat_name)
            
            # Показываем прогресс каждые 10 чатов
            if i % 10 == 0:
                cached_chats = len(self._message_ids_cache)
                total_ids = sum(len(ids) for ids in self._message_ids_cache.values())
                logger.info(f"📊 Кэш: {cached_chats} чатов, {total_ids} ID сообщений")
        
        total_cached = sum(len(ids) for ids in self._message_ids_cache.values())
        logger.info(f"✅ Кэш построен: {len(self._message_ids_cache)} чатов, {total_cached} ID сообщений")

    async def index_once(self) -> AsyncGenerator[Message, None]:
        """Выполняет индексацию новых сообщений с индивидуальной логикой для каждого чата."""
        from datetime import timedelta, UTC
        
        await self.assert_authorized()
        
        # Загружаем список разрешенных чатов
        allowed_chats = self._load_allowed_chats()
        if allowed_chats:
            logger.info(f"📋 Загружен список из {len(allowed_chats)} разрешенных чатов")
        else:
            logger.info("📋 Список разрешенных чатов пуст, индексируем все чаты")
        
        # Предварительно строим кэш для всех разрешенных чатов
        if allowed_chats:
            self._build_message_ids_cache(allowed_chats)
        
        processed_dialogs = 0
        filtered_dialogs = 0
        total_new_messages = 0  # Общее количество новых сообщений
        processed_chats_count = 0  # Количество чатов с новыми сообщениями
        
        # Если есть список разрешенных чатов, показываем его размер
        if allowed_chats:
            logger.info(f"📊 Будем обрабатывать до {len(allowed_chats)} разрешенных чатов из всех диалогов")
        else:
            logger.info("📊 Обрабатываем все диалоги (список разрешенных пуст)")
        
        # Обрабатываем диалоги с фильтрацией
        async for dialog in self.client.iter_dialogs():
            dialog_name = getattr(dialog, "name", None) or getattr(
                dialog.entity, "title", None
            ) or str(getattr(dialog.entity, "id", "unknown"))
            
            # Фильтруем чаты по списку разрешенных
            if allowed_chats and dialog_name not in allowed_chats:
                filtered_dialogs += 1
                if filtered_dialogs % 50 == 0:
                    logger.info(f"⏭️  Пропущено {filtered_dialogs} чатов (не в списке разрешенных)")
                continue
            
            processed_dialogs += 1
            
            logger.info(f"💬 [{processed_dialogs}] Обрабатываем чат: {dialog_name}")
            
            # Определяем дату начала индексации для конкретного чата
            now_utc = datetime.now(UTC)
            chat_indexed_date = self._get_latest_indexed_date(dialog_name)
            if chat_indexed_date.tzinfo is None:
                chat_indexed_date = chat_indexed_date.replace(tzinfo=UTC)
            
            # Специальная логика для Marketapp чатов - проверяем, нужно ли полная индексация
            if "marketapp" in dialog_name.lower():
                # Проверяем, есть ли уже проиндексированные сообщения
                cached_message_ids = self._load_message_ids_cache(dialog_name)
                cached_count = len(cached_message_ids)
                
                if cached_count == 0:
                    # Если нет проиндексированных сообщений, начинаем с года назад
                    from datetime import timedelta
                    chat_indexed_date = now_utc - timedelta(days=365)
                    logger.info(f"📅 Marketapp чат '{dialog_name}': полная индексация с {chat_indexed_date} (первая индексация, все сообщения за год)")
                else:
                    # Если есть проиндексированные сообщения, продолжаем с последней даты
                    logger.info(f"📅 Marketapp чат '{dialog_name}': индексация новых сообщений с {chat_indexed_date} (уже проиндексировано {cached_count} сообщений)")
            else:
                # Обычная логика для остальных чатов
                time_diff = now_utc - chat_indexed_date
                if time_diff.total_seconds() < 3600:  # Менее часа
                    logger.info(f"📅 Начинаем индексацию с {chat_indexed_date} (последние сообщения, {time_diff.total_seconds()/60:.1f} мин назад)")
                elif time_diff.total_seconds() < 86400:  # Менее дня
                    logger.info(f"📅 Начинаем индексацию с {chat_indexed_date} (недавние сообщения, {time_diff.total_seconds()/3600:.1f} ч назад)")
                else:
                    logger.info(f"📅 Начинаем индексацию с {chat_indexed_date} (старые сообщения, {time_diff.days} дн назад)")
            
            message_count = 0
            skipped_count = 0
            
            # Определяем лимит сообщений в зависимости от периода индексации
            days_since_last = (now_utc - chat_indexed_date).days
            
            # Специальные лимиты для чатов Marketapp - скачиваем все сообщения
            if "marketapp" in dialog_name.lower():
                max_messages = 50000  # Очень высокий лимит для Marketapp чатов
                min_messages = 0      # Минимум 0 - индексируем только новые
                logger.info(f"📊 Специальный лимит для Marketapp чата '{dialog_name}': максимум {max_messages} (все сообщения)")
            elif days_since_last <= 7:
                # За последние 7 дней - индексируем только новые сообщения
                max_messages = 2000    # Максимум для предотвращения перегрузки
                min_messages = 0       # Минимум 0 - индексируем только новые
            else:
                # За более длительный период - тоже только новые сообщения
                max_messages = 2000    # Максимум для предотвращения перегрузки
                min_messages = 0       # Минимум 0 - индексируем только новые
            
            # Логируем лимит только для обычных чатов (не Marketapp)
            if "marketapp" not in dialog_name.lower():
                logger.info(f"📊 Лимит сообщений для чата '{dialog_name}': максимум {max_messages} (только новые сообщения)")
            
            # Проверяем, сколько сообщений уже есть в кэше для этого чата
            cached_message_ids = self._load_message_ids_cache(dialog_name)
            cached_count = len(cached_message_ids)
            
            # Проверяем новые сообщения даже если в кэше уже достаточно сообщений
            if cached_count >= min_messages:
                logger.info(f"  🔍 Чат '{dialog_name}' имеет {cached_count} сообщений в кэше, проверяем новые сообщения...")
            
            # Сначала пробуем получить новые сообщения
            async for message in self.iter_messages_since(dialog.entity, chat_indexed_date):
                # Ограничиваем по максимуму (2000 сообщений)
                if message_count >= max_messages:
                    logger.info(f"  ✅ Достигнут лимит {max_messages} сообщений для чата '{dialog_name}'")
                    break
                
                # Быстрая проверка дубликатов через кэш (без сетевых вызовов)
                if self._message_exists(message.id, dialog_name):
                    skipped_count += 1
                    if skipped_count % 100 == 0:
                        logger.info(f"  ⏭️  Пропущено {skipped_count} дубликатов")
                    continue
                
                # Логируем первые несколько новых сообщений для отладки
                if message_count < 5:
                    logger.info(f"  🔍 Новое сообщение {message.id} от {getattr(message, 'date', 'N/A')}")
                
                message_count += 1
                if message_count % 50 == 0:  # Логируем каждые 50 сообщений
                    msg_date = getattr(message, "date", None)
                    if msg_date:
                        logger.info(f"  📝 Обработано {message_count} новых сообщений, последнее: {msg_date}")
                
                # Сообщения обрабатываются через основной поток индексации
                yield message
            
            # Индексируем только новые сообщения, не загружаем старые
            logger.info(f"  ✅ Обработано {message_count} новых сообщений для чата '{dialog_name}'")
            
            if message_count > 0:
                # Обновляем общую статистику
                total_new_messages += message_count
                processed_chats_count += 1
                
                status = "✅"
                logger.info(f"{status} Чат '{dialog_name}': обработано {message_count} новых сообщений, пропущено {skipped_count} дубликатов")
                
                # Отправляем специальный маркер для удаления старых сообщений
                yield {"type": "chat_completed", "chat_name": dialog_name, "new_messages": message_count}
            else:
                logger.info(f"ℹ️  Чат '{dialog_name}': новых сообщений нет, пропущено {skipped_count} дубликатов")
        
        # Итоговая статистика
        logger.info(f"📊 ИТОГО: обработано {processed_dialogs} чатов, пропущено {filtered_dialogs} чатов (не в списке разрешенных)")
        logger.info(f"📈 НОВЫЕ СООБЩЕНИЯ: {total_new_messages} сообщений проиндексировано в {processed_chats_count} чатах")
