from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Awaitable, Callable, Dict, List, Optional

from .utils import format_seconds
from .telegram_utils import safe_edit_message

logger = logging.getLogger(__name__)

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import ContextTypes
except Exception:  # pragma: no cover - optional dependency
    Update = InlineKeyboardButton = InlineKeyboardMarkup = None  # type: ignore
    ContextTypes = None  # type: ignore


class ReindexHandlersMixin:
    _list_chats: Optional[Callable[[], Awaitable[List[str]]]]
    _index_last: Optional[Callable[[List[str], int], Awaitable[int]]]
    _reindex_state: Dict[int, Dict]
    _dump_state: Dict[int, Dict]
    _dump_since: Optional[Callable[[int], Awaitable[int]]]
    _index_state: Dict[int, Dict]
    _index_dumped: Optional[Callable[[int], Awaitable[int]]]
    _get_theme_chats: Optional[Callable[[str], Awaitable[List[str]]]]
    _list_themes: Optional[Callable[[], Awaitable[List[str]]]]
    _get_active_theme_name: Optional[Callable[[], Awaitable[str]]]
    _safe_reply: Callable[[Update, ContextTypes.DEFAULT_TYPE, str], Awaitable[None]]
    _tele_indexer: Optional[object]  # TelethonIndexer instance
    _retriever: Optional[object]  # Retriever instance
    _telethon_service: Optional[object]  # TelethonService instance

    async def _cmd_reindex(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._list_chats or not self._index_last:
            await self._safe_reply(update, context, "Режим /reindex недоступен.")
            return
        chat_id = update.effective_chat.id if update.effective_chat else 0
        try:
            theme_name: Optional[str] = None
            if self._list_themes and self._get_active_theme_name:
                try:
                    theme_name = await self._get_active_theme_name()
                except Exception:
                    theme_name = None
            if (
                theme_name
                and self._list_themes
                and hasattr(self, "_get_theme_chats")
                and self._get_theme_chats
            ):
                all_themes = await self._list_themes()
                if theme_name in all_themes:
                    selected_titles = await self._get_theme_chats(theme_name)
                    if selected_titles:
                        self._reindex_state[chat_id] = {
                            "step": "choose_interval",
                            "selected_titles": selected_titles,
                        }
                        buttons = [
                            [
                                InlineKeyboardButton(
                                    "🕐 За час", callback_data="reindex_hour"
                                ),
                                InlineKeyboardButton(
                                    "📅 За день", callback_data="reindex_day"
                                ),
                                InlineKeyboardButton(
                                    "📆 За неделю", callback_data="reindex_week"
                                ),
                            ],
                            [
                                InlineKeyboardButton(
                                    "🗓️ За месяц", callback_data="reindex_month"
                                ),
                                InlineKeyboardButton(
                                    "✏️ Количество месяцев",
                                    callback_data="reindex_months",
                                ),
                            ],
                        ]
                        await self._safe_reply(
                            update,
                            context,
                            f"Активная тема: '{theme_name}'. Будут индексироваться {len(selected_titles)} чатов. Выберите интервал:",
                            reply_markup=InlineKeyboardMarkup(buttons),
                        )
                        return
            titles: List[str] = await self._list_chats()
        except Exception as exc:  # pragma: no cover - authorization/network errors
            logger.exception("Failed to list chats: %s", exc)
            await self._safe_reply(
                update,
                context,
                "Не удалось получить список чатов. Убедитесь, что Telethon авторизован (создана сессия).",
            )
            return
        if not titles:
            await self._safe_reply(update, context, "Список чатов пуст.")
            return

        self._reindex_state[chat_id] = {
            "step": "choose_chats",
            "chat_titles": titles,
            "selected_titles": [],
            "current_page": 0,
        }

        await self._show_reindex_chat_selection(update, chat_id)

    async def _show_reindex_chat_selection(
        self, update: Update, chat_id: int, edit_message: bool = False
    ) -> None:
        state = self._reindex_state.get(chat_id)
        if not state:
            return

        available = state.get("chat_titles", [])
        selected = state.get("selected_titles", [])
        current_page = state.get("current_page", 0)

        total = len(available)
        per_page = getattr(self, "CHATS_PER_PAGE", 8)
        total_pages = (total + per_page - 1) // per_page or 1

        current_page = max(0, min(current_page, total_pages - 1))
        state["current_page"] = current_page

        start = current_page * per_page
        end = min(start + per_page, total)

        buttons: List[List[InlineKeyboardButton]] = []
        for i in range(start, end):
            title = available[i]
            icon = "✅" if title in selected else "⬜"
            max_len = getattr(self, "MAX_CHAT_NAME_LENGTH", 30)
            display = title[:max_len] + ("..." if len(title) > max_len else "")
            buttons.append(
                [
                    InlineKeyboardButton(
                        f"{icon} {display}", callback_data=f"reidx_chat_{i}"
                    )
                ]
            )

        if total_pages > 1:
            nav: List[InlineKeyboardButton] = []
            if current_page > 0:
                nav.append(InlineKeyboardButton("⬅️ Назад", callback_data="reidx_prev"))
            nav.append(
                InlineKeyboardButton(
                    f"{current_page + 1}/{total_pages}", callback_data="reidx_info"
                )
            )
            if current_page < total_pages - 1:
                nav.append(InlineKeyboardButton("➡️ Вперёд", callback_data="reidx_next"))
            buttons.append(nav)

        buttons.append(
            [
                InlineKeyboardButton("✅ Готово", callback_data="reidx_confirm"),
                InlineKeyboardButton("❌ Отмена", callback_data="reidx_cancel"),
            ]
        )

        keyboard = InlineKeyboardMarkup(buttons)
        text = f"Выберите чаты для индексации:\nВыбрано: {len(selected)} из {total}"
        if total_pages > 1:
            text += f"\nСтраница: {current_page + 1}/{total_pages}"

        if edit_message and getattr(update, "callback_query", None):
            await update.callback_query.edit_message_text(text, reply_markup=keyboard)
        else:
            await update.message.reply_text(text, reply_markup=keyboard)

    async def _cmd_dump(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._dump_since:
            await self._safe_reply(update, context, "Дамп недоступен.")
            return
        chat_id = update.effective_chat.id if update.effective_chat else 0
        args = getattr(context, "args", [])
        if args and args[0].isdigit():
            days = int(args[0])
            if days <= 0:
                await self._safe_reply(update, context, "Некорректное число дней.")
                return
            keyboard = InlineKeyboardMarkup(
                [[InlineKeyboardButton("❌ Отмена", callback_data="dump_cancel")]]
            )
            msg = await update.message.reply_text(
                "Начинаю выгрузку сообщений…", reply_markup=keyboard
            )
            self._dump_state[chat_id] = {}
            await self._run_dump(chat_id, context, msg, days, keyboard)
            return
        self._dump_state[chat_id] = {"step": "days"}
        buttons = [
            [
                InlineKeyboardButton("7 дней", callback_data="dump_days_7"),
                InlineKeyboardButton("30 дней", callback_data="dump_days_30"),
                InlineKeyboardButton("90 дней", callback_data="dump_days_90"),
            ]
        ]
        await self._safe_reply(
            update,
            context,
            "За сколько дней нужно скачать сообщения? Введите число или выберите вариант:",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def _cmd_index(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id if update.effective_chat else 0
        args = getattr(context, "args", [])
        
        if args and args[0].lower() == "all":
            keyboard = InlineKeyboardMarkup(
                [[InlineKeyboardButton("❌ Отмена", callback_data="index_cancel")]]
            )
            msg = await update.message.reply_text(
                "Начинаю полную индексацию всех сообщений…", reply_markup=keyboard
            )
            self._index_state[chat_id] = {"full_index": True}
            await self._run_full_index(chat_id, context, msg, keyboard)
            return
            
        if not self._index_dumped:
            await self._safe_reply(update, context, "Индексация недоступна.")
            return
            
        if args and args[0].isdigit():
            days = int(args[0])
            if days <= 0:
                await self._safe_reply(update, context, "Некорректное число дней.")
                return
            keyboard = InlineKeyboardMarkup(
                [[InlineKeyboardButton("❌ Отмена", callback_data="index_cancel")]]
            )
            msg = await update.message.reply_text(
                "Начинаю построение индекса…", reply_markup=keyboard
            )
            self._index_state[chat_id] = {}
            await self._run_index(chat_id, context, msg, days, keyboard)
            return
        # Set a short-lived state asking for a numeric day count.
        # Add an expiry to prevent stale state from hijacking normal messages.
        try:
            from datetime import UTC, datetime, timedelta
            expires_at = (datetime.now(UTC) + timedelta(minutes=10)).isoformat()
        except Exception:
            expires_at = None  # fallback if timezone not available
        state_payload = {"step": "days"}
        if expires_at:
            state_payload["expires_at"] = expires_at
        self._index_state[chat_id] = state_payload
        buttons = [
            [
                InlineKeyboardButton("7 дней", callback_data="index_days_7"),
                InlineKeyboardButton("30 дней", callback_data="index_days_30"),
                InlineKeyboardButton("90 дней", callback_data="index_days_90"),
            ],
            [
                InlineKeyboardButton("🔄 Все сообщения", callback_data="index_all")
            ]
        ]
        await self._safe_reply(
            update,
            context,
            "За сколько дней нужно построить индекс? Введите число или выберите вариант:",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def _run_full_index(
        self,
        chat_id: int,
        context: ContextTypes.DEFAULT_TYPE,
        msg,
        keyboard: InlineKeyboardMarkup,
    ) -> None:
        """Выполняет полную индексацию всех сообщений через TelethonIndexer."""
        state = self._index_state.get(chat_id, {})
        state["cancel"] = False
        self._index_state[chat_id] = state

        UPDATE_INTERVAL = 2
        start_time = datetime.now(UTC)
        last_update = start_time

        async def _progress_worker(count: int) -> None:
            nonlocal last_update
            now = datetime.now(UTC)
            if (now - last_update).total_seconds() < UPDATE_INTERVAL:
                return
            if state.get("cancel"):
                return
            elapsed = (now - start_time).total_seconds()
            speed = count / elapsed if elapsed > 0 else 0
            text = (
                f"Полная индексация… обработано сообщений: {count}, "
                f"скорость: {speed:.1f}/с"
            )
            try:
                await safe_edit_message(msg, text, reply_markup=keyboard)
                last_update = now
            except Exception:
                pass

        def progress_cb(count: int) -> None:
            asyncio.create_task(_progress_worker(count))

        def is_cancelled() -> bool:
            return bool(state.get("cancel"))

        try:
            if hasattr(self, '_tele_indexer') and self._tele_indexer:
                messages_count = 0
                
                vector_index = None
                if hasattr(self, '_retriever') and self._retriever:
                    vector_index = getattr(self._retriever, 'index', None)
                    if not vector_index and hasattr(self._retriever, 'vector_index'):
                        vector_index = getattr(self._retriever, 'vector_index', None)
                
                if not vector_index:
                    logger.error("VectorIndex недоступен в _run_full_index")
                    await safe_edit_message(msg, "❌ Ошибка: VectorIndex недоступен.")
                    return
                
                raw_storage = None
                if hasattr(self, '_tele_indexer') and hasattr(self._tele_indexer, '_raw_storage'):
                    raw_storage = self._tele_indexer._raw_storage
                elif hasattr(self, '_telethon_service') and self._telethon_service:
                    raw_storage = getattr(self._telethon_service, '_raw_storage', None)
                    # Если raw_storage не найден напрямую, попробуем через tele_indexer
                    if not raw_storage and hasattr(self._telethon_service, '_tele_indexer'):
                        tele_indexer = getattr(self._telethon_service, '_tele_indexer', None)
                        if tele_indexer:
                            raw_storage = getattr(tele_indexer, '_raw_storage', None)
                
                if not raw_storage:
                    logger.warning("RawStorage недоступен в _run_full_index, продолжаем без сохранения")
                
                logger.info("Начинаю полную индексацию через TelethonIndexer.index_once()")
                
                # Импортируем функцию извлечения данных
                from utils.message_extractor import extract_message_data
                
                async for message in self._tele_indexer.index_once():
                    if is_cancelled():
                        break
                    
                    # Используем новую функцию извлечения расширенной структуры
                    msg_data = extract_message_data(message)
                    
                    if raw_storage and msg_data["text"].strip():
                        try:
                            raw_storage.save(msg_data["chat"], msg_data)
                        except Exception as e:
                            logger.warning(f"Failed to store message in raw storage: {e}")
                    
                    if msg_data["text"].strip():
                        try:
                            await vector_index.add(
                                f"msg_{msg_data['id']}_{msg_data['date']}",
                                msg_data["text"],
                                {
                                    "chat": msg_data["chat"], 
                                    "date": msg_data["date"], 
                                    "theme": "default"
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Failed to index message in vector_index: {e}")
                    
                    messages_count += 1
                    if messages_count % 100 == 0:
                        progress_cb(messages_count)
                        logger.info(f"Проиндексировано {messages_count} сообщений")
                        
                await safe_edit_message(msg, f"✅ Полная индексация завершена! Обработано {messages_count} сообщений.")
                logger.info(f"Полная индексация завершена успешно: {messages_count} сообщений")
            else:
                logger.error("TelethonIndexer недоступен в _run_full_index")
                await safe_edit_message(msg, "❌ Ошибка: TelethonIndexer недоступен.")
        except Exception as e:
            logger.exception("Full indexing failed: %s", e)
            await safe_edit_message(msg, f"❌ Ошибка при индексации: {str(e)}")
        finally:
            self._index_state.pop(chat_id, None)

    async def _cb_reindex_select(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await self._safe_query_answer(query)
        data = query.data or ""
        chat_id = update.effective_chat.id if update.effective_chat else 0
        state = self._reindex_state.get(chat_id)
        if not state or state.get("step") != "choose_chats":
            await query.edit_message_text(
                "Сессия индексации прервана. Запустите /reindex заново."
            )
            return

        if data == "reidx_prev":
            state["current_page"] = state.get("current_page", 0) - 1
            await self._show_reindex_chat_selection(update, chat_id, edit_message=True)
            return
        if data == "reidx_next":
            state["current_page"] = state.get("current_page", 0) + 1
            await self._show_reindex_chat_selection(update, chat_id, edit_message=True)
            return
        if data.startswith("reidx_chat_"):
            idx = int(data.split("_")[-1])
            chats = state.get("chat_titles", [])
            if 0 <= idx < len(chats):
                title = chats[idx]
                selected = state.setdefault("selected_titles", [])
                if title in selected:
                    selected.remove(title)
                else:
                    selected.append(title)
            await self._show_reindex_chat_selection(update, chat_id, edit_message=True)
            return
        if data == "reidx_confirm":
            selected = state.get("selected_titles", [])
            if not selected:
                await self._safe_query_answer(
                    query, "Не выбрано ни одного чата", show_alert=True
                )
                return
            state["step"] = "choose_interval"
            buttons = [
                [
                    InlineKeyboardButton("🕐 За час", callback_data="reindex_hour"),
                    InlineKeyboardButton("📅 За день", callback_data="reindex_day"),
                    InlineKeyboardButton("📆 За неделю", callback_data="reindex_week"),
                ],
                [
                    InlineKeyboardButton("🗓️ За месяц", callback_data="reindex_month"),
                    InlineKeyboardButton(
                        "✏️ Количество месяцев", callback_data="reindex_months"
                    ),
                ],
            ]
            await query.edit_message_text(
                "Выберите интервал для индексации выбранных чатов:",
                reply_markup=InlineKeyboardMarkup(buttons),
            )
            return
        if data == "reidx_cancel":
            self._reindex_state.pop(chat_id, None)
            await query.edit_message_text("Выбор чатов отменён.")
            return

    async def _cb_reindex_interval(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await self._safe_query_answer(query)
        data = query.data
        chat_id = update.effective_chat.id if update.effective_chat else 0
        state = self._reindex_state.get(chat_id)
        if not state:
            await query.edit_message_text(
                "Сессия индексации прервана. Запустите /reindex заново."
            )
            return
        selected = state.get("selected_titles", [])
        if not selected:
            await query.edit_message_text("Сначала выберите чаты через /reindex")
            return
        now = datetime.now(UTC)
        if data == "reindex_hour":
            since = now - timedelta(hours=1)
            label = "за последний час"
        elif data == "reindex_day":
            since = now - timedelta(days=1)
            label = "за последние 24 часа"
        elif data == "reindex_week":
            since = now - timedelta(days=7)
            label = "за последнюю неделю"
        elif data == "reindex_month":
            since = now - timedelta(days=30)
            label = "за последний месяц"
        elif data == "reindex_months":
            state["step"] = "custom_months"
            await query.edit_message_text("Введите количество месяцев для индексации:")
            return
        else:
            await query.edit_message_text("Неизвестный интервал.")
            return
        try:
            if not hasattr(context.application, "bot_data"):
                context.application.bot_data = {}
            keyboard = InlineKeyboardMarkup(
                [[InlineKeyboardButton("❌ Отмена", callback_data="reindex_cancel")]]
            )
            msg = await query.edit_message_text(
                f"Начинаю индексацию {label}…", reply_markup=keyboard
            )
            index_since_cb = getattr(self, "_index_since", None)
            if not index_since_cb:
                await safe_edit_message(
                    msg,
                    "Функция индексации по интервалу недоступна в этой сборке."
                )
                return

            count_total_cb = getattr(self, "_count_messages_since", None)
            total: Optional[int] = None
            if count_total_cb:
                try:
                    total = await count_total_cb(selected, since)
                    state["total"] = total
                except Exception:
                    total = None

            state["cancel"] = False
            UPDATE_INTERVAL = 2
            start_time = datetime.now(UTC)
            last_update = start_time

            async def _progress_worker(count: int) -> None:
                nonlocal last_update
                now = datetime.now(UTC)
                if (now - last_update).total_seconds() < UPDATE_INTERVAL:
                    return
                if state.get("cancel"):
                    return
                elapsed = (now - start_time).total_seconds()
                speed = count / elapsed if elapsed > 0 else 0
                if total and total > 0:
                    percent = min(100.0, count / total * 100)
                    remaining = (total - count) / speed if speed > 0 else 0
                    text = (
                        f"Индексация {label}… {percent:.1f}% ({count}/{total}), "
                        f"скорость: {speed:.1f}/с, осталось: {format_seconds(remaining)}"
                    )
                else:
                    text = (
                        f"Индексация {label}… обработано сообщений: {count}, "
                        f"скорость: {speed:.1f}/с"
                    )
                try:
                    await safe_edit_message(msg, text, reply_markup=keyboard)
                    last_update = now
                except Exception:
                    pass

            def progress_cb(count: int) -> None:
                asyncio.create_task(_progress_worker(count))

            def is_cancelled() -> bool:
                return bool(state.get("cancel"))

            count = await index_since_cb(
                selected,
                since,
                progress_cb=progress_cb,
                is_cancelled=is_cancelled,
            )
            if state.get("cancel"):
                await safe_edit_message(msg, f"Операция отменена. Обработано сообщений: {count}")
            else:
                await safe_edit_message(msg, f"Готово. Проиндексировано сообщений: {count}")
            self._reindex_state.pop(chat_id, None)
            try:
                await self._show_main_menu(chat_id, context)
            except Exception:
                pass
        except Exception as exc:
            logger.exception("Interval reindex failed: %s", exc)
            await query.edit_message_text("Ошибка при индексации. Подробности в логах.")

    async def _cb_reindex_cancel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await self._safe_query_answer(query)
        chat_id = update.effective_chat.id if update.effective_chat else 0
        state = self._reindex_state.get(chat_id)
        if state is not None:
            state["cancel"] = True
            try:
                await query.edit_message_text("Операция прерывается…")
            except Exception:
                pass

    async def _run_dump(
        self,
        chat_id: int,
        context: ContextTypes.DEFAULT_TYPE,
        msg,
        days: int,
        keyboard: InlineKeyboardMarkup,
    ) -> None:
        state = self._dump_state.get(chat_id, {})
        state["cancel"] = False
        self._dump_state[chat_id] = state

        count_total_cb = getattr(self, "_count_dump_since", None)
        total: Optional[int] = None
        if count_total_cb:
            try:
                total = await count_total_cb(days)
                state["total"] = total
            except Exception:
                total = None

        UPDATE_INTERVAL = 2
        start_time = datetime.now(UTC)
        last_update = start_time

        async def _progress_worker(count: int) -> None:
            nonlocal last_update
            now = datetime.now(UTC)
            if (now - last_update).total_seconds() < UPDATE_INTERVAL:
                return
            if state.get("cancel"):
                return
            elapsed = (now - start_time).total_seconds()
            speed = count / elapsed if elapsed > 0 else 0
            if total and total > 0:
                percent = min(100.0, count / total * 100)
                remaining = (total - count) / speed if speed > 0 else 0
                text = (
                    f"Выгружаю сообщения… {percent:.1f}% ({count}/{total}), "
                    f"скорость: {speed:.1f}/с, осталось: {format_seconds(remaining)}"
                )
            else:
                text = (
                    f"Выгружаю сообщения… сохранено: {count}, "
                    f"скорость: {speed:.1f}/с"
                )
            try:
                await safe_edit_message(msg, text, reply_markup=keyboard)
                last_update = now
            except Exception:
                pass

        def progress_cb(count: int) -> None:
            asyncio.create_task(_progress_worker(count))

        def is_cancelled() -> bool:
            return bool(state.get("cancel"))

        try:
            count = await self._dump_since(
                days, progress_cb=progress_cb, is_cancelled=is_cancelled
            )
            if state.get("cancel"):
                await safe_edit_message(
                    msg,
                    f"Операция отменена. Сохранено сообщений: {count}",
                    reply_markup=None,
                )
            else:
                await safe_edit_message(
                    msg,
                    f"Готово. Сохранено сообщений: {count}",
                    reply_markup=None,
                )
        except Exception as exc:
            logger.exception("Dump failed: %s", exc)
            await safe_edit_message(msg, "Ошибка при выгрузке сообщений")
        finally:
            self._dump_state.pop(chat_id, None)
            try:
                await self._show_main_menu(chat_id, context)
            except Exception:
                pass

    async def _cb_dump_days(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await self._safe_query_answer(query)
        data = query.data or ""
        chat_id = update.effective_chat.id if update.effective_chat else 0
        state = self._dump_state.get(chat_id)
        if not state or state.get("step") != "days":
            await query.edit_message_text(
                "Сессия выгрузки прервана. Запустите /dump заново."
            )
            return
        try:
            days = int(data.rsplit("_", 1)[-1])
            if days <= 0:
                raise ValueError
        except Exception:
            await query.edit_message_text("Некорректное число дней.")
            return
        if not self._dump_since:
            await query.edit_message_text("Выгрузка недоступна.")
            self._dump_state.pop(chat_id, None)
            return
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("❌ Отмена", callback_data="dump_cancel")]]
        )
        msg = await query.edit_message_text(
            "Начинаю выгрузку сообщений…", reply_markup=keyboard
        )
        await self._run_dump(chat_id, context, msg, days, keyboard)

    async def _run_index(
        self,
        chat_id: int,
        context: ContextTypes.DEFAULT_TYPE,
        msg,
        days: int,
        keyboard: InlineKeyboardMarkup,
    ) -> None:
        state = self._index_state.get(chat_id, {})
        state["cancel"] = False
        self._index_state[chat_id] = state

        UPDATE_INTERVAL = 2
        start_time = datetime.now(UTC)
        last_update = start_time

        async def _progress_worker(count: int) -> None:
            nonlocal last_update
            now = datetime.now(UTC)
            if (now - last_update).total_seconds() < UPDATE_INTERVAL:
                return
            if state.get("cancel"):
                return
            elapsed = (now - start_time).total_seconds()
            speed = count / elapsed if elapsed > 0 else 0
            text = (
                f"Индексация… обработано сообщений: {count}, "
                f"скорость: {speed:.1f}/с"
            )
            try:
                await safe_edit_message(msg, text, reply_markup=keyboard)
                last_update = now
            except Exception:
                pass

        def progress_cb(count: int) -> None:
            asyncio.create_task(_progress_worker(count))

        def is_cancelled() -> bool:
            return bool(state.get("cancel"))

        try:
            count = await self._index_dumped(
                days, progress_cb=progress_cb, is_cancelled=is_cancelled
            )
            if state.get("cancel"):
                await safe_edit_message(
                    msg,
                    f"Операция отменена. Проиндексировано сообщений: {count}",
                    reply_markup=None,
                )
            else:
                await safe_edit_message(
                    msg,
                    f"Готово. Проиндексировано сообщений: {count}",
                    reply_markup=None,
                )
        except Exception as exc:
            logger.exception("Indexing failed: %s", exc)
            await safe_edit_message(msg, "Ошибка при индексации сообщений")
        finally:
            self._index_state.pop(chat_id, None)
            try:
                await self._show_main_menu(chat_id, context)
            except Exception:
                pass

    async def _cb_index_days(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await self._safe_query_answer(query)
        data = query.data or ""
        chat_id = update.effective_chat.id if update.effective_chat else 0
        state = self._index_state.get(chat_id)
        if not state or state.get("step") != "days":
            await query.edit_message_text(
                "Сессия индексации прервана. Запустите /index заново."
            )
            return
        try:
            days = int(data.rsplit("_", 1)[-1])
            if days <= 0:
                raise ValueError
        except Exception:
            await query.edit_message_text("Некорректное число дней.")
            return
        if not self._index_dumped:
            await query.edit_message_text("Индексация недоступна.")
            self._index_state.pop(chat_id, None)
            return
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("❌ Отмена", callback_data="index_cancel")]]
        )
        msg = await query.edit_message_text(
            "Начинаю построение индекса…", reply_markup=keyboard
        )
        await self._run_index(chat_id, context, msg, days, keyboard)

    async def _cb_index_all(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Обработчик кнопки 'Все сообщения' для полной индексации."""
        query = update.callback_query
        await self._safe_query_answer(query)
        chat_id = update.effective_chat.id if update.effective_chat else 0
        state = self._index_state.get(chat_id)
        if not state or state.get("step") != "days":
            await query.edit_message_text(
                "Сессия индексации прервана. Запустите /index заново."
            )
            return

        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("❌ Отмена", callback_data="index_cancel")]]
        )
        msg = await query.edit_message_text(
            "Начинаю полную индексацию всех сообщений…", reply_markup=keyboard
        )
        self._index_state[chat_id] = {"full_index": True}
        await self._run_full_index(chat_id, context, msg, keyboard)

    async def _cb_dump_cancel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await self._safe_query_answer(query)
        chat_id = update.effective_chat.id if update.effective_chat else 0
        state = self._dump_state.get(chat_id)
        if state is not None:
            state["cancel"] = True
            try:
                await query.edit_message_text("Операция прерывается…")
            except Exception:
                pass

    async def _cb_index_cancel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await self._safe_query_answer(query)
        chat_id = update.effective_chat.id if update.effective_chat else 0
        state = self._index_state.get(chat_id)
        if state is not None:
            state["cancel"] = True
            try:
                await query.edit_message_text("Операция прерывается…")
            except Exception:
                pass
