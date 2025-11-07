"""Telegram bot module for AI assistant.

This module defines a small wrapper around ``python-telegram-bot`` to interact
with the user. It exposes a :class:`TelegramBot` class that routes incoming
messages to a callback and returns the model's response.

The real project would add keyboards for confirmation of commands and more
advanced error handling.  Here we only provide the minimal asynchronous
interface required by the specification.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from finance.finrl_agent import AnalysisResult
from retriever.retriever import Retriever
from services.event_bus import AsyncEventBus

from .persistence import PersistedDict as _PersistedDict
from .reindex_handlers import ReindexHandlersMixin
from .telethon_handlers import TelethonHandlersMixin
from .theme_callbacks import ThemeCallbacksMixin
from .theme_chat_callbacks import ThemeChatCallbacksMixin
from .theme_commands import ThemeCommandsMixin
from .utils import format_seconds, split_long
from .telegram_utils import safe_edit_message

StreamCallback = Callable[[str], Awaitable[None]]

logger = logging.getLogger(__name__)


class BroadcastExecutor:
    """Simple notification helper to broadcast trading signals via Telegram."""

    def __init__(self, bot: "TelegramBot" | None = None):
        self._bot = bot
        self._alert_handler: Optional["TradingAlertHandler"] = None

    def attach_bot(self, bot: "TelegramBot") -> None:
        self._bot = bot
        # Инициализируем TradingAlertHandler после прикрепления бота
        if self._bot:
            from .trading_alert_handler import TradingAlertHandler
            self._alert_handler = TradingAlertHandler(self._bot)

    async def broadcast_trade_signal(self, alert_payload) -> None:
        if self._bot is None:
            logger.warning("Broadcast executor missing bot instance")
            return

        chat_id = int(os.getenv("TRADING_ALERT_CHAT_ID", "0"))
        if not chat_id:
            logger.warning("TRADING_ALERT_CHAT_ID not configured")
            return

        if isinstance(alert_payload, dict):
            signal_data = alert_payload
        else:
            signal_data = alert_payload.model_dump(mode="json")  # type: ignore[attr-defined]

        signal_data.setdefault("id", signal_data.get("id") or signal_data.get("external_id") or "unknown")

        if self._alert_handler and self._bot._application:
            try:
                await self._alert_handler.send_alert(chat_id, signal_data)
                logger.info("interactive_trading_alert_sent", symbol=signal_data.get("symbol"))
                return
            except Exception as exc:  # pragma: no cover - fallback path
                logger.exception("interactive_alert_failed", error=str(exc))

        await self._send_simple_alert(chat_id, signal_data)

    async def _send_simple_alert(self, chat_id: int, alert_payload) -> None:
        """Fallback метод для отправки простого алерта без кнопок."""
        if isinstance(alert_payload, dict):
            symbol = alert_payload.get("symbol", "-")
            timeframe = alert_payload.get("timeframe", "-")
            direction = alert_payload.get("direction", "-")
            entry = alert_payload.get("entry", "-")
            confidence = alert_payload.get("confidence")
        else:
            symbol = getattr(alert_payload, "symbol", "-")
            timeframe = getattr(alert_payload, "timeframe", "-")
            direction = getattr(alert_payload, "direction", "-")
            entry = getattr(alert_payload, "entry", "-")
            confidence = getattr(alert_payload, "confidence", None)
        message = (
            f"⚡️ Trading Alert\n"
            f"Symbol: {symbol}\n"
            f"Timeframe: {timeframe}\n"
            f"Direction: {direction}\n"
            f"Entry: {entry}\n"
            f"Confidence: {confidence if confidence is not None else '-'}\n"
        )
        if self._bot._application is None:
            logger.warning("Telegram application not started; cannot send alert")
            return
        async with self._bot._application.bot as bot_client:
            await bot_client.send_message(chat_id=chat_id, text=message)


async def create_broadcast_executor() -> BroadcastExecutor:
    return BroadcastExecutor()


try:  # ``python-telegram-bot`` is an optional dependency in this template.
    from telegram import (
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        Update,
        Message,
        CallbackQuery,
    )
    from telegram.constants import ParseMode
    from telegram.error import TelegramError, TimedOut
    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        CommandHandler,
        ContextTypes,
        Defaults,
        MessageHandler,
        filters,
    )
    from telegram.helpers import escape_markdown
    from telegram.request import HTTPXRequest
except Exception:  # pragma: no cover - library may be missing in tests
    Update = ContextTypes = MessageHandler = filters = Application = None  # type: ignore
    InlineKeyboardButton = InlineKeyboardMarkup = CallbackQueryHandler = None  # type: ignore
    Message = CallbackQuery = None  # type: ignore
    ParseMode = Defaults = None  # type: ignore

    class TelegramError(Exception):  # type: ignore
        pass

    def escape_markdown(text: str, **kwargs):  # type: ignore
        return text


STATE_FILE = "bot_state.json"
MAX_SEARCH_RESULTS = 20


class TelegramBot(
    TelethonHandlersMixin,
    ReindexHandlersMixin,
    ThemeCommandsMixin,
    ThemeCallbacksMixin,
    ThemeChatCallbacksMixin,
):
    """Simple Telegram bot wrapper.

    Parameters
    ----------
    token:
        API token obtained from BotFather.
    on_message:
        Coroutine executed for every incoming message. It should accept the
        message text, ``chat_id`` and return a reply that will be sent back to
        the user.
    on_message_stream:
        Optional coroutine for streaming replies. It receives the message text,
        ``chat_id`` and a callback that should be invoked with partial pieces of
        the response. Each invocation of the callback updates the previously
        sent message in chat.
    """

    # Pagination constants
    CHATS_PER_PAGE = 8  # Number of chats to show per page
    MAX_CHAT_NAME_LENGTH = 30  # Maximum length for chat names in buttons

    async def _safe_reply(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, **kwargs
    ) -> None:
        """Safely reply to update, handling both direct commands and menu callbacks."""
        if update.message:
            await update.message.reply_text(text, **kwargs)
        elif update.callback_query and update.callback_query.message:
            # For menu callbacks, edit the message
            await update.callback_query.edit_message_text(text, **kwargs)
        else:
            # Fallback - try to send to chat_id if available
            chat_id = None
            if update.callback_query:
                chat_id = (
                    update.callback_query.message.chat_id
                    if update.callback_query.message
                    else None
                )
            if chat_id and context.bot:
                await context.bot.send_message(chat_id=chat_id, text=text, **kwargs)

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик ошибок для Application."""
        logger.error("Telegram bot error occurred", exc_info=context.error)
        
        # Пытаемся отправить уведомление об ошибке пользователю
        if update and hasattr(update, 'effective_chat') and update.effective_chat:
            try:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Произошла ошибка при обработке запроса. Попробуйте позже."
                )
            except Exception:
                # Если не удалось отправить сообщение, просто логируем
                logger.error("Failed to send error notification to user")

    async def _safe_query_answer(self, query: Any, *args: Any, **kwargs: Any) -> None:
        """Answer a callback query and log telegram errors."""
        try:
            await query.answer(*args, **kwargs)
        except TelegramError as exc:  # type: ignore[misc]
            logger.warning("Callback query answer failed: %s", exc)

    async def _send_or_edit_menu(
        self,
        chat_id: int,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
        markup: "InlineKeyboardMarkup",
        *,
        source: Optional[Union["CallbackQuery", "Message"]] = None,
        log_context: str,
    ) -> bool:
        """Send a new menu message or edit the existing one when possible."""

        message = None
        if source is not None:
            message = getattr(source, "message", source)

        if message is not None:
            if CallbackQuery is not None and isinstance(source, CallbackQuery):
                if await safe_edit_message(message, text, reply_markup=markup):
                    return True
            else:
                edit_method = getattr(message, "edit_text", None)
                if callable(edit_method):
                    try:
                        await edit_method(text, reply_markup=markup)
                        return True
                    except Exception:
                        logger.exception("Failed to edit message in %s", log_context)

        bot = getattr(context, "bot", None)
        if bot is None:
            logger.warning("Context missing bot in %s", log_context)
            return False

        await bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=markup,
        )
        return True

    def __init__(
        self,
        token: str,
        on_message: Callable[[str, int], Awaitable[str]],
        on_message_stream: Optional[
            Callable[[str, int, StreamCallback], Awaitable[None]]
        ] = None,
        list_chats: Optional[Callable[[], Awaitable[List[str]]]] = None,
        index_last: Optional[Callable[[List[str], int], Awaitable[int]]] = None,
        summarize_interval: Optional[Callable[[int], Awaitable[str]]] = None,
        set_summary_interval: Optional[Callable[[int], Awaitable[str]]] = None,
        list_themes: Optional[Callable[[], Awaitable[List[str]]]] = None,
        set_active_theme: Optional[Callable[[str], Awaitable[None]]] = None,
        get_active_theme_name: Optional[Callable[[], Awaitable[str]]] = None,
        run_finance_analysis: Optional[
            Callable[[List[str]], Awaitable[AnalysisResult]]
        ] = None,
        telethon_auth_request_code: Optional[Callable[[str], Awaitable[None]]] = None,
        telethon_auth_sign_in: Optional[
            Callable[[str, str, Optional[str]], Awaitable[Dict[str, str]]]
        ] = None,
        telethon_is_authorized: Optional[Callable[[], Awaitable[bool]]] = None,
        create_new_theme: Optional[Callable[[str, List[str]], Awaitable[bool]]] = None,
        delete_theme_by_name: Optional[Callable[[str], Awaitable[bool]]] = None,
        get_theme_chats: Optional[Callable[[str], Awaitable[List[str]]]] = None,
        add_chat_to_theme_by_name: Optional[
            Callable[[str, str], Awaitable[bool]]
        ] = None,
        remove_chat_from_theme_by_name: Optional[
            Callable[[str, str], Awaitable[bool]]
        ] = None,
        refresh_chat_cache: Optional[Callable[[], Awaitable[bool]]] = None,
        dump_since: Optional[Callable[[int], Awaitable[int]]] = None,
        index_dumped: Optional[Callable[[int], Awaitable[int]]] = None,
        publish_chronicle: Optional[Callable[[], Awaitable[None]]] = None,
        retriever: Optional[Retriever] = None,
        index_state_path: Optional[Path] = None,
        summarize_cluster: Optional[Callable[[List[str]], Awaitable[str]]] = None,
        summarize_as_agent: Optional[Callable[[List[str]], Awaitable[str]]] = None,
        summarize_url: Optional[Callable[[str], Awaitable[str]]] = None,
        bus: Optional[AsyncEventBus] = None,
        tele_indexer: Optional[object] = None,
        telethon_service: Optional[object] = None,
    ):
        self.token = token
        self.on_message = on_message
        self.on_message_stream = on_message_stream
        self.app: Optional[Application] = None
        self._broadcast_executor: BroadcastExecutor | None = None
        # Optional reindex flow callbacks
        self._list_chats = list_chats
        self._index_last = index_last
        # Optional summarize callback
        self._summarize_interval = summarize_interval
        # Optional onboarding + themes
        self._list_themes = list_themes
        self._set_active_theme = set_active_theme
        self._get_active_theme_name = get_active_theme_name
        self._run_finance_analysis = run_finance_analysis
        # Telethon auth callbacks
        self._telethon_auth_request_code = telethon_auth_request_code
        self._telethon_auth_sign_in = telethon_auth_sign_in
        self._telethon_is_authorized = telethon_is_authorized
        # Theme management callbacks
        self._create_new_theme = create_new_theme
        self._delete_theme_by_name = delete_theme_by_name
        self._get_theme_chats = get_theme_chats
        self._add_chat_to_theme_by_name = add_chat_to_theme_by_name
        self._remove_chat_from_theme_by_name = remove_chat_from_theme_by_name
        self._refresh_chat_cache = refresh_chat_cache
        self._dump_since = dump_since
        self._index_dumped = index_dumped
        self._publish_chronicle = publish_chronicle
        self._retriever = retriever
        self._index_state_path = index_state_path
        self._summarize_cluster = summarize_cluster
        self._summarize_as_agent = summarize_as_agent
        self._summarize_url = summarize_url
        self._bus = bus
        self._tele_indexer = tele_indexer
        self._telethon_service = telethon_service
        # Callback to change summary interval
        self._set_summary_interval = set_summary_interval
        # Central registry of bot commands with descriptions and usage examples
        self._commands: Dict[str, Dict[str, object]] = {}
        self._commands["help"] = {
            "handler": self._cmd_help,
            "description": "— показать эту справку",
            "usage": "/help",
        }
        self._commands["menu"] = {
            "handler": self._cmd_menu,
            "description": "— открыть главное меню",
            "usage": "/menu",
        }
        self._commands["status"] = {
            "handler": self._cmd_status,
            "description": "— показать активную тему, статус Telethon и размер индекса",
            "usage": "/status",
        }
        if self._publish_chronicle:
            self._commands["chronicle"] = {
                "handler": self._cmd_chronicle,
                "description": "— опубликовать обзор тем",
                "usage": "/chronicle",
            }
        if self._retriever:
            self._commands["search"] = {
                "handler": self._cmd_search,
                "description": (
                    f"[N] запрос — поиск по индексу и вывод топ-N сообщений (макс. {MAX_SEARCH_RESULTS})",
                ),
                "usage": "/search 5 интересная тема",
            }
        if self._list_chats and self._index_last:
            self._commands["reindex"] = {
                "handler": self._cmd_reindex,
                "description": "— интерактивная индексация: выбрать чаты и интервал (час/день/неделя)",
                "usage": "/reindex",
            }
        if self._dump_since:
            self._commands["dump"] = {
                "handler": self._cmd_dump,
                "description": "— скачать сообщения за N дней (число можно указать сразу)",
                "usage": "/dump 7",
            }
        if self._index_dumped:
            self._commands["index"] = {
                "handler": self._cmd_index,
                "description": "— построить индекс по скачанным сообщениям за N дней или /index all для полной индексации",
                "usage": "/index 7\n/index all",
            }
        if self._summarize_interval:
            self._commands["summary"] = {
                "handler": self._cmd_summary,
                "description": "— сводка сообщений за N часов или выбор периода из меню",
                "usage": "/summary 3",
            }
        if self._set_summary_interval:
            self._commands["set_summary_interval"] = {
                "handler": self._cmd_set_summary_interval,
                "description": "— задать интервал периодических сводок в часах (число можно указать сразу)",
                "usage": "/set_summary_interval 12",
            }
        if self._summarize_as_agent:
            self._commands["agent_report"] = {
                "handler": self._cmd_agent_report,
                "description": "тексты — агентский отчёт по сообщениям",
                "usage": "/agent_report сообщение1 | сообщение2",
            }
        if self._summarize_url:
            self._commands["sumurl"] = {
                "handler": self._cmd_sumurl,
                "description": "ссылка — краткое содержание страницы",
                "usage": "/sumurl https://example.com",
            }
        if self._summarize_cluster:
            self._commands["cluster"] = {
                "handler": self._cmd_cluster,
                "description": "тексты — краткое резюме темы по сообщениям",
                "usage": "/cluster сообщение1 | сообщение2",
            }
        if self._run_finance_analysis:
            self._commands["signals"] = {
                "handler": self._cmd_signals,
                "description": "тикеры через пробел или запятую — финансовый анализ указанных тикеров",
                "usage": "/signals GAZP SBER\n/signals GAZP,SBER",
            }
        if self._list_themes:
            self._commands["themes"] = {
                "handler": self._cmd_themes,
                "description": "— управление темами",
                "usage": "/themes",
            }
        if self._create_new_theme:
            self._commands["create_theme"] = {
                "handler": self._cmd_create_theme,
                "description": "<название> — создать новую тему",
                "usage": "/create_theme Новая тема",
            }
        if self._delete_theme_by_name:
            self._commands["delete_theme"] = {
                "handler": self._cmd_delete_theme,
                "description": "— удалить тему",
                "usage": "/delete_theme",
            }
        if self._get_theme_chats:
            self._commands["edit_theme"] = {
                "handler": self._cmd_edit_theme,
                "description": "— редактировать чаты в теме",
                "usage": "/edit_theme",
            }
        if self._set_active_theme:
            self._commands["switch_theme"] = {
                "handler": self._cmd_switch_theme,
                "description": "— переключить активную тему",
                "usage": "/switch_theme",
            }
        if self._list_chats and self._create_new_theme and self._set_active_theme:
            self._commands["start"] = {
                "handler": self._cmd_start,
                "description": "— начальная настройка: создать тему и выбрать чаты",
                "usage": "/start",
            }
        self._commands["cancel"] = {
            "handler": self._cmd_cancel,
            "description": "— отменить текущую операцию",
            "usage": "/cancel",
        }
        # Help sections with command categories and examples
        self._help_sections = {
            "main": {
                "title": "Основные",
                "commands": ["help", "menu", "status", "start", "cancel"],
            },
            "search": {"title": "Поиск", "commands": ["search", "sumurl"]},
            "index": {
                "title": "Индекс",
                "commands": [
                    "reindex",
                    "dump",
                    "index",
                    "summary",
                    "agent_report",
                    "cluster",
                    "chronicle",
                    "set_summary_interval",
                ],
            },
            "themes": {
                "title": "Темы",
                "commands": [
                    "themes",
                    "create_theme",
                    "delete_theme",
                    "edit_theme",
                    "switch_theme",
                ],
            },
            "finance": {"title": "Финансы", "commands": ["signals"]},
        }
        for section in self._help_sections.values():
            section["commands"] = [
                name for name in section["commands"] if name in self._commands
            ]
        self._register_telethon()
        # chat_id -> {step, chat_titles, selected_titles}
        self._reindex_state: Dict[int, Dict] = _PersistedDict(
            save_callback=self._save_states
        )
        self._auth_state: Dict[int, Dict] = _PersistedDict(
            save_callback=self._save_states
        )
        self._theme_state: Dict[int, Dict] = _PersistedDict(
            save_callback=self._save_states
        )
        self._dump_state: Dict[int, Dict] = _PersistedDict(
            save_callback=self._save_states
        )
        self._index_state: Dict[int, Dict] = _PersistedDict(
            save_callback=self._save_states
        )
        self._summary_state: Dict[int, Dict] = _PersistedDict(
            save_callback=self._save_states
        )
        self._set_interval_state: Dict[int, Dict] = _PersistedDict(
            save_callback=self._save_states
        )
        self._cluster_state: Dict[int, bool] = {}
        self._agent_report_state: Dict[int, bool] = {}
        # initial setup state per chat
        self._setup_state: Dict[int, Dict] = _PersistedDict(
            save_callback=self._save_states
        )
        self._state_file = STATE_FILE
        self._load_states()
        if self._broadcast_executor:
            self._broadcast_executor.attach_bot(self)

    def set_broadcast_executor(self, executor: BroadcastExecutor) -> None:
        self._broadcast_executor = executor
        executor.attach_bot(self)

    async def send_message(self, chat_id: int, text: str, **kwargs: Any) -> Any:
        """Delegate to the underlying telegram bot if available.

        Some services expect the bot to expose ``send_message`` directly.
        This method waits briefly for the application to be initialized and
        then calls ``app.bot.send_message``.
        """
        # Avoid race at startup: wait up to 5s for app/bot to initialize
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 5.0
        while True:
            app = getattr(self, "app", None)
            bot = getattr(app, "bot", None) if app is not None else None
            if bot is not None:
                return await bot.send_message(chat_id=chat_id, text=text, **kwargs)
            if loop.time() >= deadline:
                raise RuntimeError("Telegram application is not started yet")
            await asyncio.sleep(0.1)

    def _save_states(self) -> None:
        data = {
            "_reindex_state": self._reindex_state,
            "_auth_state": self._auth_state,
            "_theme_state": self._theme_state,
            "_dump_state": self._dump_state,
            "_index_state": self._index_state,
            "_summary_state": self._summary_state,
            "_set_interval_state": self._set_interval_state,
            "_setup_state": self._setup_state,
        }
        try:
            tmp = self._state_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self._state_file)
        except Exception as exc:
            logger.exception("Failed to save bot state: %s", exc)

    def _load_states(self) -> None:
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.exception("Failed to load bot state: %s", exc)
            return

        now = datetime.now(UTC).timestamp()
        for name in [
            "_reindex_state",
            "_auth_state",
            "_theme_state",
            "_dump_state",
            "_index_state",
            "_summary_state",
            "_set_interval_state",
            "_setup_state",
        ]:
            raw = data.get(name, {})
            cleaned = {}
            for chat_id, state in raw.items():
                try:
                    chat_id_int = int(chat_id)
                except Exception:
                    continue
                expires = (
                    state.get("expires_at")
                    or state.get("expire_at")
                    or state.get("expires")
                    or state.get("expiry")
                )
                if isinstance(expires, str):
                    try:
                        expires = datetime.fromisoformat(expires).timestamp()
                    except Exception:
                        expires = None
                if isinstance(expires, (int, float)) and expires < now:
                    continue
                if any(
                    state.get(k) in (True, "true", "done", "completed")
                    for k in ("completed", "finished", "done", "status")
                ):
                    continue
                cleaned[chat_id_int] = state
            setattr(
                self,
                name,
                _PersistedDict(cleaned, save_callback=self._save_states),
            )

        # Persist cleaned states
        self._save_states()

    async def start(self) -> None:
        """Start polling Telegram for new messages."""
        if Application is None:  # pragma: no cover - informative error
            raise RuntimeError("python-telegram-bot is required to run the bot")

        defaults = Defaults(parse_mode=ParseMode.MARKDOWN)
        
        # Настраиваем HTTPXRequest с увеличенными таймаутами
        request = HTTPXRequest(
            connect_timeout=10,
            read_timeout=60,
            write_timeout=60,
            pool_timeout=60
        )
        
        self.app = Application.builder().token(self.token).defaults(defaults).request(request).build()
        
        # Добавляем обработчики ошибок
        self.app.add_error_handler(self._error_handler)
        
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )
        # Register command handlers from central registry
        for name, meta in self._commands.items():
            handler = meta.get("handler")
            if handler:
                self.app.add_handler(CommandHandler(name, handler))
        # New: reindex by interval callback and cancellation
        if "reindex" in self._commands:
            self.app.add_handler(
                CallbackQueryHandler(self._cb_reindex_interval, pattern=r"^reindex_")
            )
            self.app.add_handler(
                CallbackQueryHandler(
                    self._cb_reindex_cancel, pattern=r"^reindex_cancel$"
                )
            )
            self.app.add_handler(
                CallbackQueryHandler(self._cb_reindex_select, pattern=r"^reidx_")
            )
        
        # Trading alert callbacks
        self.app.add_handler(
            CallbackQueryHandler(self._cb_trading_alert, pattern=r"^(take|skip|details)_")
        )
        if "dump" in self._commands:
            self.app.add_handler(
                CallbackQueryHandler(self._cb_dump_days, pattern=r"^dump_days_\d+$")
            )
            self.app.add_handler(
                CallbackQueryHandler(self._cb_dump_cancel, pattern=r"^dump_cancel$")
            )
        if "index" in self._commands:
            self.app.add_handler(
                CallbackQueryHandler(self._cb_index_days, pattern=r"^index_days_\d+$")
            )
            self.app.add_handler(
                CallbackQueryHandler(self._cb_index_all, pattern=r"^index_all$")
            )
            self.app.add_handler(
                CallbackQueryHandler(self._cb_index_cancel, pattern=r"^index_cancel$")
            )
        # /summary related callbacks
        if "summary" in self._commands:
            self.app.add_handler(
                CallbackQueryHandler(
                    self._cb_summary, pattern=r"^summary:(?:\d+|custom)$"
                )
            )
        if "set_summary_interval" in self._commands:
            self.app.add_handler(
                CallbackQueryHandler(
                    self._cb_set_summary_interval,
                    pattern=r"^summary_int_(?:\d+|custom)$",
                )
            )
        # Theme management callback handlers
        self.app.add_handler(
            CallbackQueryHandler(self._cb_theme_action, pattern=r"^theme_")
        )
        self.app.add_handler(
            CallbackQueryHandler(self._cb_chat_toggle, pattern=r"^chat_")
        )
        self.app.add_handler(CallbackQueryHandler(self._cb_chat_nav, pattern=r"^nav_"))
        # Initial setup callbacks
        if "start" in self._commands:
            self.app.add_handler(
                CallbackQueryHandler(self._cb_setup, pattern=r"^setup_")
            )
        # Help callbacks
        self.app.add_handler(CallbackQueryHandler(self._cb_help, pattern=r"^help_"))
        # Menu callbacks
        self.app.add_handler(CallbackQueryHandler(self._cb_menu, pattern=r"^menu_"))

        logger.info("Telegram bot started")
        # Fully-async startup path for PTB v20+
        try:
            if hasattr(self.app, "initialize"):
                await self.app.initialize()
            await self.app.start()
            if hasattr(self.app, "updater") and self.app.updater is not None:  # type: ignore[attr-defined]
                await self.app.updater.start_polling()  # type: ignore[attr-defined]
                # Wait until stop signal
                if hasattr(self.app.updater, "wait_until_shutdown"):
                    await self.app.updater.wait_until_shutdown()  # type: ignore[attr-defined]
                else:  # pragma: no cover - very old versions
                    await asyncio.Event().wait()
            else:
                # If no updater attribute, fall back to idle
                await self.app.run_polling()  # type: ignore[attr-defined]
        finally:
            try:
                if hasattr(self.app, "stop"):
                    await self.app.stop()
                if hasattr(self.app, "shutdown"):
                    await self.app.shutdown()
                if self._bus is not None:
                    await self._bus.join()
            except Exception:
                pass

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process incoming messages and send back a reply."""
        if update.message is None:  # pragma: no cover - safety check
            return
        text = update.message.text or ""
        chat = getattr(update, "effective_chat", None)
        chat_id = chat.id if chat else 0
        msg_id = update.message.message_id
        logger.info(
            "received_message",
            extra={
                "update_id": update.update_id,
                "chat_id": chat_id,
                "message_id": msg_id,
                "text": text,
            },
        )
        
        if self._bus:
            try:
                from events.models import MessageReceived
                await self._bus.publish("incoming", MessageReceived(chat_id=chat_id, message_id=msg_id, text=text))
                logger.info("Message published to event bus", extra={"chat_id": chat_id, "message_id": msg_id})
            except Exception as exc:
                logger.exception("Failed to publish message to event bus: %s", exc)
        
        # Reindex state machine per chat
        # Initial setup flow
        if chat_id in self._setup_state:
            state = self._setup_state[chat_id]
            step = state.get("step")
            if step == "theme":
                theme_name = text.strip()
                if not theme_name:
                    await update.message.reply_text(
                        "Название темы не может быть пустым. Введите снова:"
                    )
                    return
                chats = []
                if self._list_chats:
                    try:
                        chats = await self._list_chats()
                    except Exception as exc:
                        logger.exception("Listing chats failed: %s", exc)
                state.update(
                    {
                        "step": "chats",
                        "theme_name": theme_name,
                        "available_chats": chats,
                        "selected_chats": [],
                        "current_page": 0,
                    }
                )
                await self._show_setup_chat_selection(update, chat_id)
            else:
                await update.message.reply_text("Используйте кнопки для выбора чатов.")
            return
        if await self._handle_telethon_auth(update, text, chat_id):
            return

        # Theme creation flow
        if chat_id in self._theme_state:
            state = self._theme_state[chat_id]
            action = state.get("action")
            if action == "create_input":
                theme_name = text.strip()
                if not theme_name:
                    await update.message.reply_text(
                        "Название темы не может быть пустым. Попробуйте ещё раз:"
                    )
                    return
                try:
                    # Get available chats
                    if self._list_chats:
                        chats = await self._list_chats()
                        if not chats:
                            await update.message.reply_text(
                                "Нет доступных чатов для добавления в тему."
                            )
                            self._theme_state.pop(chat_id, None)
                            return

                        # Update state for chat selection
                        state.update(
                            {
                                "action": "create",
                                "theme_name": theme_name,
                                "available_chats": chats,
                                "selected_chats": [],
                                "current_page": 0,
                            }
                        )

                        # Show chat selection
                        await self._show_chat_selection(update, chat_id)
                    else:
                        await update.message.reply_text("Создание тем недоступно.")
                        self._theme_state.pop(chat_id, None)
                except Exception as exc:
                    logger.exception("Error in theme creation flow: %s", exc)
                    await update.message.reply_text("Ошибка при создании темы.")
                    self._theme_state.pop(chat_id, None)
                return

        if chat_id in self._cluster_state:
            self._cluster_state.pop(chat_id, None)
            if not self._summarize_cluster:
                await update.message.reply_text("Кластеризация недоступна.")
                return
            msgs = [m.strip() for m in text.split("|") if m.strip()]
            try:
                summary = await self._summarize_cluster(msgs)
                await update.message.reply_text(summary or "Нет результата")
            except Exception as exc:
                logger.exception("Cluster command failed: %s", exc)
                await update.message.reply_text("Не удалось выполнить кластеризацию.")
            return

        if chat_id in self._agent_report_state:
            self._agent_report_state.pop(chat_id, None)
            if not self._summarize_as_agent:
                await update.message.reply_text("Агентский отчёт недоступен.")
                return
            msgs = [m.strip() for m in text.split("|") if m.strip()]
            try:
                report = await self._summarize_as_agent(msgs)
                await update.message.reply_text(report or "Нет результата")
            except Exception as exc:
                logger.exception("Agent report command failed: %s", exc)
                await update.message.reply_text("Не удалось подготовить отчёт.")
            return

        if chat_id in self._reindex_state:
            state = self._reindex_state[chat_id]
            step = state.get("step")
            if step == "choose_chats":
                try:
                    indices = [int(x.strip()) for x in text.split(",") if x.strip()]
                except Exception:
                    await update.message.reply_text(
                        "Не удалось разобрать список. Введите индексы через запятую, например: 1,3,5"
                    )
                    return
                titles = state.get("chat_titles", [])
                selected: List[str] = []
                for i in indices:
                    if 1 <= i <= len(titles):
                        selected.append(titles[i - 1])
                if not selected:
                    await update.message.reply_text(
                        "Список пуст. Повторите ввод индексов."
                    )
                    return
                state["selected_titles"] = selected
                state["step"] = "choose_interval"
                # Show the interval keyboard again in case user missed it above
                from telegram import InlineKeyboardButton, InlineKeyboardMarkup

                buttons = [
                    [
                        InlineKeyboardButton("🕐 За час", callback_data="reindex_hour"),
                        InlineKeyboardButton("📅 За день", callback_data="reindex_day"),
                        InlineKeyboardButton(
                            "📆 За неделю", callback_data="reindex_week"
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "🗓️ За месяц", callback_data="reindex_month"
                        ),
                        InlineKeyboardButton(
                            "✏️ Количество месяцев", callback_data="reindex_months"
                        ),
                    ],
                ]
                await update.message.reply_text(
                    "Выберите интервал для индексации выбранных чатов:",
                    reply_markup=InlineKeyboardMarkup(buttons),
                )
                return
            elif step == "custom_months":
                try:
                    months = int(text.strip())
                    if months <= 0:
                        raise ValueError
                except Exception:
                    await update.message.reply_text(
                        "Введите положительное число месяцев"
                    )
                    return
                index_since_cb = getattr(self, "_index_since", None)
                if not index_since_cb:
                    await update.message.reply_text(
                        "Функция индексации по интервалу недоступна в этой сборке."
                    )
                    self._reindex_state.pop(chat_id, None)
                    return
                keyboard = InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "❌ Отмена", callback_data="reindex_cancel"
                            )
                        ]
                    ]
                )
                msg = await update.message.reply_text(
                    "Начинаю индексацию…", reply_markup=keyboard
                )
                since = datetime.now(UTC) - timedelta(days=30 * months)

                count_total_cb = getattr(self, "_count_messages_since", None)
                total: Optional[int] = None
                if count_total_cb:
                    try:
                        total = await count_total_cb(
                            state.get("selected_titles", []), since
                        )
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
                            f"Индексация… {percent:.1f}% ({count}/{total}), "
                            f"скорость: {speed:.1f}/с, осталось: {format_seconds(remaining)}"
                        )
                    else:
                        text = (
                            f"Индексация… обработано сообщений: {count}, "
                            f"скорость: {speed:.1f}/с"
                        )
                    try:
                        success = await safe_edit_message(msg, text, reply_markup=keyboard)
                        if success:
                            last_update = now
                    except Exception:
                        pass

                def progress_cb(count: int) -> None:
                    asyncio.create_task(_progress_worker(count))

                def is_cancelled() -> bool:
                    return bool(state.get("cancel"))

                try:
                    count = await index_since_cb(
                        state.get("selected_titles", []),
                        since,
                        progress_cb=progress_cb,
                        is_cancelled=is_cancelled,
                    )
                    if state.get("cancel"):
                        await safe_edit_message(
                            msg,
                            f"Операция отменена. Обработано сообщений: {count}",
                            reply_markup=None,
                        )
                    else:
                        await safe_edit_message(
                            msg,
                            f"Готово. Проиндексировано сообщений: {count}",
                            reply_markup=None,
                        )
                except Exception as exc:
                    logger.exception("Reindex failed: %s", exc)
                    await safe_edit_message(msg, "Ошибка при индексации. Подробности в логах.")
                finally:
                    self._reindex_state.pop(chat_id, None)
                return
            elif step == "choose_n":
                try:
                    n = int(text.strip())
                    if n <= 0:
                        raise ValueError
                except Exception:
                    await update.message.reply_text("Введите положительное число N")
                    return
                if not self._index_last:
                    await update.message.reply_text("Индексация недоступна.")
                    self._reindex_state.pop(chat_id, None)
                    return
                await update.message.reply_text(
                    "Начинаю индексацию… это может занять несколько минут"
                )
                try:
                    count = await self._index_last(state.get("selected_titles", []), n)
                    await update.message.reply_text(
                        f"Готово. Проиндексировано сообщений: {count}"
                    )
                except Exception as exc:  # pragma: no cover - runtime errors
                    logger.exception("Reindex failed: %s", exc)
                    await update.message.reply_text(
                        "Индексация завершилась с ошибками. Часть чатов могла быть пропущена. Подробности см. в логах."
                    )
                finally:
                    self._reindex_state.pop(chat_id, None)
                    try:
                        await self._show_main_menu(chat_id, context)
                    except Exception:
                        pass
                return

        if chat_id in self._dump_state:
            state = self._dump_state[chat_id]
            step = state.get("step")
            if step == "days":
                try:
                    days = int(text.strip())
                    if days <= 0:
                        raise ValueError
                except Exception:
                    await update.message.reply_text("Введите положительное число дней")
                    return
                if not self._dump_since:
                    await update.message.reply_text("Выгрузка недоступна.")
                    self._dump_state.pop(chat_id, None)
                    return
                keyboard = InlineKeyboardMarkup(
                    [[InlineKeyboardButton("❌ Отмена", callback_data="dump_cancel")]]
                )
                msg = await update.message.reply_text(
                    "Начинаю выгрузку сообщений…", reply_markup=keyboard
                )
                state["cancel"] = False

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
                        success = await safe_edit_message(msg, text, reply_markup=keyboard)
                        if success:
                            last_update = now
                    except Exception:
                        pass

                def progress_cb(count: int) -> None:
                    asyncio.create_task(_progress_worker(count))

                def is_cancelled() -> bool:
                    return bool(state.get("cancel"))

                try:
                    count = await self._dump_since(
                        days,
                        progress_cb=progress_cb,
                        is_cancelled=is_cancelled,
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
                return

        if chat_id in self._index_state:
            state = self._index_state[chat_id]
            step = state.get("step")
            if step == "days":
                try:
                    days = int(text.strip())
                    if days <= 0:
                        raise ValueError
                except Exception:
                    await update.message.reply_text("Введите положительное число дней")
                    return
                if not self._index_dumped:
                    await update.message.reply_text("Индексация недоступна.")
                    self._index_state.pop(chat_id, None)
                    return
                keyboard = InlineKeyboardMarkup(
                    [[InlineKeyboardButton("❌ Отмена", callback_data="index_cancel")]]
                )
                msg = await update.message.reply_text(
                    "Начинаю построение индекса…", reply_markup=keyboard
                )
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
                    text = (
                        f"Индексация… обработано сообщений: {count}, "
                        f"скорость: {speed:.1f}/с"
                    )
                    try:
                        success = await safe_edit_message(msg, text, reply_markup=keyboard)
                        if success:
                            last_update = now
                    except Exception:
                        pass

                def progress_cb(count: int) -> None:
                    asyncio.create_task(_progress_worker(count))

                def is_cancelled() -> bool:
                    return bool(state.get("cancel"))

                try:
                    count = await self._index_dumped(
                        days,
                        progress_cb=progress_cb,
                        is_cancelled=is_cancelled,
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
                return

        if chat_id in self._summary_state:
            state = self._summary_state[chat_id]
            if state.get("step") == "custom_hours":
                try:
                    hours = int(text.strip())
                    if hours <= 0:
                        raise ValueError
                except Exception:
                    await update.message.reply_text(
                        "Введите положительное целое число часов"
                    )
                    return
                await update.message.reply_text("Готовлю сводку…")
                try:
                    summary = await self._summarize_interval(hours)
                    if not summary:
                        await update.message.reply_text(
                            "Нет данных для выбранного периода."
                        )
                    else:
                        for chunk in split_long(summary, 3500):
                            await update.message.reply_text(chunk)
                except Exception as exc:
                    logger.exception("Summary callback failed: %s", exc)
                    await update.message.reply_text(
                        "Ошибка при суммаризации. Проверьте логи."
                    )
                finally:
                    self._summary_state.pop(chat_id, None)
                    try:
                        await self._show_main_menu(chat_id, context)
                    except Exception:
                        pass
                return

        if chat_id in self._set_interval_state:
            state = self._set_interval_state[chat_id]
            if state.get("step") == "await_hours":
                try:
                    hours = int(text.strip())
                    if hours <= 0:
                        raise ValueError
                except Exception:
                    await update.message.reply_text(
                        "Введите положительное целое число часов"
                    )
                    return
                try:
                    msg = "Интервал обновлён"
                    if self._set_summary_interval:
                        res = await self._set_summary_interval(hours)
                        if res:
                            msg = res
                except Exception as exc:
                    logger.exception("Failed to set summary interval: %s", exc)
                    msg = "Не удалось обновить интервал"
                await update.message.reply_text(msg)
                self._set_interval_state.pop(chat_id, None)
                return

        # Default chat handling
        if self.on_message_stream:
            logger.info("dispatch_to_on_message_stream", extra={"chat_id": chat_id})
            typing_task = None
            if context.bot:

                async def _keep_typing() -> None:
                    while True:
                        try:
                            await context.bot.send_chat_action(
                                chat_id=chat_id, action="typing"
                            )
                        except Exception:
                            pass
                        await asyncio.sleep(4)

                typing_task = asyncio.create_task(_keep_typing())
            try:
                msg = await update.message.reply_text("🤖 Генерирую ответ…", parse_mode=None)
            except Exception as e:
                logger.error(f"Failed to send initial message: {e}")
                return
            buffer = ""
            last_edit = datetime.now() - timedelta(seconds=1)

            async def _stream_cb(chunk: str) -> None:
                nonlocal buffer, last_edit
                buffer += chunk
                now = datetime.now()
                if now - last_edit >= timedelta(seconds=1):
                    last_edit = now
                    try:
                        await safe_edit_message(msg, buffer, parse_mode=None)
                    except Exception:
                        try:
                            await msg.delete()
                        except Exception:
                            pass

            try:
                logger.info("on_message_stream_start", extra={"chat_id": chat_id})
                await self.on_message_stream(text, chat_id, _stream_cb)
                logger.info("on_message_stream_end", extra={"chat_id": chat_id})
                now = datetime.now()
                diff = now - last_edit
                if diff < timedelta(seconds=1):
                    await asyncio.sleep((timedelta(seconds=1) - diff).total_seconds())
                try:
                    if len(buffer) > 3500:
                        chunks = split_long(buffer)
                        await safe_edit_message(msg, chunks[0], parse_mode=None)
                        for chunk in chunks[1:]:
                            await update.message.reply_text(chunk, parse_mode=None)
                    else:
                        await safe_edit_message(msg, buffer, parse_mode=None)
                except Exception:
                    try:
                        await msg.delete()
                    except Exception:
                        pass
            except Exception as exc:
                logger.exception("Streaming handler failed: %s", exc)
                try:
                    await safe_edit_message(msg, "Произошла ошибка при получении ответа.", parse_mode=None)
                except Exception:
                    try:
                        await msg.delete()
                    except Exception:
                        pass
            finally:
                if typing_task:
                    typing_task.cancel()
                    with contextlib.suppress(Exception):
                        await typing_task
        else:
            typing_task = None
            if context.bot:

                async def _keep_typing() -> None:
                    while True:
                        try:
                            await context.bot.send_chat_action(
                                chat_id=chat_id, action="typing"
                            )
                        except Exception:
                            pass
                        await asyncio.sleep(4)

            typing_task = asyncio.create_task(_keep_typing())
            logger.info("on_message_start", extra={"chat_id": chat_id})
            reply = await self.on_message(text, chat_id)
            logger.info(
                "on_message_end",
                extra={"chat_id": chat_id, "reply_length": len(reply)},
            )
            if typing_task:
                typing_task.cancel()
                with contextlib.suppress(Exception):
                    await typing_task
            await update.message.reply_text(reply, parse_mode=None)
            logger.info("reply_sent", extra={"chat_id": chat_id})

    async def _cmd_search(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /search command to retrieve similar messages from the index."""
        if not getattr(self, "_retriever", None):
            await self._safe_reply(update, context, "Поиск недоступен.")
            return
        args = context.args or []
        top_n = 5
        warn = None
        query = ""
        if args:
            if args[0].isdigit():
                try:
                    requested = max(1, int(args[0]))
                except Exception:
                    requested = 5
                if requested > MAX_SEARCH_RESULTS:
                    warn = (
                        f"Можно запросить максимум {MAX_SEARCH_RESULTS} результатов. "
                        f"Показываю {MAX_SEARCH_RESULTS}."
                    )
                top_n = min(requested, MAX_SEARCH_RESULTS)
                query = " ".join(args[1:])
            else:
                query = " ".join(args)
        query = query.strip()
        if not query:
            await self._safe_reply(update, context, "Использование: /search [N] запрос")
            return
        if warn:
            await self._safe_reply(update, context, warn)
        chat = getattr(update, "effective_chat", None)
        chat_id = chat.id if chat else 0
        if context.bot:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        try:
            results = await self._retriever.query(query, top_k=top_n)
        except FileNotFoundError:
            await self._safe_reply(
                update,
                context,
                "Индекс не найден. Попробуйте выполнить /reindex.",
            )
            return
        except (ConnectionError, OSError) as exc:  # pragma: no cover - network errors
            logger.exception("Search network problem: %s", exc)
            await self._safe_reply(
                update,
                context,
                "Не удалось выполнить поиск из-за проблем сети. Повторите попытку позже.",
            )
            return
        except Exception as exc:  # pragma: no cover - runtime errors
            logger.exception("Search failed: %s", exc)
            await self._safe_reply(update, context, "Ошибка поиска.")
            return
        if not results:
            await self._safe_reply(
                update,
                context,
                "Ничего не найдено. Попробуйте уточнить запрос или выполните /reindex.",
            )
            return
        lines = []
        for i, r in enumerate(results[:top_n], start=1):
            chat = r.metadata.get("chat", "")
            date = r.metadata.get("date", "")
            msg_ids = r.metadata.get("message_ids", "")
            first_id = msg_ids.split(",")[0] if msg_ids else ""
            text = escape_markdown(r.text.strip().replace("\n", " ")[:200], version=2)
            link = ""
            if chat and first_id:
                if str(chat).lstrip("-").isdigit():
                    link = f"tg://msg?chat_id={chat}&message_id={first_id}"
                else:
                    link = f"https://t.me/{chat}/{first_id}"
                text = f"[{text}]({link})"
            lines.append(f"{i}. [{chat}] {date}: {text}")
        for chunk in split_long("\n".join(lines)):
            await self._safe_reply(
                update, context, chunk, parse_mode=ParseMode.MARKDOWN_V2
            )

    async def _cmd_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await self._show_help_categories(update, context)

    def _build_help_categories_keyboard(self) -> InlineKeyboardMarkup:
        buttons: List[List[InlineKeyboardButton]] = []
        for key, section in self._help_sections.items():
            cmds = [c for c in section["commands"] if c in self._commands]
            if cmds:
                buttons.append(
                    [
                        InlineKeyboardButton(
                            section["title"], callback_data=f"help_cat_{key}"
                        )
                    ]
                )
        buttons.append(
            [InlineKeyboardButton("⬅️ Назад в меню", callback_data="menu_main")]
        )
        return InlineKeyboardMarkup(buttons)

    async def _show_help_categories(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        lines = ["Выберите категорию команд:"]
        if self._refresh_chat_cache:
            lines.append("🔄 Список чатов кешируется на 30 минут для быстрой работы")
        lines.append(
            "Просто отправьте текст — получаете ответ от LLM (с учётом индекса, если он есть).",
        )
        await self._safe_reply(
            update,
            context,
            "\n".join(lines),
            reply_markup=self._build_help_categories_keyboard(),
        )

    async def _cb_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await self._safe_query_answer(query)
        data = query.data
        if data == "help_main":
            await self._show_help_categories(update, context)
            return
        if data.startswith("help_cat_"):
            key = data[len("help_cat_") :]
            section = self._help_sections.get(key)
            if not section:
                await query.edit_message_text("Неизвестная категория.")
                return
            cmds = [c for c in section["commands"] if c in self._commands]
            lines = [f"Команды категории «{section['title']}»:"]
            for c in cmds:
                meta = self._commands.get(c, {})
                desc = html.escape(str(meta.get("description", "")))
                usage = meta.get("usage")
                line = f"<b>/{c}</b> {desc}"
                if usage:
                    line += f"\nПример: <code>{html.escape(str(usage))}</code>"
                lines.append(line)
            buttons = [
                [InlineKeyboardButton("⬅️ Назад", callback_data="help_main")],
                [InlineKeyboardButton("⬅️ В меню", callback_data="menu_main")],
            ]
            await self._safe_reply(
                update,
                context,
                "\n\n".join(lines),
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(buttons),
            )
            return
        await query.edit_message_text("Неизвестный запрос.")

    async def _cmd_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Begin interactive initial setup of theme and chats."""
        if self._list_chats and self._create_new_theme and self._set_active_theme:
            chat_id = update.effective_chat.id if update.effective_chat else 0
            # If themes can be listed, offer selection or creation first
            if self._list_themes:
                try:
                    themes = await self._list_themes()
                except Exception:
                    logger.exception("Failed to list themes on /start")
                    themes = []
                if themes:
                    self._setup_state[chat_id] = {"step": "choose_theme"}
                    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

                    buttons = [
                        [InlineKeyboardButton(f"📁 {name}", callback_data=f"setup_theme:{name}")]
                        for name in themes
                    ]
                    buttons.append(
                        [InlineKeyboardButton("➕ Создать новую", callback_data="setup_new_theme")]
                    )
                    buttons.append(
                        [InlineKeyboardButton("❌ Отмена", callback_data="setup_cancel")]
                    )
                    await update.message.reply_text(
                        "Выберите существующую тему или создайте новую:",
                        reply_markup=InlineKeyboardMarkup(buttons),
                    )
                    return
            # Fallback: ask to create new theme by name
            self._setup_state[chat_id] = {"step": "theme"}
            await update.message.reply_text("Введите название новой темы:")
        else:
            await update.message.reply_text(
                "Привет! Используйте /help, чтобы узнать доступные команды."
            )

    async def _show_setup_chat_selection(
        self, update: Update, chat_id: int, edit_message: bool = False
    ) -> None:
        """Show chat selection interface for initial setup."""
        state = self._setup_state.get(chat_id)
        if not state:
            return

        available_chats = state.get("available_chats", [])
        selected_chats = state.get("selected_chats", [])
        current_page = state.get("current_page", 0)

        total_chats = len(available_chats)
        per_page = self.CHATS_PER_PAGE
        total_pages = (total_chats + per_page - 1) // per_page if per_page else 1
        current_page = max(0, min(current_page, max(total_pages - 1, 0)))
        state["current_page"] = current_page
        start_idx = current_page * per_page
        end_idx = min(start_idx + per_page, total_chats)

        # Map indices to chats for callbacks
        state["chat_index_map"] = {i: c for i, c in enumerate(available_chats)}

        buttons: List[List[InlineKeyboardButton]] = []
        for i in range(start_idx, end_idx):
            chat = available_chats[i]
            icon = "✅" if chat in selected_chats else "⬜"
            display = chat[: self.MAX_CHAT_NAME_LENGTH] + (
                "..." if len(chat) > self.MAX_CHAT_NAME_LENGTH else ""
            )
            buttons.append(
                [
                    InlineKeyboardButton(
                        f"{icon} {display}", callback_data=f"setup_toggle_{i}"
                    )
                ]
            )

        nav_buttons: List[InlineKeyboardButton] = []
        if total_pages > 1:
            if current_page > 0:
                nav_buttons.append(
                    InlineKeyboardButton("⬅️", callback_data="setup_prev")
                )
            nav_buttons.append(
                InlineKeyboardButton(
                    f"{current_page + 1}/{total_pages}", callback_data="setup_info"
                )
            )
            if current_page < total_pages - 1:
                nav_buttons.append(
                    InlineKeyboardButton("➡️", callback_data="setup_next")
                )
            buttons.append(nav_buttons)

        buttons.append(
            [
                InlineKeyboardButton("💾 Сохранить", callback_data="setup_save"),
                InlineKeyboardButton("❌ Отмена", callback_data="setup_cancel"),
            ]
        )

        keyboard = InlineKeyboardMarkup(buttons)
        text = (
            f"Выбор чатов для темы: {state.get('theme_name', '')}\n"
            f"Выбрано чатов: {len(selected_chats)} из {total_chats}"
        )
        if edit_message and hasattr(update, "callback_query"):
            await update.callback_query.edit_message_text(text, reply_markup=keyboard)
        else:
            await update.message.reply_text(text, reply_markup=keyboard)

    async def _cb_setup(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle callbacks for initial setup chat selection."""
        query = update.callback_query
        await self._safe_query_answer(query)
        data = query.data
        chat_id = query.message.chat_id if query.message else 0
        state = self._setup_state.get(chat_id)
        if not state:
            await query.edit_message_text("Сессия настройки не найдена.")
            return

        # Initial theme choice step
        if state.get("step") == "choose_theme":
            if data and data.startswith("setup_theme:"):
                theme_name = data.split(":", 1)[1]
                try:
                    if self._set_active_theme:
                        await self._set_active_theme(theme_name)
                    await query.edit_message_text(
                        f"Тема «{html.escape(theme_name)}» активирована."
                    )
                except Exception as exc:
                    logger.exception("Activate theme failed: %s", exc)
                    await query.edit_message_text("Не удалось активировать тему.")
                finally:
                    self._setup_state.pop(chat_id, None)
                    try:
                        await self._show_main_menu(chat_id, context)
                    except Exception:
                        pass
                return
            if data == "setup_new_theme":
                self._setup_state[chat_id] = {"step": "theme"}
                await query.edit_message_text("Введите название новой темы:")
                return
            if data == "setup_cancel":
                await query.edit_message_text("Настройка отменена.")
                self._setup_state.pop(chat_id, None)
                return

        if data.startswith("setup_toggle_"):
            idx = int(data[len("setup_toggle_") :])
            chat = state.get("chat_index_map", {}).get(idx)
            if chat:
                selected = state.get("selected_chats", [])
                if chat in selected:
                    selected.remove(chat)
                else:
                    selected.append(chat)
                state["selected_chats"] = selected
            await self._show_setup_chat_selection(update, chat_id, edit_message=True)
        elif data == "setup_prev":
            state["current_page"] = max(0, state.get("current_page", 0) - 1)
            await self._show_setup_chat_selection(update, chat_id, edit_message=True)
        elif data == "setup_next":
            total_chats = len(state.get("available_chats", []))
            total_pages = (total_chats + self.CHATS_PER_PAGE - 1) // self.CHATS_PER_PAGE
            state["current_page"] = min(
                total_pages - 1, state.get("current_page", 0) + 1
            )
            await self._show_setup_chat_selection(update, chat_id, edit_message=True)
        elif data == "setup_info":
            # Just acknowledge page indicator
            await self._safe_query_answer(query)
        elif data == "setup_save":
            theme = state.get("theme_name", "")
            chats = state.get("selected_chats", [])
            ok = False
            if self._create_new_theme:
                try:
                    ok = await self._create_new_theme(theme, chats)
                except Exception as exc:
                    logger.exception("Create theme failed: %s", exc)
            if ok and self._set_active_theme:
                try:
                    await self._set_active_theme(theme)
                except Exception as exc:
                    logger.exception("Set active theme failed: %s", exc)
            if ok:
                await query.edit_message_text("Настройка завершена.")
            else:
                await query.edit_message_text("Не удалось сохранить тему.")
            self._setup_state.pop(chat_id, None)
        elif data == "setup_cancel":
            await query.edit_message_text("Настройка отменена.")
            self._setup_state.pop(chat_id, None)
        else:
            await query.edit_message_text("Неизвестное действие.")

    async def _cb_trading_alert(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Обработка callback'ов для торговых алертов."""
        try:
            # Получаем TradingAlertHandler из BroadcastExecutor
            if hasattr(self, '_broadcast_executor') and self._broadcast_executor:
                alert_handler = self._broadcast_executor._alert_handler
                if alert_handler:
                    await alert_handler.handle_callback(update, context)
                else:
                    await update.callback_query.answer("Обработчик алертов недоступен")
            else:
                await update.callback_query.answer("Система алертов не инициализирована")
        except Exception as e:
            logger.error(f"Error handling trading alert callback: {e}")
            await update.callback_query.answer("Ошибка обработки действия")

    async def _show_main_menu(
        self,
        chat_id: int,
        context: ContextTypes.DEFAULT_TYPE,
        source: Optional[Union["CallbackQuery", "Message"]] = None,
    ) -> None:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        # Новое упрощенное главное меню
        buttons = []
        
        # Статус системы - всегда доступен
        buttons.append(
            [InlineKeyboardButton("📊 Статус системы", callback_data="menu_status")]
        )
        
        # Поиск и анализ - объединенная категория
        if any([
            self._retriever,
            self._summarize_interval,
            self._summarize_as_agent,
            self._summarize_cluster,
            self._summarize_url,
            self._run_finance_analysis
        ]):
            buttons.append(
                [InlineKeyboardButton("🔍 Поиск и анализ", callback_data="menu_search_analysis")]
            )
        
        # Управление данными - новая категория
        if any([
            self._list_chats and self._index_last,
            self._dump_since,
            self._index_dumped,
            self._publish_chronicle,
            self._set_summary_interval
        ]):
            buttons.append(
                [InlineKeyboardButton("🗂 Управление данными", callback_data="menu_data_management")]
            )
        
        # Темы - упрощенная категория
        if self._list_themes or self._set_active_theme or self._create_new_theme:
            buttons.append(
                [InlineKeyboardButton("📂 Темы", callback_data="menu_themes")]
            )
        
        # Настройки
        if self._telethon_auth_request_code:
            buttons.append(
                [InlineKeyboardButton("⚙️ Настройки", callback_data="menu_settings")]
            )
        
        # Помощь
        buttons.append([InlineKeyboardButton("❓ Помощь", callback_data="menu_help")])
        
        # Заголовок с эмодзи
        text = "🤖 AI Assistant\n━━━━━━━━━━━━━━━━━━━━"
        
        markup = InlineKeyboardMarkup(buttons)

        await self._send_or_edit_menu(
            chat_id,
            context,
            text,
            markup,
            source=source,
            log_context="_show_main_menu",
        )

    async def _show_search_analysis_menu(
        self,
        chat_id: int,
        context: ContextTypes.DEFAULT_TYPE,
        source: Optional[Union["CallbackQuery", "Message"]] = None,
    ) -> None:
        """Show unified search and analysis menu."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        
        buttons = []
        
        # Поиск по индексу
        if self._retriever:
            buttons.append(
                [InlineKeyboardButton("🔍 Поиск по индексу", callback_data="menu_search")]
            )
        
        # Сводка за период
        if self._summarize_interval:
            buttons.append(
                [InlineKeyboardButton("📝 Сводка за период", callback_data="menu_summary")]
            )
        
        # Агентский отчёт
        if self._summarize_as_agent:
            buttons.append(
                [InlineKeyboardButton("🤖 Агентский отчёт", callback_data="menu_agent_report")]
            )
        
        # Кластеризация
        if self._summarize_cluster:
            buttons.append(
                [InlineKeyboardButton("🧮 Кластеризация", callback_data="menu_cluster")]
            )
        
        # Анализ ссылки
        if self._summarize_url:
            buttons.append(
                [InlineKeyboardButton("🔗 Анализ ссылки", callback_data="menu_sumurl")]
            )
        
        # Финансовые сигналы
        if self._run_finance_analysis:
            buttons.append(
                [InlineKeyboardButton("📈 Финансовые сигналы", callback_data="menu_signals")]
            )
        
        buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="menu_main")])
        
        text = "🔍 Поиск и анализ\n━━━━━━━━━━━━━━━━━━━━"
        
        markup = InlineKeyboardMarkup(buttons)

        await self._send_or_edit_menu(
            chat_id,
            context,
            text,
            markup,
            source=source,
            log_context="_show_search_analysis_menu",
        )

    async def _show_search_menu(
        self, chat_id: int, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Legacy search menu - redirect to new unified menu."""
        await self._show_search_analysis_menu(chat_id, context)

    async def _show_data_management_menu(
        self,
        chat_id: int,
        context: ContextTypes.DEFAULT_TYPE,
        source: Optional[Union["CallbackQuery", "Message"]] = None,
    ) -> None:
        """Show data management menu with all indexing and data operations."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        
        buttons = []
        
        # Индексация
        if self._list_chats and self._index_last:
            buttons.append(
                [InlineKeyboardButton("📥 Индексация", callback_data="menu_reindex")]
            )
        
        # Выгрузка данных
        if self._dump_since:
            buttons.append(
                [InlineKeyboardButton("💾 Выгрузка данных", callback_data="menu_dump")]
            )
        
        # Построение индекса
        if self._index_dumped:
            buttons.append(
                [InlineKeyboardButton("🗂 Построение индекса", callback_data="menu_index")]
            )
        
        # Обзор тем
        if self._publish_chronicle:
            buttons.append(
                [InlineKeyboardButton("📰 Обзор тем", callback_data="menu_chronicle")]
            )
        
        # Настройка интервалов
        if self._set_summary_interval:
            buttons.append(
                [InlineKeyboardButton("⏱ Настройка интервалов", callback_data="menu_set_summary_interval")]
            )
        
        buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="menu_main")])
        
        text = "🗂 Управление данными\n━━━━━━━━━━━━━━━━━━━━"
        
        markup = InlineKeyboardMarkup(buttons)

        await self._send_or_edit_menu(
            chat_id,
            context,
            text,
            markup,
            source=source,
            log_context="_show_data_management_menu",
        )

    async def _show_index_menu(
        self, chat_id: int, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Legacy index menu - redirect to new data management menu."""
        await self._show_data_management_menu(chat_id, context)

    async def _show_theme_menu(
        self,
        chat_id: int,
        context: ContextTypes.DEFAULT_TYPE,
        source: Optional[Union["CallbackQuery", "Message"]] = None,
    ) -> None:
        """Show simplified theme management menu with context."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        
        # Получаем информацию о текущей теме
        current_theme = "неизвестна"
        theme_chats = 0
        if self._get_active_theme_name:
            try:
                current_theme = await self._get_active_theme_name()
                if self._get_theme_chats:
                    theme_chats = len(await self._get_theme_chats(current_theme))
            except Exception:
                logger.exception("Error getting current theme info")
        
        buttons = []
        
        # Показываем текущую тему с количеством чатов
        if current_theme != "неизвестна":
            theme_text = f"🎯 Текущая: {current_theme}"
            if theme_chats > 0:
                theme_text += f" ({theme_chats} чатов)"
            buttons.append([InlineKeyboardButton(theme_text, callback_data="theme_info_current")])
        
        # Все темы
        if self._list_themes:
            buttons.append(
                [InlineKeyboardButton("📁 Все темы", callback_data="menu_themes_list")]
            )
        
        # Создать тему
        if self._create_new_theme:
            buttons.append(
                [InlineKeyboardButton("➕ Создать тему", callback_data="menu_create_theme")]
            )
        
        # Редактировать тему
        if self._get_theme_chats:
            buttons.append(
                [InlineKeyboardButton("✏️ Редактировать тему", callback_data="menu_edit_theme")]
            )
        
        # Быстрая смена темы
        if self._list_themes and self._set_active_theme:
            buttons.append(
                [InlineKeyboardButton("🔄 Быстрая смена", callback_data="menu_cycle_theme")]
            )
        
        buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="menu_main")])
        
        text = "📂 Темы\n━━━━━━━━━━━━━━━━━━━━"
        
        markup = InlineKeyboardMarkup(buttons)

        await self._send_or_edit_menu(
            chat_id,
            context,
            text,
            markup,
            source=source,
            log_context="_show_theme_menu",
        )

    async def _show_settings_menu(
        self,
        chat_id: int,
        context: ContextTypes.DEFAULT_TYPE,
        source: Optional[Union["CallbackQuery", "Message"]] = None,
    ) -> None:
        """Show settings menu with system configuration options."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        
        buttons = []
        
        # Авторизация Telethon
        if self._telethon_auth_request_code and self._telethon_auth_sign_in:
            buttons.append(
                [InlineKeyboardButton("🔐 Авторизация Telethon", callback_data="menu_telethon")]
            )
        
        # Статус системы
        buttons.append(
            [InlineKeyboardButton("📊 Статус системы", callback_data="menu_status")]
        )
        
        # Обновить кеш
        if self._refresh_chat_cache:
            buttons.append(
                [InlineKeyboardButton("🔄 Обновить кеш", callback_data="menu_refresh_cache")]
            )
        
        # Отменить операции
        buttons.append(
            [InlineKeyboardButton("❌ Отменить операции", callback_data="menu_cancel_operations")]
        )
        
        buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="menu_main")])
        
        text = "⚙️ Настройки\n━━━━━━━━━━━━━━━━━━━━"
        
        markup = InlineKeyboardMarkup(buttons)

        await self._send_or_edit_menu(
            chat_id,
            context,
            text,
            markup,
            source=source,
            log_context="_show_settings_menu",
        )

    async def _show_finance_menu(
        self, chat_id: int, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Legacy finance menu - redirect to search analysis menu."""
        await self._show_search_analysis_menu(chat_id, context)

    async def _cmd_menu(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show main menu - redirect to new UX design."""
        chat_id = update.effective_chat.id if update.effective_chat else 0
        await self._show_main_menu(chat_id, context)

    async def _cb_menu(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await self._safe_query_answer(query)
        data = query.data
        chat_id = query.message.chat_id if query.message else 0
        
        # Новые меню
        if data == "menu_status":
            await self._show_status_menu(chat_id, context, update, query)
        elif data == "menu_search_analysis":
            await self._show_search_analysis_menu(chat_id, context, query)
        elif data == "menu_data_management":
            await self._show_data_management_menu(chat_id, context, query)
        elif data == "menu_themes":
            await self._show_theme_menu(chat_id, context, query)
        elif data == "menu_settings":
            await self._show_settings_menu(chat_id, context, query)
        
        # Статус системы
        elif data == "status_refresh":
            await self._show_status_menu(chat_id, context, update, query)
        elif data == "status_detailed":
            await self._run_with_chat_id(self._cmd_status, chat_id, context)
        
        # Поиск и анализ
        elif data == "menu_search":
            await self._run_with_chat_id(
                self._cmd_search, chat_id, context, text="/search"
            )
        elif data == "menu_summary":
            await self._show_summary_menu(chat_id, context)
        elif data == "menu_agent_report":
            await self._run_with_chat_id(self._cmd_agent_report, chat_id, context)
        elif data == "menu_cluster":
            await self._run_with_chat_id(self._cmd_cluster, chat_id, context)
        elif data == "menu_sumurl":
            await self._run_with_chat_id(
                self._cmd_sumurl, chat_id, context, text="/sumurl"
            )
        elif data == "menu_signals":
            await self._run_with_chat_id(
                self._cmd_signals, chat_id, context, text="/signals"
            )
        
        # Управление данными
        elif data == "menu_reindex":
            await self._start_reindex(chat_id, context)
        elif data == "menu_dump":
            await self._run_with_chat_id(
                self._cmd_dump, chat_id, context, text="/dump"
            )
        elif data == "menu_index":
            await self._run_with_chat_id(
                self._cmd_index, chat_id, context, text="/index"
            )
        elif data == "menu_chronicle":
            await self._run_with_chat_id(self._cmd_chronicle, chat_id, context)
        elif data == "menu_set_summary_interval":
            await self._run_with_chat_id(
                self._cmd_set_summary_interval,
                chat_id,
                context,
                text="/set_summary_interval",
            )
        
        # Темы
        elif data == "theme_info_current":
            await self._run_with_chat_id(self._cmd_status, chat_id, context)
        elif data == "menu_themes_list":
            await self._run_with_chat_id(self._cmd_themes, chat_id, context)
        elif data == "menu_create_theme":
            await self._run_with_chat_id(
                self._cmd_create_theme, chat_id, context, text="/create_theme"
            )
        elif data == "menu_edit_theme":
            await self._run_with_chat_id(self._cmd_edit_theme, chat_id, context)
        elif data == "menu_cycle_theme":
            await self._run_with_chat_id(self._cmd_quick_switch_theme, chat_id, context)
        
        # Настройки
        elif data == "menu_telethon":
            await self._run_with_chat_id(self._cmd_telethon_auth, chat_id, context)
        elif data == "menu_refresh_cache":
            await self._refresh_chat_cache() if self._refresh_chat_cache else None
            await self._show_settings_menu(chat_id, context, query)
        elif data == "menu_cancel_operations":
            await self._run_with_chat_id(self._cmd_cancel, chat_id, context)
        
        # Помощь
        elif data == "menu_help":
            await self._run_with_chat_id(self._cmd_help, chat_id, context)
        
        # Навигация
        elif data == "menu_main":
            await self._show_main_menu(chat_id, context, query)
        
        # Legacy поддержка
        elif data == "menu_cat_search":
            await self._show_search_analysis_menu(chat_id, context, query)
        elif data == "menu_cat_index":
            await self._show_data_management_menu(chat_id, context, query)
        elif data == "menu_cat_themes":
            await self._show_theme_menu(chat_id, context, query)
        elif data == "menu_cat_finance":
            await self._show_search_analysis_menu(chat_id, context, query)
        elif data == "menu_cat_settings":
            await self._show_settings_menu(chat_id, context, query)
        
        else:
            await query.edit_message_text("Неизвестный пункт меню.")

    async def _show_status_menu(
        self,
        chat_id: int,
        context: ContextTypes.DEFAULT_TYPE,
        update: Optional[Update] = None,
        source: Optional[Union["CallbackQuery", "Message"]] = None,
    ) -> None:
        """Show comprehensive system status with context information."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        
        # Собираем информацию о системе
        theme = "неизвестна"
        theme_chats = 0
        if self._get_active_theme_name:
            try:
                theme = await self._get_active_theme_name()
                if self._get_theme_chats:
                    theme_chats = len(await self._get_theme_chats(theme))
            except Exception:
                logger.exception("Error getting active theme info")
        
        telethon_status = "недоступен"
        if self._telethon_is_authorized:
            try:
                auth = await self._telethon_is_authorized()
                telethon_status = "авторизован" if auth else "не авторизован"
            except Exception:
                logger.exception("Error checking telethon auth")
                telethon_status = "ошибка"
        
        index_size = 0
        if self._retriever and getattr(self._retriever, "index", None):
            index_size = len(getattr(self._retriever.index, "entries", []))
        
        last_indexed = "—"
        if self._index_state_path:
            try:
                self._index_state_path.parent.mkdir(parents=True, exist_ok=True)
                if self._index_state_path.exists():
                    ts_str = self._index_state_path.read_text(encoding="utf-8").strip()
                    if ts_str:
                        dt = datetime.fromisoformat(ts_str)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=UTC)
                        last_indexed = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                logger.exception("Failed to read last indexed state")
        
        # Форматируем статус
        text = (
            "📊 Статус системы\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🎯 Активная тема: {theme}\n"
            f"📈 Размер индекса: {index_size:,} сообщений\n"
            f"🔄 Последняя индексация: {last_indexed}\n"
            f"🔐 Telethon: {telethon_status}\n"
        )
        
        if theme_chats > 0:
            text += f"📂 Чатов в теме: {theme_chats}\n"
        
        # Кнопки действий
        buttons = []
        buttons.append([InlineKeyboardButton("🔄 Обновить", callback_data="status_refresh")])
        buttons.append([InlineKeyboardButton("📊 Подробно", callback_data="status_detailed")])
        buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="menu_main")])

        markup = InlineKeyboardMarkup(buttons)
        handled = await self._send_or_edit_menu(
            chat_id,
            context,
            text,
            markup,
            source=source,
            log_context="_show_status_menu",
        )

        if not handled and update is not None:
            await self._safe_reply(update, context, text)

    async def _cmd_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show system status - redirect to new status menu."""
        chat = getattr(update, "effective_chat", None)
        chat_id = getattr(chat, "id", 0)
        await self._show_status_menu(chat_id, context, update)

    async def _cmd_cancel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Cancel any ongoing multi-step interaction for this chat."""
        chat = getattr(update, "effective_chat", None)
        chat_id = getattr(chat, "id", 0)
        states = [
            self._reindex_state,
            self._auth_state,
            self._theme_state,
            self._dump_state,
            self._index_state,
            self._summary_state,
            self._set_interval_state,
            self._cluster_state,
            self._agent_report_state,
            self._setup_state,
        ]
        # Set cancellation flag for long-running tasks
        for state in (self._reindex_state, self._dump_state, self._index_state):
            if chat_id in state:
                state[chat_id]["cancel"] = True
        for state in states:
            state.pop(chat_id, None)
        await self._safe_reply(update, context, "Операция отменена.")

    async def _show_summary_menu(
        self, chat_id: int, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._summarize_interval:
            await context.bot.send_message(
                chat_id=chat_id, text="Суммаризация недоступна."
            )
            return
        buttons = [
            [
                InlineKeyboardButton("1 час", callback_data="summary:1"),
                InlineKeyboardButton("6 часов", callback_data="summary:6"),
            ],
            [
                InlineKeyboardButton("1 день", callback_data="summary:24"),
                InlineKeyboardButton("1 неделя", callback_data="summary:168"),
            ],
            [InlineKeyboardButton("✏️ Своё значение", callback_data="summary:custom")],
        ]
        keyboard = InlineKeyboardMarkup(buttons)
        await context.bot.send_message(
            chat_id=chat_id,
            text="Выберите период для суммаризации:",
            reply_markup=keyboard,
        )

    async def _cmd_summary(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = getattr(update, "effective_chat", None)
        chat_id = chat.id if chat else 0
        if context.bot:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        args = getattr(context, "args", [])
        if args and args[0].isdigit():
            hours = int(args[0])
            try:
                summary = (
                    await self._summarize_interval(hours)
                    if self._summarize_interval
                    else ""
                )
            except FileNotFoundError:
                await self._safe_reply(
                    update, context, "Индекс отсутствует. Сначала выполните /reindex."
                )
                await self._show_main_menu(chat_id, context)
                return
            except (
                ConnectionError,
                OSError,
            ) as exc:  # pragma: no cover - network errors
                logger.exception("Summary network problem: %s", exc)
                await self._safe_reply(
                    update,
                    context,
                    "Не удалось получить данные из-за проблем сети. Попробуйте позже.",
                )
                await self._show_main_menu(chat_id, context)
                return
            except Exception as exc:  # pragma: no cover - unexpected errors
                logger.exception("Summary command failed: %s", exc)
                await self._safe_reply(update, context, "Ошибка при суммаризации.")
                await self._show_main_menu(chat_id, context)
                return
            if not summary:
                await self._safe_reply(
                    update, context, "Нет данных для выбранного периода."
                )
                await self._show_main_menu(chat_id, context)
                return
            for chunk in split_long(summary, 3500):
                await self._safe_reply(update, context, chunk)
            await self._show_main_menu(chat_id, context)
            return

        await self._show_summary_menu(chat_id, context)

    async def _cmd_sumurl(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._summarize_url:
            await self._safe_reply(update, context, "Суммаризация ссылок недоступна.")
            return
        chat = getattr(update, "effective_chat", None)
        chat_id = chat.id if chat else 0
        if context.bot:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        args = getattr(context, "args", [])
        if not args:
            await self._safe_reply(update, context, "Укажите ссылку.")
            await self._show_main_menu(chat_id, context)
            return
        url = args[0]
        try:
            summary = await self._summarize_url(url)
        except Exception as exc:  # pragma: no cover - network errors
            logger.exception("Link summary failed: %s", exc)
            await self._safe_reply(update, context, "Не удалось получить сводку.")
            await self._show_main_menu(chat_id, context)
            return
        if not summary:
            await self._safe_reply(update, context, "Сводка пуста.")
            await self._show_main_menu(chat_id, context)
            return
        for chunk in split_long(summary, 3500):
            await self._safe_reply(update, context, chunk)
        await self._show_main_menu(chat_id, context)

    async def _cmd_agent_report(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._summarize_as_agent:
            await self._safe_reply(update, context, "Агентский отчёт недоступен.")
            return
        chat = getattr(update, "effective_chat", None)
        chat_id = chat.id if chat else 0
        self._agent_report_state[chat_id] = True
        await self._safe_reply(
            update,
            context,
            "Отправьте сообщения через '|' для агентского отчёта.",
        )

    async def _cmd_cluster(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._summarize_cluster:
            await self._safe_reply(update, context, "Кластеризация недоступна.")
            return
        chat = getattr(update, "effective_chat", None)
        chat_id = chat.id if chat else 0
        self._cluster_state[chat_id] = True
        await self._safe_reply(
            update,
            context,
            "Отправьте сообщения через '|' для кластеризации.",
        )

    async def _cmd_chronicle(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._publish_chronicle:
            await self._safe_reply(update, context, "Хроника недоступна.")
            return
        chat = getattr(update, "effective_chat", None)
        chat_id = chat.id if chat else 0
        if context.bot:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        try:
            await self._publish_chronicle()
            await self._safe_reply(update, context, "Обзор тем опубликован.")
        except Exception:
            logger.exception("Chronicle command failed")
            await self._safe_reply(update, context, "Не удалось подготовить обзор тем.")

    async def _start_reindex(
        self, chat_id: int, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if context.bot:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        await self._run_with_chat_id(super()._cmd_reindex, chat_id, context)

    async def _run_with_chat_id(
        self,
        handler,
        chat_id: int,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        text: Optional[str] = None,
        args: Optional[List[str]] = None,
    ) -> None:
        class DummyMessage:
            def __init__(self, chat_id: int, text: str):
                self.chat_id = chat_id
                self.text = text

            def __str__(self) -> str:  # pragma: no cover - debugging helper
                return self.text or f"<DummyMessage chat_id={self.chat_id}>"

            async def reply_text(self, text: str, **kwargs) -> None:
                if context.bot:
                    await context.bot.send_message(
                        chat_id=self.chat_id, text=text, **kwargs
                    )
                else:  # pragma: no cover - bot is optional in some tests
                    logger.warning("Cannot send message without bot instance")

        message_text = text or ""
        dummy_update = type(
            "U",
            (),
            {
                "message": DummyMessage(chat_id, message_text),
                "effective_chat": type("C", (), {"id": chat_id})(),
            },
        )()

        had_args = hasattr(context, "args")
        previous_args = getattr(context, "args", None)
        new_args = list(args) if args is not None else []
        setattr(context, "args", new_args)
        try:
            await handler(dummy_update, context)
        finally:
            if had_args:
                setattr(context, "args", previous_args)
            else:
                with contextlib.suppress(AttributeError):
                    delattr(context, "args")

    async def _cb_summary(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        logger.info("_cb_summary called with update: %s", update)
        if not self._summarize_interval:
            logger.warning(
                "Summary callback called but summarize_interval is not available"
            )
            return
        query = update.callback_query
        if not query:
            logger.warning("Summary callback called but no callback_query found")
            return
        await self._safe_query_answer(query)
        data = query.data or ""
        logger.info("Summary callback received with data: %s", data)
        chat_id = update.effective_chat.id if update.effective_chat else 0
        if data == "summary:custom":
            self._summary_state[chat_id] = {"step": "custom_hours"}
            await query.edit_message_text("Введите количество часов для суммаризации:")
            return
        try:
            _, hours_str = data.split(":", 1)
            hours = int(hours_str)
            logger.info("Parsed hours: %d", hours)
        except Exception as exc:
            logger.error("Failed to parse summary callback data '%s': %s", data, exc)
            await query.edit_message_text("Не удалось разобрать период.")
            return
        await query.edit_message_text("Готовлю сводку…")
        if context.bot:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        try:
            logger.info("Calling _summarize_interval with %d hours", hours)
            summary = await self._summarize_interval(hours)
        except FileNotFoundError:
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="Индекс отсутствует. Сначала выполните /reindex.",
            )
            await self._show_main_menu(chat_id, context)
            return
        except (ConnectionError, OSError) as exc:  # pragma: no cover - network errors
            logger.exception("Summary network problem: %s", exc)
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="Не удалось получить данные из-за проблем сети. Попробуйте позже.",
            )
            await self._show_main_menu(chat_id, context)
            return
        except Exception as exc:
            logger.exception("Summary callback failed: %s", exc)
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="Ошибка при суммаризации.",
            )
            await self._show_main_menu(chat_id, context)
            return
        logger.info(
            "Summary result: %s... (length: %d)",
            summary[:100] if summary else "None",
            len(summary) if summary else 0,
        )
        if not summary:
            logger.warning("Empty summary returned for %d hours", hours)
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="Нет данных для выбранного периода.",
            )
            await self._show_main_menu(chat_id, context)
            return
        chunks = split_long(summary, 1500)  # Еще больше уменьшаем лимит
        logger.info(f"Split summary into {len(chunks)} chunks")
        
        # Если получился только один фрагмент, но он слишком длинный, принудительно разбиваем
        if len(chunks) == 1 and len(chunks[0]) > 2000:
            logger.warning(f"Single chunk too long ({len(chunks[0])} chars), forcing split")
            text = chunks[0]
            chunks = []
            for i in range(0, len(text), 1000):  # Разбиваем по 1000 символов
                chunk = text[i:i+1000]
                if chunk.strip():  # Добавляем только непустые фрагменты
                    chunks.append(chunk.strip())
            logger.info(f"Force split into {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            # Дополнительная проверка размера
            if len(chunk) > 3000:
                logger.warning(f"Chunk {i+1} is too long ({len(chunk)} chars), splitting further")
                sub_chunks = split_long(chunk, 1000)
                for j, sub_chunk in enumerate(sub_chunks):
                    if len(sub_chunk) > 3000:
                        logger.warning(f"Sub-chunk {j+1} still too long, truncating")
                        sub_chunk = sub_chunk[:2900] + "..."
                    
                    logger.debug(f"Sending sub-chunk {j+1}: {len(sub_chunk)} chars")
                    
                    try:
                        await context.bot.send_message(
                            chat_id=query.message.chat_id,
                            text=sub_chunk,
                            parse_mode=None,
                            disable_web_page_preview=True,
                        )
                    except Exception as sub_e:
                        logger.error(f"Failed to send sub-chunk {j+1}: {sub_e}")
                        # В крайнем случае отправляем урезанную версию
                        await context.bot.send_message(
                            chat_id=query.message.chat_id,
                            text=sub_chunk[:1000] + "...",
                            parse_mode=None,
                            disable_web_page_preview=True,
                        )
            else:
                logger.debug(f"Sending chunk {i+1}/{len(chunks)}: {len(chunk)} chars")
                
                try:
                    # Force plain text to avoid Telegram parse errors on arbitrary content
                    await context.bot.send_message(
                        chat_id=query.message.chat_id,
                        text=chunk,
                        parse_mode=None,
                        disable_web_page_preview=True,
                    )
                except Exception as e:
                    logger.error(f"Failed to send chunk {i+1}: {e}")
                    # Если сообщение все еще слишком длинное, попробуем разбить еще больше
                    if "Message is too long" in str(e) or "Text is too long" in str(e):
                        logger.warning("Message still too long, splitting further")
                        sub_chunks = split_long(chunk, 1000)
                        for j, sub_chunk in enumerate(sub_chunks):
                            try:
                                await context.bot.send_message(
                                    chat_id=query.message.chat_id,
                                    text=sub_chunk,
                                    parse_mode=None,
                                    disable_web_page_preview=True,
                                )
                            except Exception as sub_e:
                                logger.error(f"Failed to send sub-chunk {j+1}: {sub_e}")
                                # В крайнем случае отправляем урезанную версию
                                await context.bot.send_message(
                                    chat_id=query.message.chat_id,
                                    text=sub_chunk[:1000] + "...",
                                    parse_mode=None,
                                    disable_web_page_preview=True,
                                )
                    else:
                        raise
        await self._show_main_menu(chat_id, context)

    async def _cb_set_summary_interval(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle selection of preset summary intervals."""
        query = update.callback_query
        if not query:
            return
        await self._safe_query_answer(query)
        data = query.data or ""
        chat_id = update.effective_chat.id if update.effective_chat else 0
        if data == "summary_int_custom":
            # Ask user for a custom value
            self._set_interval_state[chat_id] = {"step": "await_hours"}
            await query.edit_message_text("Введите интервал сводки в часах:")
            return
        try:
            hours = int(data.split("_", 2)[-1])
        except Exception:
            await query.edit_message_text("Не удалось разобрать период.")
            return
        try:
            msg = "Интервал обновлён"
            if self._set_summary_interval:
                res = await self._set_summary_interval(hours)
                if res:
                    msg = res
        except Exception as exc:
            logger.exception("Failed to set summary interval: %s", exc)
            msg = "Не удалось обновить интервал"
        await query.edit_message_text(msg)
        self._set_interval_state.pop(chat_id, None)

    async def _cmd_set_summary_interval(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /set_summary_interval command by offering common options."""
        chat_id = update.effective_chat.id if update.effective_chat else 0

        if not hasattr(self, "_set_interval_state"):
            self._set_interval_state = _PersistedDict(  # type: ignore[attr-defined]
                save_callback=self._save_states
            )

        args = getattr(context, "args", [])
        if args and args[0].isdigit():
            hours = int(args[0])
            msg = "Интервал обновлён"
            try:
                if self._set_summary_interval:
                    res = await self._set_summary_interval(hours)
                    if res:
                        msg = res
            except Exception as exc:
                logger.exception("Failed to set summary interval: %s", exc)
                msg = "Не удалось обновить интервал"
            if update.message:
                await update.message.reply_text(msg)
            return

        self._set_interval_state[chat_id] = {"step": "await_hours"}
        if update.message:
            keyboard = [
                [
                    InlineKeyboardButton("1 ч", callback_data="summary_int_1"),
                    InlineKeyboardButton("6 ч", callback_data="summary_int_6"),
                ],
                [
                    InlineKeyboardButton("12 ч", callback_data="summary_int_12"),
                    InlineKeyboardButton("24 ч", callback_data="summary_int_24"),
                ],
                [
                    InlineKeyboardButton(
                        "Другое значение", callback_data="summary_int_custom"
                    )
                ],
            ]
            markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "Выберите интервал сводки:", reply_markup=markup
            )

    async def _cmd_signals(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._run_finance_analysis:
            await update.message.reply_text(
                "Финансовый анализ недоступен. Установите зависимости, например: `pip install yfinance finrl`"
            )
            return
        message = getattr(update, "effective_message", None) or getattr(
            update, "message", None
        )
        text = (getattr(message, "text", "") or "").strip()
        # format: /signals TICKER1 TICKER2 or /signals TICKER1,TICKER2
        parts = text.split(maxsplit=1)
        tickers: List[str] = []
        if len(parts) == 2 and parts[1].strip():
            tickers = [
                t.strip().upper() for t in re.split(r"[\s,]+", parts[1]) if t.strip()
            ]
        if not tickers:
            await update.message.reply_text(
                "Укажите тикеры через пробел или запятую: /signals AAPL MSFT или /signals AAPL,MSFT"
            )
            return
        chat = getattr(update, "effective_chat", None)
        chat_id = chat.id if chat else 0
        if context.bot:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        await update.message.reply_text("Готовлю анализ…")
        try:
            result = await self._run_finance_analysis(tickers)
            mode_msg = (
                "Используется продвинутый режим (FinRL)."
                if result.advanced
                else "Используется упрощённый режим."
            )
            await update.message.reply_text(mode_msg)
            report = result.report_markdown
            for chunk in split_long(report, 3500):
                await update.message.reply_text(chunk)
        except (ConnectionError, OSError) as exc:  # pragma: no cover - network errors
            logger.exception("Signals network problem: %s", exc)
            await update.message.reply_text(
                "Не удалось получить данные для анализа из-за проблем сети. Попробуйте позже."
            )
        except Exception as exc:
            logger.exception("Signals failed: %s", exc)
            await update.message.reply_text("Ошибка анализа.")
