from __future__ import annotations

import logging
from html import escape
from typing import Awaitable, Callable, Dict, List, Optional

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, constants
    from telegram.ext import ContextTypes
except Exception:  # pragma: no cover - optional dependency
    Update = InlineKeyboardButton = InlineKeyboardMarkup = None  # type: ignore
    ContextTypes = None  # type: ignore


logger = logging.getLogger(__name__)


class ThemeCommandsMixin:
    _list_themes: Optional[Callable[[], Awaitable[List[str]]]]
    _create_new_theme: Optional[Callable[[str, List[str]], Awaitable[bool]]]
    _delete_theme_by_name: Optional[Callable[[str], Awaitable[bool]]]
    _get_theme_chats: Optional[Callable[[str], Awaitable[List[str]]]]
    _add_chat_to_theme_by_name: Optional[Callable[[str, str], Awaitable[bool]]]
    _remove_chat_from_theme_by_name: Optional[Callable[[str, str], Awaitable[bool]]]
    _set_active_theme: Optional[Callable[[str], Awaitable[None]]]
    _theme_state: Dict[int, Dict]
    _list_chats: Optional[Callable[[], Awaitable[List[str]]]]
    _refresh_chat_cache: Optional[Callable[[], Awaitable[bool]]]

    # ========================= Theme Management Commands =========================

    async def _cmd_themes(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show all themes with management buttons."""
        if not self._list_themes:
            await update.message.reply_text("Управление темами недоступно.")
            return

        try:
            themes = await self._list_themes()
        except (ConnectionError, OSError) as exc:  # pragma: no cover - network errors
            logger.exception("Theme list network problem: %s", exc)
            await update.message.reply_text(
                "Не удалось получить список тем из-за проблем сети. Попробуйте позже."
            )
            return
        except Exception as exc:
            logger.exception("Error listing themes: %s", exc)
            await update.message.reply_text("Ошибка при получении списка тем.")
            return
        if not themes:
            await update.message.reply_text(
                "Нет созданных тем.\nИспользуйте /create_theme <название> для создания новой темы."
            )
            return

        # Store theme index mapping for all operations
        chat_id = update.effective_chat.id if update.effective_chat else 0

        # Clear any existing state to avoid conflicts
        if chat_id in self._theme_state:
            logger.debug(
                "Clearing existing theme state for chat %s: %s",
                chat_id,
                self._theme_state[chat_id],
            )
            del self._theme_state[chat_id]

        theme_index_map = {i: theme for i, theme in enumerate(themes)}
        logger.debug(
            "Creating main_theme_map for chat %s: %s", chat_id, theme_index_map
        )
        # Store in theme_state for callback handling
        self._theme_state[chat_id] = {"main_theme_map": theme_index_map}
        logger.debug("Stored main_theme_map in state: %s", self._theme_state[chat_id])

        buttons = []
        for i, theme in enumerate(themes):
            # Truncate long theme names for display
            display_name = theme[:25] + "..." if len(theme) > 25 else theme
            buttons.append(
                [
                    InlineKeyboardButton(
                        f"📁 {display_name}", callback_data=f"theme_info_{i}"
                    ),
                    InlineKeyboardButton("✏️", callback_data=f"theme_edit_{i}"),
                    InlineKeyboardButton("🗑️", callback_data=f"theme_delete_{i}"),
                ]
            )

        buttons.append(
            [
                InlineKeyboardButton(
                    "➕ Создать новую тему", callback_data="theme_create"
                )
            ]
        )

        keyboard = InlineKeyboardMarkup(buttons)
        await update.message.reply_text("Управление темами:", reply_markup=keyboard)

    async def _cmd_create_theme(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Create new theme."""
        if not self._create_new_theme or not self._list_chats:
            await update.message.reply_text("Создание тем недоступно.")
            return

        # Parse theme name from command
        text = (update.message.text or "").strip()
        parts = text.split(maxsplit=1)

        if len(parts) < 2:
            await update.message.reply_text(
                "Использование: /create_theme <название темы>"
            )
            return

        theme_name = parts[1].strip()
        if not theme_name:
            await update.message.reply_text("Название темы не может быть пустым.")
            return

        chat_id = update.effective_chat.id if update.effective_chat else 0
        try:
            # Get available chats
            chats = await self._list_chats()
        except (ConnectionError, OSError) as exc:  # pragma: no cover - network errors
            logger.exception("Create theme network problem: %s", exc)
            await update.message.reply_text(
                "Не удалось получить список чатов из-за проблем сети. Попробуйте позже."
            )
            return
        except Exception as exc:
            logger.exception("Error creating theme: %s", exc)
            await update.message.reply_text("Ошибка при создании темы.")
            return
        if not chats:
            await update.message.reply_text(
                "Нет доступных чатов для добавления в тему."
            )
            return

        # Store state for chat selection
        self._theme_state[chat_id] = {
            "action": "create",
            "theme_name": theme_name,
            "available_chats": chats,
            "selected_chats": [],
            "current_page": 0,
        }

        # Show chat selection
        await self._show_chat_selection(update, chat_id)

    async def _cmd_delete_theme(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Delete theme with confirmation."""
        if not self._delete_theme_by_name or not self._list_themes:
            await update.message.reply_text("Удаление тем недоступно.")
            return

        try:
            themes = await self._list_themes()
        except (ConnectionError, OSError) as exc:  # pragma: no cover - network errors
            logger.exception("Delete theme network problem: %s", exc)
            await update.message.reply_text(
                "Не удалось получить список тем из-за проблем сети. Попробуйте позже."
            )
            return
        except Exception as exc:
            logger.exception("Error in delete theme: %s", exc)
            await update.message.reply_text("Ошибка при получении списка тем.")
            return
        if not themes:
            await update.message.reply_text("Нет тем для удаления.")
            return
        # Store theme index mapping to avoid long callback_data
        chat_id = update.effective_chat.id if update.effective_chat else 0

        # Clear any existing state to avoid conflicts
        if chat_id in self._theme_state:
            logger.debug(
                "Clearing existing theme state for chat %s: %s",
                chat_id,
                self._theme_state[chat_id],
            )
            del self._theme_state[chat_id]

        theme_index_map = {i: theme for i, theme in enumerate(themes)}
        logger.debug(
            "Creating delete_theme_map for chat %s: %s", chat_id, theme_index_map
        )
        # Store temporarily in theme_state for callback handling
        self._theme_state[chat_id] = {"delete_theme_map": theme_index_map}
        logger.debug("Stored delete_theme_map in state: %s", self._theme_state[chat_id])

        buttons = []
        for i, theme in enumerate(themes):
            # Truncate long theme names for display
            display_name = theme[:40] + "..." if len(theme) > 40 else theme
            buttons.append(
                [
                    InlineKeyboardButton(
                        f"🗑️ {display_name}", callback_data=f"theme_delete_{i}"
                    )
                ]
            )

        keyboard = InlineKeyboardMarkup(buttons)
        await update.message.reply_text(
            "Выберите тему для удаления:", reply_markup=keyboard
        )

    async def _cmd_edit_theme(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Edit theme chats."""
        if not self._get_theme_chats or not self._list_themes or not self._list_chats:
            await update.message.reply_text("Редактирование тем недоступно.")
            return

        try:
            themes = await self._list_themes()
            if not themes:
                await update.message.reply_text("Нет тем для редактирования.")
                return

            # Store theme index mapping to avoid long callback_data
            chat_id = update.effective_chat.id if update.effective_chat else 0
            theme_index_map = {i: theme for i, theme in enumerate(themes)}
            # Store temporarily in theme_state for callback handling
            if chat_id not in self._theme_state:
                self._theme_state[chat_id] = {}
            self._theme_state[chat_id]["edit_theme_map"] = theme_index_map

            buttons = []
            for i, theme in enumerate(themes):
                # Truncate long theme names for display
                display_name = theme[:40] + "..." if len(theme) > 40 else theme
                buttons.append(
                    [
                        InlineKeyboardButton(
                            f"✏️ {display_name}", callback_data=f"theme_edit_start_{i}"
                        )
                    ]
                )

            keyboard = InlineKeyboardMarkup(buttons)
            await update.message.reply_text(
                "Выберите тему для редактирования:", reply_markup=keyboard
            )
        except Exception as exc:
            logger.exception("Error in edit theme: %s", exc)
            await update.message.reply_text("Ошибка при получении списка тем.")

    async def _cmd_quick_switch_theme(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Quickly cycle to the next available theme."""
        if not self._set_active_theme or not self._list_themes:
            await update.message.reply_text("Переключение тем недоступно.")
            return

        try:
            themes = await self._list_themes()
            if not themes:
                await update.message.reply_text("Нет доступных тем.")
                return

            get_name = getattr(self, "_get_active_theme_name", None)
            current = await get_name() if get_name else None
            idx = 0
            if current in themes:
                idx = (themes.index(current) + 1) % len(themes)
            next_theme = themes[idx]
            await self._set_active_theme(next_theme)
            await update.message.reply_text(
                f"Активная тема: {escape(next_theme)}",
                parse_mode=constants.ParseMode.HTML,
            )
        except Exception as exc:
            logger.exception("Error in quick switch theme: %s", exc)
            await update.message.reply_text("Ошибка при переключении темы.")

    async def _cmd_switch_theme(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Switch active theme."""
        if not self._set_active_theme or not self._list_themes:
            await update.message.reply_text("Переключение тем недоступно.")
            return

        try:
            themes = await self._list_themes()
            if not themes:
                await update.message.reply_text("Нет доступных тем.")
                return

            # Store theme index mapping to avoid long callback_data
            chat_id = update.effective_chat.id if update.effective_chat else 0
            theme_index_map = {i: theme for i, theme in enumerate(themes)}
            # Store temporarily in theme_state for callback handling
            if chat_id not in self._theme_state:
                self._theme_state[chat_id] = {}
            self._theme_state[chat_id]["switch_theme_map"] = theme_index_map

            buttons = []
            for i, theme in enumerate(themes):
                # Truncate long theme names for display
                display_name = theme[:40] + "..." if len(theme) > 40 else theme
                buttons.append(
                    [
                        InlineKeyboardButton(
                            f"🔄 {display_name}", callback_data=f"theme_switch_{i}"
                        )
                    ]
                )

            keyboard = InlineKeyboardMarkup(buttons)
            await update.message.reply_text(
                "Выберите активную тему:", reply_markup=keyboard
            )
        except Exception as exc:
            logger.exception("Error in switch theme: %s", exc)
            await update.message.reply_text("Ошибка при получении списка тем.")
