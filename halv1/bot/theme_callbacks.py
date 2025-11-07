from __future__ import annotations

import logging
from typing import Awaitable, Callable, Dict, List, Optional

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import ContextTypes
except Exception:  # pragma: no cover - optional dependency
    Update = InlineKeyboardButton = InlineKeyboardMarkup = None  # type: ignore
    ContextTypes = None  # type: ignore


logger = logging.getLogger(__name__)


class ThemeCallbacksMixin:
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
    # ========================= Theme Callback Handlers =========================

    async def _cb_theme_action(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle theme management callbacks."""
        query = update.callback_query
        await self._safe_query_answer(query)

        data = query.data
        chat_id = update.effective_chat.id if update.effective_chat else 0
        state = self._theme_state.get(chat_id)
        if data != "theme_create" and not state:
            await query.edit_message_text(
                "Сессия управления темами истекла. Вызовите /themes заново."
            )
            return

        logger.debug("=== THEME CALLBACK DEBUG ===")
        logger.debug("Raw callback_data: '%s'", data)
        logger.debug("Chat ID: %s", chat_id)
        logger.debug("Current theme_state for chat: %s", state or {})

        try:
            if data == "theme_create":
                await query.edit_message_text("Введите название новой темы:")
                self._theme_state[chat_id] = {"action": "create_input"}

            elif data.startswith("theme_info_"):
                theme_index_str = data[11:]  # Remove "theme_info_" prefix
                try:
                    theme_index = int(theme_index_str)
                    # Get theme name from stored mapping
                    state = self._theme_state.get(chat_id, {})
                    main_theme_map = state.get("main_theme_map", {})
                    theme = main_theme_map.get(theme_index)

                    if theme and self._get_theme_chats:
                        chats = await self._get_theme_chats(theme)
                        chat_list = (
                            "\n".join(f"• {chat}" for chat in chats)
                            if chats
                            else "Нет чатов"
                        )
                        await query.edit_message_text(
                            f"📁 Тема: {theme}\n\nЧаты:\n{chat_list}"
                        )
                    else:
                        await query.edit_message_text("Ошибка: тема не найдена.")
                except ValueError:
                    await query.edit_message_text("Ошибка: неверный индекс темы.")

            elif data.startswith("theme_edit_"):
                theme_index_str = data[11:]  # Remove "theme_edit_" prefix
                try:
                    theme_index = int(theme_index_str)
                    # Get theme name from stored mapping
                    state = self._theme_state.get(chat_id, {})
                    main_theme_map = state.get("main_theme_map", {})
                    theme = main_theme_map.get(theme_index)

                    if theme:
                        await self._start_theme_editing(query, chat_id, theme)
                        # Clean up the main mapping after use since we'll create specific edit mapping
                        if "main_theme_map" in state:
                            del state["main_theme_map"]
                    else:
                        await query.edit_message_text("Ошибка: тема не найдена.")
                except ValueError:
                    await query.edit_message_text("Ошибка: неверный индекс темы.")

            elif data.startswith("theme_delete_confirm_"):
                theme_index_str = data[21:]  # Remove "theme_delete_confirm_" prefix
                logger.debug("Processing theme_delete_confirm callback: %s", data)
                logger.debug(
                    "Extracted index string: '%s' (length: %d)",
                    theme_index_str,
                    len(theme_index_str),
                )
                logger.debug("Index string bytes: %s", theme_index_str.encode("utf-8"))
                try:
                    # Clean up any potential whitespace or special characters
                    clean_index_str = theme_index_str.strip()
                    logger.debug("Cleaned index string: '%s'", clean_index_str)
                    theme_index = int(clean_index_str)
                    logger.debug("Successfully parsed theme_index: %d", theme_index)
                    # Get theme name from stored mapping
                    state = self._theme_state.get(chat_id, {})
                    delete_theme_map = state.get("delete_theme_map", {})
                    theme = delete_theme_map.get(theme_index)

                    if theme and self._delete_theme_by_name:
                        success = await self._delete_theme_by_name(theme)
                        if success:
                            await query.edit_message_text(f"Тема '{theme}' удалена.")
                        else:
                            await query.edit_message_text(
                                f"Ошибка при удалении темы '{theme}'."
                            )
                    else:
                        await query.edit_message_text("Ошибка: тема не найдена.")

                    # Clean up temporary mapping
                    if chat_id in self._theme_state:
                        state = self._theme_state[chat_id]
                        if "delete_theme_map" in state:
                            del state["delete_theme_map"]
                        if not state:  # If empty, remove the entire entry
                            del self._theme_state[chat_id]
                except ValueError as e:
                    logger.exception(
                        "Failed to parse theme index for confirm '%s': %s",
                        theme_index_str,
                        e,
                    )
                    await query.edit_message_text(
                        f"Ошибка: неверный индекс темы '{theme_index_str}'."
                    )

            elif data.startswith("theme_delete_"):
                theme_index_str = data[13:]  # Remove "theme_delete_" prefix
                logger.debug(
                    "Processing theme_delete callback: %s, extracted index: '%s'",
                    data,
                    theme_index_str,
                )
                try:
                    theme_index = int(theme_index_str)
                    logger.debug("Parsed theme_index: %d", theme_index)
                    # Get theme name from stored mapping - check both main menu and delete command mappings
                    state = self._theme_state.get(chat_id, {})
                    delete_theme_map = state.get("delete_theme_map", {})
                    main_theme_map = state.get("main_theme_map", {})

                    # Try delete mapping first, then main mapping
                    theme = delete_theme_map.get(theme_index) or main_theme_map.get(
                        theme_index
                    )
                    logger.debug("Found theme for index %d: %s", theme_index, theme)
                    logger.debug("Delete theme map: %s", delete_theme_map)
                    logger.debug("Main theme map: %s", main_theme_map)

                    if theme:
                        # If coming from main menu, create delete mapping for confirmation step
                        if not delete_theme_map and main_theme_map:
                            state["delete_theme_map"] = main_theme_map.copy()
                            # Clean up main mapping
                            if "main_theme_map" in state:
                                del state["main_theme_map"]

                        # Safety check: ensure theme_index is actually a number
                        logger.debug(
                            "Creating confirm button with theme_index: %d (type: %s)",
                            theme_index,
                            type(theme_index),
                        )
                        if not isinstance(theme_index, int):
                            logger.error(
                                "ERROR: theme_index is not an integer: %s", theme_index
                            )
                            await query.edit_message_text(
                                "Ошибка: неверный тип индекса темы."
                            )
                            return

                        confirm_callback = f"theme_delete_confirm_{theme_index}"
                        logger.debug(
                            "Generated confirm callback_data: '%s'", confirm_callback
                        )

                        buttons = [
                            [
                                InlineKeyboardButton(
                                    "✅ Да, удалить", callback_data=confirm_callback
                                )
                            ],
                            [
                                InlineKeyboardButton(
                                    "❌ Отмена", callback_data="theme_cancel"
                                )
                            ],
                        ]
                        keyboard = InlineKeyboardMarkup(buttons)
                        await query.edit_message_text(
                            f"Удалить тему '{theme}'?", reply_markup=keyboard
                        )
                    else:
                        all_mappings = {**delete_theme_map, **main_theme_map}
                        await query.edit_message_text(
                            f"Ошибка: тема не найдена для индекса {theme_index}. Доступные: {list(all_mappings.keys())}"
                        )
                except ValueError as e:
                    logger.exception(
                        "Failed to parse theme index '%s': %s", theme_index_str, e
                    )
                    # Check if this is the "confirm_X" issue
                    if "confirm_" in theme_index_str:
                        logger.error(
                            "DETECTED CONFIRM PREFIX ISSUE: '%s' contains 'confirm_'",
                            theme_index_str,
                        )
                        logger.error("Original callback_data: '%s'", data)
                        logger.error("This suggests a double prefixing issue")
                    await query.edit_message_text(
                        f"Ошибка: неверный индекс темы '{theme_index_str}'."
                    )

            elif data.startswith("theme_switch_"):
                theme_index_str = data[13:]  # Remove "theme_switch_" prefix
                try:
                    theme_index = int(theme_index_str)
                    # Get theme name from stored mapping - check both switch command and main menu mappings
                    state = self._theme_state.get(chat_id, {})
                    switch_theme_map = state.get("switch_theme_map", {})
                    main_theme_map = state.get("main_theme_map", {})

                    # Try switch mapping first, then main mapping
                    theme = switch_theme_map.get(theme_index) or main_theme_map.get(
                        theme_index
                    )

                    if theme and self._set_active_theme:
                        await self._set_active_theme(theme)
                        await query.edit_message_text(
                            f"✅ Активная тема переключена на '{theme}'."
                        )
                        # Clean up the mappings
                        mappings_to_clean = ["switch_theme_map", "main_theme_map"]
                        for mapping in mappings_to_clean:
                            if mapping in state:
                                del state[mapping]
                    else:
                        await query.edit_message_text("❌ Не удалось переключить тему.")
                except ValueError:
                    await query.edit_message_text("Ошибка: неверный индекс темы.")

            elif data.startswith("theme_edit_start_"):
                theme_index_str = data[17:]  # Remove "theme_edit_start_" prefix
                try:
                    theme_index = int(theme_index_str)
                    # Get theme name from stored mapping
                    state = self._theme_state.get(chat_id, {})
                    edit_theme_map = state.get("edit_theme_map", {})
                    theme = edit_theme_map.get(theme_index)

                    if theme:
                        await self._start_theme_editing(query, chat_id, theme)
                        # Clean up the mapping after use
                        if "edit_theme_map" in state:
                            del state["edit_theme_map"]
                    else:
                        await query.edit_message_text("Ошибка: тема не найдена.")
                except ValueError:
                    await query.edit_message_text("Ошибка: неверный индекс темы.")

            elif data == "theme_cancel":
                await query.edit_message_text("Операция отменена.")
                # Clean up any temporary mappings
                state = self._theme_state.get(chat_id, {})
                mappings_to_clean = [
                    "delete_theme_map",
                    "edit_theme_map",
                    "switch_theme_map",
                    "main_theme_map",
                ]
                for mapping in mappings_to_clean:
                    if mapping in state:
                        del state[mapping]
                if not state:  # If state is empty, remove the chat_id entry
                    self._theme_state.pop(chat_id, None)

        except Exception as exc:
            logger.exception("Error in theme callback: %s", exc)
            await query.edit_message_text("Произошла ошибка.")
