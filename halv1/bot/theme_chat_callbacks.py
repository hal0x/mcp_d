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


class ThemeChatCallbacksMixin:
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

    async def _cb_chat_toggle(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle chat selection toggle callbacks."""
        query = update.callback_query
        await self._safe_query_answer(query)

        data = query.data
        chat_id = update.effective_chat.id if update.effective_chat else 0

        if chat_id not in self._theme_state:
            await query.edit_message_text("Сессия истекла. Начните заново.")
            return

        state = self._theme_state[chat_id]

        try:
            if data.startswith("chat_toggle_"):
                chat_index_str = data[12:]  # Remove "chat_toggle_" prefix
                try:
                    chat_index = int(chat_index_str)
                    chat_index_map = state.get("chat_index_map", {})
                    chat_name = chat_index_map.get(chat_index)

                    if chat_name:
                        selected = state.get("selected_chats", [])

                        if chat_name in selected:
                            selected.remove(chat_name)
                        else:
                            selected.append(chat_name)

                        state["selected_chats"] = selected
                        await self._show_chat_selection(
                            update, chat_id, edit_message=True
                        )
                    else:
                        await query.edit_message_text("Ошибка: чат не найден.")
                except ValueError:
                    await query.edit_message_text("Ошибка: неверный индекс чата.")

            elif data == "chat_save":
                await self._save_theme_changes(query, chat_id)

            elif data == "chat_cancel":
                await query.edit_message_text("Операция отменена.")
                self._theme_state.pop(chat_id, None)

            elif data == "chat_select_all":
                available = state.get("available_chats", [])
                state["selected_chats"] = available.copy()
                await self._show_chat_selection(update, chat_id, edit_message=True)

            elif data == "chat_clear_all":
                state["selected_chats"] = []
                await self._show_chat_selection(update, chat_id, edit_message=True)

        except Exception as exc:
            logger.exception("Error in chat toggle callback: %s", exc)
            await query.edit_message_text("Произошла ошибка.")

    async def _cb_chat_nav(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle chat navigation callbacks."""
        query = update.callback_query
        await self._safe_query_answer(query)

        data = query.data
        chat_id = update.effective_chat.id if update.effective_chat else 0

        if chat_id not in self._theme_state:
            await query.edit_message_text("Сессия истекла. Начните заново.")
            return

        state = self._theme_state[chat_id]

        try:
            if data == "nav_prev":
                current_page = state.get("current_page", 0)
                state["current_page"] = max(0, current_page - 1)
                await self._show_chat_selection(update, chat_id, edit_message=True)

            elif data == "nav_next":
                current_page = state.get("current_page", 0)
                available_chats = state.get("available_chats", [])
                total_pages = (
                    len(available_chats) + self.CHATS_PER_PAGE - 1
                ) // self.CHATS_PER_PAGE
                state["current_page"] = min(total_pages - 1, current_page + 1)
                await self._show_chat_selection(update, chat_id, edit_message=True)

            elif data == "nav_info":
                # Info button clicked - just show current page info
                current_page = state.get("current_page", 0)
                available_chats = state.get("available_chats", [])
                total_pages = (
                    len(available_chats) + self.CHATS_PER_PAGE - 1
                ) // self.CHATS_PER_PAGE
                await self._safe_query_answer(
                    query, f"Страница {current_page + 1} из {total_pages}"
                )

            elif data == "nav_refresh":
                if self._refresh_chat_cache:
                    await query.edit_message_text("🔄 Обновляем список чатов...")
                    success = await self._refresh_chat_cache()
                    if success:
                        # Get fresh chat list
                        if self._list_chats:
                            fresh_chats = await self._list_chats()
                            state["available_chats"] = fresh_chats
                            state["current_page"] = 0  # Reset to first page
                            await self._show_chat_selection(
                                update, chat_id, edit_message=True
                            )
                        else:
                            await query.edit_message_text(
                                "✅ Список чатов обновлён, но не удалось получить новый список."
                            )
                    else:
                        await query.edit_message_text(
                            "❌ Не удалось обновить список чатов."
                        )
                else:
                    await query.edit_message_text("Обновление списка чатов недоступно.")

        except Exception as exc:
            logger.exception("Error in chat navigation callback: %s", exc)
            await query.edit_message_text("Произошла ошибка.")

    # ========================= Helper Methods =========================

    async def _show_chat_selection(
        self, update: Update, chat_id: int, edit_message: bool = False
    ) -> None:
        """Show chat selection interface with pagination."""
        state = self._theme_state.get(chat_id)
        if not state:
            return

        available_chats = state.get("available_chats", [])
        selected_chats = state.get("selected_chats", [])
        theme_name = state.get("theme_name", "")
        action = state.get("action", "")
        current_page = state.get("current_page", 0)

        # Store chat index mapping to avoid long callback_data
        chat_index_map = {i: chat for i, chat in enumerate(available_chats)}
        state["chat_index_map"] = chat_index_map

        # Calculate pagination
        total_chats = len(available_chats)
        total_pages = (total_chats + self.CHATS_PER_PAGE - 1) // self.CHATS_PER_PAGE

        # Ensure current page is valid
        current_page = max(0, min(current_page, total_pages - 1))
        state["current_page"] = current_page

        # Get chats for current page
        start_idx = current_page * self.CHATS_PER_PAGE
        end_idx = min(start_idx + self.CHATS_PER_PAGE, total_chats)

        logger.debug(
            "_show_chat_selection: theme=%s, action=%s, page=%d/%d",
            theme_name,
            action,
            current_page + 1,
            total_pages,
        )

        # Build buttons for chats on current page
        buttons = []
        for i in range(start_idx, end_idx):
            chat = available_chats[i]
            is_selected = chat in selected_chats
            icon = "✅" if is_selected else "⬜"
            # Truncate long chat names for display
            display_name = (
                chat[: self.MAX_CHAT_NAME_LENGTH] + "..."
                if len(chat) > self.MAX_CHAT_NAME_LENGTH
                else chat
            )
            buttons.append(
                [
                    InlineKeyboardButton(
                        f"{icon} {display_name}", callback_data=f"chat_toggle_{i}"
                    )
                ]
            )

        # Add navigation buttons if needed
        nav_buttons = []
        if total_pages > 1:
            if current_page > 0:
                nav_buttons.append(
                    InlineKeyboardButton("⬅️ Назад", callback_data="nav_prev")
                )
            nav_buttons.append(
                InlineKeyboardButton(
                    f"{current_page + 1}/{total_pages}", callback_data="nav_info"
                )
            )
            if current_page < total_pages - 1:
                nav_buttons.append(
                    InlineKeyboardButton("➡️ Вперёд", callback_data="nav_next")
                )
            buttons.append(nav_buttons)

        # Add utility buttons
        utility_buttons = []
        if self._refresh_chat_cache:
            utility_buttons.append(
                InlineKeyboardButton("🔄 Обновить список", callback_data="nav_refresh")
            )
        utility_buttons.append(
            InlineKeyboardButton("✅ Все", callback_data="chat_select_all")
        )
        utility_buttons.append(
            InlineKeyboardButton("🗑️ Очистить", callback_data="chat_clear_all")
        )
        if utility_buttons:  # Only add if there are utility buttons
            buttons.append(utility_buttons)

        # Add control buttons
        control_buttons = [
            InlineKeyboardButton("💾 Сохранить", callback_data="chat_save"),
            InlineKeyboardButton("❌ Отмена", callback_data="chat_cancel"),
        ]
        buttons.append(control_buttons)

        keyboard = InlineKeyboardMarkup(buttons)

        selected_count = len(selected_chats)
        text = f"{'Создание' if action == 'create' else 'Редактирование'} темы: {theme_name}\n"
        text += f"Выбрано чатов: {selected_count} из {total_chats}\n"
        if total_pages > 1:
            text += f"Страница: {current_page + 1}/{total_pages}\n"
        text += "\nВыберите чаты для темы:"

        if edit_message and hasattr(update, "callback_query"):
            await update.callback_query.edit_message_text(text, reply_markup=keyboard)
        else:
            await update.message.reply_text(text, reply_markup=keyboard)

    async def _start_theme_editing(self, query, chat_id: int, theme: str) -> None:
        """Start editing a theme."""
        try:
            if self._get_theme_chats and self._list_chats:
                current_chats = await self._get_theme_chats(theme)
                all_chats = await self._list_chats()

                self._theme_state[chat_id] = {
                    "action": "edit",
                    "theme_name": theme,
                    "available_chats": all_chats,
                    "selected_chats": current_chats.copy(),
                    "current_page": 0,
                }

                # Create a fake update object for _show_chat_selection
                fake_update = type(
                    "obj", (object,), {"callback_query": query, "message": None}
                )

                await self._show_chat_selection(fake_update, chat_id, edit_message=True)
        except Exception as exc:
            logger.exception("Error starting theme edit: %s", exc)
            await query.edit_message_text("Ошибка при запуске редактирования темы.")

    async def _save_theme_changes(self, query, chat_id: int) -> None:
        """Save theme changes."""
        state = self._theme_state.get(chat_id)
        if not state:
            await query.edit_message_text("Ошибка: состояние сессии не найдено.")
            return

        theme_name = state.get("theme_name", "")
        selected_chats = state.get("selected_chats", [])
        action = state.get("action", "")

        logger.info(
            "Saving theme changes: action=%s, theme=%s, chats=%s",
            action,
            theme_name,
            selected_chats,
        )

        try:
            if action == "create":
                if self._create_new_theme:
                    success = await self._create_new_theme(theme_name, selected_chats)
                    logger.info("Create theme result: %s", success)
                    if success:
                        await query.edit_message_text(
                            f"✅ Тема '{theme_name}' создана с {len(selected_chats)} чатами."
                        )
                    else:
                        await query.edit_message_text(
                            f"❌ Не удалось создать тему '{theme_name}'."
                        )
            elif action == "edit":
                if self._create_new_theme:  # set_theme under the hood
                    success = await self._create_new_theme(theme_name, selected_chats)
                    logger.info("Edit theme result: %s", success)
                    if success:
                        await query.edit_message_text(
                            f"✅ Тема '{theme_name}' обновлена с {len(selected_chats)} чатами."
                        )
                    else:
                        await query.edit_message_text(
                            f"❌ Не удалось обновить тему '{theme_name}'."
                        )

        except Exception as exc:
            logger.exception("Error saving theme: %s", exc)
            await query.edit_message_text("Ошибка при сохранении темы.")
        finally:
            self._theme_state.pop(chat_id, None)
