from __future__ import annotations

import logging
from typing import List

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import ContextTypes
except Exception:  # pragma: no cover - telegram may be missing
    Update = InlineKeyboardButton = InlineKeyboardMarkup = ContextTypes = None  # type: ignore

logger = logging.getLogger(__name__)


class TelethonHandlersMixin:
    def _register_telethon(self) -> None:
        if self._telethon_auth_request_code and self._telethon_auth_sign_in:
            self._commands["telethon_auth"] = {
                "handler": self._cmd_telethon_auth,
                "description": "— интерактивная авторизация Telethon",
                "usage": "/telethon_auth",
            }
            self._help_sections["settings"] = {
                "title": "Настройки",
                "commands": ["telethon_auth"],
            }

    async def _handle_telethon_auth(
        self, update: Update, text: str, chat_id: int
    ) -> bool:
        if chat_id not in self._auth_state:
            return False
        state = self._auth_state[chat_id]
        step = state.get("step")
        if step == "phone":
            phone = text.strip()
            if not phone:
                await update.message.reply_text(
                    "Введите номер телефона в международном формате"
                )
                return True
            try:
                if self._telethon_auth_request_code:
                    await self._telethon_auth_request_code(phone)
                state["phone"] = phone
                state["step"] = "code"
                await update.message.reply_text(
                    "Код отправлен. Введите код из Telegram"
                )
            except Exception as exc:
                logger.exception("Auth phone step failed: %s", exc)
                await update.message.reply_text(
                    "Не удалось отправить код. Проверьте номер и попробуйте снова."
                )
                self._auth_state.pop(chat_id, None)
            return True
        if step == "code":
            code = text.strip().replace(" ", "")
            try:
                if self._telethon_auth_sign_in:
                    res = await self._telethon_auth_sign_in(
                        state.get("phone", ""), code, None
                    )
                else:
                    res = {"ok": "false", "error": "no handler"}
                if res.get("ok") == "true":
                    await update.message.reply_text("Авторизация завершена")
                    self._auth_state.pop(chat_id, None)
                    return True
                if res.get("need_password") == "true":
                    state["step"] = "password"
                    state["code"] = code
                    await update.message.reply_text("Включена 2FA. Введите пароль:")
                    return True
                await update.message.reply_text(
                    "Не удалось авторизоваться: " + (res.get("error") or "")
                )
            except Exception as exc:
                logger.exception("Auth code step failed: %s", exc)
                await update.message.reply_text(
                    "Ошибка авторизации. Попробуйте ещё раз позже."
                )
                self._auth_state.pop(chat_id, None)
            return True
        if step == "password":
            password = text
            try:
                if self._telethon_auth_sign_in:
                    res = await self._telethon_auth_sign_in(
                        state.get("phone", ""), state.get("code", ""), password
                    )
                else:
                    res = {"ok": "false", "error": "no handler"}
                if res.get("ok") == "true":
                    await update.message.reply_text("Авторизация завершена")
                else:
                    await update.message.reply_text(
                        "Неверный пароль или ошибка авторизации"
                    )
            except Exception as exc:
                logger.exception("Auth password step failed: %s", exc)
                await update.message.reply_text(
                    "Ошибка авторизации. Попробуйте ещё раз позже."
                )
            finally:
                self._auth_state.pop(chat_id, None)
            return True
        return False

    async def _show_settings_menu(
        self, chat_id: int, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        buttons: List[List[InlineKeyboardButton]] = []
        if self._telethon_auth_request_code and self._telethon_auth_sign_in:
            buttons.append(
                [
                    InlineKeyboardButton(
                        "🔐 Авторизация Telethon", callback_data="menu_telethon"
                    )
                ]
            )
        buttons.append([InlineKeyboardButton("📊 Статус", callback_data="menu_status")])
        buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="menu_main")])
        await context.bot.send_message(
            chat_id=chat_id,
            text="Настройки:",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def _cmd_telethon_auth(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not (self._telethon_auth_request_code and self._telethon_auth_sign_in):
            await update.message.reply_text("Авторизация недоступна.")
            return
        if self._telethon_is_authorized and await self._telethon_is_authorized():
            await update.message.reply_text("Уже авторизовано.")
            return
        chat_id = update.effective_chat.id if update.effective_chat else 0
        self._auth_state[chat_id] = {"step": "phone"}
        await update.message.reply_text(
            "Введите номер телефона в международном формате (например, +6688...):"
        )
