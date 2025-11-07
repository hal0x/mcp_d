from __future__ import annotations

import asyncio
import pathlib
import sys
from typing import Any, List, cast

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from telegram.ext import CallbackContext, ExtBot

from bot.telegram_bot import TelegramBot


class DummyMessage:
    def __init__(self, text: str = ""):
        self.text = text
        self.replies: List[str] = []

    async def reply_text(self, text: str, **kwargs: Any) -> None:
        self.replies.append(text)


class DummySentMessage:
    def __init__(self) -> None:
        self.edits: List[str] = []

    async def edit_text(self, text: str, **kwargs: Any) -> None:
        self.edits.append(text)


class DummyQuery:
    def __init__(self, data: str):
        self.data = data
        self.edits: List[str] = []

    async def answer(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def edit_message_text(self, text: str, **kwargs: Any) -> DummySentMessage:
        self.edits.append(text)
        return DummySentMessage()


class DummyContext:
    pass


def test_edit_theme_flow_callbacks() -> None:
    async def main() -> None:
        call_log: List[str] = []
        create_args: dict[str, str | List[str]] = {}

        async def list_themes() -> List[str]:
            call_log.append("list_themes")
            return ["work"]

        async def list_chats() -> List[str]:
            call_log.append("list_chats")
            return ["chat1", "chat2"]

        async def get_theme_chats(name: str) -> List[str]:
            call_log.append("get_theme_chats")
            return ["chat1"]

        async def create_new_theme(name: str, chats: List[str]) -> bool:
            call_log.append("create_new_theme")
            create_args["name"] = name
            create_args["chats"] = chats
            return True

        async def send_message(text: str, chat_id: int) -> str:
            return ""

        bot = TelegramBot(
            "token",
            send_message,
            list_chats=list_chats,
            list_themes=list_themes,
            create_new_theme=create_new_theme,
            get_theme_chats=get_theme_chats,
        )

        ctx = cast(
            CallbackContext[
                ExtBot[None], dict[Any, Any], dict[Any, Any], dict[Any, Any]
            ],
            DummyContext(),
        )
        update = type(
            "Update",
            (),
            {
                "message": DummyMessage(),
                "effective_chat": type("Chat", (), {"id": 1})(),
            },
        )()
        await bot._cmd_themes(update, ctx)

        query = DummyQuery("theme_edit_0")
        update_cb = type(
            "Update",
            (),
            {"callback_query": query, "effective_chat": type("Chat", (), {"id": 1})()},
        )()
        await bot._cb_theme_action(update_cb, ctx)

        toggle_query = DummyQuery("chat_toggle_1")
        update_toggle = type(
            "Update",
            (),
            {
                "callback_query": toggle_query,
                "effective_chat": type("Chat", (), {"id": 1})(),
            },
        )()
        await bot._cb_chat_toggle(update_toggle, ctx)

        save_query = DummyQuery("chat_save")
        update_save = type(
            "Update",
            (),
            {
                "callback_query": save_query,
                "effective_chat": type("Chat", (), {"id": 1})(),
            },
        )()
        await bot._cb_chat_toggle(update_save, ctx)

        assert call_log == [
            "list_themes",
            "get_theme_chats",
            "list_chats",
            "create_new_theme",
        ]
        assert create_args["name"] == "work"
        assert create_args["chats"] == ["chat1", "chat2"]
        assert 1 not in bot._theme_state

    asyncio.run(main())


def test_select_all_and_clear() -> None:
    async def main() -> None:
        async def list_chats() -> List[str]:
            return ["chat1", "chat2", "chat3"]

        async def send_message(text: str, chat_id: int) -> str:
            return ""

        bot = TelegramBot("token", send_message, list_chats=list_chats)

        chat_id = 1
        bot._theme_state[chat_id] = {
            "available_chats": ["chat1", "chat2", "chat3"],
            "selected_chats": ["chat1"],
            "theme_name": "work",
            "action": "edit",
            "current_page": 0,
        }

        ctx = cast(
            CallbackContext[
                ExtBot[None], dict[Any, Any], dict[Any, Any], dict[Any, Any]
            ],
            DummyContext(),
        )

        select_all_query = DummyQuery("chat_select_all")
        update_select_all = type(
            "Update",
            (),
            {
                "callback_query": select_all_query,
                "effective_chat": type("Chat", (), {"id": chat_id})(),
            },
        )()
        await bot._cb_chat_toggle(update_select_all, ctx)

        assert set(bot._theme_state[chat_id]["selected_chats"]) == {
            "chat1",
            "chat2",
            "chat3",
        }

        clear_query = DummyQuery("chat_clear_all")
        update_clear = type(
            "Update",
            (),
            {
                "callback_query": clear_query,
                "effective_chat": type("Chat", (), {"id": chat_id})(),
            },
        )()
        await bot._cb_chat_toggle(update_clear, ctx)

        assert bot._theme_state[chat_id]["selected_chats"] == []

    asyncio.run(main())
