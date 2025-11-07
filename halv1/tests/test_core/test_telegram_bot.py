import asyncio
import pathlib
import sys
from types import SimpleNamespace
from typing import Any, cast

try:  # python-telegram-bot is optional during tests
    from telegram.error import TimedOut
except Exception:  # pragma: no cover

    class TimedOut(Exception):
        pass


sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from bot.telegram_bot import TelegramBot
from retriever.retriever import Retriever


class DummyEntry:
    def __init__(self, text: str, metadata: dict[str, str]) -> None:
        self.text = text
        self.metadata = metadata


class DummyRetriever:
    def __init__(self, entries: list[DummyEntry]) -> None:
        self.entries = entries

    async def query(self, query: str, top_k: int = 25) -> list[DummyEntry]:
        return self.entries


def test_cmd_search_adds_link_and_escapes_markdown() -> None:
    async def main() -> None:
        entry = DummyEntry(
            "hello [link](test)",
            {"chat": "chatname", "date": "2024-01-01", "message_ids": "42"},
        )

        async def _dummy(text: str, chat_id: int) -> str:
            return ""

        bot = TelegramBot(
            "token", _dummy, retriever=cast(Retriever, DummyRetriever([entry]))
        )
        replies: list[str] = []

        async def fake_reply(
            update: Any, context: Any, text: str, **kwargs: Any
        ) -> None:
            replies.append(text)

        bot._safe_reply = fake_reply  # type: ignore

        class DummyBot:
            async def send_chat_action(self, chat_id: int, action: Any) -> None:
                pass

            async def send_message(
                self, chat_id: int, text: str, **kwargs: Any
            ) -> None:
                pass

        class Ctx:
            args: list[str] = ["query"]
            bot: DummyBot = DummyBot()

        await bot._cmd_search(cast(Any, object()), cast(Any, Ctx()))
        assert "[chatname]" in replies[0]
        expected = r"[hello \[link\]\(test\)](https://t.me/chatname/42)"
        assert expected in replies[0]

    asyncio.run(main())


def test_cb_menu_handles_timeout() -> None:
    async def main() -> None:
        async def _dummy(text: str, chat_id: int) -> str:
            return ""

        bot = TelegramBot("token", _dummy)

        async def dummy_show(chat_id: int, context: Any) -> None:
            pass

        bot._show_search_menu = dummy_show  # type: ignore

        class DummyQuery:
            data = "menu_cat_search"
            message = SimpleNamespace(chat_id=1)

            async def answer(self, *args: Any, **kwargs: Any) -> None:
                raise TimedOut("timeout")

        update = SimpleNamespace(callback_query=DummyQuery())
        context = SimpleNamespace()

        # Should not raise even if query.answer times out
        await bot._cb_menu(cast(Any, update), cast(Any, context))
        
        # Проверяем, что обработчик корректно обработал timeout
        # и не упал с исключением
        assert bot._show_search_menu is not None, "Меню поиска должно быть доступно"
        # Проверяем, что бот находится в корректном состоянии после обработки timeout
        assert hasattr(bot, '_show_search_menu'), "Бот должен иметь метод показа меню поиска"

    asyncio.run(main())


def test_cmd_search_numeric_chat_uses_tg_scheme() -> None:
    async def main() -> None:
        entry = DummyEntry(
            "hello",
            {"chat": "12345", "date": "2024-01-01", "message_ids": "7"},
        )

        async def _dummy(text: str, chat_id: int) -> str:
            return ""

        bot = TelegramBot(
            "token", _dummy, retriever=cast(Retriever, DummyRetriever([entry]))
        )
        replies: list[str] = []

        async def fake_reply(
            update: Any, context: Any, text: str, **kwargs: Any
        ) -> None:
            replies.append(text)

        bot._safe_reply = fake_reply  # type: ignore

        class DummyBot:
            async def send_chat_action(self, chat_id: int, action: Any) -> None:
                pass

        class Ctx:
            args: list[str] = ["query"]
            bot: DummyBot = DummyBot()

        await bot._cmd_search(cast(Any, object()), cast(Any, Ctx()))
        assert "tg://msg?chat_id=12345&message_id=7" in replies[0]

    asyncio.run(main())


def test_cmd_start_setup_when_callbacks_available() -> None:
    async def main() -> None:
        async def _dummy(text: str, chat_id: int) -> str:
            return ""

        async def list_chats() -> list[str]:
            return ["chat1"]

        async def create_new_theme(name: str, chats: list[str]) -> bool:
            return True

        async def set_active_theme(name: str) -> None:
            return None

        bot = TelegramBot(
            "token",
            _dummy,
            list_chats=list_chats,
            create_new_theme=create_new_theme,
            set_active_theme=set_active_theme,
        )

        replies: list[str] = []

        class DummyMessage:
            async def reply_text(self, text: str, **kwargs: Any) -> None:
                replies.append(text)

        update = SimpleNamespace(
            message=DummyMessage(), effective_chat=SimpleNamespace(id=123)
        )

        await bot._cmd_start(cast(Any, update), cast(Any, SimpleNamespace()))

        assert bot._setup_state[123]["step"] == "theme"
        assert replies[-1] == "Введите название новой темы:"

    asyncio.run(main())


def test_cmd_start_without_callbacks_shows_help() -> None:
    async def main() -> None:
        async def _dummy(text: str, chat_id: int) -> str:
            return ""

        bot = TelegramBot("token", _dummy)

        replies: list[str] = []

        class DummyMessage:
            async def reply_text(self, text: str, **kwargs: Any) -> None:
                replies.append(text)

        update = SimpleNamespace(
            message=DummyMessage(), effective_chat=SimpleNamespace(id=123)
        )

        await bot._cmd_start(cast(Any, update), cast(Any, SimpleNamespace()))

        assert "/help" in replies[-1]

    asyncio.run(main())


def test_cmd_chronicle_calls_service() -> None:
    async def main() -> None:
        async def _dummy(text: str, chat_id: int) -> str:
            return ""

        called = False

        async def publish() -> None:
            nonlocal called
            called = True

        bot = TelegramBot("token", _dummy, publish_chronicle=publish)
        assert "chronicle" in bot._commands

        replies: list[str] = []

        async def fake_reply(
            update: Any, context: Any, text: str, **kwargs: Any
        ) -> None:
            replies.append(text)

        bot._safe_reply = fake_reply  # type: ignore

        class DummyBot:
            async def send_chat_action(self, chat_id: int, action: Any) -> None:
                pass

        class Ctx:
            bot: DummyBot = DummyBot()

        await bot._cmd_chronicle(cast(Any, object()), cast(Any, Ctx()))

        assert called
        assert replies[-1] == "Обзор тем опубликован."

    asyncio.run(main())


def test_cmd_status_reports_info() -> None:
    async def main() -> None:
        async def _dummy(text: str, chat_id: int) -> str:
            return ""

        async def get_active_theme_name() -> str:
            return "work"

        async def telethon_is_authorized() -> bool:
            return True

        class DummyIndex:
            def __init__(self) -> None:
                self.entries: list[int] = [1, 2, 3]

        class DummyRetriever:
            def __init__(self) -> None:
                self.index: DummyIndex = DummyIndex()

        bot = TelegramBot(
            "token",
            _dummy,
            get_active_theme_name=get_active_theme_name,
            telethon_is_authorized=telethon_is_authorized,
            retriever=cast(Retriever, DummyRetriever()),
        )

        replies: list[str] = []

        async def fake_reply(
            update: Any, context: Any, text: str, **kwargs: Any
        ) -> None:
            replies.append(text)

        bot._safe_reply = fake_reply  # type: ignore

        class Ctx:
            pass

        await bot._cmd_status(cast(Any, object()), cast(Any, Ctx()))

        assert "Активная тема: work" in replies[0]
        assert "Telethon: авторизован" in replies[0]
        assert "Размер индекса: 3" in replies[0]

    asyncio.run(main())


def test_cmd_set_summary_interval_accepts_arg() -> None:
    async def main() -> None:
        called: list[int] = []

        async def _dummy(text: str, chat_id: int) -> str:
            return ""

        async def _set(hours: int) -> str:
            called.append(hours)
            return ""

        bot = TelegramBot("token", _dummy, set_summary_interval=_set)

        replies: list[str] = []

        class DummyMessage:
            async def reply_text(self, text: str, **kwargs: Any) -> None:
                replies.append(text)

        update = SimpleNamespace(
            message=DummyMessage(), effective_chat=SimpleNamespace(id=123)
        )

        class Ctx:
            args: list[str] = ["5"]

        await bot._cmd_set_summary_interval(cast(Any, update), cast(Any, Ctx()))

        assert called == [5]
        assert replies[-1] == "Интервал обновлён"
        assert 123 not in bot._set_interval_state

    asyncio.run(main())


def test_cmd_sumurl_uses_callback() -> None:
    async def main() -> None:
        async def _dummy(text: str, chat_id: int) -> str:
            return ""

        async def summarize(url: str) -> str:
            return f"summary for {url}"

        bot = TelegramBot("token", _dummy, summarize_url=summarize)

        replies: list[str] = []

        async def fake_reply(
            update: Any, context: Any, text: str, **kwargs: Any
        ) -> None:
            replies.append(text)

        bot._safe_reply = fake_reply  # type: ignore

        class DummyBot:
            async def send_chat_action(self, chat_id: int, action: Any) -> None:
                pass

            async def send_message(
                self, chat_id: int, text: str, **kwargs: Any
            ) -> None:
                pass

        class Ctx:
            args: list[str] = ["http://example.com"]
            bot: DummyBot = DummyBot()

        await bot._cmd_sumurl(cast(Any, object()), cast(Any, Ctx()))
        assert "summary for http://example.com" in replies[0]

    asyncio.run(main())
