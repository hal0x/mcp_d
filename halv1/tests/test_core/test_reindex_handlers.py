import asyncio
import pathlib
import sys
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, cast

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from bot.telegram_bot import TelegramBot


class DummyMessage:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.replies: list[str] = []

    async def reply_text(self, text: str, **kwargs: Any) -> "DummySentMessage":
        self.replies.append(text)
        return DummySentMessage()


class DummySentMessage:
    def __init__(self) -> None:
        self.edits: list[str] = []

    async def edit_text(self, text: str, **kwargs: Any) -> None:
        self.edits.append(text)


class DummyQuery:
    def __init__(self, data: str) -> None:
        self.data = data
        self.edits: list[str] = []

    async def answer(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def edit_message_text(self, text: str, **kwargs: Any) -> DummySentMessage:
        self.edits.append(text)
        return DummySentMessage()


class DummyContext:
    def __init__(self) -> None:
        self.application = SimpleNamespace(bot_data={})
        self.bot = None
        self.args: list[str] = []


def test_reindex_state_flow() -> None:
    async def main() -> None:
        call_log: list[str] = []

        async def list_chats() -> list[str]:
            call_log.append("list_chats")
            return ["chat1", "chat2"]

        async def index_last(chats: list[str], n: int) -> int:
            return 0

        async def index_since(
            chats: list[str],
            since: int,
            progress_cb: Callable[[int], Awaitable[None]] | None = None,
            is_cancelled: Callable[[], bool] | None = None,
        ) -> int:
            call_log.append("index_since")
            return 0

        async def on_message(text: str, chat_id: int) -> str:
            return ""

        bot = TelegramBot(
            "token",
            on_message,
            list_chats=list_chats,
            index_last=index_last,
        )
        bot._index_since = index_since  # type: ignore[attr-defined]

        ctx = cast(Any, DummyContext())
        update = type(
            "Update",
            (),
            {
                "message": DummyMessage(),
                "effective_chat": type("Chat", (), {"id": 1})(),
            },
        )()
        await bot._cmd_reindex(update, ctx)
        assert bot._reindex_state[1]["step"] == "choose_chats"

        # Select first chat via callback
        query_toggle = DummyQuery("reidx_chat_0")
        update_toggle = type(
            "Update",
            (),
            {
                "callback_query": query_toggle,
                "effective_chat": type("Chat", (), {"id": 1})(),
            },
        )()
        await bot._cb_reindex_select(update_toggle, ctx)
        assert bot._reindex_state[1]["selected_titles"] == ["chat1"]

        # Confirm selection
        query_confirm = DummyQuery("reidx_confirm")
        update_confirm = type(
            "Update",
            (),
            {
                "callback_query": query_confirm,
                "effective_chat": type("Chat", (), {"id": 1})(),
            },
        )()
        await bot._cb_reindex_select(update_confirm, ctx)
        assert bot._reindex_state[1]["step"] == "choose_interval"

        # Choose interval
        query = DummyQuery("reindex_day")
        update_cb = type(
            "Update",
            (),
            {"callback_query": query, "effective_chat": type("Chat", (), {"id": 1})()},
        )()
        await bot._cb_reindex_interval(update_cb, ctx)

        assert call_log == ["list_chats", "index_since"]
        assert 1 not in bot._reindex_state

    asyncio.run(main())


def test_dump_index_with_args() -> None:
    async def main() -> None:
        dump_calls: list[int] = []
        index_calls: list[int] = []

        async def dump_since(
            days: int,
            progress_cb: Callable[[int], Awaitable[None]] | None = None,
            is_cancelled: Callable[[], bool] | None = None,
        ) -> int:
            dump_calls.append(days)
            return 1

        async def index_dumped(
            days: int,
            progress_cb: Callable[[int], Awaitable[None]] | None = None,
            is_cancelled: Callable[[], bool] | None = None,
        ) -> int:
            index_calls.append(days)
            return 1

        async def on_message(text: str, chat_id: int) -> str:
            return ""

        bot = TelegramBot(
            "token",
            on_message,
            dump_since=dump_since,
            index_dumped=index_dumped,
        )

        chat = type("Chat", (), {"id": 1})()
        ctx = cast(Any, DummyContext())

        ctx.args = ["5"]
        update_dump = type(
            "Update", (), {"message": DummyMessage(), "effective_chat": chat}
        )()
        await bot._cmd_dump(update_dump, ctx)

        ctx.args = ["10"]
        update_index = type(
            "Update", (), {"message": DummyMessage(), "effective_chat": chat}
        )()
        await bot._cmd_index(update_index, ctx)

        assert dump_calls == [5]
        assert index_calls == [10]
        assert 1 not in bot._dump_state
        assert 1 not in bot._index_state

    asyncio.run(main())


def test_reindex_via_menu() -> None:
    async def main() -> None:
        async def list_chats() -> list[str]:
            return ["chat1", "chat2"]

        async def index_last(chats: list[str], n: int) -> int:
            return 0

        async def on_message(text: str, chat_id: int) -> str:
            return ""

        bot = TelegramBot(
            "token",
            on_message,
            list_chats=list_chats,
            index_last=index_last,
        )

        class DummyBot:
            def __init__(self) -> None:
                self.sent: list[str] = []

            async def send_message(
                self, chat_id: int, text: str, **kwargs: Any
            ) -> None:
                self.sent.append(text)

            async def send_chat_action(self, chat_id: int, action: str) -> None:
                pass

        ctx = cast(Any, SimpleNamespace(bot=DummyBot()))

        await bot._start_reindex(1, ctx)

        assert bot._reindex_state[1]["step"] == "choose_chats"
        assert any("Выберите" in msg for msg in ctx.bot.sent)

    asyncio.run(main())


def test_dump_and_index_callbacks() -> None:
    async def main() -> None:
        dump_calls: list[int] = []
        index_calls: list[int] = []

        async def dump_since(
            days: int,
            progress_cb: Callable[[int], Awaitable[None]] | None = None,
            is_cancelled: Callable[[], bool] | None = None,
        ) -> int:
            dump_calls.append(days)
            return 1

        async def index_dumped(
            days: int,
            progress_cb: Callable[[int], Awaitable[None]] | None = None,
            is_cancelled: Callable[[], bool] | None = None,
        ) -> int:
            index_calls.append(days)
            return 1

        async def on_message(text: str, chat_id: int) -> str:
            return ""

        bot = TelegramBot(
            "token",
            on_message,
            dump_since=dump_since,
            index_dumped=index_dumped,
        )

        ctx = cast(Any, DummyContext())
        chat = type("Chat", (), {"id": 1})()

        update_dump = type(
            "Update", (), {"message": DummyMessage(), "effective_chat": chat}
        )()
        await bot._cmd_dump(update_dump, ctx)
        query_dump = DummyQuery("dump_days_7")
        update_dump_cb = type(
            "Update", (), {"callback_query": query_dump, "effective_chat": chat}
        )()
        await bot._cb_dump_days(update_dump_cb, ctx)

        update_index = type(
            "Update", (), {"message": DummyMessage(), "effective_chat": chat}
        )()
        await bot._cmd_index(update_index, ctx)
        query_index = DummyQuery("index_days_30")
        update_index_cb = type(
            "Update", (), {"callback_query": query_index, "effective_chat": chat}
        )()
        await bot._cb_index_days(update_index_cb, ctx)

        assert dump_calls == [7]
        assert index_calls == [30]
        assert 1 not in bot._dump_state
        assert 1 not in bot._index_state

    asyncio.run(main())
