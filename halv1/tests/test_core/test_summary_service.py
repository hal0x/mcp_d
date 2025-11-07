import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from index.vector_index import VectorEntry
from services.summary_service import SummaryService
from tasks.scheduler import TaskScheduler


class DummyIndex:
    entries: list[Any] = []


class DummyThemeStore:
    def get_chats(self, theme: str) -> dict[str, str]:
        return {}


class DummyTelethonService:
    async def index_since(
        self, *args: Any, **kwargs: Any
    ) -> None:  # pragma: no cover - stub
        pass


def _dummy() -> None:
    return None


def test_summary_service_reads_user_and_timezone(tmp_path: Path) -> None:
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        "summary:\n  user_name: '@bot'\n  timezone: 'Europe/Moscow'\n",
        encoding="utf-8",
    )
    service = SummaryService(
        summarizer=cast(Any, object()),
        vector_index=cast(Any, DummyIndex()),
        theme_store=cast(Any, DummyThemeStore()),
        telethon_service=cast(Any, DummyTelethonService()),
        scheduler=TaskScheduler(),
        summary_interval_file=tmp_path / "interval.yaml",
        get_active_theme=lambda: "default",
        get_bot=_dummy,
        summary_chat_id=None,
        summary_interval=3600,
        user_name=None,
        timezone=None,
        config_path=cfg,
    )
    assert service.user_name == "@bot"
    assert service.timezone == "Europe/Moscow"


def test_summary_service_config_missing(tmp_path: Path) -> None:
    service = SummaryService(
        summarizer=cast(Any, object()),
        vector_index=cast(Any, DummyIndex()),
        theme_store=cast(Any, DummyThemeStore()),
        telethon_service=cast(Any, DummyTelethonService()),
        scheduler=TaskScheduler(),
        summary_interval_file=tmp_path / "interval.yaml",
        get_active_theme=lambda: "default",
        get_bot=_dummy,
        summary_chat_id=None,
        summary_interval=3600,
        user_name=None,
        timezone=None,
        config_path=tmp_path / "missing.yaml",
    )
    assert service.user_name == "@user"
    assert service.timezone == "UTC"


def test_set_summary_interval_updates_file_and_scheduler(tmp_path: Path) -> None:
    async def run() -> None:
        sched = TaskScheduler()
        interval_file = tmp_path / "interval.yaml"
        service = SummaryService(
            summarizer=cast(Any, object()),
            vector_index=cast(Any, DummyIndex()),
            theme_store=cast(Any, DummyThemeStore()),
            telethon_service=cast(Any, DummyTelethonService()),
            scheduler=sched,
            summary_interval_file=interval_file,
            get_active_theme=lambda: "default",
            get_bot=_dummy,
            summary_chat_id=None,
            summary_interval=3600,
            user_name="@user",
            timezone="UTC",
        )
        await service.start()
        old_task = service.scheduler_task
        result = await service.set_summary_interval(2)
        assert result.startswith("Интервал сводки")
        data = yaml.safe_load(interval_file.read_text(encoding="utf-8"))
        assert data["summary_interval_seconds"] == 7200
        assert service.summary_interval == 7200
        assert service.scheduler_task is not old_task
        await service.stop()

    asyncio.run(run())


def test_summarize_interval_skips_invalid_dates(tmp_path: Path) -> None:
    async def run() -> None:
        valid_iso = datetime.now(UTC).isoformat()
        index = DummyIndex()
        index.entries = [
            VectorEntry(
                "bad", "ignored", [1.0], {"date": "invalid", "theme": "default"}
            ),
            VectorEntry(
                "good",
                "used",
                [1.0],
                {"date": valid_iso, "theme": "default", "chat": "chat"},
            ),
        ]

        calls: dict[str, Any] = {}

        class DummySummarizer:
            def summarize_as_agent(
                self, **kwargs: Any
            ) -> str:  # pragma: no cover - simple
                calls.update(kwargs)
                return "summary"

        service = SummaryService(
            summarizer=cast(Any, DummySummarizer()),
            vector_index=cast(Any, index),
            theme_store=cast(Any, DummyThemeStore()),
            telethon_service=cast(Any, DummyTelethonService()),
            scheduler=TaskScheduler(),
            summary_interval_file=tmp_path / "interval.yaml",
            get_active_theme=lambda: "default",
            get_bot=_dummy,
            summary_chat_id=None,
            summary_interval=3600,
            user_name="user",
            timezone="UTC",
        )

        result = await service.summarize_interval(24)
        assert result == "summary"
        assert "ignored" not in calls.get("messages_block", "")

    asyncio.run(run())
