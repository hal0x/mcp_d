from __future__ import annotations

import asyncio
import pathlib
import sys
from typing import Iterable

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from index.insight_store import InsightStore
from index.summarizer import Summarizer
from services.chronicle_service import ChronicleService
from tasks.scheduler import TaskScheduler


class DummySummarizer(Summarizer):
    def __init__(self) -> None:
        pass

    def summarize(self, texts: Iterable[str]) -> str:
        return " | ".join(texts)


class DummySendBot:
    def __init__(self, sent: list[tuple[int, str]]) -> None:
        self._sent = sent

    async def send_message(self, chat_id: int, text: str) -> None:
        self._sent.append((chat_id, text))


class DummyApp:
    def __init__(self, sent: list[tuple[int, str]]) -> None:
        self.bot = DummySendBot(sent)


class DummyBot:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []
        self.app = DummyApp(self.sent)


def test_chronicle_service_publishes(tmp_path: pathlib.Path) -> None:
    store = InsightStore(str(tmp_path / "insights.json"))
    store.set_clusters(
        {
            "c1": {
                "centroid": [],
                "medoid": {
                    "chunk_id": "1",
                    "text": "hello",
                    "embedding": [],
                    "metadata": {},
                },
                "summary": "",
                "members": [],
                "timeline": [
                    {
                        "chunk_id": "1",
                        "text": "hello",
                        "embedding": [],
                        "metadata": {},
                        "timestamp": 0,
                    }
                ],
            }
        }
    )
    bot = DummyBot()
    scheduler = TaskScheduler()
    service = ChronicleService(
        DummySummarizer(),
        store,
        scheduler,
        lambda: bot,
        summary_chat_id=123,
        days_interval=1,
    )
    asyncio.run(service.publish_chronicle())
    assert bot.sent == [(123, "Хроника событий:\n1970-01-01T00:00:00+00:00: hello")]
