import asyncio

from services.telethon_service import TelethonService


class DummyIndexer:
    def __init__(self):
        self.calls = 0

    async def list_dialogs(self):
        self.calls += 1
        return [("chat1", 1), ("chat2", 2)]

    async def ensure_connected(self):
        pass


class DummyRaw:
    pass


class DummyVector:
    pass


class DummyCluster:
    pass


class DummyThemeStore:
    pass


def test_list_chats_caches_and_refreshes(tmp_path):
    async def run() -> None:
        indexer = DummyIndexer()
        service = TelethonService(
            indexer,
            DummyRaw(),
            DummyVector(),
            DummyCluster(),
            DummyThemeStore(),
            tmp_path / "state.txt",
            lambda: "default",
        )
        chats1 = await service.list_chats()
        chats2 = await service.list_chats()
        assert chats1 == ["chat1", "chat2"]
        assert chats2 == chats1
        assert indexer.calls == 1
        await service.list_chats(force_refresh=True)
        assert indexer.calls == 2

    asyncio.run(run())
