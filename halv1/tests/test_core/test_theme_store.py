import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from index.raw_storage import _sanitize_component
from index.theme_store import ThemeStore, sanitize_name


def test_add_and_remove_chat(tmp_path: pathlib.Path) -> None:
    store_path = tmp_path / "themes.json"
    store = ThemeStore(str(store_path))

    store.set_theme("Work", {"chat1": "Chat 1"})
    store.add_chat_to_theme("Work", "chat2", "Chat 2")
    assert store.get_chats("Work") == {"chat1": "Chat 1", "chat2": "Chat 2"}

    removed = store.remove_chat_from_theme("Work", "chat1")
    assert removed is True
    assert store.get_chats("Work") == {"chat2": "Chat 2"}

    not_removed = store.remove_chat_from_theme("Work", "missing")
    assert not_removed is False


def test_migrate_chat_names_to_sanitized(tmp_path: pathlib.Path) -> None:
    store_path = tmp_path / "themes.json"
    store = ThemeStore(str(store_path))

    raw_names = ["chat/1", " chat?2 "]
    # emulate legacy list-based storage
    store._themes[sanitize_name("Work")] = raw_names  # type: ignore[assignment]

    store.migrate_chat_names_to_sanitized()
    expected = {_sanitize_component(n): n.strip() for n in raw_names}
    assert store.get_chats("Work") == expected
