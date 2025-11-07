"""SQLite models and migrations for memory storage."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

__all__ = ["MemoryItem", "init_db", "apply_migrations"]


@dataclass
class MemoryItem:
    """Simple representation of a memory entry.

    The model stores the raw ``content`` alongside its vector ``embedding`` and
    a collection of optional metadata fields used throughout the memory
    subsystem:

    - ``timestamp``: when the memory was recorded.
    - ``modality``: origin of the data, e.g. ``text`` or ``image``.
    - ``entities`` / ``topics``: extracted tags linked via graph edges.
    - ``importance`` and ``recall_score``: numeric signals for scheduling.
    - ``schema_id``: foreign key to an entry in the ``schemas`` table.
    - ``frozen``: whether the memory is protected from pruning.
    - ``source``: identifier of the memory's origin.
    """

    content: str
    embedding: Sequence[float]
    timestamp: float | int | None = None
    modality: str | None = None
    entities: list[str] | None = None
    topics: list[str] | None = None
    importance: float | None = None
    recall_score: float | None = None
    schema_id: int | None = None
    frozen: bool = False
    source: str | None = None


MIGRATIONS: list[Path] = [
    Path(__file__).with_name("migrations").joinpath("0001_initial.sql"),
]


def apply_migrations(
    conn: sqlite3.Connection,
    migrations: Iterable[Path] = MIGRATIONS,
    *,
    enable_hnsw: bool = True,
) -> None:
    """Apply SQL migration files to ``conn``."""

    cur = conn.cursor()
    for path in migrations:
        script = path.read_text(encoding="utf-8")
        if not enable_hnsw:
            script = script.replace(
                "CREATE VIRTUAL TABLE IF NOT EXISTS items_hnsw USING hnsw(\n    embedding FLOAT16[384]\n);",
                "CREATE TABLE IF NOT EXISTS items_hnsw (\n    rowid INTEGER PRIMARY KEY,\n    embedding BLOB\n);",
            )
        cur.executescript(script)
    conn.commit()


def init_db(path: str | Path) -> sqlite3.Connection:
    """Return a connection to ``path`` and ensure schema exists."""
    
    # Создаем директорию если она не существует
    import os
    dir_path = os.path.dirname(str(path))
    if dir_path:  # Только если есть директория
        os.makedirs(dir_path, exist_ok=True)
    
    # Создаем соединение с базой данных (SQLite создаст файл автоматически)
    # Разрешаем использование соединения из разных потоков, так как память
    # может обращаться к БД из планировщиков/воркеров.
    conn = sqlite3.connect(
        path,
        check_same_thread=False,
    )
    
    # Проверяем что файл является валидной базой данных
    try:
        conn.execute("SELECT 1")
    except sqlite3.DatabaseError:
        # Если файл не является базой данных, создаем новую
        conn.close()
        os.remove(path)
        conn = sqlite3.connect(
            path,
            check_same_thread=False,
        )
    # Базовые PRAGMA для более дружелюбной конкуррентности
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        # Если какие-то PRAGMA недоступны — тихо игнорируем
        pass
    try:
        conn.enable_load_extension(True)
    except AttributeError:  # pragma: no cover - depends on SQLite build
        pass
    hnsw_ok = True
    try:
        conn.load_extension("hnsw")
    except Exception:  # pragma: no cover - optional extension
        hnsw_ok = False
    apply_migrations(conn, enable_hnsw=hnsw_ok)
    return conn
