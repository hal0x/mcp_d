"""Raw message storage to JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _sanitize_component(name: str) -> str:
    """Return a filesystem-safe single path component derived from ``name``.

    Replaces unsafe characters with ``_`` and avoids absolute/parent paths.
    """
    name = name.strip().replace("\\", "/")
    # Keep alphanumerics and a small safe set, replace others
    safe = "".join(ch if ch.isalnum() or ch in " _-." else "_" for ch in name)
    # Collapse slashes to underscores to avoid nested/absolute paths
    safe = safe.replace("/", "_")
    if not safe or safe == ".":
        safe = "unknown"
    if safe.startswith("."):
        safe = "_" + safe.lstrip(".")
    return safe


class RawStorage:
    """Persist raw Telegram messages into daily JSON files."""

    def __init__(self, base_path: str = "db/raw") -> None:
        self.base_path = Path(base_path)

    def save(self, chat_name: str, message: Dict[str, Any]) -> Path:
        """Append *message* to the file for ``chat_name`` and return the path."""
        date = message.get("date", "unknown")[:10]
        safe_chat = _sanitize_component(str(chat_name))
        file_path = self.base_path / safe_chat / f"{date}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("a", encoding="utf-8") as f:
            json.dump(message, f, ensure_ascii=False)
            f.write("\n")
    def trim_old_messages(self, chat_name: str, max_messages: int = 2000) -> int:
        """Удаляет старые сообщения из чата, оставляя только последние max_messages.
        
        Returns:
            Количество удаленных сообщений
        """
        safe_chat = _sanitize_component(str(chat_name))
        chat_dir = self.base_path / safe_chat
        
        if not chat_dir.exists():
            return 0
        
        # Собираем все файлы сообщений и сортируем по дате
        message_files = []
        for file_path in chat_dir.glob("*.json"):
            try:
                # Читаем количество сообщений в файле
                with file_path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                    message_files.append((file_path, len(lines)))
            except Exception:
                continue
        
        if not message_files:
            return 0
        
        # Сортируем по дате файла (старые первыми)
        message_files.sort(key=lambda x: x[0].name)
        
        total_messages = sum(count for _, count in message_files)
        if total_messages <= max_messages:
            return 0
        
        messages_to_remove = total_messages - max_messages
        removed_count = 0
        
        # Удаляем сообщения из старых файлов
        for file_path, count in message_files:
            if removed_count >= messages_to_remove:
                break
            
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                if removed_count + count <= messages_to_remove:
                    # Удаляем весь файл
                    file_path.unlink()
                    removed_count += count
                else:
                    # Удаляем часть сообщений из файла
                    lines_to_keep = count - (messages_to_remove - removed_count)
                    with file_path.open("w", encoding="utf-8") as f:
                        f.writelines(lines[-lines_to_keep:])
                    removed_count += count - lines_to_keep
                    
            except Exception:
                continue
        
        return removed_count
