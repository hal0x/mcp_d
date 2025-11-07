from __future__ import annotations

from typing import List


def split_long(text: str, max_len: int = 3500) -> List[str]:
    """Split *text* into chunks not exceeding *max_len* characters.

    The function tries to split on newline boundaries to avoid breaking
    lines in the middle.  If no newline is found within the limit, the text
    is split strictly at ``max_len``.
    """
    if not text:
        return []
    if len(text) <= max_len:
        return [text]
    
    chunks: List[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        
        # Ищем лучшую точку разбивки
        split_pos = remaining.rfind("\n", 0, max_len + 1)
        if split_pos <= 0:
            split_pos = remaining.rfind(" ", 0, max_len + 1)
        if split_pos <= 0:
            split_pos = remaining.rfind(".", 0, max_len + 1)
        if split_pos <= 0:
            split_pos = remaining.rfind(",", 0, max_len + 1)
        if split_pos <= 0:
            split_pos = remaining.rfind(";", 0, max_len + 1)
        if split_pos <= 0:
            split_pos = remaining.rfind(":", 0, max_len + 1)
        if split_pos <= 0:
            # Если ничего не найдено, принудительно разбиваем
            split_pos = max_len
        
        chunk = remaining[:split_pos].strip()
        if chunk:  # Добавляем только непустые части
            chunks.append(chunk)
        
        remaining = remaining[split_pos:].strip()
    
    return chunks


def format_seconds(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS`` or ``MM:SS``."""
    if seconds < 0:
        seconds = 0
    total = int(seconds)
    mins, sec = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return f"{hrs:d}:{mins:02d}:{sec:02d}"
    return f"{mins:d}:{sec:02d}"
