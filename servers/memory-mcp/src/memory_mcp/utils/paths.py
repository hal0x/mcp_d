"""Утилиты для работы с путями проекта.

Централизованные функции для поиска корня проекта и резолвинга путей.
"""

from __future__ import annotations

from pathlib import Path


def find_project_root(start_path: Path | None = None) -> Path:
    """
    Находит корень проекта по наличию файла pyproject.toml.

    Поднимается вверх по директориям от start_path до тех пор, пока не найдёт
    pyproject.toml. Если файл не найден, возвращает текущую рабочую директорию.

    Args:
        start_path: Начальный путь для поиска. Если None, используется Path.cwd().

    Returns:
        Path к корню проекта (директория с pyproject.toml) или Path.cwd()
        если pyproject.toml не найден.

    Examples:
        >>> from pathlib import Path
        >>> root = find_project_root(Path(__file__).parent)
        >>> assert (root / "pyproject.toml").exists()
    """
    if start_path is None:
        start_path = Path.cwd()

    # Нормализуем путь
    current_dir = Path(start_path).resolve()

    # Поднимаемся вверх по директориям
    project_root = current_dir
    while project_root.parent != project_root:
        if (project_root / "pyproject.toml").exists():
            return project_root
        project_root = project_root.parent

    # Если не нашли, возвращаем текущую рабочую директорию
    return Path.cwd()

