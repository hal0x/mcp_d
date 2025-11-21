"""Template manager for quality analyzer prompts."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """Loads prompt templates from disk and provides formatted strings."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.cache: dict[str, str] = {}

    def load(self, name: str) -> str:
        if name in self.cache:
            return self.cache[name]

        possible_extensions = [".txt", ".md", ".jinja", ""]
        errors = []

        for ext in possible_extensions:
            path = self.base_dir / f"{name}{ext}"
            if not path.exists():
                continue
            try:
                content = path.read_text(encoding="utf-8")
                self.cache[name] = content
                return content
            except Exception as exc:  # pragma: no cover - редкие ошибки чтения
                errors.append((path, exc))

        if errors:
            for path, exc in errors:
                logger.warning(
                    "Не удалось прочитать шаблон %s (%s): %s", name, path, exc
                )
        else:
            logger.warning(
                "Промпт %s не найден (%s + %s)",
                name,
                self.base_dir,
                possible_extensions,
            )

        default = ""
        self.cache[name] = default
        return default

    def format(self, name: str, **kwargs) -> str:
        template = self.load(name)
        return template.format(**kwargs)


DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "templates" / "prompts"
