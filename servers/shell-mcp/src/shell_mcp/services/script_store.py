from __future__ import annotations

import builtins
import json
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path

from ..config import get_settings

_EXTENSIONS = {
    "python": ".py",
    "bash": ".sh",
    "sh": ".sh",
    "shell": ".sh",
    "node": ".mjs",
}


class ScriptStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.root = Path(settings.SAVED_SCRIPTS_DIR).expanduser()
        self.temp_root = Path(settings.TEMP_SCRIPTS_DIR).expanduser()
        self.temp_max_age_days = settings.TEMP_SCRIPT_MAX_AGE_DAYS
        
        # Создаем директории
        self.root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.root / "index.json"
        self.temp_index_path = self.temp_root / "index.json"
        self._lock = threading.Lock()
        
        if not self.index_path.exists():
            self._write_index({})
        if not self.temp_index_path.exists():
            self._write_temp_index({})

    def save(self, name: str, language: str, code: str) -> dict[str, str]:
        slug = self._slugify(name)
        with self._lock:
            index = self._read_index()
            slug = self._resolve_slug_collision(slug, name, index)
            ext = _EXTENSIONS.get(language.lower(), ".txt")
            script_path = self.root / f"{slug}{ext}"
            script_path.write_text(code, encoding="utf-8")

            meta = {
                "name": name,
                "slug": slug,
                "language": language,
                "path": str(script_path),
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            index[slug] = meta
            self._write_index(index)
            return meta

    def save_temp(self, name: str, language: str, code: str) -> dict[str, str]:
        """Сохраняет скрипт как временный."""
        slug = self._slugify(name)
        with self._lock:
            temp_index = self._read_temp_index()
            slug = self._resolve_slug_collision(slug, name, temp_index)
            ext = _EXTENSIONS.get(language.lower(), ".txt")
            script_path = self.temp_root / f"{slug}{ext}"
            script_path.write_text(code, encoding="utf-8")

            meta = {
                "name": name,
                "slug": slug,
                "language": language,
                "path": str(script_path),
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "is_temp": True,
            }
            temp_index[slug] = meta
            self._write_temp_index(temp_index)
            return meta

    def promote_temp_to_permanent(self, slug: str) -> dict[str, str]:
        """Переводит временный скрипт в постоянный."""
        with self._lock:
            temp_index = self._read_temp_index()
            permanent_index = self._read_index()
            
            if slug not in temp_index:
                raise KeyError(f"Temporary script '{slug}' not found")
            
            temp_meta = temp_index[slug]
            
            # Создаем постоянный скрипт
            ext = _EXTENSIONS.get(temp_meta["language"].lower(), ".txt")
            permanent_path = self.root / f"{slug}{ext}"
            
            # Копируем содержимое файла
            temp_path = Path(temp_meta["path"])
            permanent_path.write_text(temp_path.read_text(encoding="utf-8"), encoding="utf-8")
            
            # Создаем метаданные для постоянного скрипта
            permanent_meta = {
                "name": temp_meta["name"],
                "slug": slug,
                "language": temp_meta["language"],
                "path": str(permanent_path),
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "promoted_from_temp": True,
            }
            
            # Добавляем в постоянный индекс
            permanent_index[slug] = permanent_meta
            self._write_index(permanent_index)
            
            # Удаляем из временного индекса и файл
            del temp_index[slug]
            self._write_temp_index(temp_index)
            temp_path.unlink(missing_ok=True)
            
            return permanent_meta

    def cleanup_old_temp_scripts(self) -> list[dict[str, str]]:
        """Очищает старые временные скрипты (старше TEMP_SCRIPT_MAX_AGE_DAYS дней)."""
        cleaned_scripts = []
        cutoff_date = datetime.utcnow() - timedelta(days=self.temp_max_age_days)
        
        with self._lock:
            temp_index = self._read_temp_index()
            scripts_to_remove = []
            
            for slug, meta in temp_index.items():
                created_at = datetime.fromisoformat(meta["created_at"].replace("Z", "+00:00"))
                if created_at < cutoff_date:
                    scripts_to_remove.append(slug)
                    cleaned_scripts.append(meta)
            
            # Удаляем старые скрипты
            for slug in scripts_to_remove:
                meta = temp_index[slug]
                script_path = Path(meta["path"])
                script_path.unlink(missing_ok=True)
                del temp_index[slug]
            
            if scripts_to_remove:
                self._write_temp_index(temp_index)
        
        return cleaned_scripts

    def list_temp(self) -> builtins.list[dict[str, str]]:
        """Возвращает список временных скриптов."""
        temp_index = self._read_temp_index()
        return sorted(
            temp_index.values(), key=lambda item: item.get("created_at", ""), reverse=True
        )

    def list(self) -> builtins.list[dict[str, str]]:
        index = self._read_index()
        return sorted(
            index.values(), key=lambda item: item.get("updated_at", ""), reverse=True
        )

    def get(self, name_or_slug: str) -> dict[str, str]:
        index = self._read_index()
        key = name_or_slug
        if key in index:
            return index[key]
        lower = name_or_slug.lower()
        for meta in index.values():
            if meta.get("name", "").lower() == lower:
                return meta
        raise KeyError(f"Saved script not found: {name_or_slug}")

    def delete(self, name_or_slug: str) -> dict[str, str]:
        """Удаляет сохраненный скрипт по имени или слагу."""
        with self._lock:
            index = self._read_index()

            # Находим ключ в индексе
            key = name_or_slug
            if key not in index:
                # Пробуем найти по имени
                lower = name_or_slug.lower()
                for slug, meta in index.items():
                    if meta.get("name", "").lower() == lower:
                        key = slug
                        break
                else:
                    raise KeyError(f"Saved script not found: {name_or_slug}")

            meta = index[key]
            script_path = Path(meta.get("path", ""))

            # Удаляем файл скрипта
            if script_path.exists():
                script_path.unlink()

            # Удаляем запись из индекса
            del index[key]
            self._write_index(index)

            return meta

    def _read_index(self) -> dict[str, dict[str, str]]:
        if not self.index_path.exists():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_index(self, data: dict[str, dict[str, str]]) -> None:
        self.index_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _read_temp_index(self) -> dict[str, dict[str, str]]:
        """Читает индекс временных скриптов."""
        if not self.temp_index_path.exists():
            return {}
        try:
            data = json.loads(self.temp_index_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_temp_index(self, data: dict[str, dict[str, str]]) -> None:
        """Записывает индекс временных скриптов."""
        self.temp_index_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def index_existing_files(self) -> list[dict[str, str]]:
        """Индексирует существующие файлы в папке скриптов, которые не зарегистрированы в индексе."""
        indexed_files = []
        
        with self._lock:
            index = self._read_index()
            existing_slugs = set(index.keys())
            
            # Сканируем все файлы в папке
            for file_path in self.root.iterdir():
                if file_path.is_file() and file_path.name != "index.json":
                    # Определяем язык по расширению
                    language = self._detect_language_by_extension(file_path.suffix)
                    if language:
                        # Создаем слаг из имени файла (без расширения)
                        slug = self._slugify(file_path.stem)
                        
                        # Если файл еще не зарегистрирован, добавляем его
                        if slug not in existing_slugs:
                            meta = {
                                "name": file_path.stem,
                                "slug": slug,
                                "language": language,
                                "path": str(file_path),
                                "updated_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() + "Z",
                            }
                            index[slug] = meta
                            indexed_files.append(meta)
                            existing_slugs.add(slug)
            
            # Сохраняем обновленный индекс
            if indexed_files:
                self._write_index(index)
        
        return indexed_files

    def _detect_language_by_extension(self, extension: str) -> str | None:
        """Определяет язык программирования по расширению файла."""
        extension_lower = extension.lower()
        for language, ext in _EXTENSIONS.items():
            if extension_lower == ext:
                return language
        return None

    @staticmethod
    def _slugify(name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")
        if not slug:
            slug = "script"
        return slug.lower()

    @staticmethod
    def _resolve_slug_collision(
        slug: str, name: str, index: dict[str, dict[str, str]]
    ) -> str:
        if slug not in index:
            return slug
        # If the same name already exists, overwrite existing entry
        for key, meta in index.items():
            if meta.get("name") == name:
                return key
        counter = 2
        candidate = f"{slug}-{counter}"
        while candidate in index:
            counter += 1
            candidate = f"{slug}-{counter}"
        return candidate
