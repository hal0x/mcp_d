"""Модуль для чтения и поиска по артифактам (markdown/JSON файлы)."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class ArtifactMetadata:
    """Метаданные артифакта."""

    path: str
    artifact_type: str  # chat_context, now_summary, report, aggregation_state
    file_name: str
    modified_time: datetime
    size: int
    content_preview: Optional[str] = None  # Первые 200 символов


@dataclass
class ArtifactSearchResult:
    """Результат поиска по артифакту."""

    artifact_path: str
    artifact_type: str
    score: float
    content: str
    snippet: str  # Фрагмент с подсветкой совпадений
    metadata: Dict[str, Any]
    line_number: Optional[int] = None  # Номер строки, если найдено в markdown


class ArtifactsReader:
    """Класс для чтения и поиска по артифактам."""

    def __init__(self, artifacts_dir: Path | str = "artifacts"):
        """
        Инициализация читателя артифактов.

        Args:
            artifacts_dir: Путь к директории с артифактами
        """
        self.artifacts_dir = Path(artifacts_dir)
        self._cache: Dict[str, Tuple[str, datetime]] = {}  # path -> (content, mtime)
        self._metadata_cache: Dict[str, ArtifactMetadata] = {}

    def scan_artifacts_directory(self) -> List[ArtifactMetadata]:
        """
        Сканирование директории артифактов и индексация файлов.

        Returns:
            Список метаданных найденных артифактов
        """
        artifacts: List[ArtifactMetadata] = []

        if not self.artifacts_dir.exists():
            logger.warning(f"Директория артифактов не найдена: {self.artifacts_dir}")
            return artifacts

        # Сканируем поддиректории
        patterns = [
            ("chat_contexts", "*.md", "chat_context"),
            ("now_summaries", "*.md", "now_summary"),
            ("reports", "*.md", "report"),
            ("reports", "*.json", "report"),
            ("smart_aggregation_state", "*.json", "aggregation_state"),
        ]

        for subdir, pattern, artifact_type in patterns:
            subdir_path = self.artifacts_dir / subdir
            if not subdir_path.exists():
                continue

            for file_path in subdir_path.rglob(pattern):
                try:
                    metadata = self._get_file_metadata(file_path, artifact_type)
                    if metadata:
                        artifacts.append(metadata)
                        self._metadata_cache[str(file_path)] = metadata
                except Exception as e:
                    logger.warning(
                        f"Ошибка при обработке файла {file_path}: {e}",
                        exc_info=True,
                    )

        logger.info(f"Найдено {len(artifacts)} артифактов в {self.artifacts_dir}")
        return artifacts

    def _get_file_metadata(
        self, file_path: Path, artifact_type: str
    ) -> Optional[ArtifactMetadata]:
        """
        Получение метаданных файла.

        Args:
            file_path: Путь к файлу
            artifact_type: Тип артифакта

        Returns:
            Метаданные или None при ошибке
        """
        try:
            stat = file_path.stat()
            modified_time = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            )

            # Читаем первые 200 символов для preview
            content_preview = None
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    preview = f.read(200)
                    if preview:
                        content_preview = preview.strip()
            except Exception:
                pass

            return ArtifactMetadata(
                path=str(file_path),
                artifact_type=artifact_type,
                file_name=file_path.name,
                modified_time=modified_time,
                size=stat.st_size,
                content_preview=content_preview,
            )
        except Exception as e:
            logger.warning(f"Ошибка при получении метаданных {file_path}: {e}")
            return None

    @lru_cache(maxsize=100)
    def read_artifact_file(self, file_path: str) -> Optional[str]:
        """
        Чтение содержимого артифакта с кэшированием.

        Args:
            file_path: Путь к файлу

        Returns:
            Содержимое файла или None при ошибке
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Файл не найден: {file_path}")
            return None

        try:
            # Проверяем кэш
            stat = path.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

            if file_path in self._cache:
                cached_content, cached_mtime = self._cache[file_path]
                if cached_mtime >= mtime:
                    return cached_content

            # Читаем файл
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Обновляем кэш
            self._cache[file_path] = (content, mtime)
            return content
        except Exception as e:
            logger.error(f"Ошибка при чтении файла {file_path}: {e}", exc_info=True)
            return None

    def get_artifact_metadata(self, file_path: str) -> Optional[ArtifactMetadata]:
        """
        Получение метаданных артифакта.

        Args:
            file_path: Путь к файлу

        Returns:
            Метаданные или None
        """
        if file_path in self._metadata_cache:
            return self._metadata_cache[file_path]

        path = Path(file_path)
        if not path.exists():
            return None

        # Определяем тип артифакта по пути
        artifact_type = "unknown"
        if "chat_contexts" in str(path):
            artifact_type = "chat_context"
        elif "now_summaries" in str(path):
            artifact_type = "now_summary"
        elif "reports" in str(path):
            artifact_type = "report"
        elif "smart_aggregation_state" in str(path):
            artifact_type = "aggregation_state"

        metadata = self._get_file_metadata(path, artifact_type)
        if metadata:
            self._metadata_cache[file_path] = metadata
        return metadata

    def search_artifacts(
        self,
        query: str,
        artifact_types: Optional[List[str]] = None,
        limit: int = 10,
        min_score: float = 0.1,
    ) -> List[ArtifactSearchResult]:
        """
        Полнотекстовый поиск по содержимому артифактов.

        Args:
            query: Поисковый запрос
            artifact_types: Фильтр по типам артифактов (None = все типы)
            limit: Максимальное количество результатов
            min_score: Минимальный score для включения в результаты

        Returns:
            Список результатов поиска, отсортированный по релевантности
        """
        # Сканируем директорию, если кэш пуст
        if not self._metadata_cache:
            self.scan_artifacts_directory()

        results: List[ArtifactSearchResult] = []
        query_lower = query.lower()
        query_words = set(re.findall(r"\w+", query_lower))

        # Ищем по всем артифактам
        for file_path, metadata in self._metadata_cache.items():
            # Фильтр по типу
            if artifact_types and metadata.artifact_type not in artifact_types:
                continue

            content = self.read_artifact_file(file_path)
            if not content:
                continue

            # Простой поиск по совпадению слов
            score, snippet, line_number = self._calculate_relevance(
                content, query, query_words
            )

            if score >= min_score:
                # Парсим JSON, если это JSON файл
                metadata_dict = {}
                if file_path.endswith(".json"):
                    try:
                        metadata_dict = json.loads(content)
                    except Exception:
                        pass

                results.append(
                    ArtifactSearchResult(
                        artifact_path=file_path,
                        artifact_type=metadata.artifact_type,
                        score=score,
                        content=content[:1000],  # Первые 1000 символов
                        snippet=snippet,
                        metadata={
                            "file_name": metadata.file_name,
                            "modified_time": metadata.modified_time.isoformat(),
                            "size": metadata.size,
                            **metadata_dict,
                        },
                        line_number=line_number,
                    )
                )

        # Сортируем по score (убывание)
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:limit]

    def _calculate_relevance(
        self, content: str, query: str, query_words: set[str]
    ) -> Tuple[float, str, Optional[int]]:
        """
        Вычисление релевантности контента запросу.

        Args:
            content: Содержимое файла
            query: Поисковый запрос
            query_words: Множество слов из запроса

        Returns:
            Кортеж (score, snippet, line_number)
        """
        content_lower = content.lower()
        content_words = set(re.findall(r"\w+", content_lower))

        # Простой TF-IDF-like scoring
        # 1. Количество совпадающих слов
        matches = len(query_words & content_words)
        if matches == 0:
            return (0.0, "", None)

        # 2. Плотность совпадений (сколько раз встречаются слова запроса)
        match_count = sum(content_lower.count(word) for word in query_words)
        total_words = len(content_words)

        # 3. Наличие точной фразы
        phrase_bonus = 2.0 if query_lower in content_lower else 1.0

        # Вычисляем score
        word_match_score = matches / len(query_words) if query_words else 0
        density_score = min(match_count / max(total_words, 1), 1.0)
        score = (word_match_score * 0.6 + density_score * 0.4) * phrase_bonus

        # Генерируем snippet
        snippet = self._generate_snippet(content, query, query_words)
        line_number = self._find_line_number(content, query)

        return (min(score, 1.0), snippet, line_number)

    def _generate_snippet(
        self, content: str, query: str, query_words: set[str], max_length: int = 200
    ) -> str:
        """
        Генерация snippet с подсветкой совпадений.

        Args:
            content: Содержимое файла
            query: Поисковый запрос
            query_words: Множество слов из запроса
            max_length: Максимальная длина snippet

        Returns:
            Snippet с совпадениями
        """
        content_lower = content.lower()
        query_lower = query.lower()

        # Ищем первое вхождение запроса или слов
        best_pos = -1
        best_length = 0

        # Сначала ищем точную фразу
        if query_lower in content_lower:
            best_pos = content_lower.find(query_lower)
            best_length = len(query)
        else:
            # Ищем первое вхождение любого слова
            for word in query_words:
                pos = content_lower.find(word)
                if pos != -1 and (best_pos == -1 or pos < best_pos):
                    best_pos = pos
                    best_length = len(word)

        if best_pos == -1:
            # Если не нашли, возвращаем начало
            return content[:max_length] + "..." if len(content) > max_length else content

        # Вырезаем фрагмент вокруг совпадения
        start = max(0, best_pos - max_length // 2)
        end = min(len(content), best_pos + best_length + max_length // 2)

        snippet = content[start:end]

        # Подсвечиваем совпадения (простая замена)
        for word in query_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            snippet = pattern.sub(f"**{word}**", snippet)

        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet

    def _find_line_number(self, content: str, query: str) -> Optional[int]:
        """
        Поиск номера строки с совпадением.

        Args:
            content: Содержимое файла
            query: Поисковый запрос

        Returns:
            Номер строки или None
        """
        query_lower = query.lower()
        content_lower = content.lower()

        # Ищем первое вхождение
        pos = content_lower.find(query_lower)
        if pos == -1:
            # Ищем первое слово
            words = query_lower.split()
            if words:
                pos = content_lower.find(words[0])

        if pos == -1:
            return None

        # Считаем номер строки
        return content[:pos].count("\n") + 1

    def clear_cache(self) -> None:
        """Очистка кэша."""
        self._cache.clear()
        self._metadata_cache.clear()
        self.read_artifact_file.cache_clear()

