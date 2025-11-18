"""Хранилище сессий поиска для интерактивного поиска."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SearchSessionStore:
    """Хранилище истории поисковых сессий и обратной связи."""

    def __init__(self, db_path: Path | str = "data/search_sessions.db"):
        """
        Инициализация хранилища сессий.

        Args:
            db_path: Путь к SQLite базе данных
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Инициализация схемы базы данных."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS search_sessions (
                        session_id TEXT PRIMARY KEY,
                        original_query TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS search_queries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        query_text TEXT NOT NULL,
                        query_type TEXT NOT NULL,  -- 'original', 'refined', 'clarified'
                        created_at TIMESTAMP NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES search_sessions(session_id)
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS search_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        query_id INTEGER,
                        result_id TEXT NOT NULL,
                        artifact_path TEXT,
                        result_type TEXT NOT NULL,  -- 'db_record', 'artifact'
                        score REAL NOT NULL,
                        content_preview TEXT,
                        created_at TIMESTAMP NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES search_sessions(session_id),
                        FOREIGN KEY (query_id) REFERENCES search_queries(id)
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS search_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        result_id TEXT NOT NULL,
                        artifact_path TEXT,
                        relevance TEXT NOT NULL,  -- 'relevant', 'irrelevant', 'partially_relevant'
                        comment TEXT,
                        created_at TIMESTAMP NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES search_sessions(session_id)
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_created ON search_sessions(created_at)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_queries_session ON search_queries(session_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_results_session ON search_results(session_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feedback_session ON search_feedback(session_id)
                """)

                conn.commit()
        except Exception as e:
            logger.error(f"Ошибка при инициализации БД сессий: {e}", exc_info=True)
            raise

    def create_session(self, original_query: str) -> str:
        """
        Создание новой поисковой сессии.

        Args:
            original_query: Исходный поисковый запрос

        Returns:
            ID созданной сессии
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO search_sessions (session_id, original_query, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, original_query, now, now),
                )

                conn.execute(
                    """
                    INSERT INTO search_queries (session_id, query_text, query_type, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, original_query, "original", now),
                )

                conn.commit()
        except Exception as e:
            logger.error(f"Ошибка при создании сессии: {e}", exc_info=True)
            raise

        return session_id

    def add_refined_query(self, session_id: str, refined_query: str) -> int:
        """
        Добавление уточненного запроса к сессии.

        Args:
            session_id: ID сессии
            refined_query: Уточненный запрос

        Returns:
            ID созданного запроса
        """
        now = datetime.now(timezone.utc)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO search_queries (session_id, query_text, query_type, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, refined_query, "refined", now),
                )

                query_id = cursor.lastrowid

                # Обновляем updated_at сессии
                conn.execute(
                    """
                    UPDATE search_sessions SET updated_at = ? WHERE session_id = ?
                    """,
                    (now, session_id),
                )

                conn.commit()
                return query_id
        except Exception as e:
            logger.error(f"Ошибка при добавлении уточненного запроса: {e}", exc_info=True)
            raise

    def save_results(
        self,
        session_id: str,
        query_id: Optional[int],
        results: List[Dict[str, Any]],
    ) -> None:
        """
        Сохранение результатов поиска.

        Args:
            session_id: ID сессии
            query_id: ID запроса (опционально)
            results: Список результатов поиска
        """
        now = datetime.now(timezone.utc)

        try:
            with sqlite3.connect(self.db_path) as conn:
                for result in results:
                    result_id = result.get("record_id") or result.get("artifact_path", "")
                    artifact_path = result.get("artifact_path")
                    result_type = "artifact" if artifact_path else "db_record"
                    score = result.get("score", 0.0)
                    content_preview = result.get("content", "")[:500]  # Первые 500 символов

                    conn.execute(
                        """
                        INSERT INTO search_results 
                        (session_id, query_id, result_id, artifact_path, result_type, score, content_preview, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            query_id,
                            result_id,
                            artifact_path,
                            result_type,
                            score,
                            content_preview,
                            now,
                        ),
                    )

                # Обновляем updated_at сессии
                conn.execute(
                    """
                    UPDATE search_sessions SET updated_at = ? WHERE session_id = ?
                    """,
                    (now, session_id),
                )

                conn.commit()
        except Exception as e:
            logger.error(f"Ошибка при сохранении результатов: {e}", exc_info=True)
            raise

    def add_feedback(
        self,
        session_id: str,
        result_id: str,
        relevance: str,
        artifact_path: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> None:
        """
        Добавление обратной связи по результату.

        Args:
            session_id: ID сессии
            result_id: ID результата
            relevance: Релевантность ('relevant', 'irrelevant', 'partially_relevant')
            artifact_path: Путь к артифакту (если применимо)
            comment: Комментарий пользователя
        """
        now = datetime.now(timezone.utc)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO search_feedback 
                    (session_id, result_id, artifact_path, relevance, comment, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, result_id, artifact_path, relevance, comment, now),
                )

                # Обновляем updated_at сессии
                conn.execute(
                    """
                    UPDATE search_sessions SET updated_at = ? WHERE session_id = ?
                    """,
                    (now, session_id),
                )

                conn.commit()
        except Exception as e:
            logger.error(f"Ошибка при добавлении обратной связи: {e}", exc_info=True)
            raise

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации о сессии.

        Args:
            session_id: ID сессии

        Returns:
            Словарь с информацией о сессии или None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM search_sessions WHERE session_id = ?
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None

                session = dict(row)

                # Получаем запросы
                cursor = conn.execute(
                    """
                    SELECT * FROM search_queries WHERE session_id = ? ORDER BY created_at
                    """,
                    (session_id,),
                )
                session["queries"] = [dict(r) for r in cursor.fetchall()]

                # Получаем результаты
                cursor = conn.execute(
                    """
                    SELECT * FROM search_results WHERE session_id = ? ORDER BY score DESC
                    """,
                    (session_id,),
                )
                session["results"] = [dict(r) for r in cursor.fetchall()]

                # Получаем обратную связь
                cursor = conn.execute(
                    """
                    SELECT * FROM search_feedback WHERE session_id = ? ORDER BY created_at
                    """,
                    (session_id,),
                )
                session["feedback"] = [dict(r) for r in cursor.fetchall()]

                return session
        except Exception as e:
            logger.error(f"Ошибка при получении сессии: {e}", exc_info=True)
            return None

    def get_feedback_for_learning(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Получение обратной связи для обучения модели.

        Args:
            limit: Максимальное количество записей

        Returns:
            Список записей обратной связи
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT f.*, s.original_query
                    FROM search_feedback f
                    JOIN search_sessions s ON f.session_id = s.session_id
                    ORDER BY f.created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                return [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Ошибка при получении обратной связи: {e}", exc_info=True)
            return []

