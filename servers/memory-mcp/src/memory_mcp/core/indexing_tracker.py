"""Модуль для отслеживания задач индексации с файловым хранилищем."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.system.paths import find_project_root

logger = logging.getLogger(__name__)


class IndexingJobTracker:
    """Класс для управления задачами индексации с файловым хранилищем."""

    def __init__(self, storage_path: str = "data/indexing_jobs.json"):
        """
        Инициализация трекера задач.

        Args:
            storage_path: Путь к JSON файлу для хранения задач
        """
        # Резолвим относительный путь
        if not os.path.isabs(storage_path):
            project_root = find_project_root(Path(__file__).parent)
            storage_path = str(project_root / storage_path)

        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _load_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Загрузить задачи из файла."""
        if not self.storage_path.exists():
            return {}

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("jobs", {})
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Ошибка при загрузке задач из {self.storage_path}: {e}")
            return {}

    def _save_jobs(self, jobs: Dict[str, Dict[str, Any]]) -> None:
        """Сохранить задачи в файл."""
        try:
            # Создаем временный файл для атомарной записи
            temp_path = self.storage_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump({"jobs": jobs}, f, indent=2, ensure_ascii=False)
            # Атомарно заменяем файл
            temp_path.replace(self.storage_path)
        except IOError as e:
            logger.error(f"Ошибка при сохранении задач в {self.storage_path}: {e}")
            raise

    def create_job(
        self,
        job_id: str,
        scope: str,
        chat: Optional[str] = None,
        chats: Optional[List[str]] = None,
        force_full: bool = False,
        recent_days: int = 7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Создать новую задачу индексации.

        Args:
            job_id: Уникальный идентификатор задачи
            scope: Область индексации ("all" или "chat")
            chat: Название чата (для scope="chat")
            chats: Список чатов (для scope="all")
            force_full: Полная переиндексация
            recent_days: Количество дней для переиндексации
            **kwargs: Дополнительные параметры

        Returns:
            Созданная задача
        """
        with self._lock:
            jobs = self._load_jobs()

            if job_id in jobs:
                logger.warning(f"Задача {job_id} уже существует, обновляем статус")
                jobs[job_id]["status"] = "started"
                jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
            else:
                job = {
                    "job_id": job_id,
                    "status": "started",
                    "scope": scope,
                    "chat": chat,
                    "chats": chats or ([] if scope == "all" else None),
                    "force_full": force_full,
                    "recent_days": recent_days,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "current_stage": "Инициализация",
                    "current_chat": None,
                    "progress": {
                        "total_chats": len(chats) if chats else (1 if chat else 0),
                        "completed_chats": 0,
                        "current_chat_sessions": 0,
                        "current_chat_messages": 0,
                    },
                    "stats": {
                        "sessions_indexed": 0,
                        "messages_indexed": 0,
                        "tasks_indexed": 0,
                    },
                    "error": None,
                    **kwargs,
                }
                jobs[job_id] = job

            self._save_jobs(jobs)
            logger.info(f"Создана задача индексации: {job_id} (scope={scope}, chat={chat})")
            return jobs[job_id]

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        current_stage: Optional[str] = None,
        current_chat: Optional[str] = None,
        progress: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Обновить задачу индексации.

        Args:
            job_id: Идентификатор задачи
            status: Новый статус
            current_stage: Текущий этап
            current_chat: Текущий обрабатываемый чат
            progress: Прогресс индексации
            stats: Статистика
            error: Сообщение об ошибке
            **kwargs: Дополнительные поля для обновления

        Returns:
            Обновленная задача или None, если задача не найдена
        """
        with self._lock:
            jobs = self._load_jobs()

            if job_id not in jobs:
                logger.warning(f"Задача {job_id} не найдена")
                return None

            job = jobs[job_id]

            if status is not None:
                job["status"] = status
                if status == "completed":
                    job["completed_at"] = datetime.now(timezone.utc).isoformat()
                elif status == "failed":
                    job["failed_at"] = datetime.now(timezone.utc).isoformat()

            if current_stage is not None:
                job["current_stage"] = current_stage

            if current_chat is not None:
                job["current_chat"] = current_chat

            if progress is not None:
                job["progress"].update(progress)

            if stats is not None:
                job["stats"].update(stats)

            if error is not None:
                job["error"] = error

            # Обновляем дополнительные поля
            for key, value in kwargs.items():
                job[key] = value

            self._save_jobs(jobs)
            return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Получить задачу по идентификатору.

        Args:
            job_id: Идентификатор задачи

        Returns:
            Задача или None, если не найдена
        """
        with self._lock:
            jobs = self._load_jobs()
            return jobs.get(job_id)

    def get_all_jobs(
        self, status: Optional[str] = None, chat: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Получить все задачи с опциональной фильтрацией.

        Args:
            status: Фильтр по статусу
            chat: Фильтр по чату

        Returns:
            Список задач
        """
        with self._lock:
            jobs = self._load_jobs()
            result = list(jobs.values())

            if status is not None:
                result = [job for job in result if job.get("status") == status]

            if chat is not None:
                result = [
                    job
                    for job in result
                    if job.get("chat") == chat
                    or chat in (job.get("chats") or [])
                    or job.get("current_chat") == chat
                ]

            return result

    def delete_job(self, job_id: str) -> bool:
        """
        Удалить задачу.

        Args:
            job_id: Идентификатор задачи

        Returns:
            True, если задача была удалена, False если не найдена
        """
        with self._lock:
            jobs = self._load_jobs()

            if job_id not in jobs:
                return False

            del jobs[job_id]
            self._save_jobs(jobs)
            logger.info(f"Удалена задача индексации: {job_id}")
            return True

    def cleanup_old_jobs(self, days: int = 7) -> int:
        """
        Удалить старые завершенные задачи.

        Args:
            days: Количество дней для хранения завершенных задач

        Returns:
            Количество удаленных задач
        """
        with self._lock:
            jobs = self._load_jobs()
            cutoff = datetime.now(timezone.utc).timestamp() - (days * 24 * 60 * 60)
            deleted = 0

            for job_id, job in list(jobs.items()):
                status = job.get("status")
                if status in ("completed", "failed"):
                    completed_at = job.get("completed_at") or job.get("failed_at")
                    if completed_at:
                        try:
                            completed_ts = datetime.fromisoformat(
                                completed_at.replace("Z", "+00:00")
                            ).timestamp()
                            if completed_ts < cutoff:
                                del jobs[job_id]
                                deleted += 1
                        except (ValueError, AttributeError):
                            pass

            if deleted > 0:
                self._save_jobs(jobs)
                logger.info(f"Удалено {deleted} старых задач индексации")

            return deleted

    def cleanup_stale_running_jobs(self, max_age_hours: int = 2) -> int:
        """
        Завершить зависшие задачи со статусом "running", которые не обновлялись долгое время.

        Args:
            max_age_hours: Максимальный возраст задачи в часах перед пометкой как зависшей

        Returns:
            Количество завершенных задач
        """
        with self._lock:
            jobs = self._load_jobs()
            cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 60 * 60)
            completed = 0

            for job_id, job in list(jobs.items()):
                status = job.get("status")
                if status == "running":
                    started_at = job.get("started_at")
                    if started_at:
                        try:
                            started_ts = datetime.fromisoformat(
                                started_at.replace("Z", "+00:00")
                            ).timestamp()
                            if started_ts < cutoff:
                                # Помечаем задачу как завершенную с ошибкой
                                job["status"] = "failed"
                                job["failed_at"] = datetime.now(timezone.utc).isoformat()
                                job["error"] = f"Задача зависла (не обновлялась более {max_age_hours} часов)"
                                job["current_stage"] = "Завершена автоматически (зависла)"
                                completed += 1
                                logger.warning(
                                    f"Завершена зависшая задача {job_id} (начата {started_at})"
                                )
                        except (ValueError, AttributeError):
                            # Если не удалось распарсить дату, помечаем как зависшую
                            job["status"] = "failed"
                            job["failed_at"] = datetime.now(timezone.utc).isoformat()
                            job["error"] = "Задача зависла (некорректная дата начала)"
                            job["current_stage"] = "Завершена автоматически (зависла)"
                            completed += 1
                            logger.warning(f"Завершена зависшая задача {job_id} (некорректная дата)")

            if completed > 0:
                self._save_jobs(jobs)
                logger.info(f"Завершено {completed} зависших задач индексации")

            return completed

