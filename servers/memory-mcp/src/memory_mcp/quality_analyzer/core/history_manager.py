#!/usr/bin/env python3
"""
Менеджер истории анализов качества

Обеспечивает иерархическое хранение и управление историей анализов:
- Сохранение результатов анализа по чатам и датам
- Сравнение анализов между собой
- Получение истории для анализа трендов
- Управление версиями анализов
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HistoryManager:
    """Менеджер истории анализов качества"""

    def __init__(self, history_dir: Path = Path("quality_analysis_history")):
        """
        Инициализация менеджера истории

        Args:
            history_dir: Директория для хранения истории
        """
        self.history_dir = history_dir

        # Создаем структуру директорий
        self.chats_dir = self.history_dir / "chats"
        self.overall_dir = self.history_dir / "overall"
        self.metadata_file = self.history_dir / "metadata.json"

        # Создаем директории если не существуют
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.chats_dir.mkdir(parents=True, exist_ok=True)
        self.overall_dir.mkdir(parents=True, exist_ok=True)

        # Загружаем метаданные
        self.metadata = self._load_metadata()

        logger.info(f"Инициализирован HistoryManager (директория: {self.history_dir})")

    def save_analysis(self, analysis_record: Dict[str, Any]) -> str:
        """
        Сохранение анализа конкретного чата

        Args:
            analysis_record: Запись анализа

        Returns:
            ID сохраненного анализа
        """
        chat_name = analysis_record.get("chat_name", "unknown")
        timestamp = analysis_record.get("timestamp", datetime.now().isoformat())

        # Создаем ID анализа
        analysis_id = f"{chat_name}_{timestamp.replace(':', '-').replace('.', '-')}"

        # Создаем директорию для чата если не существует
        chat_dir = self.chats_dir / chat_name
        chat_dir.mkdir(parents=True, exist_ok=True)

        # Сохраняем файл анализа
        analysis_file = chat_dir / f"{analysis_id}.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis_record, f, ensure_ascii=False, indent=2)

        # Обновляем метаданные
        self._update_chat_metadata(chat_name, analysis_id, analysis_record)

        logger.info(f"Анализ сохранен: {analysis_file}")
        return analysis_id

    def save_overall_analysis(self, overall_record: Dict[str, Any]) -> str:
        """
        Сохранение общего анализа

        Args:
            overall_record: Запись общего анализа

        Returns:
            ID сохраненного анализа
        """
        timestamp = overall_record.get("timestamp", datetime.now().isoformat())

        # Создаем ID анализа
        analysis_id = f"overall_{timestamp.replace(':', '-').replace('.', '-')}"

        # Сохраняем файл анализа
        analysis_file = self.overall_dir / f"{analysis_id}.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(overall_record, f, ensure_ascii=False, indent=2)

        # Обновляем метаданные
        self._update_overall_metadata(analysis_id, overall_record)

        logger.info(f"Общий анализ сохранен: {analysis_file}")
        return analysis_id

    def get_history(
        self,
        chat_name: Optional[str] = None,
        limit: int = 10,
        days_back: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Получение истории анализов

        Args:
            chat_name: Фильтр по чату (если None - все чаты)
            limit: Максимальное количество записей
            days_back: Количество дней назад для фильтрации

        Returns:
            Список записей истории
        """
        logger.info(f"Получение истории (чат: {chat_name}, лимит: {limit})")

        if chat_name:
            return self._get_chat_history(chat_name, limit, days_back)
        else:
            return self._get_all_history(limit, days_back)

    def _get_chat_history(
        self,
        chat_name: str,
        limit: int,
        days_back: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Получение истории конкретного чата"""
        chat_dir = self.chats_dir / chat_name

        if not chat_dir.exists():
            logger.warning(f"Директория чата не найдена: {chat_dir}")
            return []

        # Получаем все файлы анализов
        analysis_files = list(chat_dir.glob("*.json"))

        # Фильтруем по дате если указано
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            filtered_files = []

            for file_path in analysis_files:
                try:
                    with open(file_path, encoding="utf-8") as f:
                        analysis_data = json.load(f)

                    timestamp_str = analysis_data.get("timestamp", "")
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if timestamp >= cutoff_date:
                            filtered_files.append(file_path)
                except Exception as e:
                    logger.warning(f"Ошибка чтения файла {file_path}: {e}")
                    continue

            analysis_files = filtered_files

        # Сортируем по дате (новые сначала)
        analysis_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Ограничиваем количество
        analysis_files = analysis_files[:limit]

        # Загружаем данные
        history = []
        for file_path in analysis_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    analysis_data = json.load(f)
                history.append(analysis_data)
            except Exception as e:
                logger.warning(f"Ошибка загрузки анализа {file_path}: {e}")

        logger.info(f"Загружено {len(history)} записей истории для чата {chat_name}")
        return history

    def _get_all_history(
        self,
        limit: int,
        days_back: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Получение всей истории"""
        all_history = []

        # Получаем историю всех чатов
        for chat_dir in self.chats_dir.iterdir():
            if chat_dir.is_dir():
                chat_history = self._get_chat_history(chat_dir.name, limit, days_back)
                all_history.extend(chat_history)

        # Получаем общую историю
        overall_history = self._get_overall_history(limit, days_back)
        all_history.extend(overall_history)

        # Сортируем по дате
        all_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Ограничиваем количество
        all_history = all_history[:limit]

        logger.info(f"Загружено {len(all_history)} записей общей истории")
        return all_history

    def _get_overall_history(
        self,
        limit: int,
        days_back: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Получение общей истории"""
        if not self.overall_dir.exists():
            return []

        # Получаем все файлы общих анализов
        analysis_files = list(self.overall_dir.glob("*.json"))

        # Фильтруем по дате если указано
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            filtered_files = []

            for file_path in analysis_files:
                try:
                    with open(file_path, encoding="utf-8") as f:
                        analysis_data = json.load(f)

                    timestamp_str = analysis_data.get("timestamp", "")
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if timestamp >= cutoff_date:
                            filtered_files.append(file_path)
                except Exception as e:
                    logger.warning(f"Ошибка чтения файла {file_path}: {e}")
                    continue

            analysis_files = filtered_files

        # Сортируем по дате (новые сначала)
        analysis_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Ограничиваем количество
        analysis_files = analysis_files[:limit]

        # Загружаем данные
        history = []
        for file_path in analysis_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    analysis_data = json.load(f)
                history.append(analysis_data)
            except Exception as e:
                logger.warning(f"Ошибка загрузки общего анализа {file_path}: {e}")

        return history

    def compare_analyses(self, analysis_ids: List[str]) -> Dict[str, Any]:
        """
        Сравнение нескольких анализов

        Args:
            analysis_ids: ID анализов для сравнения

        Returns:
            Результаты сравнения
        """
        logger.info(f"Сравнение анализов: {analysis_ids}")

        analyses = []

        # Загружаем анализы
        for analysis_id in analysis_ids:
            analysis = self._load_analysis_by_id(analysis_id)
            if analysis:
                analyses.append(analysis)
            else:
                logger.warning(f"Анализ не найден: {analysis_id}")

        if len(analyses) < 2:
            return {
                "error": "Недостаточно анализов для сравнения",
                "available_analyses": len(analyses),
            }

        # Выполняем сравнение
        comparison = self._perform_comparison(analyses)

        logger.info(f"Сравнение завершено для {len(analyses)} анализов")
        return comparison

    def _load_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Загрузка анализа по ID"""
        # Пробуем найти в чатах
        for chat_dir in self.chats_dir.iterdir():
            if chat_dir.is_dir():
                analysis_file = chat_dir / f"{analysis_id}.json"
                if analysis_file.exists():
                    try:
                        with open(analysis_file, encoding="utf-8") as f:
                            return json.load(f)
                    except Exception as e:
                        logger.warning(f"Ошибка загрузки анализа {analysis_file}: {e}")

        # Пробуем найти в общих анализах
        analysis_file = self.overall_dir / f"{analysis_id}.json"
        if analysis_file.exists():
            try:
                with open(analysis_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Ошибка загрузки общего анализа {analysis_file}: {e}")

        return None

    def _perform_comparison(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Выполнение сравнения анализов"""
        comparison = {
            "compared_analyses": len(analyses),
            "comparison_timestamp": datetime.now().isoformat(),
            "metrics_comparison": {},
            "trends": {},
            "recommendations": [],
        }

        # Сравниваем метрики
        metrics_comparison = {}
        for i, analysis in enumerate(analyses):
            metrics = analysis.get("metrics", {})
            basic_metrics = metrics.get("basic", {})

            metrics_comparison[f"analysis_{i+1}"] = {
                "timestamp": analysis.get("timestamp", ""),
                "chat_name": analysis.get("chat_name", "unknown"),
                "average_score": basic_metrics.get("average_score", 0),
                "success_rate": basic_metrics.get("success_rate", 0),
                "total_queries": basic_metrics.get("total_queries", 0),
            }

        comparison["metrics_comparison"] = metrics_comparison

        # Анализируем тренды
        scores = [m["average_score"] for m in metrics_comparison.values()]
        if len(scores) >= 2:
            if scores[-1] > scores[0]:
                comparison["trends"]["score_trend"] = "improving"
            elif scores[-1] < scores[0]:
                comparison["trends"]["score_trend"] = "declining"
            else:
                comparison["trends"]["score_trend"] = "stable"

        # Генерируем рекомендации
        if comparison["trends"].get("score_trend") == "declining":
            comparison["recommendations"].append(
                "Качество поиска снижается - требуется анализ причин"
            )
        elif comparison["trends"].get("score_trend") == "improving":
            comparison["recommendations"].append(
                "Качество поиска улучшается - продолжайте текущие практики"
            )

        return comparison

    def _load_metadata(self) -> Dict[str, Any]:
        """Загрузка метаданных"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Ошибка загрузки метаданных: {e}")

        return {
            "chats": {},
            "overall": {},
            "last_updated": datetime.now().isoformat(),
        }

    def _save_metadata(self):
        """Сохранение метаданных"""
        self.metadata["last_updated"] = datetime.now().isoformat()

        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения метаданных: {e}")

    def _update_chat_metadata(
        self,
        chat_name: str,
        analysis_id: str,
        analysis_record: Dict[str, Any],
    ):
        """Обновление метаданных чата"""
        if "chats" not in self.metadata:
            self.metadata["chats"] = {}

        if chat_name not in self.metadata["chats"]:
            self.metadata["chats"][chat_name] = {
                "total_analyses": 0,
                "last_analysis": None,
                "analyses": [],
            }

        chat_metadata = self.metadata["chats"][chat_name]
        chat_metadata["total_analyses"] += 1
        chat_metadata["last_analysis"] = analysis_id

        # Добавляем анализ в список
        analysis_summary = {
            "id": analysis_id,
            "timestamp": analysis_record.get("timestamp", ""),
            "total_queries": analysis_record.get("total_queries", 0),
            "average_score": analysis_record.get("metrics", {})
            .get("basic", {})
            .get("average_score", 0),
        }

        chat_metadata["analyses"].append(analysis_summary)

        # Ограничиваем количество записей в метаданных
        if len(chat_metadata["analyses"]) > 50:
            chat_metadata["analyses"] = chat_metadata["analyses"][-50:]

        self._save_metadata()

    def _update_overall_metadata(
        self,
        analysis_id: str,
        overall_record: Dict[str, Any],
    ):
        """Обновление метаданных общих анализов"""
        if "overall" not in self.metadata:
            self.metadata["overall"] = {
                "total_analyses": 0,
                "last_analysis": None,
                "analyses": [],
            }

        overall_metadata = self.metadata["overall"]
        overall_metadata["total_analyses"] += 1
        overall_metadata["last_analysis"] = analysis_id

        # Добавляем анализ в список
        analysis_summary = {
            "id": analysis_id,
            "timestamp": overall_record.get("timestamp", ""),
            "total_chats": overall_record.get("total_chats", 0),
            "average_score": overall_record.get("overall_metrics", {}).get(
                "average_score", 0
            ),
        }

        overall_metadata["analyses"].append(analysis_summary)

        # Ограничиваем количество записей в метаданных
        if len(overall_metadata["analyses"]) > 50:
            overall_metadata["analyses"] = overall_metadata["analyses"][-50:]

        self._save_metadata()

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики истории"""
        stats = {
            "total_chats": len(self.metadata.get("chats", {})),
            "total_chat_analyses": sum(
                chat.get("total_analyses", 0)
                for chat in self.metadata.get("chats", {}).values()
            ),
            "total_overall_analyses": self.metadata.get("overall", {}).get(
                "total_analyses", 0
            ),
            "last_updated": self.metadata.get("last_updated", ""),
        }

        return stats
