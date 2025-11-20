#!/usr/bin/env python3
"""
Кластеризация сессий в темы

Автоматическая группировка похожих сессий на основе семантического сходства.
Вдохновлено системой schemas из архитектуры HALv1.
"""

import asyncio
import logging
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SessionClusterer:
    """Кластеризация сессий в тематические группы"""

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        min_cluster_size: int = 2,
        max_cluster_size: int = 50,
        use_hdbscan: bool = True,
    ):
        """
        Инициализация кластеризатора

        Args:
            similarity_threshold: Порог косинусного сходства для группировки (0-1)
            min_cluster_size: Минимальный размер кластера
            max_cluster_size: Максимальный размер кластера
            use_hdbscan: Использовать HDBSCAN (True) или простой порог (False)
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.use_hdbscan = use_hdbscan

        logger.info(
            f"Инициализирован SessionClusterer "
            f"(threshold={similarity_threshold}, "
            f"min_size={min_cluster_size}, "
            f"use_hdbscan={use_hdbscan})"
        )

    def cluster_sessions(
        self,
        sessions: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        Кластеризация сессий в темы

        Args:
            sessions: Список сессий с метаданными
            embeddings: Список эмбеддингов (если None, берутся из sessions)

        Returns:
            Словарь с результатами кластеризации
        """
        if not sessions:
            logger.warning("Нет сессий для кластеризации")
            return {"clusters": [], "noise": [], "stats": {}}

        logger.info(f"Начало кластеризации {len(sessions)} сессий")

        # Извлекаем эмбеддинги
        if embeddings is None:
            embeddings = self._extract_embeddings(sessions)

        if not embeddings or len(embeddings) != len(sessions):
            logger.error("Не удалось извлечь эмбеддинги")
            return {"clusters": [], "noise": [], "stats": {}}

        # Выполняем кластеризацию
        if self.use_hdbscan and len(sessions) >= self.min_cluster_size * 2:
            labels = self._cluster_hdbscan(embeddings)
        else:
            labels = self._cluster_threshold(embeddings)

        # Группируем сессии по кластерам
        clusters_dict = {}
        noise_sessions = []

        for session, label in zip(sessions, labels):
            if label == -1:  # Noise/outliers
                noise_sessions.append(session)
            else:
                if label not in clusters_dict:
                    clusters_dict[label] = []
                clusters_dict[label].append(session)

        # Формируем результат
        clusters = []
        for cluster_id, cluster_sessions in clusters_dict.items():
            # Фильтруем слишком маленькие и большие кластеры
            if len(cluster_sessions) < self.min_cluster_size:
                noise_sessions.extend(cluster_sessions)
                continue

            if len(cluster_sessions) > self.max_cluster_size:
                # Большие кластеры разбиваем дополнительно
                logger.warning(
                    f"Кластер {cluster_id} слишком большой ({len(cluster_sessions)}), "
                    f"требуется дополнительная кластеризация"
                )
                # Рекурсивная кластеризация больших кластеров
                sub_clusters = self._recursive_cluster_large_cluster(cluster_sessions, embeddings, sessions)
                # Добавляем подкластеры в результат
                for sub_cluster in sub_clusters:
                    clusters.append(sub_cluster)
                continue

            cluster_data = {
                "cluster_id": f"cluster-{cluster_id}",
                "session_count": len(cluster_sessions),
                "sessions": cluster_sessions,
                "session_ids": [s.get("session_id") for s in cluster_sessions],
            }

            # Вычисляем метаданные кластера
            cluster_data.update(self._compute_cluster_metadata(cluster_sessions))

            clusters.append(cluster_data)

        # Статистика
        stats = {
            "total_sessions": len(sessions),
            "clustered_sessions": len(sessions) - len(noise_sessions),
            "noise_sessions": len(noise_sessions),
            "num_clusters": len(clusters),
            "avg_cluster_size": (
                sum(len(c["sessions"]) for c in clusters) / len(clusters)
                if clusters
                else 0
            ),
            "min_cluster_size": min((len(c["sessions"]) for c in clusters), default=0),
            "max_cluster_size": max((len(c["sessions"]) for c in clusters), default=0),
        }

        logger.info(
            f"Кластеризация завершена: {stats['num_clusters']} кластеров, "
            f"{stats['noise_sessions']} outliers"
        )

        return {"clusters": clusters, "noise": noise_sessions, "stats": stats}

    def _extract_embeddings(self, sessions: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Извлечение эмбеддингов из сессий

        Args:
            sessions: Список сессий

        Returns:
            Список эмбеддингов
        """
        embeddings = []

        for session in sessions:
            # Пробуем разные ключи
            embedding = None

            if "embedding" in session and session["embedding"]:
                embedding = session["embedding"]
            elif "embeddings" in session and session["embeddings"]:
                embedding = session["embeddings"]

            # Если нет прямого эмбеддинга, пробуем вычислить из метаданных
            if embedding is None:
                logger.warning(f"Нет эмбеддинга для сессии {session.get('session_id')}")
                # Возвращаем нулевой вектор (будет noise)
                embedding = [0.0] * 768  # Стандартный размер

            embeddings.append(embedding)

        return embeddings

    def _cluster_hdbscan(self, embeddings: List[List[float]]) -> List[int]:
        """
        HDBSCAN кластеризация

        Args:
            embeddings: Эмбеддинги

        Returns:
            Метки кластеров (-1 для noise)
        """
        try:
            import hdbscan

            embeddings_array = np.array(embeddings)

            # Создаём кластеризатор
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=max(1, self.min_cluster_size // 2),
                metric="euclidean",
                cluster_selection_epsilon=1.0 - self.similarity_threshold,
                cluster_selection_method="eom",
            )

            # Выполняем кластеризацию
            labels = clusterer.fit_predict(embeddings_array)

            logger.info(
                f"HDBSCAN: найдено {len(set(labels)) - (1 if -1 in labels else 0)} кластеров"
            )

            return labels.tolist()

        except Exception as e:
            logger.error(f"Ошибка HDBSCAN: {e}, откат на threshold clustering")
            return self._cluster_threshold(embeddings)

    def _cluster_threshold(self, embeddings: List[List[float]]) -> List[int]:
        """
        Простая кластеризация по порогу сходства

        Args:
            embeddings: Эмбеддинги

        Returns:
            Метки кластеров (-1 для noise)
        """
        embeddings_array = np.array(embeddings)
        n = len(embeddings)

        # Вычисляем матрицу косинусного сходства
        similarity_matrix = cosine_similarity(embeddings_array)

        # Инициализируем метки
        labels = [-1] * n
        current_cluster_id = 0

        # Жадный алгоритм кластеризации
        for i in range(n):
            if labels[i] != -1:
                continue

            # Находим все похожие сессии
            similar_indices = []
            for j in range(n):
                if i != j and similarity_matrix[i][j] >= self.similarity_threshold:
                    similar_indices.append(j)

            # Если нашли похожие, создаём кластер
            if len(similar_indices) >= self.min_cluster_size - 1:
                labels[i] = current_cluster_id
                for j in similar_indices:
                    if labels[j] == -1:
                        labels[i] = current_cluster_id

                current_cluster_id += 1

        logger.info(f"Threshold: найдено {current_cluster_id} кластеров")

        return labels

    def _compute_cluster_metadata(
        self, sessions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Вычисление метаданных кластера

        Args:
            sessions: Сессии в кластере

        Returns:
            Метаданные кластера
        """
        # Чаты
        chats = [s.get("chat") or s.get("meta", {}).get("chat_name") for s in sessions]
        chat_counts = Counter(filter(None, chats))

        # Временной диапазон
        dates = []
        for s in sessions:
            date_str = s.get("start_time") or s.get("meta", {}).get("start_time_utc")
            if date_str:
                try:
                    from ..utils.datetime_utils import parse_datetime_utc

                    result = parse_datetime_utc(date_str, return_none_on_error=True, use_zoneinfo=True)
                    if result:
                        dates.append(result)
                except:
                    pass

        time_range = None
        if dates:
            min_date = min(dates)
            max_date = max(dates)
            time_range = {
                "start": min_date.isoformat(),
                "end": max_date.isoformat(),
                "span_days": (max_date - min_date).days,
            }

        # Темы/топики (если есть)
        all_topics = []
        for s in sessions:
            topics = s.get("topics", [])
            for topic in topics:
                title = topic.get("title") if isinstance(topic, dict) else topic
                if title:
                    all_topics.append(title)

        topic_counts = Counter(all_topics)
        top_topics = [topic for topic, _count in topic_counts.most_common(5)]

        # Сущности
        all_entities = []
        for s in sessions:
            entities = s.get("entities", [])
            all_entities.extend(entities)

        entity_counts = Counter(all_entities)
        top_entities = [entity for entity, _count in entity_counts.most_common(10)]

        # Языки
        languages = [
            s.get("meta", {}).get("dominant_language", "unknown") for s in sessions
        ]
        language_counts = Counter(languages)
        dominant_language = (
            language_counts.most_common(1)[0][0] if language_counts else "unknown"
        )

        # Общее количество сообщений
        total_messages = sum(
            s.get("message_count", 0)
            or s.get("meta", {}).get("messages_total", 0)
            or len(s.get("messages", []))
            for s in sessions
        )

        return {
            "chats": dict(chat_counts.most_common()),
            "dominant_chat": chat_counts.most_common(1)[0][0] if chat_counts else None,
            "time_range": time_range,
            "top_topics": top_topics,
            "top_entities": top_entities,
            "dominant_language": dominant_language,
            "total_messages": total_messages,
        }

    def generate_cluster_summary(
        self, cluster: Dict[str, Any], embedding_client=None
    ) -> Dict[str, Any]:
        """
        Генерация сводки для кластера

        Args:
            cluster: Данные кластера
            embedding_client: Embedding client для генерации LLM сводки (опционально)

        Returns:
            Кластер с добавленной сводкой
        """
        cluster_id = cluster["cluster_id"]
        cluster["sessions"]

        logger.info(f"Генерация сводки для кластера {cluster_id}")

        # Базовая сводка на основе метаданных
        summary = self._generate_basic_summary(cluster)

        # Если есть embedding_client, генерируем детальную сводку через ClusterSummarizer
        if embedding_client:
            try:
                from .cluster_summarizer import ClusterSummarizer
                
                summarizer = ClusterSummarizer(embedding_client)
                # ClusterSummarizer.summarize_cluster - async метод
                # Пытаемся запустить через asyncio.run
                try:
                    llm_summary = asyncio.run(summarizer.summarize_cluster(cluster))
                except RuntimeError:
                    # Если event loop уже запущен, пропускаем LLM генерацию
                    logger.warning(
                        "Не удалось запустить async генерацию сводки: event loop уже запущен. "
                        "Используйте async версию метода или вызывайте из синхронного контекста."
                    )
                    llm_summary = None
                
                # Обновляем сводку данными из LLM
                if llm_summary:
                    if llm_summary.get("title"):
                        summary["llm_title"] = llm_summary["title"]
                    if llm_summary.get("description"):
                        summary["llm_description"] = llm_summary["description"]
                    if llm_summary.get("key_insights"):
                        summary["key_insights"] = llm_summary["key_insights"]
            except Exception as e:
                logger.error(f"Ошибка генерации LLM сводки: {e}")

        cluster["summary"] = summary

        return cluster

    def _generate_basic_summary(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        """
        Базовая сводка на основе метаданных

        Args:
            cluster: Данные кластера

        Returns:
            Сводка
        """
        session_count = cluster["session_count"]
        dominant_chat = cluster.get("dominant_chat", "N/A")
        top_topics = cluster.get("top_topics", [])
        top_entities = cluster.get("top_entities", [])
        time_range = cluster.get("time_range", {})

        # Формируем описание
        title_parts = []

        if top_topics:
            title_parts.append(top_topics[0])
        elif top_entities:
            title_parts.append(f"Обсуждение: {', '.join(top_entities[:3])}")
        else:
            title_parts.append(f"Кластер {cluster['cluster_id']}")

        title = " - ".join(title_parts)

        description_parts = [
            f"Объединяет {session_count} сессий",
        ]

        if dominant_chat:
            description_parts.append(f"в основном из чата '{dominant_chat}'")

        if time_range and "span_days" in time_range:
            span = time_range["span_days"]
            if span > 0:
                description_parts.append(f"охватывая {span} дней")

        description = " ".join(description_parts) + "."

        return {
            "title": title,
            "description": description,
            "topics": top_topics,
            "entities": top_entities,
            "session_count": session_count,
        }

    def _recursive_cluster_large_cluster(
        self,
        cluster_sessions: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]],
        all_sessions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Рекурсивная кластеризация большого кластера на подкластеры.

        Args:
            cluster_sessions: Сессии большого кластера
            embeddings: Эмбеддинги всех сессий (опционально)
            all_sessions: Все сессии (для получения эмбеддингов)

        Returns:
            Список подкластеров
        """
        if len(cluster_sessions) <= self.max_cluster_size:
            # Кластер уже достаточно маленький
            return []

        logger.info(
            f"Рекурсивная кластеризация кластера из {len(cluster_sessions)} сессий"
        )

        # Получаем эмбеддинги для сессий этого кластера
        cluster_embeddings = None
        if embeddings is not None:
            # Находим индексы сессий в общем списке
            session_ids = {s.get("session_id") for s in cluster_sessions}
            cluster_embeddings = [
                emb
                for i, s in enumerate(all_sessions)
                if s.get("session_id") in session_ids and i < len(embeddings)
                for emb in [embeddings[i]]
            ]

        # Рекурсивно кластеризуем подкластер
        sub_cluster_result = self.cluster_sessions(cluster_sessions, cluster_embeddings)
        sub_clusters = sub_cluster_result.get("clusters", [])

        logger.info(f"Большой кластер разбит на {len(sub_clusters)} подкластеров")
        return sub_clusters

