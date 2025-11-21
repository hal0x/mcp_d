"""Кластеризация сессий по темам."""

import logging
from typing import Any, Dict, List, Optional

from ...memory.storage.vector.qdrant_collections import QdrantCollectionsManager
from ...utils.system.naming import slugify

logger = logging.getLogger(__name__)


class ClusteringManager:
    """Кластеризация сессий по темам."""

    def __init__(
        self,
        qdrant_manager: Optional[QdrantCollectionsManager],
        sessions_collection: Optional[str],
        clusters_collection: Optional[str],
        session_clusterer: Optional[Any],
        cluster_summarizer: Optional[Any],
    ):
        """Инициализирует менеджер кластеризации.

        Args:
            qdrant_manager: Менеджер Qdrant коллекций
            sessions_collection: Имя коллекции сессий
            clusters_collection: Имя коллекции кластеров
            session_clusterer: Кластеризатор сессий
            cluster_summarizer: Саммаризатор кластеров
        """
        self.qdrant_manager = qdrant_manager
        self.sessions_collection = sessions_collection
        self.clusters_collection = clusters_collection
        self.session_clusterer = session_clusterer
        self.cluster_summarizer = cluster_summarizer

    async def cluster_chat_sessions(
        self, chat_name: str, summaries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Кластеризация сессий чата и сохранение результатов

        Args:
            chat_name: Название чата
            summaries: Список саммаризаций сессий

        Returns:
            Статистика кластеризации
        """
        if not self.session_clusterer or not self.cluster_summarizer:
            return {"clusters_count": 0, "sessions_clustered": 0}

        # Получаем эмбеддинги из Qdrant для сессий
        session_ids = [s["session_id"] for s in summaries]

        try:
            result = None
            if self.qdrant_manager and self.sessions_collection:
                result = self.qdrant_manager.get(
                    collection_name=self.sessions_collection, ids=session_ids
                )
            else:
                logger.warning("Qdrant недоступен, кластеризация невозможна")
                return {"clusters_count": 0, "sessions_clustered": 0}

            if not result["ids"]:
                logger.warning(f"Не найдены эмбеддинги для сессий чата {chat_name}")
                return {"clusters_count": 0, "sessions_clustered": 0}

            # Преобразуем в формат для кластеризации
            sessions_data = []
            embeddings_list = []

            for i, session_id in enumerate(result["ids"]):
                sessions_data.append(
                    {
                        "session_id": session_id,
                        "metadata": result["metadatas"][i],
                        "document": result["documents"][i],
                    }
                )
                embeddings_list.append(result["embeddings"][i])

            # Выполняем кластеризацию
            logger.info(f"Кластеризация {len(sessions_data)} сессий чата {chat_name}")
            clustering_result = self.session_clusterer.cluster_sessions(
                sessions_data, embeddings_list
            )

            clusters = clustering_result.get("clusters", [])
            logger.info(f"Найдено {len(clusters)} кластеров")

            # Сохраняем кластеры в Qdrant и обновляем метаданные сессий
            clusters_saved = 0
            sessions_clustered = 0

            for cluster in clusters:
                cluster_id = f"{slugify(chat_name)}-cluster-{cluster['cluster_id']}"

                # Обновляем метаданные сессий, добавляя информацию о кластере
                for session_id in cluster["session_ids"]:
                    try:
                        # Получаем текущую метаданные сессии
                        session_data = None
                        if self.qdrant_manager and self.sessions_collection:
                            session_data = self.qdrant_manager.get(
                                collection_name=self.sessions_collection,
                                ids=[session_id],
                            )

                        if session_data and session_data.get("ids"):
                            metadata = session_data["metadatas"][0].copy()
                            metadata["cluster_id"] = cluster_id
                            metadata["cluster_label"] = cluster.get("label", "")

                            # Обновляем метаданные через upsert (Qdrant не имеет отдельного update)
                            if self.qdrant_manager and self.sessions_collection:
                                # Получаем текущие данные для обновления
                                current_data = self.qdrant_manager.get(
                                    collection_name=self.sessions_collection,
                                    ids=[session_id],
                                )
                                if current_data and current_data.get("ids"):
                                    # Обновляем через upsert с новыми метаданными
                                    self.qdrant_manager.upsert(
                                        collection_name=self.sessions_collection,
                                        ids=[session_id],
                                        embeddings=current_data.get("embeddings", [[]])[:1]
                                        or [[]],
                                        metadatas=[metadata],
                                        documents=current_data.get("documents", [""])[:1]
                                        or [""],
                                    )
                                    sessions_clustered += 1
                    except Exception as e:
                        logger.error(
                            f"Ошибка при обновлении метаданных сессии {session_id}: {e}"
                        )

                # Генерируем эмбеддинг для кластера (среднее эмбеддингов сессий)
                cluster_embedding = [0.0] * len(embeddings_list[0])
                for session_id in cluster["session_ids"]:
                    try:
                        idx = session_ids.index(session_id)
                        session_emb = embeddings_list[idx]
                        for i, val in enumerate(session_emb):
                            cluster_embedding[i] += val
                    except ValueError:
                        continue

                # Нормализуем
                n = len(cluster["session_ids"])
                if n > 0:
                    cluster_embedding = [val / n for val in cluster_embedding]

                # Создаём документ для кластера
                cluster_doc = (
                    f"Кластер: {cluster.get('label', 'Без названия')}\n"
                    f"Ключевые слова: {', '.join(cluster.get('keywords', []))}\n"
                    f"Топики: {', '.join(cluster.get('topics', []))}\n"
                    f"Сущности: {', '.join(cluster.get('entities', []))}"
                )

                # Метаданные кластера
                cluster_metadata = {
                    "cluster_id": cluster_id,
                    "chat": chat_name,
                    "label": cluster.get("label", ""),
                    "size": cluster.get("size", 0),
                    "coherence": cluster.get("coherence", 0.0),
                    "session_ids": ",".join(
                        cluster["session_ids"][:10]
                    ),  # Первые 10
                }

                # Сохраняем кластер
                try:
                    if self.qdrant_manager and self.clusters_collection:
                        self.qdrant_manager.upsert(
                            collection_name=self.clusters_collection,
                            ids=[cluster_id],
                            documents=[cluster_doc],
                            embeddings=[cluster_embedding],
                            metadatas=[cluster_metadata],
                        )
                    clusters_saved += 1
                    logger.info(
                        f"Сохранён кластер {cluster_id}: {cluster.get('label', '')}"
                    )
                except Exception as e:
                    logger.error(f"Ошибка при сохранении кластера {cluster_id}: {e}")

            return {
                "clusters_count": clusters_saved,
                "sessions_clustered": sessions_clustered,
                "total_sessions": len(sessions_data),
                "noise_sessions": clustering_result.get("noise_count", 0),
            }

        except Exception as e:
            logger.error(f"Ошибка при кластеризации чата {chat_name}: {e}")
            return {"clusters_count": 0, "sessions_clustered": 0}

    def get_clusters(
        self, chat: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Получить список кластеров

        Args:
            chat: Фильтр по чату (опционально)
            limit: Максимальное количество кластеров

        Returns:
            Список кластеров
        """
        if not self.qdrant_manager or not self.clusters_collection:
            return []

        try:
            where_filter = {"chat": chat} if chat else None

            result = self.qdrant_manager.get(
                collection_name=self.clusters_collection,
                where=where_filter,
                limit=limit,
            )

            clusters = []
            for i, cluster_id in enumerate(result["ids"]):
                clusters.append(
                    {
                        "cluster_id": cluster_id,
                        "metadata": result["metadatas"][i],
                        "document": result["documents"][i],
                    }
                )

            return clusters
        except Exception as e:
            logger.error(f"Ошибка при получении кластеров: {e}")
            return []

    def get_cluster_sessions(self, cluster_id: str) -> List[Dict[str, Any]]:
        """
        Получить сессии, принадлежащие кластеру

        Args:
            cluster_id: ID кластера

        Returns:
            Список сессий
        """
        if not self.qdrant_manager or not self.sessions_collection:
            return []

        try:
            result = self.qdrant_manager.get(
                collection_name=self.sessions_collection,
                where={"cluster_id": cluster_id},
            )

            sessions = []
            for i, session_id in enumerate(result["ids"]):
                sessions.append(
                    {
                        "session_id": session_id,
                        "metadata": result["metadatas"][i],
                        "document": result["documents"][i],
                    }
                )

            return sessions
        except Exception as e:
            logger.error(f"Ошибка при получении сессий кластера {cluster_id}: {e}")
            return []

