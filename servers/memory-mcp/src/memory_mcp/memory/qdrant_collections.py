"""Менеджер коллекций Qdrant для замены ChromaDB."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..config import get_settings

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:
    QdrantClient = None
    qmodels = None

logger = logging.getLogger(__name__)


class QdrantCollectionsManager:
    """Менеджер для работы с несколькими коллекциями Qdrant (замена ChromaDB)."""

    def __init__(self, url: Optional[str] = None, vector_size: int = 1024):
        """Инициализирует менеджер коллекций Qdrant.

        Args:
            url: URL Qdrant сервера (если None, берется из настроек)
            vector_size: Размерность векторов
        """
        if url is None:
            settings = get_settings()
            url = settings.get_qdrant_url()
        
        self.url = url
        self.vector_size = vector_size
        self.client: Optional[QdrantClient] = None
        self._collections: Dict[str, str] = {}
        
        if url and QdrantClient is not None:
            try:
                self.client = QdrantClient(url=url, check_compatibility=False)
                logger.info("QdrantCollectionsManager подключен к %s", url)
            except Exception as exc:
                logger.warning("Не удалось подключиться к Qdrant: %s", exc)
                self.client = None
        elif url:
            logger.warning("qdrant-client не установлен; QdrantCollectionsManager отключен")

    def available(self) -> bool:
        """Проверяет доступность Qdrant."""
        return self.client is not None

    def ensure_collection(self, collection_name: str, force_recreate: bool = False) -> bool:
        """Создает коллекцию, если она не существует.

        Args:
            collection_name: Имя коллекции
            force_recreate: Пересоздать коллекцию, если она существует

        Returns:
            True если коллекция создана/существует
        """
        if not self.client or qmodels is None:
            return False
        
        try:
            try:
                info = self.client.get_collection(collection_name)
                if force_recreate:
                    logger.info(f"Пересоздание коллекции {collection_name}")
                    self.client.recreate_collection(
                        collection_name=collection_name,
                        vectors_config=qmodels.VectorParams(
                            size=self.vector_size,
                            distance=qmodels.Distance.COSINE,
                        ),
                    )
                else:
                    existing_size = info.config.params.vectors.size
                    if existing_size != self.vector_size:
                        logger.warning(
                            f"Размерность коллекции {collection_name} не совпадает "
                            f"({existing_size} != {self.vector_size}), пересоздаем"
                        )
                        self.client.recreate_collection(
                            collection_name=collection_name,
                            vectors_config=qmodels.VectorParams(
                                size=self.vector_size,
                                distance=qmodels.Distance.COSINE,
                            ),
                        )
                self._collections[collection_name] = collection_name
                return True
            except Exception:
                logger.info(f"Создание коллекции {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_size,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
                self._collections[collection_name] = collection_name
                return True
        except Exception as exc:
            logger.error(f"Ошибка при создании коллекции {collection_name}: {exc}")
            return False

    def upsert(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: Optional[List[str]] = None,
    ) -> bool:
        """Добавляет или обновляет записи в коллекции (аналог ChromaDB upsert).

        Args:
            collection_name: Имя коллекции
            ids: Список ID записей
            embeddings: Список векторов эмбеддингов
            metadatas: Список метаданных
            documents: Список документов (опционально, сохраняется в payload)

        Returns:
            True если успешно
        """
        if not self.client or qmodels is None:
            return False
        
        if not self.ensure_collection(collection_name):
            return False
        
        if len(ids) != len(embeddings) or len(ids) != len(metadatas):
            logger.error("Несоответствие размеров: ids, embeddings, metadatas должны быть одинаковой длины")
            return False
        
        try:
            points = []
            for i, record_id in enumerate(ids):
                payload = dict(metadatas[i])
                if documents and i < len(documents):
                    payload["document"] = documents[i]
                
                points.append(
                    qmodels.PointStruct(
                        id=record_id,
                        vector=embeddings[i] if i < len(embeddings) else [],
                        payload=payload,
                    )
                )
            
            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
            return True
        except Exception as exc:
            logger.error(f"Ошибка при upsert в коллекцию {collection_name}: {exc}")
            return False

    def get(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Получает записи из коллекции (аналог ChromaDB get).

        Args:
            collection_name: Имя коллекции
            ids: Список ID для получения (если указан, where игнорируется)
            where: Фильтр по метаданным (ChromaDB-стиль)
            limit: Максимальное количество результатов

        Returns:
            Словарь с ключами: ids, embeddings, metadatas, documents
        """
        if not self.client or qmodels is None:
            return {"ids": [], "embeddings": [], "metadatas": [], "documents": []}
        
        try:
            if ids:
                points = self.client.retrieve(
                    collection_name=collection_name,
                    ids=ids,
                    with_payload=True,
                    with_vectors=True,
                )
            else:
                filter_conditions = self._build_filter(where) if where else None
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_conditions,
                    limit=limit or 10,
                    with_payload=True,
                    with_vectors=True,
                )
                points = scroll_result[0]
            
            result_ids = []
            result_embeddings = []
            result_metadatas = []
            result_documents = []
            
            for point in points:
                result_ids.append(str(point.id))
                if point.vector:
                    result_embeddings.append(list(point.vector))
                else:
                    result_embeddings.append([])
                
                payload = dict(point.payload or {})
                document = payload.pop("document", "")
                result_documents.append(document)
                result_metadatas.append(payload)
            
            return {
                "ids": result_ids,
                "embeddings": result_embeddings,
                "metadatas": result_metadatas,
                "documents": result_documents,
            }
        except Exception as exc:
            logger.error(f"Ошибка при get из коллекции {collection_name}: {exc}")
            return {"ids": [], "embeddings": [], "metadatas": [], "documents": []}

    def delete_collection(self, collection_name: str) -> bool:
        """Удаляет коллекцию.

        Args:
            collection_name: Имя коллекции

        Returns:
            True если успешно
        """
        if not self.client:
            return False
        
        try:
            self.client.delete_collection(collection_name)
            if collection_name in self._collections:
                del self._collections[collection_name]
            return True
        except Exception as exc:
            logger.error(f"Ошибка при удалении коллекции {collection_name}: {exc}")
            return False

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Удаляет записи из коллекции.

        Args:
            collection_name: Имя коллекции
            ids: Список ID для удаления
            where: Фильтр по метаданным

        Returns:
            Количество удаленных записей
        """
        if not self.client or qmodels is None:
            return 0
        
        try:
            if ids:
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=qmodels.PointIdsList(points=ids),
                )
                return len(ids)
            elif where:
                filter_conditions = self._build_filter(where)
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_conditions,
                    limit=10000,
                    with_payload=False,
                    with_vectors=False,
                )
                points_to_delete = [str(point.id) for point in scroll_result[0]]
                if points_to_delete:
                    self.client.delete(
                        collection_name=collection_name,
                        points_selector=qmodels.PointIdsList(points=points_to_delete),
                    )
                return len(points_to_delete)
            return 0
        except Exception as exc:
            logger.error(f"Ошибка при удалении из коллекции {collection_name}: {exc}")
            return 0

    def count(self, collection_name: str, where: Optional[Dict[str, Any]] = None) -> int:
        """Подсчитывает количество записей в коллекции.

        Args:
            collection_name: Имя коллекции
            where: Фильтр по метаданным

        Returns:
            Количество записей
        """
        if not self.client:
            return 0
        
        try:
            if where:
                filter_conditions = self._build_filter(where)
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_conditions,
                    limit=1,
                    with_payload=False,
                    with_vectors=False,
                )
                total = 0
                offset = None
                while True:
                    scroll_result = self.client.scroll(
                        collection_name=collection_name,
                        scroll_filter=filter_conditions,
                        limit=1000,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False,
                    )
                    points = scroll_result[0]
                    if not points:
                        break
                    total += len(points)
                    offset = scroll_result[1]
                    if offset is None:
                        break
                return total
            else:
                info = self.client.get_collection(collection_name)
                return info.points_count
        except Exception as exc:
            logger.error(f"Ошибка при подсчете записей в коллекции {collection_name}: {exc}")
            return 0

    def _build_filter(self, where: Dict[str, Any]) -> Optional[qmodels.Filter]:
        """Строит фильтр Qdrant из ChromaDB-стиля where.

        Args:
            where: Фильтр в стиле ChromaDB

        Returns:
            Фильтр Qdrant или None
        """
        if not qmodels or not where:
            return None
        
        must_conditions = []
        
        for key, value in where.items():
            if key == "$and":
                and_conditions = []
                for condition in value:
                    sub_filter = self._build_filter(condition)
                    if sub_filter:
                        and_conditions.append(sub_filter)
                if and_conditions:
                    must_conditions.append(qmodels.Filter(must=and_conditions))
            elif key == "$or":
                or_conditions = []
                for condition in value:
                    sub_filter = self._build_filter(condition)
                    if sub_filter:
                        or_conditions.append(sub_filter)
                if or_conditions:
                    must_conditions.append(qmodels.Filter(should=or_conditions))
            elif isinstance(value, dict):
                if "$eq" in value:
                    must_conditions.append(
                        qmodels.FieldCondition(
                            key=key,
                            match=qmodels.MatchValue(value=value["$eq"]),
                        )
                    )
                elif "$ne" in value:
                    logger.warning(f"Qdrant не поддерживает $ne для {key}, пропускаем")
            else:
                must_conditions.append(
                    qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchValue(value=value),
                    )
                )
        
        if not must_conditions:
            return None
        
        return qmodels.Filter(must=must_conditions)

    def close(self) -> None:
        """Закрывает соединение с Qdrant."""
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass

