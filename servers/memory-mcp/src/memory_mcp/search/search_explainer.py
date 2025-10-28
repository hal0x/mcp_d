#!/usr/bin/env python3
"""
SearchExplainer - Объяснение результатов поиска

Вдохновлено архитектурой HALv1:
- Декомпозиция scores (BM25, Vector, RRF)
- Connection graph через TypedGraphMemory
- Объяснение релевантности
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    """Декомпозиция score результата поиска"""

    doc_id: str
    final_score: float

    # BM25 компоненты
    bm25_score: float
    bm25_rank: Optional[int]

    # Vector компоненты
    vector_similarity: float
    vector_distance: float
    vector_rank: Optional[int]

    # RRF компоненты
    rrf_score: float
    rrf_vector_contribution: float
    rrf_bm25_contribution: float

    # Метаданные
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "doc_id": self.doc_id,
            "final_score": self.final_score,
            "bm25": {
                "score": self.bm25_score,
                "rank": self.bm25_rank,
            },
            "vector": {
                "similarity": self.vector_similarity,
                "distance": self.vector_distance,
                "rank": self.vector_rank,
            },
            "rrf": {
                "score": self.rrf_score,
                "vector_contribution": self.rrf_vector_contribution,
                "bm25_contribution": self.rrf_bm25_contribution,
            },
            "metadata": self.metadata,
        }


@dataclass
class ConnectionPath:
    """Путь связей между запросом и результатом"""

    query_entity: str
    result_entity: str
    path_length: int
    path_nodes: List[str]
    path_edges: List[str]
    path_strength: float  # Сила связи (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "query_entity": self.query_entity,
            "result_entity": self.result_entity,
            "path_length": self.path_length,
            "path_nodes": self.path_nodes,
            "path_edges": self.path_edges,
            "path_strength": self.path_strength,
        }


@dataclass
class RelevanceExplanation:
    """Полное объяснение релевантности результата"""

    doc_id: str
    query: str
    rank: int

    # Декомпозиция score
    score_breakdown: ScoreBreakdown

    # Связи через граф
    connection_paths: List[ConnectionPath]

    # Текстовое объяснение
    explanation_text: str

    # Факторы релевантности
    relevance_factors: Dict[str, float]  # factor -> weight

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "doc_id": self.doc_id,
            "query": self.query,
            "rank": self.rank,
            "score_breakdown": self.score_breakdown.to_dict(),
            "connection_paths": [cp.to_dict() for cp in self.connection_paths],
            "explanation_text": self.explanation_text,
            "relevance_factors": self.relevance_factors,
        }


class ScoreDecomposer:
    """Декомпозиция scores для объяснения"""

    def __init__(self, alpha: float = 0.6, k: int = 60):
        """
        Инициализация

        Args:
            alpha: Вес векторного поиска в RRF
            k: Параметр RRF
        """
        self.alpha = alpha
        self.k = k

        logger.info(f"Инициализирован ScoreDecomposer (alpha={alpha}, k={k})")

    def decompose(
        self,
        doc_id: str,
        final_score: float,
        bm25_results: List[Tuple[str, float]],
        vector_results: List[Tuple[str, float]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScoreBreakdown:
        """
        Декомпозиция score результата

        Args:
            doc_id: ID документа
            final_score: Финальный RRF score
            bm25_results: Результаты BM25 поиска [(doc_id, score), ...]
            vector_results: Результаты векторного поиска [(doc_id, score), ...]
            metadata: Метаданные документа

        Returns:
            Декомпозиция score
        """
        # Ищем позиции в результатах
        bm25_rank = None
        bm25_score = 0.0
        for rank, (bid, score) in enumerate(bm25_results):
            if bid == doc_id:
                bm25_rank = rank
                bm25_score = score
                break

        vector_rank = None
        vector_similarity = 0.0
        vector_distance = 0.0
        for rank, (vid, sim) in enumerate(vector_results):
            if vid == doc_id:
                vector_rank = rank
                vector_similarity = sim
                # Обратная конвертация: distance = (1 / similarity) - 1
                vector_distance = (1.0 / sim) - 1.0 if sim > 0 else float("inf")
                break

        # Вычисляем RRF вклады
        rrf_vector_contribution = 0.0
        rrf_bm25_contribution = 0.0

        if vector_rank is not None:
            rrf_vector_contribution = self.alpha / (self.k + vector_rank + 1)

        if bm25_rank is not None:
            rrf_bm25_contribution = (1.0 - self.alpha) / (self.k + bm25_rank + 1)

        rrf_score = rrf_vector_contribution + rrf_bm25_contribution

        return ScoreBreakdown(
            doc_id=doc_id,
            final_score=final_score,
            bm25_score=bm25_score,
            bm25_rank=bm25_rank,
            vector_similarity=vector_similarity,
            vector_distance=vector_distance,
            vector_rank=vector_rank,
            rrf_score=rrf_score,
            rrf_vector_contribution=rrf_vector_contribution,
            rrf_bm25_contribution=rrf_bm25_contribution,
            metadata=metadata or {},
        )

    def explain_score(self, breakdown: ScoreBreakdown) -> str:
        """
        Текстовое объяснение score

        Args:
            breakdown: Декомпозиция score

        Returns:
            Текстовое объяснение
        """
        explanation = []

        # Финальный score
        explanation.append(
            f"📊 Финальный score: {breakdown.final_score:.4f} (RRF fusion)"
        )
        explanation.append("")

        # BM25 компонент
        if breakdown.bm25_rank is not None:
            explanation.append("🔤 BM25 (лексический поиск):")
            explanation.append(
                f"   Score: {breakdown.bm25_score:.4f} | Rank: #{breakdown.bm25_rank + 1}"
            )
            explanation.append(
                f"   Вклад в RRF: {breakdown.rrf_bm25_contribution:.4f} ({(1-self.alpha)*100:.0f}% веса)"
            )
        else:
            explanation.append("🔤 BM25: не найдено")

        explanation.append("")

        # Vector компонент
        if breakdown.vector_rank is not None:
            explanation.append("🧠 Vector (семантический поиск):")
            explanation.append(
                f"   Similarity: {breakdown.vector_similarity:.4f} | Distance: {breakdown.vector_distance:.4f}"
            )
            explanation.append(f"   Rank: #{breakdown.vector_rank + 1}")
            explanation.append(
                f"   Вклад в RRF: {breakdown.rrf_vector_contribution:.4f} ({self.alpha*100:.0f}% веса)"
            )
        else:
            explanation.append("🧠 Vector: не найдено")

        explanation.append("")

        # Анализ доминирующего фактора
        if breakdown.rrf_vector_contribution > breakdown.rrf_bm25_contribution:
            dominant = "векторный поиск"
            ratio = breakdown.rrf_vector_contribution / (
                breakdown.rrf_bm25_contribution or 0.001
            )
        else:
            dominant = "BM25"
            ratio = breakdown.rrf_bm25_contribution / (
                breakdown.rrf_vector_contribution or 0.001
            )

        explanation.append(
            f"🎯 Доминирующий фактор: {dominant} (в {ratio:.1f}x раз сильнее)"
        )

        return "\n".join(explanation)


class ConnectionGraphBuilder:
    """Построение графа связей между запросом и результатами"""

    def __init__(self, typed_graph_memory=None):
        """
        Инициализация

        Args:
            typed_graph_memory: TypedGraphMemory для поиска связей
        """
        self.graph = typed_graph_memory
        logger.info("Инициализирован ConnectionGraphBuilder")

    def find_connections(
        self,
        query_entities: List[str],
        result_id: str,
        max_paths: int = 3,
        max_depth: int = 3,
    ) -> List[ConnectionPath]:
        """
        Поиск путей связей между сущностями запроса и результата

        Args:
            query_entities: Сущности из запроса
            result_id: ID результата
            max_paths: Максимум путей
            max_depth: Максимальная глубина поиска

        Returns:
            Список путей связей
        """
        if not self.graph:
            logger.warning(
                "TypedGraphMemory не инициализирован, пропускаем поиск связей"
            )
            return []

        connections = []

        # Получаем узел результата из графа
        result_node = self.graph.get_node(result_id)
        if not result_node:
            logger.debug(f"Узел {result_id} не найден в графе")
            return []

        # Для каждой сущности запроса ищем путь к результату
        for query_entity in query_entities:
            try:
                # Ищем узел сущности в графе
                entity_nodes = [
                    n
                    for n in self.graph.get_nodes_by_type("Entity")
                    if query_entity.lower() in n.label.lower()
                ]

                if not entity_nodes:
                    continue

                entity_node = entity_nodes[0]

                # Ищем путь между узлами
                paths = self.graph.find_paths(
                    entity_node.id, result_id, max_depth=max_depth
                )

                for path in paths[:max_paths]:
                    # Извлекаем информацию о пути
                    path_nodes = [node.label for node in path]
                    path_edges = self._extract_edge_types(path)
                    path_strength = self._calculate_path_strength(path)

                    connection = ConnectionPath(
                        query_entity=query_entity,
                        result_entity=result_node.label,
                        path_length=len(path) - 1,
                        path_nodes=path_nodes,
                        path_edges=path_edges,
                        path_strength=path_strength,
                    )
                    connections.append(connection)

            except Exception as e:
                logger.error(f"Ошибка поиска связей для {query_entity}: {e}")
                continue

        return connections[:max_paths]

    def _extract_edge_types(self, path: List[Any]) -> List[str]:
        """Извлечение типов рёбер из пути"""
        if not self.graph or len(path) < 2:
            return []

        edge_types = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]

            # Получаем рёбра между узлами
            neighbors = self.graph.get_neighbors(source.id)
            for neighbor in neighbors:
                if neighbor.id == target.id:
                    # Получаем тип ребра (упрощённая версия)
                    edge_types.append("relates_to")
                    break

        return edge_types

    def _calculate_path_strength(self, path: List[Any]) -> float:
        """Вычисление силы пути"""
        if len(path) <= 1:
            return 1.0

        # Сила обратно пропорциональна длине пути
        # Короткие пути = сильные связи
        return 1.0 / len(path)


class MarkdownExporter:
    """Экспорт объяснений в Markdown"""

    @staticmethod
    def export_explanation(explanation: RelevanceExplanation) -> str:
        """
        Экспорт одного объяснения в Markdown

        Args:
            explanation: Объяснение релевантности

        Returns:
            Markdown текст
        """
        lines = []

        # Заголовок
        lines.append(f"## 🔍 Результат #{explanation.rank + 1}")
        lines.append("")
        lines.append(f"**Запрос**: `{explanation.query}`")
        lines.append(f"**Документ**: `{explanation.doc_id}`")
        lines.append(f"**Score**: `{explanation.score_breakdown.final_score:.4f}`")
        lines.append("")

        # Score breakdown
        breakdown = explanation.score_breakdown
        lines.append("### 📊 Декомпозиция Score")
        lines.append("")
        lines.append("| Компонент | Значение | Rank | Вклад в RRF |")
        lines.append("|-----------|----------|------|-------------|")

        # BM25
        if breakdown.bm25_rank is not None:
            lines.append(
                f"| **BM25** | {breakdown.bm25_score:.4f} | "
                f"#{breakdown.bm25_rank + 1} | {breakdown.rrf_bm25_contribution:.4f} |"
            )
        else:
            lines.append("| **BM25** | - | - | 0.0000 |")

        # Vector
        if breakdown.vector_rank is not None:
            lines.append(
                f"| **Vector** | {breakdown.vector_similarity:.4f} | "
                f"#{breakdown.vector_rank + 1} | {breakdown.rrf_vector_contribution:.4f} |"
            )
        else:
            lines.append("| **Vector** | - | - | 0.0000 |")

        # RRF
        lines.append(f"| **RRF Total** | - | - | {breakdown.rrf_score:.4f} |")
        lines.append("")

        # Связи через граф
        if explanation.connection_paths:
            lines.append("### 🕸️  Связи через граф знаний")
            lines.append("")
            for i, path in enumerate(explanation.connection_paths, 1):
                lines.append(f"{i}. **{path.query_entity}** → **{path.result_entity}**")
                lines.append(f"   - Путь: `{' → '.join(path.path_nodes)}`")
                lines.append(
                    f"   - Длина: {path.path_length}, Сила: {path.path_strength:.2f}"
                )
                lines.append("")

        # Факторы релевантности
        if explanation.relevance_factors:
            lines.append("### 🎯 Факторы релевантности")
            lines.append("")
            lines.append("| Фактор | Вес |")
            lines.append("|--------|-----|")
            for factor, weight in sorted(
                explanation.relevance_factors.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"| {factor} | {weight:.4f} |")
            lines.append("")

        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def export_batch(
        explanations: List[RelevanceExplanation], output_file: str
    ) -> None:
        """
        Экспорт нескольких объяснений в Markdown файл

        Args:
            explanations: Список объяснений
            output_file: Путь к выходному файлу
        """
        with open(output_file, "w", encoding="utf-8") as f:
            # Заголовок документа
            f.write("# 🔍 Объяснение результатов поиска\n\n")

            if explanations:
                f.write(f"**Запрос**: `{explanations[0].query}`\n")
                f.write(f"**Результатов**: {len(explanations)}\n\n")
                f.write("---\n\n")

            # Экспорт каждого объяснения
            for explanation in explanations:
                f.write(MarkdownExporter.export_explanation(explanation))

        logger.info(f"Экспортировано {len(explanations)} объяснений в {output_file}")


class SearchExplainer:
    """Основной класс для объяснения результатов поиска"""

    def __init__(
        self,
        alpha: float = 0.6,
        k: int = 60,
        typed_graph_memory=None,
    ):
        """
        Инициализация

        Args:
            alpha: Вес векторного поиска в RRF
            k: Параметр RRF
            typed_graph_memory: TypedGraphMemory для поиска связей
        """
        self.score_decomposer = ScoreDecomposer(alpha=alpha, k=k)
        self.connection_builder = ConnectionGraphBuilder(typed_graph_memory)
        self.markdown_exporter = MarkdownExporter()

        logger.info("Инициализирован SearchExplainer")

    def explain_result(
        self,
        doc_id: str,
        query: str,
        rank: int,
        final_score: float,
        bm25_results: List[Tuple[str, float]],
        vector_results: List[Tuple[str, float]],
        metadata: Optional[Dict[str, Any]] = None,
        query_entities: Optional[List[str]] = None,
    ) -> RelevanceExplanation:
        """
        Полное объяснение релевантности результата

        Args:
            doc_id: ID документа
            query: Текст запроса
            rank: Позиция в результатах (0-based)
            final_score: Финальный score
            bm25_results: Результаты BM25
            vector_results: Результаты векторного поиска
            metadata: Метаданные документа
            query_entities: Сущности из запроса (для графа связей)

        Returns:
            Объяснение релевантности
        """
        # 1. Декомпозиция score
        score_breakdown = self.score_decomposer.decompose(
            doc_id=doc_id,
            final_score=final_score,
            bm25_results=bm25_results,
            vector_results=vector_results,
            metadata=metadata,
        )

        # 2. Поиск связей через граф
        connection_paths = []
        if query_entities:
            connection_paths = self.connection_builder.find_connections(
                query_entities=query_entities,
                result_id=doc_id,
                max_paths=3,
                max_depth=3,
            )

        # 3. Генерация текстового объяснения
        explanation_text = self._generate_explanation_text(
            query=query,
            rank=rank,
            score_breakdown=score_breakdown,
            connection_paths=connection_paths,
        )

        # 4. Вычисление факторов релевантности
        relevance_factors = self._compute_relevance_factors(
            score_breakdown=score_breakdown,
            connection_paths=connection_paths,
        )

        return RelevanceExplanation(
            doc_id=doc_id,
            query=query,
            rank=rank,
            score_breakdown=score_breakdown,
            connection_paths=connection_paths,
            explanation_text=explanation_text,
            relevance_factors=relevance_factors,
        )

    def _generate_explanation_text(
        self,
        query: str,
        rank: int,
        score_breakdown: ScoreBreakdown,
        connection_paths: List[ConnectionPath],
    ) -> str:
        """Генерация текстового объяснения"""
        lines = []

        # Заголовок
        lines.append(f"🔍 Объяснение результата #{rank + 1}")
        lines.append(f'Запрос: "{query}"')
        lines.append(f"Документ: {score_breakdown.doc_id}")
        lines.append("=" * 70)
        lines.append("")

        # Score breakdown
        lines.append(self.score_decomposer.explain_score(score_breakdown))
        lines.append("")

        # Связи через граф
        if connection_paths:
            lines.append("🕸️  Связи через граф знаний:")
            lines.append("")
            for i, path in enumerate(connection_paths, 1):
                lines.append(f"   {i}. {path.query_entity} → {path.result_entity}")
                lines.append(f"      Путь: {' → '.join(path.path_nodes)}")
                lines.append(
                    f"      Длина: {path.path_length}, Сила: {path.path_strength:.2f}"
                )
                lines.append("")
        else:
            lines.append("🕸️  Связи через граф: не найдены")
            lines.append("")

        return "\n".join(lines)

    def _compute_relevance_factors(
        self,
        score_breakdown: ScoreBreakdown,
        connection_paths: List[ConnectionPath],
    ) -> Dict[str, float]:
        """Вычисление весов факторов релевантности"""
        factors = {}

        # Вклад BM25
        if score_breakdown.bm25_rank is not None:
            factors["bm25_lexical_match"] = score_breakdown.rrf_bm25_contribution

        # Вклад Vector
        if score_breakdown.vector_rank is not None:
            factors[
                "vector_semantic_similarity"
            ] = score_breakdown.rrf_vector_contribution

        # Вклад графа связей
        if connection_paths:
            avg_path_strength = sum(p.path_strength for p in connection_paths) / len(
                connection_paths
            )
            factors["graph_connections"] = avg_path_strength * 0.1  # Меньший вес

        return factors

    def explain_batch(
        self,
        results: List[Dict[str, Any]],
        query: str,
        bm25_results: List[Tuple[str, float]],
        vector_results: List[Tuple[str, float]],
        query_entities: Optional[List[str]] = None,
        max_explain: int = 5,
    ) -> List[RelevanceExplanation]:
        """
        Объяснение нескольких результатов

        Args:
            results: Список результатов поиска
            query: Текст запроса
            bm25_results: Результаты BM25
            vector_results: Результаты векторного поиска
            query_entities: Сущности из запроса
            max_explain: Максимум результатов для объяснения

        Returns:
            Список объяснений
        """
        explanations = []

        for rank, result in enumerate(results[:max_explain]):
            try:
                explanation = self.explain_result(
                    doc_id=result.get("id", ""),
                    query=query,
                    rank=rank,
                    final_score=result.get("score", 0.0),
                    bm25_results=bm25_results,
                    vector_results=vector_results,
                    metadata=result.get("metadata", {}),
                    query_entities=query_entities,
                )
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Ошибка объяснения результата {rank}: {e}")
                continue

        return explanations

    def export_to_markdown(
        self, explanations: List[RelevanceExplanation], output_file: str
    ) -> None:
        """
        Экспорт объяснений в Markdown файл

        Args:
            explanations: Список объяснений
            output_file: Путь к выходному файлу
        """
        self.markdown_exporter.export_batch(explanations, output_file)
