"""Analytics module for building a knowledge graph on top of chat summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import yaml

from ..core.lmstudio_client import LMStudioEmbeddingClient

try:
    import chromadb  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    chromadb = None  # type: ignore


_FRONT_MATTER_PATTERN = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_HEADING_PATTERN = re.compile(r"^##+\s+(.*)", re.MULTILINE)
_LINK_PATTERN = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<target>[^\)]+)\)")


@dataclass(slots=True)
class SummaryDocument:
    """Container for a single markdown summary document."""

    chat_name: str
    path: Path
    metadata: dict[str, Any]
    content: str
    headings: list[str] = field(default_factory=list)
    links: list[tuple[str, str]] = field(default_factory=list)

    @property
    def embedding_text(self) -> str:
        """Build a compact representation for semantic similarity queries."""
        parts: list[str] = []
        meta = self.metadata
        purpose = meta.get("purpose")
        if purpose:
            parts.append(str(purpose))
        topics = meta.get("topics_canon") or []
        if topics:
            parts.append("; ".join(str(topic) for topic in topics))
        tags = meta.get("tags_glossary") or []
        if tags:
            tag_titles = [tag.get("tag", "") for tag in tags if tag.get("tag")]
            if tag_titles:
                parts.append("; ".join(tag_titles))
        parts.extend(self.headings[:5])
        trimmed_body = re.sub(r"\s+", " ", self.content).strip()
        if trimmed_body:
            parts.append(trimmed_body[:800])
        return " \n".join(parts)


@dataclass(slots=True)
class Insight:
    """Single insight description with supporting data."""

    title: str
    detail: str
    score: float
    nodes: Sequence[str] = ()


@dataclass(slots=True)
class InsightGraphResult:
    """Aggregated result of insight analysis."""

    graph: nx.Graph
    insights: Sequence[Insight]
    metrics: dict[str, Any]


class SummaryInsightAnalyzer:
    """Analyse markdown summaries and build a knowledge graph."""

    def __init__(
        self,
        summaries_dir: Path | str = Path("artifacts/reports"),
        chroma_path: Path | str = Path("./chroma_db"),
        *,
        embedding_client: LMStudioEmbeddingClient | None = None,
        chroma_client: Any | None = None,
        similarity_threshold: float = 0.76,
        max_similar_results: int = 8,
    ) -> None:
        self.summaries_dir = Path(summaries_dir)
        self.chroma_path = Path(chroma_path)
        self.similarity_threshold = similarity_threshold
        self.max_similar_results = max_similar_results
        self.embedding_client = embedding_client or LMStudioEmbeddingClient()
        self._own_embedding_client = embedding_client is None
        self.chroma_client = chroma_client or (
            chromadb.PersistentClient(path=str(self.chroma_path)) if chromadb else None
        )
        self.summary_collection = None
        if self.chroma_client is not None:
            try:
                self.summary_collection = self.chroma_client.get_collection(
                    "telegram_summaries"
                )
            except Exception:
                self.summary_collection = None

    async def __aenter__(self) -> SummaryInsightAnalyzer:
        if self._own_embedding_client:
            await self.embedding_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._own_embedding_client:
            await self.embedding_client.__aexit__(exc_type, exc_val, exc_tb)

    def load_documents(self) -> list[SummaryDocument]:
        """Load all markdown summaries from the configured directory."""
        documents: list[SummaryDocument] = []
        if not self.summaries_dir.exists():
            return documents
        for file_path in sorted(self.summaries_dir.glob("*.md")):
            if file_path.is_dir():
                continue
            metadata, body = self._split_front_matter(file_path)
            chat_name = metadata.get("chat") or file_path.stem
            headings = self._extract_headings(body)
            links = self._extract_links(body)
            documents.append(
                SummaryDocument(
                    chat_name=chat_name,
                    path=file_path,
                    metadata=metadata,
                    content=body.strip(),
                    headings=headings,
                    links=links,
                )
            )
        return documents

    async def analyze(self) -> InsightGraphResult:
        """Run full analysis and build the insight graph."""
        documents = self.load_documents()
        graph = self._build_graph(documents)
        if documents:
            await self._enrich_with_similarity(graph, documents)
        metrics = self._collect_metrics(graph)
        insights = self._derive_insights(graph, metrics)
        return InsightGraphResult(graph=graph, insights=insights, metrics=metrics)

    def export_graphml(self, result: InsightGraphResult, output_path: Path) -> Path:
        """Persist the graph into GraphML for downstream tools."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a clean copy of the graph for GraphML export
        clean_graph = nx.Graph()

        # Copy nodes with serializable attributes only
        for node, attrs in result.graph.nodes(data=True):
            clean_attrs = {}
            for key, value in attrs.items():
                if key == "metadata" and isinstance(value, dict):
                    # Convert metadata dict to string for GraphML compatibility
                    clean_attrs[key] = str(value)
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    clean_attrs[key] = value
                else:
                    # Convert other complex types to string
                    clean_attrs[key] = str(value)
            clean_graph.add_node(node, **clean_attrs)

        # Copy edges with serializable attributes only
        for source, target, attrs in result.graph.edges(data=True):
            clean_attrs = {}
            for key, value in attrs.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    clean_attrs[key] = value
                else:
                    clean_attrs[key] = str(value)
            clean_graph.add_edge(source, target, **clean_attrs)

        nx.write_graphml(clean_graph, output_path)
        return output_path

    @staticmethod
    def generate_report(result: InsightGraphResult) -> str:
        """Produce a markdown report for the computed insights."""
        lines: list[str] = ["# Insight Graph Summary", ""]
        graph_info = result.metrics.get("graph", {})
        lines.append(f"- Узлов: {graph_info.get('nodes', 0)}")
        lines.append(f"- Связей: {graph_info.get('edges', 0)}")
        density = graph_info.get("density", 0.0)
        lines.append(f"- Плотность: {density:.3f}")
        lines.append("")
        lines.append("## Топ инсайтов")
        if not result.insights:
            lines.append("- Инсайты не обнаружены")
        else:
            for insight in result.insights:
                nodes_text = ", ".join(insight.nodes)
                lines.append(
                    f"- **{insight.title}** (score={insight.score:.2f}) — {insight.detail}"
                )
                if nodes_text:
                    lines.append(f"  - Узлы: {nodes_text}")
        centrality = result.metrics.get("centrality", {})
        if centrality:
            lines.append("")
            lines.append("## Центральность")
            for key, values in centrality.items():
                top_values = list(values.items())[:5]
                if not top_values:
                    continue
                lines.append(
                    f"- {key}: "
                    + ", ".join(f"{node} ({score:.2f})" for node, score in top_values)
                )
        return "\n".join(lines)

    def _build_graph(self, documents: Sequence[SummaryDocument]) -> nx.Graph:
        graph = nx.Graph()
        for doc in documents:
            graph.add_node(
                doc.chat_name,
                type="chat",
                path=str(doc.path),
                metadata=doc.metadata,
            )
            for tag in self._extract_tags(doc.metadata):
                node_id = f"tag::{tag}"
                graph.add_node(node_id, type="tag", tag=tag)
                graph.add_edge(doc.chat_name, node_id, relation="tag")
            for topic in self._extract_topics(doc):
                node_id = f"topic::{topic}"
                graph.add_node(node_id, type="topic", topic=topic)
                graph.add_edge(doc.chat_name, node_id, relation="topic")
            for participant in self._extract_participants(doc.metadata):
                node_id = f"person::{participant}"
                graph.add_node(node_id, type="participant", person=participant)
                graph.add_edge(doc.chat_name, node_id, relation="participant")
            for label, target in doc.links:
                if target.startswith("sessions/"):
                    session_id = f"session::{target.split('/')[-1]}"
                    graph.add_node(
                        session_id, type="session", label=label, target=target
                    )
                    graph.add_edge(doc.chat_name, session_id, relation="session")
        return graph

    async def _enrich_with_similarity(
        self, graph: nx.Graph, documents: Sequence[SummaryDocument]
    ) -> None:
        if self.summary_collection is None:
            return
        embeddings = await self.embedding_client.generate_embeddings(
            [doc.embedding_text for doc in documents]
        )
        for doc, embedding in zip(documents, embeddings):
            try:
                results = self.summary_collection.query(
                    query_embeddings=[embedding],
                    n_results=self.max_similar_results,
                    include=["metadatas", "distances"],
                )
            except Exception:
                continue
            metadatas = results.get("metadatas") or []
            distances = results.get("distances") or []
            if not metadatas or not metadatas[0]:
                continue
            for metadata, distance in zip(metadatas[0], distances[0]):
                other_chat = metadata.get("chat_name") or metadata.get("chat")
                if not other_chat or other_chat == doc.chat_name:
                    continue
                # Используем логарифмическую схожесть для лучшего различия
                similarity = 1.0 / (1.0 + float(distance) ** 0.5)
                if similarity < self.similarity_threshold:
                    continue
                if other_chat not in graph:
                    graph.add_node(other_chat, type="chat", metadata={})
                existing = graph.get_edge_data(doc.chat_name, other_chat, default={})
                if existing.get("weight", 0.0) < similarity:
                    graph.add_edge(
                        doc.chat_name,
                        other_chat,
                        relation="similarity",
                        weight=similarity,
                    )

    def _collect_metrics(self, graph: nx.Graph) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "graph": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0.0,
                "components": nx.number_connected_components(graph) if graph else 0,
            },
        }
        if graph.number_of_nodes() == 0:
            return metrics
        chat_nodes = [
            n for n, data in graph.nodes(data=True) if data.get("type") == "chat"
        ]
        if chat_nodes:
            degree_centrality = nx.degree_centrality(graph)
            betweenness = nx.betweenness_centrality(graph, normalized=True)
            try:
                pagerank = nx.pagerank(graph, alpha=0.85)
            except ModuleNotFoundError:
                pagerank = {}
            metrics["centrality"] = {
                "degree": self._sort_metrics(degree_centrality, chat_nodes),
                "betweenness": self._sort_metrics(betweenness, chat_nodes),
            }
            if pagerank:
                metrics["centrality"]["pagerank"] = self._sort_metrics(
                    pagerank, chat_nodes
                )
        return metrics

    def _derive_insights(
        self, graph: nx.Graph, metrics: dict[str, Any]
    ) -> list[Insight]:
        insights: list[Insight] = []
        centrality = metrics.get("centrality", {})
        if centrality:
            top_bridgers = list(centrality.get("betweenness", {}).items())[:3]
            if top_bridgers:
                score = sum(val for _, val in top_bridgers) / len(top_bridgers)
                nodes = [node for node, _ in top_bridgers]
                detail = "Узлы с наибольшим посредничеством связывают разрозненные темы"
                insights.append(
                    Insight(
                        title="Ключевые хабы обсуждений",
                        detail=detail,
                        score=score,
                        nodes=nodes,
                    )
                )
        tag_edges = [
            (u, v)
            for u, v, data in graph.edges(data=True)
            if data.get("relation") == "tag"
        ]
        if tag_edges:
            tag_counts: dict[str, int] = {}
            for _, tag_node in tag_edges:
                tag_counts[tag_node] = tag_counts.get(tag_node, 0) + 1
            top_tag = max(tag_counts.items(), key=lambda item: item[1])
            related_chats = [chat for chat, tag in tag_edges if tag == top_tag[0]]
            insights.append(
                Insight(
                    title="Доминирующий тег",
                    detail=f"Тег {top_tag[0].split('::', 1)[-1]} объединяет {top_tag[1]} чатов",
                    score=float(top_tag[1]),
                    nodes=related_chats,
                )
            )
        similarity_edges = [
            (u, v, data)
            for u, v, data in graph.edges(data=True)
            if data.get("relation") == "similarity"
        ]
        if similarity_edges:
            strongest = max(
                similarity_edges, key=lambda item: item[2].get("weight", 0.0)
            )
            insights.append(
                Insight(
                    title="Сильнейшая схожесть",
                    detail=f"Чаты {strongest[0]} и {strongest[1]} имеют высокую семантическую близость",
                    score=float(strongest[2].get("weight", 0.0)),
                    nodes=[strongest[0], strongest[1]],
                )
            )
        return insights

    @staticmethod
    def _sort_metrics(
        values: dict[str, float], only_nodes: Iterable[str]
    ) -> dict[str, float]:
        filtered = {node: values.get(node, 0.0) for node in only_nodes}
        return dict(sorted(filtered.items(), key=lambda item: item[1], reverse=True))

    @staticmethod
    def _split_front_matter(file_path: Path) -> tuple[dict[str, Any], str]:
        metadata: dict[str, Any] = {}
        with open(file_path, encoding="utf-8") as fh:
            text = fh.read()
        match = _FRONT_MATTER_PATTERN.match(text)
        if match:
            try:
                metadata = yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError:
                metadata = {}
            body = text[match.end() :]
        else:
            body = text
        return metadata, body

    @staticmethod
    def _extract_headings(body: str) -> list[str]:
        return [heading.strip() for heading in _HEADING_PATTERN.findall(body)]

    @staticmethod
    def _extract_links(body: str) -> list[tuple[str, str]]:
        return [
            (match.group("label"), match.group("target"))
            for match in _LINK_PATTERN.finditer(body)
        ]

    @staticmethod
    def _extract_tags(metadata: dict[str, Any]) -> list[str]:
        tags = metadata.get("tags_glossary")
        if not isinstance(tags, list):
            return []
        result = []
        for tag in tags:
            if isinstance(tag, dict) and tag.get("tag"):
                result.append(str(tag["tag"]).strip())
        return result

    @staticmethod
    def _extract_topics(doc: SummaryDocument) -> list[str]:
        topics = doc.metadata.get("topics_canon")
        result: list[str] = []
        if isinstance(topics, list):
            result.extend(str(topic).strip() for topic in topics if str(topic).strip())
        for heading in doc.headings:
            if heading.startswith("📅"):
                continue
            clean_heading = heading.strip("# ")
            if clean_heading and clean_heading not in result:
                result.append(clean_heading)
        return result

    @staticmethod
    def _extract_participants(metadata: dict[str, Any]) -> list[str]:
        participants = metadata.get("participants")
        if isinstance(participants, list):
            return [
                str(participant).strip()
                for participant in participants
                if str(participant).strip()
            ]
        return []


async def build_insight_graph(
    summaries_dir: Path | str = Path("artifacts/reports"),
    chroma_path: Path | str = Path("./chroma_db"),
    *,
    similarity_threshold: float = 0.76,
) -> InsightGraphResult:
    """Convenience coroutine to run insight analysis in a single call."""
    async with SummaryInsightAnalyzer(
        summaries_dir=summaries_dir,
        chroma_path=chroma_path,
        similarity_threshold=similarity_threshold,
    ) as analyzer:
        return await analyzer.analyze()
