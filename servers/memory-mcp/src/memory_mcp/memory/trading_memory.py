"""Trading memory helpers built on top of TypedGraphMemory."""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import uuid4

from ..utils.datetime_utils import parse_datetime_utc
from .graph_types import DocChunkNode, EdgeType, GraphEdge, GraphNode, NodeType
from .typed_graph import TypedGraphMemory


class TradingMemory:
    """Provides CRUD helpers for trading signals persisted alongside the graph."""

    def __init__(self, graph: TypedGraphMemory):
        self._graph = graph
        self._conn = graph.conn
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trading_signals (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                direction TEXT,
                entry REAL,
                confidence REAL,
                context TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trading_signal_performance (
                signal_id TEXT PRIMARY KEY,
                pnl REAL,
                result TEXT,
                closed_at TEXT,
                notes TEXT,
                FOREIGN KEY(signal_id) REFERENCES trading_signals(id) ON DELETE CASCADE
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals(symbol)
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ store
    def store_signal(
        self,
        *,
        symbol: str,
        signal_type: str,
        direction: Optional[str],
        entry: Optional[float],
        confidence: Optional[float],
        context: dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> dict[str, Any]:
        signal_id = str(uuid4())
        ts = timestamp or datetime.now(timezone.utc)
        cursor = self._conn.cursor()
        sanitized_context = self._serialize_context(context)
        cursor.execute(
            """
            INSERT INTO trading_signals
            (id, timestamp, symbol, signal_type, direction, entry, confidence, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal_id,
                ts.isoformat(),
                symbol.upper(),
                signal_type,
                direction,
                entry,
                confidence,
                json.dumps(sanitized_context, ensure_ascii=False),
            ),
        )
        self._conn.commit()

        # Добавляем узел в граф знаний
        label = f"{signal_type} {symbol.upper()}"
        node = GraphNode(
            id=signal_id,
            type=NodeType.TRADING_SIGNAL,
            label=label,
            properties={
                "symbol": symbol.upper(),
                "signal_type": signal_type,
                "direction": direction,
                "entry": entry,
                "confidence": confidence,
                "context": sanitized_context,
                "timestamp": ts.isoformat(),
            },
            created_at=ts.isoformat(),
            updated_at=ts.isoformat(),
        )
        self._graph.add_node(node)

        self._link_symbol(signal_id, symbol.upper(), ts)
        self._link_telegram_context(signal_id, ts, sanitized_context)

        return {
            "signal_id": signal_id,
            "timestamp": ts,
            "symbol": symbol.upper(),
            "signal_type": signal_type,
            "direction": direction,
            "entry": entry,
            "confidence": confidence,
            "context": sanitized_context,
        }

    # ---------------------------------------------------------------- search
    def search_patterns(
        self,
        *,
        query: Optional[str],
        symbol: Optional[str],
        timeframe: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        cursor = self._conn.cursor()
        clauses: list[str] = []
        params: list[Any] = []

        if query:
            like = f"%{query.lower()}%"
            clauses.append(
                "(LOWER(symbol) LIKE ? OR LOWER(signal_type) LIKE ? OR LOWER(context) LIKE ?)"
            )
            params.extend([like, like, like])

        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())

        if timeframe and timeframe.lower() != "all":
            delta = self._resolve_timeframe(timeframe)
            if delta:
                cutoff = datetime.now(timezone.utc) - delta
                clauses.append("timestamp >= ?")
                params.append(cutoff.isoformat())

        where_sql = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = (
            "SELECT id, timestamp, symbol, signal_type, direction, entry, confidence, context "
            "FROM trading_signals"
            f"{where_sql} ORDER BY timestamp DESC LIMIT ?"
        )
        params.append(limit)

        rows = cursor.execute(sql, params).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            context_json = {}
            if row[7]:
                try:
                    context_json = json.loads(row[7])
                except json.JSONDecodeError:
                    context_json = {"raw": row[7]}
            results.append(
                {
                    "signal_id": row[0],
                    "timestamp": parse_datetime_utc(row[1], use_zoneinfo=False) if row[1] else datetime.now(timezone.utc),
                    "symbol": row[2],
                    "signal_type": row[3],
                    "direction": row[4],
                    "entry": row[5],
                    "confidence": row[6],
                    "context": context_json,
                }
            )
        return results

    # ---------------------------------------------------------------- perf
    def get_performance(self, signal_id: str) -> dict[str, Any] | None:
        cursor = self._conn.cursor()
        row = cursor.execute(
            "SELECT id, timestamp, symbol, signal_type, direction, entry, confidence, context "
            "FROM trading_signals WHERE id = ?",
            (signal_id,),
        ).fetchone()
        if not row:
            return None
        performance = cursor.execute(
            "SELECT pnl, result, closed_at, notes FROM trading_signal_performance WHERE signal_id = ?",
            (signal_id,),
        ).fetchone()
        context_json = {}
        if row[7]:
            try:
                context_json = json.loads(row[7])
            except json.JSONDecodeError:
                context_json = {"raw": row[7]}
        result: dict[str, Any] = {
            "signal": {
                "signal_id": row[0],
                "timestamp": parse_datetime_utc(row[1], use_zoneinfo=False) if row[1] else datetime.now(timezone.utc),
                "symbol": row[2],
                "signal_type": row[3],
                "direction": row[4],
                "entry": row[5],
                "confidence": row[6],
                "context": context_json,
            }
        }
        if performance:
            closed_at = (
                parse_datetime_utc(performance[2], return_none_on_error=True) if performance[2] else None
            )
            result["performance"] = {
                "pnl": performance[0],
                "result": performance[1],
                "closed_at": closed_at,
                "notes": performance[3],
            }
        return result

    @staticmethod
    def _serialize_context(context: dict[str, Any]) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [convert(v) for v in value]
            if isinstance(value, datetime):
                return value.astimezone(timezone.utc).isoformat()
            if isinstance(value, (int, float, str, bool)) or value is None:
                return value
            return str(value)

        return {str(k): convert(v) for k, v in context.items()}

    def _link_symbol(self, signal_id: str, symbol: str, timestamp: datetime) -> None:
        symbol_node_id = f"symbol:{symbol}"
        if symbol_node_id not in self._graph.graph:
            symbol_node = GraphNode(
                id=symbol_node_id,
                type=NodeType.ENTITY,
                label=symbol,
                properties={
                    "entity_type": "asset",
                    "symbol": symbol,
                    "source": "trading_memory",
                },
                created_at=timestamp.isoformat(),
                updated_at=timestamp.isoformat(),
            )
            with suppress(Exception):
                self._graph.add_node(symbol_node)

        edge = GraphEdge(
            id=f"edge:{signal_id}:{symbol_node_id}",
            source_id=signal_id,
            target_id=symbol_node_id,
            type=EdgeType.ASSOCIATED_WITH,
            properties={"relation": "trading_signal_asset"},
        )
        with suppress(Exception):
            self._graph.add_edge(edge)

    def _link_telegram_context(
        self,
        signal_id: str,
        timestamp: datetime,
        context: dict[str, Any],
    ) -> None:
        telegram_ctx = context.get("telegram")
        if not isinstance(telegram_ctx, dict):
            telegram_ctx = {}

        chat_id = self._first_non_empty(
            telegram_ctx.get("chat_id"),
            context.get("telegram_chat_id"),
        )
        chat_username = self._first_non_empty(
            telegram_ctx.get("chat_username"),
            telegram_ctx.get("chat_slug"),
            context.get("telegram_chat_username"),
        )
        chat_title = self._first_non_empty(
            telegram_ctx.get("chat_title"),
            context.get("telegram_chat"),
            context.get("chat_title"),
        )

        if not any((chat_id, chat_username, chat_title)):
            return

        chat_identifier = self._slugify(chat_id or chat_username or chat_title)
        chat_node_id = f"telegram:chat:{chat_identifier}"
        chat_properties = {
            "chat_id": chat_id,
            "chat_username": chat_username,
            "chat_title": chat_title,
            "entity_type": "telegram_chat",
            "source": "telegram",
        }
        chat_properties.update(
            {
                key: value
                for key, value in telegram_ctx.items()
                if key
                not in {
                    "chat_id",
                    "chat_username",
                    "chat_slug",
                    "chat_title",
                    "message_id",
                    "message_url",
                    "message_text",
                    "message_timestamp",
                    "author",
                }
            }
        )
        chat_node = GraphNode(
            id=chat_node_id,
            type=NodeType.ENTITY,
            label=chat_title or chat_username or chat_identifier,
            properties={k: v for k, v in chat_properties.items() if v is not None},
            created_at=timestamp.isoformat(),
            updated_at=timestamp.isoformat(),
        )
        with suppress(Exception):
            self._graph.add_node(chat_node)

        chat_edge = GraphEdge(
            id=f"edge:{signal_id}:{chat_node_id}:telegram",
            source_id=signal_id,
            target_id=chat_node_id,
            type=EdgeType.ASSOCIATED_WITH,
            properties={"relation": "telegram_chat"},
        )
        with suppress(Exception):
            self._graph.add_edge(chat_edge)

        message_id = self._first_non_empty(
            telegram_ctx.get("message_id"),
            context.get("telegram_message_id"),
        )
        if not message_id:
            return

        message_url = self._first_non_empty(
            telegram_ctx.get("message_url"),
            context.get("telegram_message_url"),
        )
        message_text = (
            self._first_non_empty(
                telegram_ctx.get("message_text"),
                context.get("telegram_message_text"),
                context.get("message_text"),
            )
            or ""
        )
        message_author = self._first_non_empty(
            telegram_ctx.get("author"),
            context.get("author"),
        )

        message_timestamp = timestamp
        raw_message_ts = self._first_non_empty(
            telegram_ctx.get("message_timestamp"),
            context.get("telegram_message_timestamp"),
        )
        if isinstance(raw_message_ts, str):
            with suppress(Exception):
                message_timestamp = parse_datetime_utc(raw_message_ts, use_zoneinfo=False)
        elif isinstance(raw_message_ts, datetime):
            message_timestamp = raw_message_ts.astimezone(timezone.utc)

        message_node_id = f"telegram:msg:{chat_identifier}:{message_id}"
        message_node = DocChunkNode(
            id=message_node_id,
            label=f"Telegram message {message_id}",
            content=message_text[:2000],
            source="telegram",
            timestamp=message_timestamp.isoformat(),
            author=message_author,
            properties={
                "chat_id": chat_id,
                "chat_username": chat_username,
                "chat_title": chat_title,
                "message_id": message_id,
                "message_url": message_url,
            },
        )
        with suppress(Exception):
            self._graph.add_node(message_node)

        message_edge = GraphEdge(
            id=f"edge:{signal_id}:{message_node_id}:telegram-message",
            source_id=signal_id,
            target_id=message_node_id,
            type=EdgeType.MENTIONS,
            properties={"relation": "telegram_message"},
        )
        with suppress(Exception):
            self._graph.add_edge(message_edge)

        chat_relation_edge = GraphEdge(
            id=f"edge:{message_node_id}:{chat_node_id}:telegram-message",
            source_id=message_node_id,
            target_id=chat_node_id,
            type=EdgeType.PART_OF,
            properties={"relation": "telegram_chat_message"},
        )
        with suppress(Exception):
            self._graph.add_edge(chat_relation_edge)

    @staticmethod
    def _first_non_empty(*values: Any) -> Any:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value
            if value not in (None, "", [], {}):
                return value
        return None

    @staticmethod
    def _slugify(value: str) -> str:
        value = value.strip().lower()
        sanitized = []
        for char in value:
            if char.isalnum():
                sanitized.append(char)
            elif sanitized and sanitized[-1] != "-":
                sanitized.append("-")
        slug = "".join(sanitized).strip("-")
        return slug or "unknown"

    @staticmethod
    def _resolve_timeframe(value: str) -> Optional[timedelta]:
        value = value.lower()
        mapping = {
            "recent": timedelta(days=7),
            "24h": timedelta(hours=24),
            "week": timedelta(days=7),
            "month": timedelta(days=30),
        }
        return mapping.get(value)


__all__ = ["TradingMemory"]
