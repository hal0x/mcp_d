"""MCP server entrypoint exposing the unified memory service."""

from __future__ import annotations

import logging
import os
import sys

from mcp.server.fastmcp import FastMCP

from . import bind
from .adapters import MemoryServiceAdapter

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("MEMORY_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def create_server() -> tuple[FastMCP, MemoryServiceAdapter]:
    """Create FastMCP server and memory adapter."""

    db_path = os.getenv("MEMORY_DB_PATH", "memory_graph.db")

    adapter = MemoryServiceAdapter(db_path=db_path)

    server = FastMCP(
        name="tg-memory-mcp",
        instructions=(
            "Инструменты для записи и поиска по памяти Telegram Dump Manager. "
            "Используйте ingest_records для загрузки событий, "
            "search_memory для поиска и fetch_record для получения полной записи. "
            "Для торговых сигналов доступны store_trading_signal, search_trading_patterns "
            "и get_signal_performance."
        ),
    )

    bind(
        server,
        ingest_adapter=lambda records: adapter.ingest(records),
        search_adapter=lambda req: adapter.search(req),
        fetch_adapter=lambda req: adapter.fetch(req),
        store_trading_signal_adapter=adapter.store_trading_signal,
        search_trading_patterns_adapter=adapter.search_trading_patterns,
        signal_performance_adapter=adapter.get_signal_performance,
        ingest_scraped_content_adapter=adapter.ingest_scraped_content,
    )

    return server, adapter


def main() -> None:
    configure_logging()
    server, adapter = create_server()
    try:
        server.run()
    finally:
        adapter.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - top-level fallback
        logging.exception("Critical error in MCP server: %s", exc)
        sys.exit(1)
