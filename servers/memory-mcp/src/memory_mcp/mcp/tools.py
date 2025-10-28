"""MCP tool bindings for the memory service (skeleton implementation)."""

from __future__ import annotations

import logging
from typing import Callable, Iterable

from mcp.server.fastmcp import FastMCP

from .schema import (
    FetchRequest,
    FetchResponse,
    GetSignalPerformanceRequest,
    GetSignalPerformanceResponse,
    IngestRequest,
    IngestResponse,
    MemoryRecordPayload,
    ScrapedContentRequest,
    ScrapedContentResponse,
    SearchRequest,
    SearchResponse,
    SearchTradingPatternsRequest,
    SearchTradingPatternsResponse,
    StoreTradingSignalRequest,
    StoreTradingSignalResponse,
)

logger = logging.getLogger(__name__)


def bind(
    mcp: FastMCP,
    *,
    ingest_adapter: Callable[[Iterable[MemoryRecordPayload]], IngestResponse],
    search_adapter: Callable[[SearchRequest], SearchResponse],
    fetch_adapter: Callable[[FetchRequest], FetchResponse],
    store_trading_signal_adapter: Callable[
        [StoreTradingSignalRequest], StoreTradingSignalResponse
    ],
    search_trading_patterns_adapter: Callable[
        [SearchTradingPatternsRequest], SearchTradingPatternsResponse
    ],
    signal_performance_adapter: Callable[
        [GetSignalPerformanceRequest], GetSignalPerformanceResponse
    ],
    ingest_scraped_content_adapter: Callable[
        [ScrapedContentRequest], ScrapedContentResponse
    ],
) -> None:
    """Register MCP tools for ingest/search/fetch operations.

    The caller is responsible for providing concrete adapters.
    """

    @mcp.tool()
    def ingest_records(request: IngestRequest) -> IngestResponse:
        """Ingest a batch of memory records."""
        return ingest_adapter(request.records)

    @mcp.tool()
    def search_memory(request: SearchRequest) -> SearchResponse:
        """Search memory for relevant records."""
        return search_adapter(request)

    @mcp.tool()
    def fetch_record(request: FetchRequest) -> FetchResponse:
        """Fetch a full memory record by identifier."""
        return fetch_adapter(request)

    @mcp.tool()
    def store_trading_signal(
        request: StoreTradingSignalRequest,
    ) -> StoreTradingSignalResponse:
        """Persist a trading signal in the memory store."""
        return store_trading_signal_adapter(request)

    @mcp.tool()
    def search_trading_patterns(
        request: SearchTradingPatternsRequest,
    ) -> SearchTradingPatternsResponse:
        """Search for stored trading patterns and signals."""
        return search_trading_patterns_adapter(request)

    @mcp.tool()
    def get_signal_performance(
        request: GetSignalPerformanceRequest,
    ) -> GetSignalPerformanceResponse:
        """Return performance metrics for a trading signal."""
        return signal_performance_adapter(request)

    @mcp.tool()
    def ingest_scraped_content(
        request: ScrapedContentRequest,
    ) -> ScrapedContentResponse:
        """Ingest scraped web content into memory."""
        return ingest_scraped_content_adapter(request)
