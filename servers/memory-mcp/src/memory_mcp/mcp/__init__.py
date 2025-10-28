"""MCP contract definitions for the unified memory service."""

from .adapters import MemoryServiceAdapter
from .schema import (
    AttachmentPayload,
    FetchRequest,
    FetchResponse,
    GetSignalPerformanceRequest,
    GetSignalPerformanceResponse,
    IngestRequest,
    IngestResponse,
    MemoryRecordPayload,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SearchTradingPatternsRequest,
    SearchTradingPatternsResponse,
    SignalPerformance,
    StoreTradingSignalRequest,
    StoreTradingSignalResponse,
    TradingSignalRecord,
)
from .server import create_server, main
from .tools import bind

__all__ = [
    "AttachmentPayload",
    "FetchRequest",
    "FetchResponse",
    "IngestRequest",
    "IngestResponse",
    "MemoryRecordPayload",
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
    "StoreTradingSignalRequest",
    "StoreTradingSignalResponse",
    "SearchTradingPatternsRequest",
    "SearchTradingPatternsResponse",
    "GetSignalPerformanceRequest",
    "GetSignalPerformanceResponse",
    "TradingSignalRecord",
    "SignalPerformance",
    "bind",
    "MemoryServiceAdapter",
    "create_server",
    "main",
]
