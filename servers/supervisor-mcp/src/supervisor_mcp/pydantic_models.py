"""Pydantic models for supervisor MCP."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Metric(BaseModel):
    """Metric data point."""
    ts: datetime
    name: str
    value: float
    tags: Dict[str, str] = Field(default_factory=dict)


class Fact(BaseModel):
    """Fact event."""
    ts: datetime
    kind: str  # Fact:Signal, Fact:Decision, Fact:Trade, Fact:Outcome, etc.
    actor: str
    correlation_id: str
    payload: Dict[str, Any]


class MCPInfo(BaseModel):
    """MCP server information."""
    name: str
    version: str
    protocol: str  # stdio, http, sse
    endpoint: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    status: str = "unknown"  # up, degraded, down
    last_seen: Optional[datetime] = None


class HealthStatus(BaseModel):
    """Health status for an MCP server."""
    name: str
    status: str  # healthy, degraded, unhealthy
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    last_check: datetime
    uptime_seconds: Optional[float] = None


class AlertRule(BaseModel):
    """Alert rule definition."""
    id: str
    name: str
    condition: str  # SQL-like condition
    severity: str  # info, warning, error, critical
    enabled: bool = True
    cooldown_minutes: int = 5
    actions: List[str] = Field(default_factory=list)  # pause_all, reduce_leverage, etc.


class Alert(BaseModel):
    """Active alert."""
    id: str
    rule_id: str
    severity: str
    message: str
    triggered_at: datetime
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


class AggregationResult(BaseModel):
    """Aggregated metrics result."""
    window: str  # 7d, 30d
    kind: str  # business, technical
    metrics: Dict[str, float]
    facts_count: int
    period_start: datetime
    period_end: datetime


# Scraping models
class ScrapeRequest(BaseModel):
    """Request for scraping a URL."""
    url: str
    options: Dict[str, Any] = Field(default_factory=dict)


class ScrapeBatchRequest(BaseModel):
    """Request for batch scraping URLs."""
    urls: List[str]
    options: Dict[str, Any] = Field(default_factory=dict)


class ScrapeSearchRequest(BaseModel):
    """Request for scraping search results."""
    query: str
    search_engine: str = "google"
    limit: int = 10
    options: Dict[str, Any] = Field(default_factory=dict)


class ScrapedContent(BaseModel):
    """Scraped content result."""
    url: str
    title: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    status: str = "success"  # success, failed, partial


class ScrapeResponse(BaseModel):
    """Response from scraping operation."""
    task_id: str
    status: str  # completed, failed, in_progress
    result: Optional[ScrapedContent] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ScrapeBatchResponse(BaseModel):
    """Response from batch scraping operation."""
    task_id: str
    status: str  # completed, failed, partial
    results: List[ScrapedContent] = Field(default_factory=list)
    failed_urls: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ScrapeSearchResponse(BaseModel):
    """Response from search scraping operation."""
    task_id: str
    status: str  # completed, failed
    results: List[ScrapedContent] = Field(default_factory=list)
    query: str
    search_engine: str
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ScrapeStatus(BaseModel):
    """Status of scraping tasks."""
    task_id: Optional[str] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    success_rate: float = 0.0
    cache_size: int = 0
    status: Optional[str] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None


class ScrapeHistory(BaseModel):
    """Scraping task history."""
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    limit: Optional[int] = None
    status_filter: Optional[str] = None
