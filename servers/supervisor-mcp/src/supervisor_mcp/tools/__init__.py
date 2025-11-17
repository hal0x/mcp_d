"""MCP tools registration for supervisor server."""

from fastmcp import FastMCP
from ..pydantic_models import (
    Metric, Fact, MCPInfo, AlertRule, 
    ScrapeRequest, ScrapeBatchRequest, ScrapeSearchRequest,
    ScrapeResponse, ScrapeBatchResponse, ScrapeSearchResponse,
    ScrapeStatus, ScrapeHistory
)
from ..services.registry import RegistryService
from ..services.health import HealthService
from ..services.metrics import MetricsService
from ..services.alerts import AlertsService
from ..services.scraper import ScraperService


def register_tools(
    mcp: FastMCP,
    registry_service: RegistryService,
    health_service: HealthService,
    metrics_service: MetricsService,
    alerts_service: AlertsService,
    scraper_service: ScraperService,
) -> None:
    """Register all MCP tools."""
    
    # Registry tools
    @mcp.tool()
    async def get_mcp_registry() -> dict:
        """Returns the registry of all MCP servers and their capabilities."""
        registry = await registry_service.get_registry()
        return {
            "mcp_servers": [
                {
                    "name": mcp.name,
                    "version": mcp.version,
                    "protocol": mcp.protocol,
                    "endpoint": mcp.endpoint,
                    "capabilities": mcp.capabilities,
                    "status": mcp.status,
                    "last_seen": mcp.last_seen.isoformat() if mcp.last_seen else None,
                }
                for mcp in registry
            ]
        }
    
    @mcp.tool()
    async def get_mcp_capabilities(name: str) -> dict:
        """Returns capabilities for a specific MCP server."""
        capabilities = await registry_service.get_capabilities(name)
        return {"name": name, "capabilities": capabilities}
    
    # Health tools
    @mcp.tool()
    async def get_health_status(name: str = None) -> dict:
        """Returns health status for MCP servers."""
        if name:
            statuses = await health_service.get_health_status(name)
        else:
            statuses = await health_service.get_health_status()
        
        return {
            "health_statuses": [
                {
                    "name": status.name,
                    "status": status.status,
                    "response_time_ms": status.response_time_ms,
                    "error": status.error,
                    "last_check": status.last_check.isoformat(),
                }
                for status in statuses
            ]
        }
    
    @mcp.tool()
    async def get_overall_health() -> dict:
        """Returns overall health summary of the MCP ecosystem."""
        return await health_service.get_overall_health()
    
    # Metrics tools
    @mcp.tool()
    async def ingest_metric(name: str, value: float, tags: dict = None, ts: str = None) -> dict:
        """Ingests a single metric data point."""
        from datetime import datetime
        
        metric = Metric(
            name=name,
            value=value,
            tags=tags or {},
            ts=datetime.fromisoformat(ts) if ts else datetime.now()
        )
        
        await metrics_service.ingest_metric(metric)
        return {"status": "success", "metric": metric.dict()}
    
    @mcp.tool()
    async def ingest_fact(kind: str, actor: str, correlation_id: str, payload: dict, ts: str = None) -> dict:
        """Ingests a single fact event."""
        from datetime import datetime
        
        fact = Fact(
            kind=kind,
            actor=actor,
            correlation_id=correlation_id,
            payload=payload,
            ts=datetime.fromisoformat(ts) if ts else datetime.now()
        )
        
        await metrics_service.ingest_fact(fact)
        return {"status": "success", "fact": fact.dict()}
    
    @mcp.tool()
    async def query_metrics(name: str = None, start_time: str = None, end_time: str = None, tags: dict = None) -> dict:
        """Queries metrics with optional filters."""
        from datetime import datetime
        
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None
        
        metrics = await metrics_service.query_metrics(
            name=name,
            start_time=start_dt,
            end_time=end_dt,
            tags=tags
        )
        
        return {
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "tags": m.tags,
                    "ts": m.ts.isoformat(),
                }
                for m in metrics
            ]
        }
    
    @mcp.tool()
    async def query_facts(kind: str = None, start_time: str = None, end_time: str = None, actor: str = None) -> dict:
        """Queries facts with optional filters."""
        from datetime import datetime
        
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None
        
        facts = await metrics_service.query_facts(
            kind=kind,
            start_time=start_dt,
            end_time=end_dt,
            actor=actor
        )
        
        return {
            "facts": [
                {
                    "kind": f.kind,
                    "actor": f.actor,
                    "correlation_id": f.correlation_id,
                    "payload": f.payload,
                    "ts": f.ts.isoformat(),
                }
                for f in facts
            ]
        }
    
    @mcp.tool()
    async def get_aggregation(kind: str = "business", window: str = "7d") -> dict:
        """Returns aggregated metrics for a time window."""
        aggregation = await metrics_service.get_aggregation(kind=kind, window=window)
        
        return {
            "window": aggregation.window,
            "kind": aggregation.kind,
            "metrics": aggregation.metrics,
            "facts_count": aggregation.facts_count,
            "period_start": aggregation.period_start.isoformat(),
            "period_end": aggregation.period_end.isoformat(),
        }
    
    # Alerts tools
    @mcp.tool()
    async def create_alert_rule(rule_id: str, name: str, condition: str, severity: str, enabled: bool = True, cooldown_minutes: int = 5, actions: list = None) -> dict:
        """Creates or updates an alert rule."""
        rule = AlertRule(
            id=rule_id,
            name=name,
            condition=condition,
            severity=severity,
            enabled=enabled,
            cooldown_minutes=cooldown_minutes,
            actions=actions or []
        )
        
        await alerts_service.create_alert_rule(rule)
        return {"status": "success", "rule": rule.dict()}
    
    @mcp.tool()
    async def get_alert_rules() -> dict:
        """Returns all alert rules."""
        rules = await alerts_service.get_alert_rules()
        return {
            "alert_rules": [
                {
                    "id": rule.id,
                    "name": rule.name,
                    "condition": rule.condition,
                    "severity": rule.severity,
                    "enabled": rule.enabled,
                    "cooldown_minutes": rule.cooldown_minutes,
                    "actions": rule.actions,
                }
                for rule in rules
            ]
        }
    
    @mcp.tool()
    async def get_active_alerts() -> dict:
        """Returns all active alerts."""
        alerts = await alerts_service.get_active_alerts()
        return {
            "active_alerts": [
                {
                    "id": alert.id,
                    "rule_id": alert.rule_id,
                    "severity": alert.severity,
                    "message": alert.message,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "acknowledged": alert.acknowledged,
                    "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    "acknowledged_by": alert.acknowledged_by,
                }
                for alert in alerts
            ]
        }
    
    @mcp.tool()
    async def acknowledge_alert(alert_id: str, acknowledged_by: str) -> dict:
        """Acknowledges an alert."""
        success = await alerts_service.acknowledge_alert(alert_id, acknowledged_by)
        return {"status": "success" if success else "not_found", "alert_id": alert_id}
    
    # Scraping tools
    @mcp.tool()
    async def scrape_url(url: str, options: dict = None) -> dict:
        """Scrape a single URL using Bright Data MCP."""
        try:
            result = await scraper_service.scrape_url(url, options or {})
            return {
                "status": "success",
                "result": result,
                "url": url
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "url": url
            }
    
    @mcp.tool()
    async def scrape_urls_batch(urls: list[str], options: dict = None) -> dict:
        """Scrape multiple URLs in batch using Bright Data MCP."""
        try:
            results = await scraper_service.scrape_urls_batch(urls, options or {})
            successful = [r for r in results if r.get("status") != "failed"]
            failed = [r for r in results if r.get("status") == "failed"]
            
            return {
                "status": "success",
                "total_urls": len(urls),
                "successful": len(successful),
                "failed": len(failed),
                "results": results
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "urls": urls
            }
    
    @mcp.tool()
    async def scrape_search_results(
        query: str, 
        search_engine: str = "google", 
        limit: int = 10, 
        options: dict = None
    ) -> dict:
        """Scrape search results for a query using Bright Data MCP."""
        try:
            results = await scraper_service.scrape_search_results(
                query, search_engine, limit, options or {}
            )
            return {
                "status": "success",
                "query": query,
                "search_engine": search_engine,
                "result_count": len(results),
                "results": results
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }
    
    @mcp.tool()
    def get_scraping_status(task_id: str = None) -> dict:
        """Get status of scraping tasks."""
        try:
            status = scraper_service.get_scraping_status(task_id)
            return {
                "status": "success",
                "data": status
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "task_id": task_id
            }
    
    @mcp.tool()
    def get_scraping_history(limit: int = None, status_filter: str = None) -> dict:
        """Get scraping task history."""
        try:
            history = scraper_service.get_scraping_history(limit, status_filter)
            return {
                "status": "success",
                "total_count": len(history),
                "limit": limit,
                "status_filter": status_filter,
                "tasks": history
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    @mcp.tool()
    def clear_scraping_cache() -> dict:
        """Clear scraping cache."""
        try:
            result = scraper_service.clear_cache()
            return {
                "status": "success",
                "cleared_items": result["cleared_items"]
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
