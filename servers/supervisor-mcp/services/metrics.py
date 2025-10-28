"""Metrics service for collecting and aggregating metrics."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from ..models import Metric, Fact, AggregationResult


class MetricsService:
    """Service for collecting and aggregating metrics."""
    
    def __init__(self):
        self._metrics: List[Metric] = []
        self._facts: List[Fact] = []
    
    async def ingest_metric(self, metric: Metric) -> None:
        """Ingest a single metric."""
        self._metrics.append(metric)
    
    async def ingest_metrics(self, metrics: List[Metric]) -> None:
        """Ingest multiple metrics."""
        self._metrics.extend(metrics)
    
    async def ingest_fact(self, fact: Fact) -> None:
        """Ingest a single fact."""
        self._facts.append(fact)
    
    async def ingest_facts(self, facts: List[Fact]) -> None:
        """Ingest multiple facts."""
        self._facts.extend(facts)
    
    async def query_metrics(
        self,
        name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Metric]:
        """Query metrics with filters."""
        filtered_metrics = self._metrics
        
        if name:
            filtered_metrics = [m for m in filtered_metrics if m.name == name]
        
        if start_time:
            filtered_metrics = [m for m in filtered_metrics if m.ts >= start_time]
        
        if end_time:
            filtered_metrics = [m for m in filtered_metrics if m.ts <= end_time]
        
        if tags:
            filtered_metrics = [
                m for m in filtered_metrics
                if all(m.tags.get(k) == v for k, v in tags.items())
            ]
        
        return filtered_metrics
    
    async def query_facts(
        self,
        kind: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        actor: Optional[str] = None
    ) -> List[Fact]:
        """Query facts with filters."""
        filtered_facts = self._facts
        
        if kind:
            filtered_facts = [f for f in filtered_facts if f.kind == kind]
        
        if start_time:
            filtered_facts = [f for f in filtered_facts if f.ts >= start_time]
        
        if end_time:
            filtered_facts = [f for f in filtered_facts if f.ts <= end_time]
        
        if actor:
            filtered_facts = [f for f in filtered_facts if f.actor == actor]
        
        return filtered_facts
    
    async def get_aggregation(
        self,
        kind: str = "business",
        window: str = "7d"
    ) -> AggregationResult:
        """Get aggregated metrics for a time window."""
        # Parse window
        if window.endswith("d"):
            days = int(window[:-1])
        else:
            days = 7
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Filter metrics and facts for the period
        period_metrics = await self.query_metrics(start_time=start_time, end_time=end_time)
        period_facts = await self.query_facts(start_time=start_time, end_time=end_time)
        
        # Calculate aggregations
        aggregated_metrics = {}
        
        if kind == "business":
            # Business metrics
            trade_facts = [f for f in period_facts if f.kind == "Fact:Trade"]
            outcome_facts = [f for f in period_facts if f.kind == "Fact:Outcome"]
            
            aggregated_metrics = {
                "total_trades": len(trade_facts),
                "total_outcomes": len(outcome_facts),
                "success_rate": len([f for f in outcome_facts if f.payload.get("success", False)]) / max(len(outcome_facts), 1),
            }
        else:
            # Technical metrics
            rpc_metrics = [m for m in period_metrics if m.name.startswith("rpc_")]
            
            if rpc_metrics:
                aggregated_metrics = {
                    "total_rpc_calls": len(rpc_metrics),
                    "avg_latency": sum(m.value for m in rpc_metrics if "latency" in m.name) / max(len([m for m in rpc_metrics if "latency" in m.name]), 1),
                    "error_rate": len([m for m in rpc_metrics if "error" in m.name]) / max(len(rpc_metrics), 1),
                }
        
        return AggregationResult(
            window=window,
            kind=kind,
            metrics=aggregated_metrics,
            facts_count=len(period_facts),
            period_start=start_time,
            period_end=end_time
        )
