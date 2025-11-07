"""Supervisor MCP client for HALv1 integration."""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional
import httpx
import structlog

logger = structlog.get_logger(__name__)


class SupervisorClient:
    """Client for sending metrics and facts to supervisor MCP."""
    
    def __init__(self, supervisor_url: str = "http://localhost:8001"):
        self.supervisor_url = supervisor_url
        self._http_client: Optional[httpx.AsyncClient] = None
        self._session_id = f"halv1_{int(time.time())}"
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=5.0)
        return self._http_client
    
    async def send_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Send a metric to supervisor."""
        try:
            client = await self._get_http_client()
            
            metric_data = {
                "name": name,
                "value": value,
                "tags": tags or {},
                "ts": (timestamp or datetime.now()).isoformat()
            }
            
            response = await client.post(
                f"{self.supervisor_url}/ingest/metric",
                json=metric_data
            )
            
            if response.status_code == 200:
                logger.debug("Metric sent successfully", metric=name, value=value)
                return True
            else:
                logger.warning("Failed to send metric", status=response.status_code)
                return False
                
        except Exception as e:
            logger.error("Error sending metric", error=str(e))
            return False
    
    async def send_fact(
        self,
        kind: str,
        actor: str,
        correlation_id: str,
        payload: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Send a fact to supervisor."""
        try:
            client = await self._get_http_client()
            
            fact_data = {
                "kind": kind,
                "actor": actor,
                "correlation_id": correlation_id,
                "payload": payload,
                "ts": (timestamp or datetime.now()).isoformat()
            }
            
            response = await client.post(
                f"{self.supervisor_url}/ingest/fact",
                json=fact_data
            )
            
            if response.status_code == 200:
                logger.debug("Fact sent successfully", kind=kind, actor=actor)
                return True
            else:
                logger.warning("Failed to send fact", status=response.status_code)
                return False
                
        except Exception as e:
            logger.error("Error sending fact", error=str(e))
            return False
    
    async def send_execution_metrics(
        self,
        operation: str,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Send execution metrics for an operation."""
        tags = {
            "operation": operation,
            "success": str(success).lower(),
            "session_id": self._session_id
        }
        
        if error:
            tags["error_type"] = type(error).__name__
        
        await self.send_metric("execution_duration_ms", duration_ms, tags)
        await self.send_metric("execution_success_rate", 1.0 if success else 0.0, tags)
        
        if error:
            await self.send_metric("execution_error_count", 1.0, tags)
    
    async def send_planning_fact(
        self,
        task: str,
        plan_steps: int,
        correlation_id: str
    ) -> None:
        """Send planning fact."""
        await self.send_fact(
            kind="Fact:Plan",
            actor="halv1",
            correlation_id=correlation_id,
            payload={
                "task": task,
                "plan_steps": plan_steps,
                "session_id": self._session_id
            }
        )
    
    async def send_execution_fact(
        self,
        step_id: str,
        tool: str,
        success: bool,
        correlation_id: str,
        result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send execution fact."""
        await self.send_fact(
            kind="Fact:Execution",
            actor="halv1",
            correlation_id=correlation_id,
            payload={
                "step_id": step_id,
                "tool": tool,
                "success": success,
                "result": result,
                "session_id": self._session_id
            }
        )
    
    async def send_decision_fact(
        self,
        decision_type: str,
        confidence: float,
        correlation_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send decision fact."""
        await self.send_fact(
            kind="Fact:Decision",
            actor="halv1",
            correlation_id=correlation_id,
            payload={
                "decision_type": decision_type,
                "confidence": confidence,
                "context": context or {},
                "session_id": self._session_id
            }
        )
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
