"""Async client for interacting with Policy MCP HTTP API."""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from ..config import get_settings
from ..models import DecisionProfile


class PolicyClient:
    """Thin async client around the Policy MCP REST interface."""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        settings = get_settings()
        self.base_url = base_url or settings.policy_url.rstrip("/")
        self.timeout = timeout or settings.policy_timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Dispose underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def upsert_profile(self, profile: DecisionProfile, activate: bool = False) -> Dict[str, Any]:
        """Create or update a decision profile in Policy MCP."""
        payload = {
            "profile_id": profile.profile_id,
            "name": profile.name,
            "description": profile.description,
            "version": profile.version,
            "weights": profile.weights,
            "thresholds": profile.thresholds,
            "risk_limits": profile.risk_limits,
            "trained_on_samples": profile.trained_on_samples,
            "confidence_score": profile.confidence_score,
            "performance_metrics": profile.performance_metrics,
            "metadata": {},
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.created_at.isoformat(),
            "activate": activate,
        }

        client = await self._get_client()
        response = await client.post(f"{self.base_url}/profiles", json=payload)
        response.raise_for_status()
        return response.json()

    async def activate_profile(self, profile_id: str) -> Dict[str, Any]:
        """Activate a profile by ID."""
        client = await self._get_client()
        response = await client.post(f"{self.base_url}/profiles/{profile_id}/activate")
        response.raise_for_status()
        return response.json()

    async def get_active_profile(self, profile_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch the currently active profile (optionally filtered by ID)."""
        client = await self._get_client()
        params = {}
        if profile_id:
            params["profile_id"] = profile_id
        response = await client.get(f"{self.base_url}/profiles/active", params=params or None)
        response.raise_for_status()
        return response.json()

    async def list_profiles(self, active_only: bool = False) -> Dict[str, Any]:
        """List profiles stored in Policy MCP."""
        client = await self._get_client()
        params = {"active_only": "true"} if active_only else None
        response = await client.get(f"{self.base_url}/profiles", params=params)
        response.raise_for_status()
        return response.json()

    async def health(self) -> Dict[str, Any]:
        """Check Policy MCP health endpoint."""
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
