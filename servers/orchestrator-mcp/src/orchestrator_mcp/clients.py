"""Client wrappers for external MCP services used by orchestrator."""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from .config import get_settings


class BaseClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class SupervisorClient(BaseClient):
    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(settings.supervisor_url, settings.default_timeout)

    async def fetch_health(self) -> Dict[str, Any]:
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


class PolicyClient(BaseClient):
    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(settings.policy_url, settings.default_timeout)

    async def check_profile(self, profile_id: str) -> Dict[str, Any]:
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/profiles/{profile_id}")
        response.raise_for_status()
        return response.json()

    async def activate_profile(self, profile_id: str) -> Dict[str, Any]:
        client = await self._get_client()
        response = await client.post(f"{self.base_url}/profiles/{profile_id}/activate")
        response.raise_for_status()
        return response.json()

    async def list_profiles(self, active_only: bool = False) -> Dict[str, Any]:
        client = await self._get_client()
        params = {"active_only": "true"} if active_only else None
        response = await client.get(f"{self.base_url}/profiles", params=params)
        response.raise_for_status()
        return response.json()

    async def configure_experiment(
        self, profile_id: str, name: str, weight: float, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/profiles/{profile_id}/experiment",
            json={"experiment_name": name, "weight": weight, "metadata": metadata or {}},
        )
        response.raise_for_status()
        return response.json()

    async def list_experiments(self) -> Dict[str, Any]:
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/profiles/experiments")
        response.raise_for_status()
        return response.json()


class LearningClient(BaseClient):
    def __init__(self) -> None:
        settings = get_settings()
        base = settings.learning_url if hasattr(settings, "learning_url") else "http://localhost:8003"
        super().__init__(base, settings.default_timeout)

    async def trigger_online_learning(self) -> Dict[str, Any]:
        client = await self._get_client()
        response = await client.post(f"{self.base_url}/orchestrator/trigger")
        response.raise_for_status()
        return response.json()

    async def list_policy_profiles(self) -> Dict[str, Any]:
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/orchestrator/policy-profiles")
        response.raise_for_status()
        return response.json()
