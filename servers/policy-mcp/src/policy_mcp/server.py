"""Policy MCP Server - Decision profiles and policy management."""

from __future__ import annotations

import asyncio
from datetime import datetime
from functools import partial
from typing import Any, Dict, Optional

from fastapi import HTTPException, status
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .db import get_session, init_db
from .pydantic_models import DecisionProfile
from .services.profile_service import ProfileService
from .tools import register_tools


class ProfileUpsertRequest(BaseModel):
    """Payload for creating/updating profiles via HTTP API."""

    profile_id: str = Field(alias="profile_id")
    name: str
    description: str
    version: str = "1.0.0"
    active: Optional[bool] = True
    weights: Dict[str, float] = Field(default_factory=dict)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    risk_limits: Dict[str, float] = Field(default_factory=dict)
    trained_on_samples: Optional[int] = None
    confidence_score: Optional[float] = None
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    activate: bool = False

    class Config:
        populate_by_name = True


def _serialize_profile(profile: DecisionProfile) -> Dict[str, Any]:
    """Convert DecisionProfile into HTTP-friendly dict."""
    rules = profile.rules or {}
    metadata = dict(profile.metadata or {})
    return {
        "profile_id": profile.id,
        "name": profile.name,
        "description": profile.description,
        "version": profile.version,
        "active": profile.active,
        "weights": rules.get("weights", {}),
        "thresholds": rules.get("thresholds", {}),
        "risk_limits": rules.get("risk_limits", {}),
        "trained_on_samples": metadata.get("trained_on_samples"),
        "confidence_score": metadata.get("confidence_score"),
        "performance_metrics": metadata.get("performance_metrics", {}),
        "metadata": metadata,
        "created_at": profile.created_at.isoformat(),
        "updated_at": profile.updated_at.isoformat(),
    }


def _upsert_payload_to_profile(
    payload: ProfileUpsertRequest,
    existing: Optional[DecisionProfile],
) -> DecisionProfile:
    """Convert HTTP payload into DecisionProfile model."""
    created_at = payload.created_at or (existing.created_at if existing else None)
    updated_at = payload.updated_at or None

    metadata = dict(payload.metadata)
    if payload.trained_on_samples is not None:
        metadata.setdefault("trained_on_samples", payload.trained_on_samples)
    if payload.confidence_score is not None:
        metadata.setdefault("confidence_score", payload.confidence_score)
    if payload.performance_metrics:
        metadata.setdefault("performance_metrics", payload.performance_metrics)

    rules = {
        "weights": payload.weights,
        "thresholds": payload.thresholds,
        "risk_limits": payload.risk_limits,
    }

    profile = DecisionProfile(
        id=payload.profile_id,
        name=payload.name,
        description=payload.description,
        active=payload.active if payload.active is not None else (existing.active if existing else True),
        version=payload.version,
        rules=rules,
        metadata=metadata,
    )

    if created_at is not None:
        profile.created_at = created_at
    if updated_at is not None:
        profile.updated_at = updated_at

    return profile


def _register_http_routes(mcp: FastMCP, profile_service: ProfileService) -> None:
    """Attach REST endpoints to the underlying FastAPI app."""
    app = mcp.http_app

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Simple health endpoint with profile inventory summary."""
        profiles = await profile_service.list_profiles(active_only=True)
        return {"status": "ok", "active_profiles": len(profiles)}

    @app.get("/profiles")
    async def list_profiles(active_only: bool = False) -> Dict[str, Any]:
        profiles = await profile_service.list_profiles(active_only=active_only)
        return {"profiles": [_serialize_profile(p) for p in profiles]}

    @app.get("/profiles/{profile_id}")
    async def get_profile(profile_id: str) -> Dict[str, Any]:
        profile = await profile_service.get_profile(profile_id)
        if profile is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
        return {"profile": _serialize_profile(profile)}

    @app.get("/profiles/active")
    async def get_active_profile(profile_id: Optional[str] = None) -> Dict[str, Any]:
        profile = await profile_service.get_active_profile(profile_id=profile_id)
        if profile is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active profile not found")
        return {"profile": _serialize_profile(profile)}

    @app.post("/profiles")
    async def upsert_profile(payload: ProfileUpsertRequest) -> Dict[str, Any]:
        existing = await profile_service.get_profile(payload.profile_id)
        profile_model = _upsert_payload_to_profile(payload, existing)

        if existing:
            updated = await profile_service.update_profile(profile_model)
            assert updated is not None  # for mypy
        else:
            updated = await profile_service.create_profile(profile_model)

        if payload.activate:
            updated = await profile_service.activate_profile(updated.id) or updated

        return {"profile": _serialize_profile(updated)}

    @app.post("/profiles/{profile_id}/activate")
    async def activate_profile(profile_id: str) -> Dict[str, Any]:
        profile = await profile_service.activate_profile(profile_id)
        if profile is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
        return {"profile": _serialize_profile(profile)}

    @app.delete("/profiles/{profile_id}")
    async def delete_profile(profile_id: str) -> Dict[str, Any]:
        deleted = await profile_service.delete_profile(profile_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
        return {"success": True}

    @app.get("/profiles/{profile_id}/versions")
    async def list_versions(profile_id: str) -> Dict[str, Any]:
        versions = await profile_service.list_profile_versions(profile_id)
        return {"profile_id": profile_id, "versions": [v.model_dump() for v in versions]}

    @app.post("/profiles/{profile_id}/rollback")
    async def rollback_profile(profile_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        profile = await profile_service.rollback_profile(
            profile_id,
            version=payload.get("version"),
            version_id=payload.get("version_id"),
        )
        if profile is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")
        return {"profile": _serialize_profile(profile)}

    @app.post("/profiles/{profile_id}/experiment")
    async def configure_experiment(profile_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        profile = await profile_service.configure_experiment(
            profile_id=profile_id,
            experiment=payload.get("experiment_name"),
            weight=float(payload.get("weight", 0.5)),
            metadata=payload.get("metadata"),
        )
        if profile is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
        return {"profile": _serialize_profile(profile)}

    @app.get("/profiles/experiments")
    async def list_experiments() -> Dict[str, Any]:
        experiments = await profile_service.list_experiments()
        return {"experiments": experiments}


def create_server(settings: Optional[Settings] = None) -> FastMCP:
    """Create and configure the policy MCP server."""
    settings = settings or get_settings()
    mcp = FastMCP("policy-mcp")

    session_provider = partial(get_session, settings=settings)
    profile_service = ProfileService(session_provider=session_provider)

    register_tools(mcp, profile_service)
    _register_http_routes(mcp, profile_service)
    return mcp


def main() -> None:
    """Main entry point."""
    import sys

    settings = get_settings()
    asyncio.run(init_db(settings=settings))
    server = create_server(settings=settings)

    if "--stdio" in sys.argv:
        server.run_stdio()
        return

    # HTTP mode
    host = "0.0.0.0"
    port = 8000

    # Parse host and port from args
    for i, arg in enumerate(sys.argv):
        if arg == "--host" and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]
        elif arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])

    asyncio.run(server.run_http_async(host=host, port=port))


if __name__ == "__main__":
    main()
