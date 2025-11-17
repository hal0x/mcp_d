"""MCP tools for Policy MCP."""

from __future__ import annotations

from typing import Optional

from fastmcp import FastMCP

from ..pydantic_models import DecisionProfile, DecisionRequest
from ..services.profile_service import ProfileService


def register_tools(mcp: FastMCP, profile_service: ProfileService) -> None:
    """Register all MCP tools."""

    @mcp.tool()
    async def create_profile(
        profile_id: str,
        name: str,
        description: str,
        rules: Optional[dict] = None,
        metadata: Optional[dict] = None,
        version: str = "1.0.0",
        active: bool = True,
    ) -> dict:
        """Create a new decision profile."""
        profile = DecisionProfile(
            id=profile_id,
            name=name,
            description=description,
            rules=rules or {},
            metadata=metadata or {},
            version=version,
            active=active,
        )
        result = await profile_service.create_profile(profile)
        return result.model_dump()

    @mcp.tool()
    async def get_profile(profile_id: str) -> dict:
        """Get a decision profile by ID."""
        profile = await profile_service.get_profile(profile_id)
        if not profile:
            return {"error": "Profile not found"}
        return profile.model_dump()

    @mcp.tool()
    async def get_active_profile(profile_id: Optional[str] = None) -> dict:
        """Get the latest active profile (optionally filtered by ID)."""
        profile = await profile_service.get_active_profile(profile_id=profile_id)
        if not profile:
            return {"error": "Active profile not found"}
        return profile.model_dump()

    @mcp.tool()
    async def list_profiles(active_only: bool = False) -> dict:
        """List all decision profiles."""
        profiles = await profile_service.list_profiles(active_only=active_only)
        return {
            "profiles": [p.model_dump() for p in profiles],
            "count": len(profiles),
        }

    @mcp.tool()
    async def update_profile(
        profile_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        rules: Optional[dict] = None,
        metadata: Optional[dict] = None,
        version: Optional[str] = None,
        active: Optional[bool] = None,
    ) -> dict:
        """Update a decision profile."""
        profile = await profile_service.get_profile(profile_id)
        if not profile:
            return {"error": "Profile not found"}

        if name is not None:
            profile.name = name
        if description is not None:
            profile.description = description
        if rules is not None:
            profile.rules = rules
        if metadata is not None:
            profile.metadata = metadata
        if version is not None:
            profile.version = version
        if active is not None:
            profile.active = active

        updated = await profile_service.update_profile(profile)
        if updated is None:
            return {"error": "Profile not found"}
        return updated.model_dump()

    @mcp.tool()
    async def activate_profile(profile_id: str) -> dict:
        """Activate a decision profile."""
        profile = await profile_service.activate_profile(profile_id)
        if not profile:
            return {"error": "Profile not found"}
        return profile.model_dump()

    @mcp.tool()
    async def delete_profile(profile_id: str) -> dict:
        """Delete a decision profile."""
        success = await profile_service.delete_profile(profile_id)
        return {"success": success}

    @mcp.tool()
    async def list_profile_versions(profile_id: str) -> dict:
        """List historical versions for a profile."""
        versions = await profile_service.list_profile_versions(profile_id)
        return {"profile_id": profile_id, "versions": [v.model_dump() for v in versions]}

    @mcp.tool()
    async def rollback_profile(profile_id: str, version: str = None, version_id: int = None) -> dict:
        """Rollback profile to a previous version."""
        try:
            profile = await profile_service.rollback_profile(
                profile_id, version=version, version_id=version_id
            )
            if not profile:
                return {"error": "Version not found"}
            return {"success": True, "profile": profile.model_dump()}
        except ValueError as exc:
            return {"error": str(exc)}

    @mcp.tool()
    async def configure_profile_experiment(
        profile_id: str,
        experiment_name: str,
        weight: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Attach experiment metadata to a profile for A/B testing."""
        profile = await profile_service.configure_experiment(
            profile_id, experiment_name, weight, metadata
        )
        if not profile:
            return {"error": "Profile not found"}
        return profile.model_dump()

    @mcp.tool()
    async def list_profile_experiments() -> dict:
        """List A/B experiment configuration across profiles."""
        experiments = await profile_service.list_experiments()
        return {"experiments": experiments}

    @mcp.tool()
    async def evaluate_decision(
        profile_id: str,
        context: dict,
        actor: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> dict:
        """Evaluate a decision request under a policy."""
        request = DecisionRequest(
            profile_id=profile_id,
            context=context,
            actor=actor,
            correlation_id=correlation_id,
        )
        result = await profile_service.evaluate(request)
        return result.model_dump()
